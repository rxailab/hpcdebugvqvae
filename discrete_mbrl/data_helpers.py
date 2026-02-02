import gzip
import hashlib
import os
import pickle
import sys
import time
import platform
import multiprocessing
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, TensorDataset
from tqdm import tqdm

from .utils import to_hashable_tensor_list

# Define constants locally to avoid import issues
DATA_DIR = './data'
MUJOCO_ENVS = [
    'Ant-v4', 'HalfCheetah-v4', 'Hopper-v4', 'Humanoid-v4',
    'Reacher-v4', 'Walker2d-v4'
]

MAX_TEST = 50_000
MAX_VALID = 5000
CACHE_DIR = 'data/cache/'

TEST_FRAC = 0.1
VALID_FRAC = 0.01


# ----------------------------
# HPC/SLURM/cgroup safe workers
# ----------------------------

def _read_int_env(*keys, default=None):
    for k in keys:
        v = os.environ.get(k, None)
        if v is None:
            continue
        try:
            return int(v)
        except ValueError:
            pass
    return default


def _cgroup_cpu_quota():
    """
    Returns an integer CPU quota if inferable from cgroups (Linux),
    otherwise None. Supports many cgroup v1/v2 setups used on HPC/container nodes.
    """
    try:
        # cgroup v2: /sys/fs/cgroup/cpu.max -> "<quota> <period>" or "max <period>"
        p = Path("/sys/fs/cgroup/cpu.max")
        if p.exists():
            quota, period = p.read_text().strip().split()
            if quota != "max":
                quota = int(quota)
                period = int(period)
                if period > 0:
                    return max(1, quota // period)

        # cgroup v1: cpu.cfs_quota_us / cpu.cfs_period_us
        quota_p = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period_p = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota_p.exists() and period_p.exists():
            quota = int(quota_p.read_text().strip())
            period = int(period_p.read_text().strip())
            if quota > 0 and period > 0:
                return max(1, quota // period)
    except Exception:
        return None
    return None


def get_safe_num_workers(requested: Optional[int], prefer: int = 1) -> int:
    """
    Decide a safe num_workers for DataLoader on the current machine.

    - On Windows: always 0.
    - On Linux/HPC: clamp to the minimum of visible CPUs, SLURM allocation,
      OMP/MKL hints, and cgroup quota.
    - If requested is None: return a small number (prefer) but not exceeding max_allowed.
    """
    if platform.system() == "Windows":
        return 0

    # Visible CPUs to the process
    try:
        visible_cpus = len(os.sched_getaffinity(0))
    except Exception:
        visible_cpus = multiprocessing.cpu_count()

    # Scheduler/cgroup limits (often smaller)
    slurm_cpus = _read_int_env("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE")
    omp_cpus = _read_int_env("OMP_NUM_THREADS", "MKL_NUM_THREADS")
    cgroup_cpus = _cgroup_cpu_quota()

    limits = [visible_cpus]
    for x in (slurm_cpus, omp_cpus, cgroup_cpus):
        if x is not None and x > 0:
            limits.append(x)

    max_allowed = max(1, min(limits))

    if requested is None:
        return min(prefer, max_allowed)

    return max(0, min(int(requested), max_allowed))


# ----------------------------
# Picklable collate function
# ----------------------------

def collate_fn_identity(x):
    """Picklable collate function to replace lambda x: x[0]"""
    return x[0]


# ----------------------------
# Cache helpers
# ----------------------------

def construct_cache_path(env_name, preprocess, randomize, seed, n):
    name = DATA_DIR + '/' + env_name
    name += '_' + str(preprocess)
    name += '_' + str(randomize)
    name += '_' + str(seed)
    if n is not None:
        name += '_' + str(n)
    name += '.pkl.gz'
    return name


def load_cache(env_name, preprocess, randomize, seed, n):
    cache_path = construct_cache_path(env_name, preprocess, randomize, seed, n)
    with gzip.open(cache_path, 'rb') as f:
        return pickle.load(f)


def save_cache(env_name, preprocess, randomize, seed, n, cache):
    cache_path = construct_cache_path(env_name, preprocess, randomize, seed, n)
    with gzip.open(cache_path, 'wb') as f:
        pickle.dump(cache, f)


# ----------------------------
# Dataset utilities
# ----------------------------

class Subset(Dataset):
    """
    Custom Subset that allows accessing the original dataset more easily.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name.startswith('_'):
            raise AttributeError(f'accessing private attribute "{name}" is prohibited')
        return getattr(self.dataset, name)


class ObsTransforms:
    """Picklable class to handle observation transformations"""

    def __init__(self, obs_mean=None, obs_std=None):
        if obs_mean is not None and obs_std is not None:
            self.flat_obs_mean = obs_mean
            self.obs_mean = obs_mean.unsqueeze(0) if len(obs_mean.shape) > 0 else obs_mean
            self.flat_obs_std = obs_std
            self.obs_std = obs_std.unsqueeze(0) if len(obs_std.shape) > 0 else obs_std
            self.has_normalization = True
        else:
            self.flat_obs_mean = None
            self.obs_mean = None
            self.flat_obs_std = None
            self.obs_std = None
            self.has_normalization = False

    def obs_transform(self, o):
        if self.has_normalization:
            return (o - self.obs_mean) / self.obs_std
        return o

    def flat_obs_transform(self, o):
        if self.has_normalization:
            return (o - self.flat_obs_mean) / self.flat_obs_std
        return o

    def rev_obs_transform(self, o):
        if self.has_normalization:
            return o * self.obs_std + self.obs_mean
        return o

    def flat_rev_obs_transform(self, o):
        if self.has_normalization:
            return o * self.flat_obs_std + self.flat_obs_mean
        return o


class NStepReplayDataset(Dataset):
    def __init__(self, env_name, n_steps, transform=None, preload=False,
                 start_idx=0, end_idx=None, extra_buffer_keys=None, cross_chunks=True):
        sanitized_env_name = env_name.replace(':', '_')
        extra_buffer_keys = extra_buffer_keys or []
        self.replay_buffer_path = f'{DATA_DIR}/{sanitized_env_name}_replay_buffer.hdf5'
        self.transform = transform
        self.preload = preload
        self.n_steps = n_steps
        self.cross_chunks = cross_chunks

        with h5py.File(self.replay_buffer_path, 'r') as buffer:
            self.act_type = buffer.get('action').dtype
            self.act_type = torch.float32 if 'float' in str(self.act_type) else torch.int64
            self.chunk_size = buffer['obs'].chunks[0]
            self.data_keys = list(sorted(buffer.keys()))
            self.extra_keys = set(self.data_keys) - \
                              set(['obs', 'action', 'next_obs', 'reward', 'done'])
            self.extra_keys = self.extra_keys.intersection(set(extra_buffer_keys or []))
            unused_keys = self.extra_keys - set(extra_buffer_keys or [])
            if len(unused_keys) > 0:
                print(f'Warning: unused keys in replay buffer: {unused_keys}')

            if buffer.attrs.get('obs_mean') is not None:
                obs_mean = torch.from_numpy(buffer.attrs['obs_mean'])
                obs_std = torch.from_numpy(buffer.attrs['obs_std'])
                self.obs_transforms = ObsTransforms(obs_mean, obs_std)
            else:
                self.obs_transforms = ObsTransforms()

            self.extra_keys = list(sorted(self.extra_keys))

            if preload:
                if end_idx is None:
                    ei = buffer['obs'].shape[0]
                else:
                    ei = end_idx + self.n_steps - 1
                self.data_buffer = {}
                self.data_buffer['obs'] = torch.from_numpy(
                    buffer['obs'][start_idx:ei]).float()
                self.data_buffer['action'] = torch.tensor(
                    buffer['action'][start_idx:ei]).to(self.act_type)
                self.data_buffer['next_obs'] = torch.from_numpy(
                    buffer['next_obs'][start_idx:ei]).float()
                self.data_buffer['reward'] = torch.tensor(
                    buffer['reward'][start_idx:ei]).float()
                self.data_buffer['done'] = torch.tensor(
                    buffer['done'][start_idx:ei]).float()

                for key in self.extra_keys:
                    self.data_buffer[key] = torch.from_numpy(
                        buffer[key][start_idx:ei])

                self.n_samples = self.data_buffer['obs'].shape[0]
            else:
                self.n_samples = int(buffer.attrs['data_idx'])

        if cross_chunks:
            self.length = self.n_samples - self.n_steps + 1
        else:
            n_full_chunks = self.n_samples // self.chunk_size
            samples_per_chunk = self.chunk_size - self.n_steps + 1
            extra_samples = max(0,
                                (self.n_samples % self.chunk_size) - self.n_steps + 1)
            self.length = n_full_chunks * samples_per_chunk + extra_samples

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.cross_chunks:
            return self.__getitem_cross_chunks(idx)
        return self.__getitem_no_cross_chunks(idx)

    def __getitem_cross_chunks(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(idx, int):
            idx = [idx]
        if isinstance(idx, (slice, range)):
            idx = list(range(idx.start, idx.stop, idx.step))

        sample_idxs = []
        for i in idx:
            sample_idxs.extend(range(i, i + self.n_steps))

        if self.preload:
            obs = self.obs_transforms.obs_transform(self.data_buffer['obs'][sample_idxs])
            action = self.data_buffer['action'][sample_idxs]
            next_obs = self.obs_transforms.obs_transform(self.data_buffer['next_obs'][sample_idxs])
            reward = self.data_buffer['reward'][sample_idxs]
            done = self.data_buffer['done'][sample_idxs]
            extra_data = [self.data_buffer[key][sample_idxs] for key in self.extra_keys]
        else:
            with h5py.File(self.replay_buffer_path, 'r') as buffer:
                obs = self.obs_transforms.obs_transform(torch.from_numpy(buffer['obs'][sample_idxs]).float())
                action = torch.tensor(buffer['action'][sample_idxs]).to(self.act_type)
                next_obs = self.obs_transforms.obs_transform(torch.from_numpy(buffer['next_obs'][sample_idxs]).float())
                reward = torch.tensor(buffer['reward'][sample_idxs]).float()
                done = torch.tensor(buffer['done'][sample_idxs]).float()
                extra_data = [torch.tensor(buffer[key][sample_idxs]) for key in self.extra_keys]

        transition_set = [obs, action, next_obs, reward, done, *extra_data]

        if self.transform:
            transition_set = self.transform(*transition_set)

        transition_set = [
            x.reshape(int(x.shape[0] / self.n_steps), self.n_steps, *x.shape[1:]).squeeze(0)
            for x in transition_set
        ]
        return transition_set

    def __getitem_no_cross_chunks(self, idx):
        raise NotImplementedError("No cross chunks mode not implemented yet")

    @property
    def obs_transform(self):
        return self.obs_transforms.obs_transform

    @property
    def flat_obs_transform(self):
        return self.obs_transforms.flat_obs_transform

    @property
    def rev_obs_transform(self):
        return self.obs_transforms.rev_obs_transform

    @property
    def flat_rev_obs_transform(self):
        return self.obs_transforms.flat_rev_obs_transform


class ReplayDataset(Dataset):
    def __init__(self, env_name, transform=None, preload=False,
                 start_idx=0, end_idx=None, extra_buffer_keys=None):
        sanitized_env_name = env_name.replace(':', '_')
        extra_buffer_keys = extra_buffer_keys or []
        self.replay_buffer_path = f'{DATA_DIR}/{sanitized_env_name}_replay_buffer.hdf5'
        self.transform = transform
        self.preload = preload

        with h5py.File(self.replay_buffer_path, 'r') as buffer:
            self.act_type = buffer.get('action').dtype
            self.act_type = torch.float32 if 'float' in str(self.act_type) else torch.int64
            self.data_keys = list(sorted(buffer.keys()))
            self.extra_keys = set(self.data_keys) - \
                              set(['obs', 'action', 'next_obs', 'reward', 'done'])
            self.extra_keys = self.extra_keys.intersection(set(extra_buffer_keys))
            unused_keys = self.extra_keys - set(extra_buffer_keys)
            if len(unused_keys) > 0:
                print(f'Warning: unused keys in replay buffer: {unused_keys}')

            if buffer.attrs.get('obs_mean') is not None:
                obs_mean = torch.from_numpy(buffer.attrs['obs_mean'])
                obs_std = torch.from_numpy(buffer.attrs['obs_std'])
                self.obs_transforms = ObsTransforms(obs_mean, obs_std)
            else:
                self.obs_transforms = ObsTransforms()

            self.extra_keys = list(sorted(self.extra_keys))

            if preload:
                if end_idx is None:
                    end_idx = buffer['obs'].shape[0]
                self.data_buffer = {}
                self.data_buffer['obs'] = torch.from_numpy(
                    buffer['obs'][start_idx:end_idx]).float()
                self.data_buffer['action'] = torch.tensor(
                    buffer['action'][start_idx:end_idx]).to(self.act_type)
                self.data_buffer['next_obs'] = torch.from_numpy(
                    buffer['next_obs'][start_idx:end_idx]).float()
                self.data_buffer['reward'] = torch.tensor(
                    buffer['reward'][start_idx:end_idx]).float()
                self.data_buffer['done'] = torch.tensor(
                    buffer['done'][start_idx:end_idx]).float()

                for key in self.extra_keys:
                    self.data_buffer[key] = torch.from_numpy(
                        buffer[key][start_idx:end_idx])

                self.length = self.data_buffer['obs'].shape[0]
            else:
                self.length = int(buffer.attrs['data_idx'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Handles numpy scalar/array conversion issues robustly.
        """
        with h5py.File(self.replay_buffer_path, 'r') as buffer:
            if isinstance(idx, (np.bool_, bool, np.integer)):
                idx = int(idx)

            obs = torch.tensor(buffer['obs'][idx]).float()

            action_data = buffer['action'][idx]
            if buffer['action'].dtype in ['int32', 'int64']:
                action = torch.tensor(action_data).long()
            else:
                action = torch.tensor(action_data).float()

            next_obs = torch.tensor(buffer['next_obs'][idx]).float()
            reward = torch.tensor(buffer['reward'][idx]).float()

            done_data = buffer['done'][idx]

            def safe_convert_to_float(data):
                try:
                    if isinstance(data, (np.bool_, bool)):
                        return float(data)

                    if hasattr(data, 'dtype') and data.dtype == bool:
                        float_data = data.astype(float)
                        if np.isscalar(float_data):
                            return float(float_data)
                        if float_data.size >= 1:
                            return float(float_data.flat[0])
                        return 0.0

                    if np.isscalar(data):
                        return float(data)

                    if hasattr(data, '__len__'):
                        if len(data) == 0:
                            return 0.0
                        if hasattr(data, 'flat'):
                            return float(data.flat[0])
                        return float(data[0])

                    return float(data)
                except (TypeError, ValueError, IndexError) as e:
                    print(f"Warning: Could not convert {type(data)} to float: {e}")
                    print(f"Data: {data}")
                    return 0.0

            done = torch.tensor(safe_convert_to_float(done_data))

            extra_data = []
            for key in self.extra_keys:
                if key in buffer:
                    value = buffer[key][idx]
                    extra_data.append(torch.tensor(safe_convert_to_float(value)))

            transition_set = [obs, action, next_obs, reward, done, *extra_data]

            if self.transform:
                transition_set = self.transform(*transition_set)

            return transition_set

    @property
    def obs_transform(self):
        return self.obs_transforms.obs_transform

    @property
    def flat_obs_transform(self):
        return self.obs_transforms.flat_obs_transform

    @property
    def rev_obs_transform(self):
        return self.obs_transforms.rev_obs_transform

    @property
    def flat_rev_obs_transform(self):
        return self.obs_transforms.flat_rev_obs_transform


# ----------------------------
# Samplers
# ----------------------------

class NStepWeakBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, n_steps, shuffle=False):
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.block_size = self.n_steps * self.batch_size
        self.dataset_length = len(dataset)
        self.n_batches = int(np.ceil(self.dataset_length / self.block_size)) * self.n_steps
        self.shuffle = shuffle
        self.batch_ids = torch.arange(self.n_batches)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(self.n_batches)
        for id in self.batch_ids:
            block_idx = id // self.n_steps
            idx = slice(block_idx * self.block_size + id % self.n_steps,
                        min((block_idx + 1) * self.block_size, self.dataset_length), self.n_steps)
            yield idx


class WeakBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = int(np.ceil(self.dataset_length / self.batch_size))
        self.shuffle = shuffle
        self.batch_ids = torch.arange(self.n_batches)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(self.n_batches)
        for id in self.batch_ids:
            idx_slice = slice(id * self.batch_size,
                              min((id + 1) * self.batch_size, self.dataset_length))
            yield idx_slice


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.shuffle = shuffle
        self.batch_ids = torch.randperm(self.dataset_length)
        self.n_batches = int(np.ceil(self.dataset_length / self.batch_size))

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        if self.shuffle:
            self.batch_ids = torch.randperm(self.dataset_length)
        for i in range(0, self.dataset_length, self.batch_size):
            idxs = self.batch_ids[i:i + self.batch_size]
            idxs = torch.sort(idxs)[0]
            yield idxs


# ----------------------------
# Transforms
# ----------------------------

def preprocess_transform(obs, action, next_obs, reward, done):
    obs[0] /= 255.0
    next_obs /= 255.0
    permute = min(obs[0].shape) != obs[0].shape[0]
    if permute:
        obs = obs.permute(0, 3, 1, 2)
        next_obs = next_obs.permute(0, 3, 1, 2)
    return obs, action, next_obs, reward, done


# ----------------------------
# DataLoader creation
# ----------------------------

def create_fast_loader(
        dataset, batch_size, shuffle=False, num_workers=0,
        n_step=1, weak_shuffle=False, pin_memory=None,
        persistent_workers=None, prefetch_factor=2, drop_last=False):
    """
    Optimized data loader with HPC-safe worker selection and HDF5-friendly defaults.
    """

    # Clamp workers safely. For HDF5 workloads, prefer=1 is usually best.
    num_workers = get_safe_num_workers(num_workers, prefer=1)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False

    sampler = None
    if weak_shuffle:
        sampler = WeakBatchSampler(dataset, batch_size, shuffle) if n_step <= 1 \
            else NStepWeakBatchSampler(dataset, batch_size, n_step, shuffle)

        dataloader_kwargs = {
            'dataset': dataset,
            'num_workers': num_workers,
            'collate_fn': collate_fn_identity,
            'sampler': sampler,
            'pin_memory': pin_memory,
            'persistent_workers': persistent_workers
        }

        if num_workers > 0 and prefetch_factor is not None:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor

        return DataLoader(**dataloader_kwargs)

    dataloader_kwargs = {
        'dataset': dataset,
        'num_workers': num_workers,
        'drop_last': drop_last,
        'shuffle': shuffle,
        'batch_size': batch_size,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers
    }

    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor

    return DataLoader(**dataloader_kwargs)


def prepare_dataloaders(env_name, batch_size=256, randomize=True, n_step=1,
                        preprocess=False, n=None, n_preload=0, seed=None,
                        valid_preload=True, preload_all=False, extra_buffer_keys=None,
                        pin_memory=None, persistent_workers=None, prefetch_factor=2,
                        num_workers: Optional[int] = None):
    """
    Prepare train/test/valid dataloaders with HPC-safe multiprocessing.

    - If num_workers is None: uses n_preload (legacy), clamped safely.
    - On Windows: forces workers=0.
    """
    extra_buffer_keys = extra_buffer_keys or []
    transform = preprocess_transform if preprocess else None

    if platform.system() == 'Windows':
        n_preload = 0
        num_workers = 0
        persistent_workers = False
        print("Windows detected - disabling multiprocessing to avoid pickle errors")

    if num_workers is None:
        num_workers = n_preload
    num_workers = get_safe_num_workers(num_workers, prefer=1)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if n_step > 1:
        dataset = NStepReplayDataset(
            env_name, n_steps=n_step, transform=transform, preload=preload_all,
            extra_buffer_keys=extra_buffer_keys)
    else:
        dataset = ReplayDataset(
            env_name, transform, preload=preload_all,
            extra_buffer_keys=extra_buffer_keys)

    if n and n < len(dataset):
        dataset.length = min(dataset.length, n)

    n_total = len(dataset)
    n_test = min(int(n_total * TEST_FRAC), MAX_TEST)
    n_valid = min(int(n_total * VALID_FRAC), MAX_VALID)
    n_train = n_total - n_test - n_valid

    train_dataset = Subset(dataset, torch.arange(n_train))
    test_dataset = Subset(dataset, torch.arange(n_train, n_total - n_valid))

    if valid_preload and not preload_all:
        if n_step > 1:
            valid_dataset = NStepReplayDataset(
                env_name, n_steps=n_step, transform=transform,
                preload=True, start_idx=n_total - n_valid, end_idx=n_total,
                extra_buffer_keys=extra_buffer_keys)
        else:
            valid_dataset = ReplayDataset(
                env_name, transform, preload=True,
                start_idx=n_total - n_valid, end_idx=n_total,
                extra_buffer_keys=extra_buffer_keys)
    else:
        valid_dataset = Subset(dataset, torch.arange(n_total - n_valid, n_total))

    if seed is not None and randomize:
        torch.manual_seed(seed)
        np.random.seed(seed)

    weak_shuffle = not preload_all
    valid_weak_shuffle = not valid_preload and not preload_all

    train_loader = create_fast_loader(
        train_dataset, batch_size, randomize, num_workers, n_step, weak_shuffle,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor, drop_last=True)

    test_loader = create_fast_loader(
        test_dataset, batch_size, False, num_workers, n_step, weak_shuffle,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor)

    valid_loader = create_fast_loader(
        valid_dataset, batch_size, False, min(num_workers, 2), n_step, valid_weak_shuffle,
        pin_memory=pin_memory, persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor)

    return train_loader, test_loader, valid_loader


def prepare_unique_obs_dataloader(args, randomize=True, preprocess=False, seed=None):
    if seed is not None and randomize:
        torch.manual_seed(seed)
        np.random.seed(seed)

    transform = preprocess_transform if preprocess else None
    unique_obs = get_unique_obs(args, cache=True, partition='all')
    dataset = TensorDataset(unique_obs)

    # Use safe worker selection
    requested = getattr(args, 'n_preload', 0)
    num_workers = get_safe_num_workers(requested, prefer=0)

    dataloader_kwargs = {
        'dataset': dataset,
        'batch_size': args.batch_size,
        'shuffle': randomize,
        'drop_last': False,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': False
    }
    if num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 2

    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader


def prepare_dataloader(env_name, partition, batch_size=256, randomize=True, n_step=1,
                       preprocess=False, n=None, n_preload=0, preload=False, seed=None,
                       extra_buffer_keys=None):
    extra_buffer_keys = extra_buffer_keys or []

    if platform.system() == 'Windows':
        n_preload = 0

    transform = preprocess_transform if preprocess else None
    if n_step > 1:
        dataset = NStepReplayDataset(
            env_name, n_steps=n_step, transform=transform,
            extra_buffer_keys=extra_buffer_keys)
    else:
        dataset = ReplayDataset(env_name, transform, extra_buffer_keys=extra_buffer_keys)

    if n and n < len(dataset):
        dataset.length = min(dataset.length, n)

    n_total = len(dataset)
    n_test = min(int(n_total * TEST_FRAC), MAX_TEST)
    n_valid = min(int(n_total * VALID_FRAC), MAX_VALID)
    n_train = n_total - n_test - n_valid

    partition_map = {
        'all': (0, n_total),
        'train': (0, n_train),
        'test': (n_train, n_total - n_valid),
        'valid': (n_total - n_valid, n_total),
    }
    start_idx, end_idx = partition_map[partition]

    if n_step > 1:
        dataset = NStepReplayDataset(
            env_name, n_steps=n_step, transform=transform, preload=preload,
            start_idx=start_idx, end_idx=end_idx, extra_buffer_keys=extra_buffer_keys)
    else:
        if preload:
            dataset = ReplayDataset(env_name, transform, preload=True,
                                    start_idx=start_idx, end_idx=end_idx,
                                    extra_buffer_keys=extra_buffer_keys)
        else:
            dataset = Subset(dataset, np.arange(start_idx, end_idx))

    if seed is not None and randomize:
        torch.manual_seed(seed)
        np.random.seed(seed)

    weak_shuffle = not preload
    num_workers = get_safe_num_workers(n_preload, prefer=1)

    dataloader = create_fast_loader(
        dataset, batch_size, randomize, num_workers, n_step, weak_shuffle)

    return dataloader


# ----------------------------
# Legacy pickle buffer loader
# ----------------------------

def load_data_buffer(env_name, preprocess=True, randomize=True, seed=0,
                     n=None, cache=True):
    if cache:
        cache_path = construct_cache_path(env_name, preprocess, randomize, seed, n)
        if os.path.exists(cache_path):
            print('Found cache, loading...')
            return load_cache(env_name, preprocess, randomize, seed, n)
    else:
        print('Data caching disabled')

    sanitized_env_name = env_name.replace(':', '_')
    replay_buffer_path = f'{DATA_DIR}/{sanitized_env_name}_replay_buffer.pkl.gz'
    with gzip.open(replay_buffer_path, 'rb') as f:
        replay_buffer = pickle.load(f)

    print('Replay buffer size:', len(replay_buffer),
          sys.getsizeof(replay_buffer[0]) * sys.getsizeof(replay_buffer))
    if n:
        replay_buffer = replay_buffer[:n]
        print('Truncated replay buffer size:', len(replay_buffer),
              sys.getsizeof(replay_buffer[0]) * sys.getsizeof(replay_buffer))

    print('Stacking data..')
    transition_data = [np.stack([x[i] for x in replay_buffer]) for i in range(len(replay_buffer[0]))]
    transition_data = [torch.from_numpy(x).float() for x in transition_data]
    transition_data[1] = transition_data[1].long()
    del replay_buffer

    if preprocess:
        transition_data[0] = (transition_data[0] / 255)
        transition_data[2] = (transition_data[2] / 255)
        permute = min(transition_data[0][0].shape) != transition_data[0][0].shape[0]
        if permute:
            transition_data[0] = transition_data[0].permute(0, 3, 1, 2)
            transition_data[2] = transition_data[2].permute(0, 3, 1, 2)

    if randomize:
        print('Randomizing data..')
        torch.manual_seed(seed)
        rand_idxs = torch.randperm(transition_data[0].shape[0])
        transition_data = [x[rand_idxs] for x in transition_data]

    if cache:
        print('Saving cache...')
        save_cache(env_name, preprocess, randomize, seed, n, transition_data)

    return transition_data


# ----------------------------
# Unique obs cache helpers
# ----------------------------

def get_md5(path, max_bytes=2 ** 20, extra_data=None):
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        md5.update(f.read(max_bytes))  # Only get the first 1MB
    if extra_data is not None:
        if isinstance(extra_data, Iterable):
            extra_data = ''.join([str(x) for x in extra_data])
        md5.update(str(extra_data).encode('utf-8'))
    return md5.hexdigest()


def get_unique_obs(args, cache=True, partition='all', early_stop_frac=1.0,
                   return_hash=False):
    if cache:
        replay_buffer_path = ReplayDataset(args.env_name).replay_buffer_path
        extra_data = []
        if getattr(args, "max_transitions", None) is not None:
            extra_data.append(args.max_transitions)
        if partition is not None:
            extra_data.append(partition)

        unique_data_hash = get_md5(replay_buffer_path, extra_data=extra_data)
        unique_data_path = os.path.join(CACHE_DIR, f'{unique_data_hash}.pkl')
        if os.path.exists(unique_data_path):
            print(f'Loading cached unique data from {unique_data_path}...')
            time_since_update = time.time() - os.path.getmtime(unique_data_path)
            if time_since_update < 60:
                time.sleep(max(0, 60 - time_since_update))
            with open(unique_data_path, 'rb') as f:
                unique_obs = pickle.load(f)
            print(f'{len(unique_obs)} unique observations loaded!')
            if return_hash:
                return unique_obs, unique_data_hash
            return unique_obs

    dataloader = prepare_dataloader(
        args.env_name, partition, batch_size=args.batch_size,
        preprocess=getattr(args, "preprocess", False),
        randomize=True, n=getattr(args, "max_transitions", None),
        n_preload=getattr(args, "n_preload", 0),
        preload=getattr(args, "preload_data", False),
        extra_buffer_keys=getattr(args, "extra_buffer_keys", None)
    )

    dataset_size = len(dataloader.dataset)
    obs_since_update = 0
    unique_obs = set()

    print('Collecting unique observations...')
    for _, batch_data in enumerate(tqdm(dataloader)):
        obs, acts, next_obs = batch_data[:3]
        obs_list = to_hashable_tensor_list(obs)
        next_obs_list = to_hashable_tensor_list(next_obs)
        obs_list = obs_list + next_obs_list

        pre_len = len(unique_obs)
        unique_obs.update(set(obs_list))
        new_len = len(unique_obs)

        if pre_len == new_len:
            obs_since_update += 1
            if obs_since_update / max(dataset_size, 1) > early_stop_frac:
                print('Stopping unique obs collection early because no new obs were found!')
                break
        else:
            obs_since_update = 0

    unique_obs = torch.stack([x._tensor for x in unique_obs])
    print(f'{len(unique_obs)} Unique observations were gathered!')

    del dataloader

    if cache:
        print(f'Saving unique data cache to {unique_data_path}...')
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(unique_data_path, 'wb') as f:
            pickle.dump(unique_obs, f)
        print('Saved!')

    if return_hash:
        return unique_obs, unique_data_hash
    return unique_obs
