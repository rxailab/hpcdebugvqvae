import argparse
from argparse import Namespace
import os
import psutil
import platform
from einops import rearrange
from gym.envs.mujoco import MujocoEnv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .visualization import states_to_imgs
# from model_construction import CONTINUOUS_ENCODER_TYPES, DISCRETE_ENCODER_TYPES, add_model_args
from .env_helpers import check_env_name

# Global configuration for observation resizing
OBS_RESIZE_CONFIG = {
    'minigrid_target_size': (48, 48),  # Standard size for all MiniGrid envs
    'resize_mode': 'bilinear',  # 'bilinear' or 'nearest'
    'cache_sizes': {}  # Cache for environment-specific sizes
}


# Add these functions to training_helpers.py after the existing functions

def get_optimized_args(args):
    """Apply GPU utilization optimizations to args with Windows multiprocessing fix"""

    # Windows multiprocessing fix - MUST come first before any other settings
    if platform.system() == 'Windows':
        print("Windows detected - disabling multiprocessing to avoid pickle errors")
        args.n_preload = 0  # Force disable multiprocessing on Windows
        # Don't override this setting later
        windows_multiprocessing_disabled = True
    else:
        windows_multiprocessing_disabled = False

    # Data loading optimizations
    '''
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # GB

        # Scale batch size based on GPU memory
        if gpu_memory >= 8:
            args.batch_size = max(args.batch_size, 1024)
            args.eval_batch_size = max(args.eval_batch_size, 512)
        if gpu_memory >= 16:
            args.batch_size = max(args.batch_size, 4096)
            args.eval_batch_size = max(args.eval_batch_size, 1024)
        if gpu_memory >= 24:
            args.batch_size = max(args.batch_size, 6144)
            args.eval_batch_size = max(args.eval_batch_size, 1536)
    '''
    # Only increase workers if NOT on Windows
    if not windows_multiprocessing_disabled:
        args.n_preload = max(args.n_preload, min(8, os.cpu_count()))

    # Enable data preloading for smaller datasets
    if not hasattr(args, 'preload_data') or not args.preload_data:
        # Enable preloading for smaller datasets to avoid HDF5 delays
        if args.max_transitions and args.max_transitions < 500000:
            args.preload_data = True
            print("Enabled data preloading to reduce HDF5 access delays")

    # Disable persistent workers on first epoch to avoid setup delays
    if not hasattr(args, 'persistent_workers'):
        args.persistent_workers = False  # Set to False initially

    # Reduce validation frequency to avoid GPU idle time
    args.log_freq = max(args.log_freq, 500)
    args.checkpoint_freq = max(args.checkpoint_freq, 5)

    # Add mixed precision training
    if not hasattr(args, 'use_amp'):
        args.use_amp = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

    # Add gradient accumulation for effective larger batch sizes
    if not hasattr(args, 'accumulation_steps'):
        effective_batch_size = 8192
        args.accumulation_steps = max(1, effective_batch_size // args.batch_size)

    # Add data loader optimizations
    if not hasattr(args, 'pin_memory'):
        args.pin_memory = torch.cuda.is_available()
    if not hasattr(args, 'persistent_workers'):
        # Only enable persistent workers if we have workers and not on Windows
        args.persistent_workers = args.n_preload > 0 and not windows_multiprocessing_disabled
    if not hasattr(args, 'prefetch_factor'):
        args.prefetch_factor = 2

    print(f"Applied GPU optimizations:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Eval batch size: {args.eval_batch_size}")
    print(f"  Data workers: {args.n_preload}")
    print(f"  Preload data: {getattr(args, 'preload_data', False)}")
    print(f"  Mixed precision: {getattr(args, 'use_amp', False)}")
    print(f"  Accumulation steps: {getattr(args, 'accumulation_steps', 1)}")
    if windows_multiprocessing_disabled:
        print(f"  Windows multiprocessing: DISABLED")

    return args


# Modify the existing get_args function to include optimizations
# In discrete_mbrl/training_helpers.py

def get_args(parser=None, apply_optimizations=True):
    """
    Get arguments for training, with optional GPU optimizations.
    Initializes an argument parser if one is not provided.
    """
    # Ensure parser is initialized before adding arguments
    parser = make_argparser(parser)

    parser.add_argument('--commitment-cost', type=float, default=0.25,
                        help='Commitment cost for VQ-VAE beta')

    args = parser.parse_args()
    args = process_args(args)

    # Apply GPU optimizations by default
    if apply_optimizations:
        args = get_optimized_args(args)

    return args


# Add GPU memory optimization utilities
def optimize_gpu_memory():
    """Call this before training to optimize GPU memory usage"""
    if torch.cuda.is_available():
        # Set memory allocation strategy
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed

        # Clear cache
        torch.cuda.empty_cache()

        print("GPU memory optimizations applied")


def setup_efficient_model(model, args):
    """Setup model for efficient training"""

    # Move to GPU
    model = model.to(args.device)

    # Enable channels_last memory format for CNN models if supported
    try:
        if hasattr(model, 'encoder') or hasattr(model, 'decoder') or \
                any('conv' in name.lower() for name, _ in model.named_modules()):
            model = model.to(memory_format=torch.channels_last)
            print("Enabled channels_last memory format")
    except:
        pass  # Some models don't support channels_last

    # Note: torch.compile is applied later in train_encoder, not here
    # This avoids issues with model attributes being lost
    return model

def apply_torch_compile(model, args):
    """Apply torch.compile separately after model setup"""
    # Only compile if using PyTorch 2.0+ and CUDA
    if hasattr(torch, 'compile') and args.device == 'cuda' and getattr(args, 'compile_model', True):
        try:
            compiled_model = torch.compile(model, mode='default')
            print("Model compiled with torch.compile")
            return compiled_model
        except Exception as e:
            print(f"torch.compile failed: {e}")
            return model
    return model

def get_obs_target_size(env_name, default_size=(48, 48)):
    """Get target observation size for an environment.

    This ensures consistent sizes across different MiniGrid variants.
    """
    # Check cache first
    if env_name in OBS_RESIZE_CONFIG['cache_sizes']:
        return OBS_RESIZE_CONFIG['cache_sizes'][env_name]

    # Determine target size based on environment
    if 'MiniGrid' in env_name:
        target_size = OBS_RESIZE_CONFIG.get('minigrid_target_size', default_size)
    else:
        target_size = None  # No resizing for non-MiniGrid environments

    # Cache the result
    OBS_RESIZE_CONFIG['cache_sizes'][env_name] = target_size
    return target_size


def fast_obs_resize(obs, target_size=None, mode=None):
    """Fast observation resizing using PyTorch's interpolate.

    Args:
        obs: Tensor of shape (B, C, H, W) or (C, H, W)
        target_size: Tuple of (H, W) for target size, or None to skip
        mode: Interpolation mode ('bilinear' or 'nearest'), uses config default if None

    Returns:
        Resized observation tensor
    """
    if target_size is None:
        return obs

    if mode is None:
        mode = OBS_RESIZE_CONFIG.get('resize_mode', 'bilinear')

    # Handle different input dimensions
    needs_squeeze = False
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
        needs_squeeze = True

    # Check if resizing is needed
    current_size = obs.shape[-2:]
    if current_size == target_size:
        if needs_squeeze:
            obs = obs.squeeze(0)
        return obs

    # Perform fast resize
    resized = F.interpolate(
        obs,
        size=target_size,
        mode=mode,
        align_corners=False if mode == 'bilinear' else None
    )

    if needs_squeeze:
        resized = resized.squeeze(0)

    return resized


def batch_obs_resize(obs_batch, env_name=None, target_size=None):
    """Efficiently resize a batch of observations.

    Args:
        obs_batch: Tensor of shape (B, C, H, W)
        env_name: Environment name to determine target size
        target_size: Override target size

    Returns:
        Resized batch
    """
    if target_size is None and env_name is not None:
        target_size = get_obs_target_size(env_name)

    if target_size is None or obs_batch.shape[-2:] == target_size:
        return obs_batch

    return F.interpolate(
        obs_batch,
        size=target_size,
        mode=OBS_RESIZE_CONFIG.get('resize_mode', 'bilinear'),
        align_corners=False if OBS_RESIZE_CONFIG.get('resize_mode', 'bilinear') == 'bilinear' else None
    )


def make_argparser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--env_name', type=str, default='MiniGrid-MultiRoom-N2-S4-v0')
    parser.add_argument('-t', '--ae_model_type', type=str, default='ae')
    parser.add_argument('-v', '--ae_model_version', type=str, default='2')
    parser.add_argument('-tet', '--trans_model_type', type=str, default='continuous')
    parser.add_argument('-tev', '--trans_model_version', type=str, default='1')
    parser.add_argument('-d', '--device', type=str, default='cuda')
    parser.add_argument('-ne', '--epochs', type=int, default=1)
    parser.add_argument('-tne', '--trans_epochs', type=int, default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=2048)
    parser.add_argument('-l', '--log_freq', type=int, default=100)
    parser.add_argument('-c', '--checkpoint_freq', type=int, default=2)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-4)
    parser.add_argument('-tlr', '--trans_learning_rate', type=float, default=None)
    parser.add_argument('-m', '--max_transitions', type=int, default=None)
    parser.add_argument('-rlc', '--recon_loss_clip', type=float, default=0)
    parser.add_argument('-u', '--all_data', dest='unique_data', action='store_false')
    parser.add_argument('--unique_data', dest='unique_data', action='store_true')
    parser.add_argument('-nl', '--no_load', dest='load', action='store_false')
    parser.add_argument('-nc', '--no_cache', dest='cache', action='store_false')
    parser.add_argument('-p', '--preprocess', action='store_true')
    parser.add_argument('-w', '--wandb', action='store_true')
    parser.add_argument('-cml', '--comet_ml', action='store_true')
    parser.add_argument('-i', '--extra_info', type=str, default=None)
    parser.add_argument('-pl', '--n_preload', type=int, default=0, help='Parallel preload data')
    parser.add_argument('-um', '--upload_model', action='store_true')
    parser.add_argument('-s', '--save', dest='save', action='store_true')
    parser.add_argument('-ntu', '--n_train_unroll', type=int, default=4)
    parser.add_argument('-pld', '--preload_data', action='store_true')
    parser.add_argument('--env_max_steps', type=int, default=None)
    parser.add_argument('--rl_unroll_steps', type=int, default=-1)
    parser.add_argument('--rl_train_steps', type=int, default=0)
    parser.add_argument('--exact_comp', action='store_true')
    parser.add_argument('--log_state_reprs', action='store_true')
    parser.add_argument('--tags', nargs='*', default=None)
    parser.add_argument('--extra_buffer_keys', nargs='*', default=[])
    parser.add_argument('--eval_policies', nargs='*', type=str,
                        default=['random'], action='store')
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--eval_unroll_steps', type=int, default=20)
    parser.add_argument('--log_norms', action='store_true')  # Log trans model norm data
    parser.add_argument('--ae_grad_clip', type=float, default=0)
    parser.add_argument('--e2e_loss', action='store_true')

    # Add observation resize arguments
    parser.add_argument('--obs_resize', type=int, nargs=2, default=None,
                        help='Target size (H W) for observation resizing')
    parser.add_argument('--obs_resize_mode', type=str, default='bilinear',
                        choices=['bilinear', 'nearest'],
                        help='Interpolation mode for resizing')
    parser.add_argument('--no_obs_resize', action='store_true',
                        help='Disable automatic observation resizing')

    add_model_args(parser)
    parser.set_defaults(
        preprocess=False, wandb=False, load=True, unique_data=False,
        cache=True, upload_model=False, save=False, preload_data=False,
        exact_comp=False, comet_ml=False, log_state_reprs=False,
        log_norms=False, e2e_loss=False, no_obs_resize=False)
    return parser


def add_model_args(parser):
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=None)
    parser.add_argument('--filter_size', type=int, default=8)
    parser.add_argument('--codebook_size', type=int, default=16)
    parser.add_argument('--ae_model_hash', type=str, default=None)

    parser.add_argument('--trans_hidden', type=int, default=256)
    parser.add_argument('--trans_depth', type=int, default=3)
    parser.add_argument('--stochastic', type=str, default='simple',
                        choices=[None, 'simple', 'categorical'])

    # Only works with ae
    parser.add_argument('--fta_tiles', type=int, default=20,
                        help='How many tiles to use in FTA')
    parser.add_argument('--fta_bound_low', type=float, default=-2,
                        help='Upper bound for FTA range')
    parser.add_argument('--fta_bound_high', type=float, default=2,
                        help='Lower bound for FTA range')
    parser.add_argument('--fta_eta', type=float, default=0.2,
                        help='Degree of fuzzyness in FTA')

    # Only works with soft_vqvae
    parser.add_argument('--repr_sparsity', type=float, default=0,
                        help='Fractional sparsity of representations post training')
    parser.add_argument('--sparsity_type', type=str, default='random',
                        choices=['random', 'identity'],
                        help='Type of sparsity mask to use')

    # Only works with universal_vq transition model
    parser.add_argument('--vq_trans_loss_type', type=str, default='mse',
                        choices=['mse', 'cross_entropy'])
    parser.add_argument('--vq_trans_1d_conv', action='store_true')
    parser.add_argument('--vq_trans_state_snap', action='store_true')

    parser.set_defaults(vq_trans_1d_conv=False, vq_trans_state_snap=False)


def validate_args(args):
    pass
    # if args.ae_model_type in CONTINUOUS_ENCODER_TYPES:
    #     if args.trans_model_type != 'continuous':
    #         print(f'Model type {args.ae_model_type} requires a continuous transition model!')
    #         print('Ending run.')
    #         sys.exit()
    # elif args.ae_model_type in DISCRETE_ENCODER_TYPES:
    #     if args.trans_model_type not in ('discrete', 'transformer', 'transformerdec'):
    #         print(f'Model type {args.ae_model_type} requires a discrete transition model!')
    #         print('Ending run.')
    #         sys.exit()
    # elif args.ae_model_type != 'flatten':
    #     raise ValueError(f'Invalid encoder type, "{args.ae_model_type}"!')

    # if 'transformer' in args.trans_model_type.lower() and \
    #    args.stochastic and args.stochastic.lower() == 'categorical':
    #     raise ValueError('Transformer type transition models cannot use categorical stochasticity type!')


def log_param_updates(args, params):
    # Skip logging during model construction to avoid network delays
    if getattr(args, '_skip_param_logging', False):
        return

    if args.wandb:
        import wandb
        if not isinstance(wandb.config, wandb.sdk.lib.preinit.PreInitObject):
            wandb.config.update(params, allow_val_change=True)
    elif args.comet_ml:
        import comet_ml
        experiment = comet_ml.get_global_experiment()
        if experiment is not None:
            experiment.log_parameters(params)


def update_arg(args, key, val):
    if args.wandb and not isinstance(args, Namespace):
        args._items.update({key: val})
    else:
        setattr(args, key, val)

    log_param_updates(args, {key: val})


def process_args(args):
    new_env_name = check_env_name(args.env_name)
    update_arg(args, 'env_name', new_env_name)
    if isinstance(args.eval_policies, str):
        update_arg(args, 'eval_policies', [args.eval_policies])

    if args.trans_learning_rate is None:
        update_arg(args, 'trans_learning_rate', args.learning_rate)

    # Configure observation resizing
    if args.obs_resize is not None:
        OBS_RESIZE_CONFIG['minigrid_target_size'] = tuple(args.obs_resize)
    if args.obs_resize_mode:
        OBS_RESIZE_CONFIG['resize_mode'] = args.obs_resize_mode

    validate_args(args)
    return args


def freeze_model(model):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False


def test_model(model, test_func, data_loader):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch_data in data_loader:
            loss = test_func(batch_data)
            if not isinstance(loss, dict):
                loss = {'loss': loss.mean()}
            losses.append(loss)
    model.train()
    return losses


def train_loop(model, trainer, train_loader, valid_loader=None, n_epochs=1,
               batch_size=128, log_freq=100, seed=0, callback=None,
               valid_callback=None, test_func=None):
    import time
    torch.manual_seed(seed)
    model.train()
    train_losses = []

    for epoch in range(n_epochs):
        #print(f'?? Starting epoch #{epoch}')
        epoch_start = time.time()

        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}",
                              unit="batch", leave=True)

        for i, batch_data in enumerate(epoch_iterator):
            train_loss, aux_data = trainer.train(batch_data)

            if not isinstance(train_loss, dict):
                train_loss = {'loss': train_loss}
            train_losses.append(train_loss)

            # Update progress bar
            if len(train_losses) > 0:
                current_loss = train_losses[-1].get('loss', 0)
                if hasattr(current_loss, 'item'):
                    current_loss = current_loss.item()
                epoch_iterator.set_postfix({'loss': f'{current_loss:.4f}'})

            if callback:
                callback(train_loss, i * batch_size, epoch, aux_data=aux_data)

            # ?? FIX: Skip validation on first few steps AND reduce frequency
            should_validate = (
                    i > 0 and  # Skip step 0
                    i % log_freq == 0 and
                    valid_loader is not None and
                    i % (log_freq * 2) == 0  # Validate only every 2nd log_freq
            )

            if should_validate:
                #print("?? Running validation...")
                valid_start = time.time()
                test_func = test_func or trainer.calculate_losses
                valid_losses = test_model(model, test_func, valid_loader)
                #print(f"??  Validation took {time.time() - valid_start:.2f}s")

                valid_loss_means = {k: np.mean([x[k].item() for x in valid_losses])
                                    for k in valid_losses[0]}

                if valid_callback:
                    #print("?? Calling valid_callback...")
                    valid_callback_start = time.time()
                    valid_callback(valid_loss_means, i, epoch)
                    #print(f"??  valid_callback took {time.time() - valid_callback_start:.2f}s")

                model.train()  # Ensure model is back in training mode

        #print(f'??  Epoch {epoch} completed in {time.time() - epoch_start:.2f}s')


def sample_recon_seqs(encoder, trans_model, dataloader, n_steps, n=4,
                      env_name=None, rev_transform=None, gif_format=False):
    # Generate sample reconstructions
    encoder.eval()
    trans_model.eval()
    device = next(encoder.parameters()).device

    # Get target size for this environment
    target_size = get_obs_target_size(env_name) if env_name else None

    sample_batch = next(iter(dataloader))
    sample_batch = [x[:n].to(device) for x in sample_batch]
    if n_steps <= 1:
        # Add dummy dimension for single step prediction
        sample_batch = [x[:, None] for x in sample_batch]
    obs, acts, next_obs = sample_batch[:3]

    # Resize observations if needed
    if target_size and not getattr(encoder, 'no_resize', False):
        obs = batch_obs_resize(obs.reshape(-1, *obs.shape[2:]), env_name=env_name)
        obs = obs.reshape(n, -1, *obs.shape[1:])
        next_obs = batch_obs_resize(next_obs.reshape(-1, *next_obs.shape[2:]), env_name=env_name)
        next_obs = next_obs.reshape(n, -1, *next_obs.shape[1:])

    init_obs = obs[:, 0]  # First step of each sequence
    z = encoder.encode(init_obs)

    # Check if this is an identity encoder (no actual encoding/decoding)
    is_identity_encoder = hasattr(encoder, '__class__') and 'Identity' in encoder.__class__.__name__

    if is_identity_encoder:
        # For identity encoders, just use the original observations
        print("Identity encoder detected - using original observations for visualization")

        # Create a simple sequence showing original observations
        dec_steps = [init_obs]
        for i in range(n_steps):
            # For identity encoder, just use the ground truth next observations
            if i < next_obs.shape[1]:
                dec_steps.append(next_obs[:, i])
            else:
                # If we run out of ground truth, repeat the last observation
                dec_steps.append(next_obs[:, -1])

        all_obs = torch.cat([obs[:, :1], next_obs], dim=1)

        # Convert to proper format for visualization
        dec_steps_tensor = torch.stack(dec_steps)


    else:
        # Normal encoder/decoder logic
        dec_steps = [torch.zeros_like(init_obs)]
        for i in range(n_steps):
            with torch.no_grad():
                z = trans_model(z, acts[:, i])[0]
                decoded = encoder.decode(z)

                # Handle spatial dimension mismatches with fast resize
                if decoded.shape != init_obs.shape:
                    decoded = fast_obs_resize(decoded, target_size=init_obs.shape[-2:])

            dec_steps.append(decoded)

        all_obs = torch.cat([obs[:, :1], next_obs], dim=1)
        dec_steps_tensor = torch.stack(dec_steps)

    # Convert states to images
    orig_shape = dec_steps_tensor.shape[:2]
    flat_dec_steps = dec_steps_tensor.reshape(-1, *dec_steps_tensor.shape[2:])
    flat_dec_steps = states_to_imgs(flat_dec_steps, env_name, transform=rev_transform)
    flat_dec_steps = torch.from_numpy(flat_dec_steps)
    dec_steps_processed = flat_dec_steps.reshape(*orig_shape, *flat_dec_steps.shape[1:])
    dec_steps_list = list(dec_steps_processed)

    orig_shape = all_obs.shape[:2]
    flat_all_obs = all_obs.reshape(-1, *all_obs.shape[2:])
    flat_all_obs = states_to_imgs(flat_all_obs, env_name, transform=rev_transform)
    flat_all_obs = torch.from_numpy(flat_all_obs)
    all_obs = flat_all_obs.reshape(*orig_shape, *flat_all_obs.shape[1:])

    if gif_format:
        sample_recons = torch.stack(dec_steps_list).transpose(0, 1)
        sample_recons = torch.cat([all_obs, sample_recons], dim=4).cpu()
        sample_recons = (sample_recons.numpy().clip(0, 1) * 255).astype(np.uint8)
        # Check for framestack, shape is (b, n, c, w, h)
        if sample_recons.shape[2] not in (1, 3):
            sample_recons = sample_recons[:, :, -1:]  # Take last frame of each framestack
    else:
        # Spread n_steps out over width, (b, c, h, n * w)
        sample_recons = torch.cat(dec_steps_list, dim=3)
        all_obs = rearrange(all_obs, 'b n c h w -> b c h (n w)')

        # Stack real and predicted obs on top of each other
        sample_recons = torch.cat([all_obs, sample_recons], dim=2).cpu()
        sample_recons = sample_recons.permute(0, 2, 3, 1).numpy()
        sample_recons = sample_recons.clip(0, 1)

        # Check for framestack
        if sample_recons.shape[-1] not in (1, 3):
            sample_recons = sample_recons[..., -1:]  # Take last frame of each framestack

    return sample_recons


def sample_recon_imgs(model, dataloader, n=4, env_name=None, rev_transform=None):
    # Generate sample reconstructions
    model.eval()

    # Get target size for this environment
    target_size = get_obs_target_size(env_name) if env_name else None

    # Check if dataloader has iter method
    if isinstance(dataloader, DataLoader):
        samples = next(iter(dataloader))[0][:n]
    elif isinstance(dataloader, torch.Tensor):
        samples = dataloader[:n]
    elif isinstance(dataloader, (list, tuple)):
        samples = torch.stack(dataloader[:n])
    else:
        raise ValueError(f'Invalid dataloader type: {type(dataloader)}!')

    device = next(model.parameters()).device

    # Resize samples if needed before encoding
    if target_size and not getattr(model, 'no_resize', False):
        samples_resized = batch_obs_resize(samples, env_name=env_name)
    else:
        samples_resized = samples

    with torch.no_grad():
        encs = model.encode(samples_resized.to(device))
        decs = model.decode(encs)

    # Handle spatial dimension mismatches between input and reconstruction
    if samples.shape != decs.shape:
        # Use fast resize instead of interpolate
        decs = batch_obs_resize(decs, target_size=samples.shape[-2:])

    # Convert states to images
    samples = states_to_imgs(samples, env_name, transform=rev_transform)
    samples = torch.from_numpy(samples)

    decs = states_to_imgs(decs, env_name, transform=rev_transform)
    decs = torch.from_numpy(decs)

    sample_recons = torch.cat([samples, decs.cpu()], dim=3)
    sample_recons = sample_recons.clip(0, 1)
    if samples.shape[1] == 1 or samples.shape[1] == 3:
        sample_recons = sample_recons.permute(0, 2, 3, 1).numpy()
    else:
        srs = sample_recons.shape
        sample_recons = sample_recons.reshape(srs[0] * srs[1], *srs[2:]).numpy()
    return sample_recons


def vec_env_random_walk(env, n_steps, progress=True):
    """ Generate random walks for n_steps. """
    env.reset()
    prog_func = tqdm if progress else lambda x: x
    for _ in prog_func(range(int(np.ceil(n_steps / env.num_envs)))):
        acts = [env.action_space.sample() for _ in range(env.num_envs)]
        env.step(acts)


def vec_env_ez_explore(env, n_steps, min_repeat=1, max_repeat=8, progress=True):
    """ Generate ez-explore walks for n_steps. """
    env.reset()

    curr_acts = np.array([0 for _ in range(env.num_envs)])
    curr_repeats = np.array([0 for _ in range(env.num_envs)])

    prog_func = tqdm if progress else lambda x: x
    for _ in prog_func(range(int(np.ceil(n_steps / env.num_envs)))):
        for i in range(env.num_envs):
            if curr_repeats[i] == 0:
                curr_acts[i] = env.action_space.sample()
                curr_repeats[i] = np.random.randint(min_repeat, max_repeat + 1)
        acts = [env.action_space.sample() for _ in range(env.num_envs)]
        env.step(acts)
        curr_repeats -= 1


# Add utility function to handle observation resizing in data loaders
class ObservationResizeWrapper:
    """Wrapper to automatically resize observations in dataloaders."""

    def __init__(self, dataset, env_name=None, target_size=None):
        self.dataset = dataset
        self.env_name = env_name
        self.target_size = target_size or get_obs_target_size(env_name)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.target_size and isinstance(data, (list, tuple)) and len(data) > 0:
            # Resize observations (typically first and third elements)
            resized_data = list(data)
            for i in [0, 2]:  # obs and next_obs positions
                if i < len(data) and torch.is_tensor(data[i]):
                    resized_data[i] = fast_obs_resize(data[i], self.target_size)
            return tuple(resized_data) if isinstance(data, tuple) else resized_data
        return data

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped dataset
        return getattr(self.dataset, name)