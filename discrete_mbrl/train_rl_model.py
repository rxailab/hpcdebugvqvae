"""
Fixed train_rl_model.py with gym/gymnasium compatibility.

The main issue is that PredictedModelWrapper and TimeLimit from gym
are not compatible with stable_baselines3 which expects gymnasium environments.
"""

import os
import sys
import time
import warnings

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit as GymnasiumTimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Import your modules
from shared.models import *
from shared.trainers import *
from .data_helpers import *
from .env_helpers import *
from .training_helpers import *
from .model_construction import *
from .utils import obs_to_img

SB3_DIR = 'mbrl_runs/'
N_EXAMPLE_ROLLOUTS = 4
N_EVAL_EPISODES = 20
EVAL_INTERVAL = 10
EVAL_UNROLL_STEPS = 20


class GymnasiumCompatWrapper(gymnasium.Env):
    """
    Wrapper to convert a gym-style environment to gymnasium-compatible.
    This fixes the compatibility issue with stable_baselines3.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env

        # Copy over spaces
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Handle metadata
        if hasattr(env, 'metadata'):
            self.metadata = env.metadata
        else:
            self.metadata = {'render_modes': []}

    def reset(self, seed=None, options=None):
        """Reset with gymnasium API (returns obs, info)"""
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)

        result = self.env.reset()

        # Handle both old and new API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        return obs, info

    def step(self, action):
        """Step with gymnasium API (returns obs, reward, terminated, truncated, info)"""
        result = self.env.step(action)

        if len(result) == 5:
            # Already new API
            return result
        elif len(result) == 4:
            # Old API: convert to new
            obs, reward, done, info = result
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        else:
            raise ValueError(f"Unexpected step result length: {len(result)}")

    def render(self):
        if hasattr(self.env, 'render'):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()

    def __getattr__(self, name):
        """Delegate attribute access to wrapped env"""
        return getattr(self.env, name)


class PredictedModelWrapperGymnasium(gymnasium.Env):
    """
    Gymnasium-compatible version of PredictedModelWrapper.
    Wraps encoder and transition model to create a world model environment.
    """

    def __init__(self, base_env, encoder, trans_model):
        super().__init__()
        self.base_env = base_env
        self.encoder = encoder
        self.trans_model = trans_model
        self.device = next(self.encoder.parameters()).device

        # Get observation shape from encoder output
        test_input = np.ones(base_env.observation_space.shape)
        test_input = preprocess_obs([test_input])
        with torch.no_grad():
            obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]

        # Define spaces
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = base_env.action_space

        self._curr_obs = None
        self.metadata = {'render_modes': []}

    def _preprocess_obs(self, obs):
        """Convert observation to tensor and encode"""
        obs = preprocess_obs([obs])
        with torch.no_grad():
            encoded = self.encoder.encode(obs.to(self.device))[0]
        return encoded.cpu()

    def _preprocess_action(self, action):
        """Convert action to tensor"""
        return preprocess_act([action])[0]

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        if seed is not None and hasattr(self.base_env, 'seed'):
            self.base_env.seed(seed)

        result = self.base_env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self._curr_obs = self._preprocess_obs(obs)
        return self._curr_obs.numpy(), info

    def step(self, action):
        """Take a step using the world model"""
        action_tensor = self._preprocess_action(action)

        with torch.no_grad():
            trans_out = self.trans_model(
                self._curr_obs.unsqueeze(0).to(self.device),
                action_tensor.unsqueeze(0).to(self.device)
            )
            next_obs, reward, gamma = [x[0].cpu() for x in trans_out]

        self._curr_obs = next_obs

        # Convert gamma to termination signal
        # gamma close to 0 means episode should end
        terminated = gamma.item() < 0.5
        truncated = False

        info = {}

        return next_obs.numpy(), reward.item(), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        if hasattr(self.base_env, 'close'):
            return self.base_env.close()


class ObsEncoderWrapperGymnasium(gymnasium.ObservationWrapper):
    """
    Gymnasium-compatible wrapper that encodes observations using the trained encoder.
    """

    def __init__(self, env, encoder):
        # First wrap if needed
        if not isinstance(env, gymnasium.Env):
            env = GymnasiumCompatWrapper(env)

        super().__init__(env)
        self.encoder = encoder
        self.device = next(self.encoder.parameters()).device

        # Get output shape
        test_input = np.ones(env.observation_space.shape)
        test_input = preprocess_obs([test_input])
        with torch.no_grad():
            obs_shape = self.encoder.encode(test_input.to(self.device)).shape[1:]

        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs):
        obs_tensor = preprocess_obs([obs])
        with torch.no_grad():
            encoded = self.encoder.encode(obs_tensor.to(self.device))[0]
        return encoded.cpu().numpy()


def traj_to_imgs(obs_buffer, episode_starts, encoder,
                 n_imgs=N_EXAMPLE_ROLLOUTS, wandb_format=True):
    """Convert trajectory observations to images for logging"""
    if isinstance(obs_buffer, np.ndarray):
        obs_buffer = torch.from_numpy(obs_buffer)
    idx = 0
    traj_imgs = []
    device = next(encoder.parameters()).device

    for _ in range(n_imgs):
        if idx >= len(obs_buffer):
            break
        traj_obs = [obs_buffer[idx]]

        for _ in range(EVAL_UNROLL_STEPS - 1):
            idx += 1
            if idx >= len(obs_buffer) or episode_starts[idx]:
                break
            traj_obs.append(obs_buffer[idx])

        traj_obs = torch.stack(traj_obs)
        decoded_obs = encoder.decode(traj_obs.to(device))
        img = obs_to_img(decoded_obs, cat=True)
        if wandb_format:
            import wandb
            img = wandb.Image(img)
        traj_imgs.append(img)
    return traj_imgs


def rl_eval(rl_model, env, encoder, n_eval_episodes=10, log=True):
    """Evaluate the RL model"""
    obs_buffer = [[] for _ in range(env.num_envs)]
    episode_starts = [[] for _ in range(env.num_envs)]

    last_episode_starts = [True] * env.num_envs

    def callback(locals, _):
        nonlocal last_episode_starts
        curr_obs = locals['observations']
        dones = locals['dones']
        for i, o in enumerate(curr_obs):
            obs_buffer[i].append(o)
            episode_starts[i].append(last_episode_starts[i])
            last_episode_starts[i] = dones[i]

    ep_rewards, ep_lengths = evaluate_policy(
        rl_model, env, callback=callback,
        n_eval_episodes=n_eval_episodes, return_episode_rewards=True
    )
    mean_reward = np.mean(ep_rewards)
    reward_std = np.std(ep_rewards)
    mean_len = np.mean(ep_lengths)

    if not log:
        return mean_reward, reward_std, mean_len

    # Flatten observation buffer
    obs_buffer = np.concatenate([np.stack(obs) for obs in obs_buffer], axis=0)
    episode_starts = np.concatenate([np.stack(starts) for starts in episode_starts], axis=0)

    traj_imgs = traj_to_imgs(obs_buffer, episode_starts, encoder)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import wandb
        wandb.log({
            'rl/eval/ep_reward_mean': np.mean(ep_rewards),
            'rl/eval/ep_reward_std': np.std(ep_rewards),
            'rl/eval/ep_mean_len': np.mean(ep_lengths),
            'rl/eval/trajectory': traj_imgs,
        })

    return mean_reward, reward_std, mean_len


def train_rl_model(args, encoder_model=None, trans_model=None):
    """Main training function for RL with world model"""

    if args.wandb:
        global wandb
        import wandb

    # Create base environment
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n

    # Get sample observation
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])

    # Load encoder
    if encoder_model is None:
        encoder_model = construct_ae_model(sample_obs.shape[1:], args)[0]
    encoder_model = encoder_model.to(args.device)
    freeze_model(encoder_model)
    encoder_model.eval()
    print('Loaded encoder')

    # Load transition model
    if trans_model is None:
        trans_model = construct_trans_model(encoder_model, args, env.action_space)[0]
    trans_model = trans_model.to(args.device)
    freeze_model(trans_model)
    trans_model.eval()
    print('Loaded transition model')

    if args.wandb:
        wandb.config.update(args, allow_val_change=True)

    # Create world model environment (gymnasium-compatible)
    world_model = PredictedModelWrapperGymnasium(env, encoder_model, trans_model)

    # Apply time limit if specified
    if args.rl_unroll_steps > 0:
        world_model = GymnasiumTimeLimit(world_model, max_episode_steps=args.rl_unroll_steps)

    # Wrap with Monitor
    world_model = Monitor(world_model)

    # Create PPO model
    model = PPO('MlpPolicy', world_model, verbose=1, device=args.device)

    # Create evaluation environment (uses real observations encoded)
    def make_eval_env():
        base_env = make_env(args.env_name, max_steps=args.env_max_steps)
        encoded_env = ObsEncoderWrapperGymnasium(base_env, encoder_model)
        return Monitor(encoded_env)

    eval_env = DummyVecEnv([make_eval_env])

    # Training callback for wandb logging
    wandb_callback = None
    if args.wandb:
        from stable_baselines3.common.callbacks import BaseCallback

        class WandbCallback(BaseCallback):
            def __init__(self, encoder, eval_env=None, verbose=0):
                super().__init__(verbose)
                self.encoder = encoder
                self.eval_env = eval_env
                self.train_step = 0

            def _on_step(self):
                return True

            def _on_rollout_end(self):
                learner = self.locals['self']
                ep_info_buffer = learner.ep_info_buffer

                if len(ep_info_buffer) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        wandb.log({
                            'rl/train/ep_reward_mean': np.mean([ep['r'] for ep in ep_info_buffer]),
                            'rl/train/ep_mean_len': np.mean([ep['l'] for ep in ep_info_buffer]),
                            'rl/train/step': learner.num_timesteps,
                        })

                # Periodic evaluation
                if self.eval_env is not None and EVAL_INTERVAL > 0 and self.train_step % EVAL_INTERVAL == 0:
                    print('Running RL Eval...')
                    mean_reward, std_reward, mean_len = rl_eval(
                        learner, self.eval_env, self.encoder,
                        n_eval_episodes=N_EVAL_EPISODES, log=True
                    )
                    print(f'Eval: reward={mean_reward:.3f}±{std_reward:.3f}, len={mean_len:.2f}')

                self.train_step += 1

        wandb_callback = WandbCallback(encoder_model, eval_env)

    # Train
    try:
        model.learn(args.rl_train_steps, callback=wandb_callback)
    except KeyboardInterrupt:
        print('Training interrupted')

    # Final evaluation
    mean_reward, std_reward, mean_len = rl_eval(
        model, eval_env, encoder_model, n_eval_episodes=N_EVAL_EPISODES, log=args.wandb
    )
    print(f'Final: reward={mean_reward:.3f}±{std_reward:.3f}, len={mean_len:.2f}')

    # Save model
    if args.save:
        save_dir = f"./models/{args.env_name}"
        os.makedirs(save_dir, exist_ok=True)
        model.save(f"{save_dir}/ppo_world_model")
        print(f"Model saved to {save_dir}/ppo_world_model")

    return model


if __name__ == '__main__':
    # Parse args
    args = get_args()

    # Setup wandb if enabled
    if args.wandb:
        import wandb

        wandb.init(
            project='discrete-model-only-rl',
            config=args,
            tags=args.tags,
            settings=wandb.Settings(start_method='thread'),
            allow_val_change=True
        )
        args = wandb.config

    # Train the model
    model = train_rl_model(args)