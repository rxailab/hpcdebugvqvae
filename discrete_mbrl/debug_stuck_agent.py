#!/usr/bin/env python3
"""
Standalone script to visualize the rollout of a trained RL model in MiniGrid.
This version is more self-contained and handles common import issues.

Usage:
    python view_rollout_standalone.py \
        --model_path ./model_free/models/MiniGrid-Empty-5x5-v0/final_model_reward_0.9961.pt \
        --env_name MiniGrid-Empty-5x5-v0 \
        --n_episodes 3

    # Save as GIF:
    python view_rollout_standalone.py \
        --model_path ./model_free/models/MiniGrid-Empty-5x5-v0/final_model_reward_0.9961.pt \
        --env_name MiniGrid-Empty-5x5-v0 \
        --save_gif rollout.gif \
        --n_episodes 1
"""

import argparse
import os
import sys
import time
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Try to import gymnasium first, fall back to gym
try:
    import gymnasium as gym
    from gymnasium import spaces

    USE_GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces

    USE_GYMNASIUM = False

# Import MiniGrid
try:
    import minigrid
    from minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
except ImportError:
    try:
        import gym_minigrid
        from gym_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper
    except ImportError:
        print("Error: Please install minigrid: pip install minigrid")
        sys.exit(1)


# ============================================================================
# Model Components (copied/adapted from shared/models.py)
# ============================================================================

class ArgmaxLayer(nn.Module):
    """Layer that applies argmax (used for discrete inputs)."""

    def forward(self, x):
        if len(x.shape) == 3:
            # x is one-hot: (batch, n_embeds, n_latents)
            return x.argmax(dim=1)  # (batch, n_latents)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def mlp(sizes, activation='relu', output_activation=None,
        discrete_input=False, n_embeds=None, embed_dim=None):
    """
    Build an MLP with the given sizes and activations.
    """
    layers = []

    # Handle discrete input (VQ-VAE style)
    if discrete_input:
        layers.append(ArgmaxLayer())
        layers.append(nn.Embedding(n_embeds, embed_dim))
        layers.append(nn.Flatten(start_dim=1))

    # Get activation function
    if activation == 'relu':
        act_fn = nn.ReLU
    elif activation == 'tanh':
        act_fn = nn.Tanh
    elif activation == 'crelu':
        act_fn = nn.ReLU  # Simplified
    else:
        act_fn = nn.ReLU

    # Build MLP layers
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act_fn())

    if output_activation is not None:
        layers.append(output_activation())

    return nn.Sequential(*layers)


# ============================================================================
# Environment Helpers
# ============================================================================

def make_env(env_name, render_mode=None, max_steps=None):
    """Create a MiniGrid environment with appropriate wrappers."""

    # Create base environment
    if USE_GYMNASIUM:
        if render_mode:
            env = gym.make(env_name, render_mode=render_mode)
        else:
            env = gym.make(env_name)
    else:
        env = gym.make(env_name)

    # Apply wrappers for image observations
    env = RGBImgObsWrapper(env)  # Get RGB images
    env = ImgObsWrapper(env)     # Remove the 'mission' from observation

    return env


def preprocess_obs(obs_list):
    """Convert observations to tensor format expected by model."""
    obs = np.array(obs_list)

    # Ensure float and normalize if needed
    if obs.dtype == np.uint8:
        obs = obs.astype(np.float32) / 255.0
    else:
        obs = obs.astype(np.float32)

    # Ensure channel-first format (B, C, H, W)
    if len(obs.shape) == 3:
        obs = np.expand_dims(obs, 0)

    if obs.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
        obs = np.transpose(obs, (0, 3, 1, 2))

    return torch.from_numpy(obs).float()


# ============================================================================
# Simple VQ-VAE Encoder (for loading weights)
# ============================================================================

class SimpleVQVAEEncoder(nn.Module):
    """
    VQ-VAE encoder that matches the structure used in training.
    Architecture is inferred from the checkpoint state dict.
    """

    def __init__(self, input_shape, embedding_dim=64, n_embeddings=16,
                 filter_size=8, n_latent_embeds=64, state_dict=None, target_n_latents=None):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embeddings = n_embeddings
        self.n_latent_embeds = n_latent_embeds
        self.codebook_size = n_embeddings

        c, h, w = input_shape

        # Try to infer architecture from state dict
        if state_dict is not None:
            self.encoder = self._build_encoder_from_state_dict(
                state_dict, c, h, target_n_latents=target_n_latents)
        else:
            # Default architecture (may not match)
            self.encoder = nn.Sequential(
                nn.Conv2d(c, 64, 8, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(64, 128, 6, stride=3, padding=2),
                nn.ReLU(),
                nn.Conv2d(128, embedding_dim, 4, stride=1, padding=0),
            )

        # Codebook
        self.codebook = nn.Embedding(n_embeddings, embedding_dim)

        # Calculate output spatial dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.encoder(dummy)
            self.latent_shape = out.shape[1:]  # (embedding_dim, h', w')
            self.n_latent_embeds = out.shape[2] * out.shape[3]
            print(f"  Encoder output shape: {out.shape}")
            print(f"  n_latent_embeds: {self.n_latent_embeds}")

    def _build_encoder_from_state_dict(self, state_dict, in_channels, input_hw, target_n_latents=None):
        """Build encoder architecture by inspecting state dict shapes."""
        # Find all encoder conv layers
        conv_layers = []
        for key in sorted(state_dict.keys()):
            if key.startswith('encoder.') and 'weight' in key and len(state_dict[key].shape) == 4:
                # Conv2d weight shape: (out_channels, in_channels, kH, kW)
                # key like: encoder.0.weight
                layer_idx = int(key.split('.')[1])
                weight = state_dict[key]
                out_ch, in_ch, kh, kw = weight.shape
                conv_layers.append({
                    'idx': layer_idx,
                    'out_channels': out_ch,
                    'in_channels': in_ch,
                    'kernel_size': (kh, kw)
                })

        conv_layers.sort(key=lambda x: x['idx'])
        print(f"  Found {len(conv_layers)} conv layers in encoder")

        if target_n_latents:
            target_spatial = int(math.sqrt(target_n_latents))
            print(f"  Target spatial output: {target_spatial}x{target_spatial} (n_latents={target_n_latents})")

        def calc_output(input_size, kernel, stride, padding):
            return (input_size - kernel + 2 * padding) // stride + 1

        # Common stride/padding pairs for different kernel sizes
        # NOTE: include (2,1) for k=8 which is common and needed for 40->18 in many Minigrid pipelines
        options = {
            8: [(4, 2), (2, 0), (2, 1), (2, 2), (2, 3), (1, 0), (1, 3), (4, 0)],
            6: [(3, 2), (2, 2), (2, 1), (1, 2), (3, 0), (1, 0), (1, 1)],
            4: [(2, 1), (1, 1), (2, 0), (1, 0), (4, 0), (1, 2)],
            3: [(1, 1), (2, 1), (1, 0)],
        }

        # Try to find valid configuration
        best_config = None
        best_diff = float('inf')

        from itertools import product

        # Get options for each layer
        layer_options = []
        for conv_info in conv_layers:
            kh = conv_info['kernel_size'][0]
            layer_options.append(options.get(kh, [(1, kh // 2)]))

        # Search all combinations
        for config in product(*layer_options):
            h = input_hw
            valid = True
            for i, (stride, padding) in enumerate(config):
                kernel = conv_layers[i]['kernel_size'][0]
                h = calc_output(h, kernel, stride, padding)
                if h <= 0:
                    valid = False
                    break

            if valid and target_n_latents:
                diff = abs(h * h - target_n_latents)
                if diff < best_diff:
                    best_diff = diff
                    best_config = config
                    if diff == 0:
                        break

        if best_config is None:
            print("  Warning: Could not find valid stride/padding configuration!")
            # Fallback: use stride=1 with same padding
            best_config = [(1, k['kernel_size'][0] // 2) for k in conv_layers]

        # Build encoder with best config
        layers = []
        h = input_hw
        for i, conv_info in enumerate(conv_layers):
            kh, kw = conv_info['kernel_size']
            in_ch = conv_info['in_channels']
            out_ch = conv_info['out_channels']
            stride, padding = best_config[i]

            h_out = calc_output(h, kh, stride, padding)
            print(
                f"    Conv layer {i}: in={in_ch}, out={out_ch}, kernel={kh}x{kw}, stride={stride}, pad={padding} -> {h}x{h} => {h_out}x{h_out}"
            )
            h = h_out

            layers.append(nn.Conv2d(in_ch, out_ch, (kh, kw), stride=stride, padding=padding))

            # Add ReLU after all but last layer
            if i < len(conv_layers) - 1:
                layers.append(nn.ReLU())

        print(f"  Final spatial: {h}x{h}, n_latents={h * h}")

        return nn.Sequential(*layers)

    def encode(self, x, return_one_hot=False, as_long=False):
        """Encode input to discrete latent codes."""
        # Get encoder output
        z_e = self.encoder(x)  # (B, embedding_dim, h, w)

        # Flatten spatial dimensions
        B, C, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # Find nearest codebook entries
        distances = torch.cdist(z_e_flat, self.codebook.weight)
        indices = distances.argmin(dim=1)  # (B * H * W,)
        indices = indices.reshape(B, H * W)  # (B, n_latents)

        if as_long:
            return indices

        if return_one_hot:
            one_hot = F.one_hot(indices, num_classes=self.n_embeddings)
            one_hot = one_hot.permute(0, 2, 1).float()  # (B, n_embeddings, n_latents)
            return one_hot

        return indices

    def forward(self, x):
        return self.encode(x, return_one_hot=True)


# ============================================================================
# Model Loading
# ============================================================================

def infer_policy_input_dim(policy_state, codebook_size, embedding_dim):
    """
    Infer the policy's expected input dimension from the first *Linear* layer,
    skipping the Embedding matrix which is (codebook_size, embedding_dim).
    """
    candidates = []
    for k, v in policy_state.items():
        if not k.endswith("weight") or v.dim() != 2:
            continue

        # Skip embedding weight (n_embeds, embed_dim)
        if v.shape == (codebook_size, embedding_dim):
            continue

        # Likely a Linear layer: (out_features, in_features)
        candidates.append((k, v.shape[1]))

    if not candidates:
        raise RuntimeError("Could not infer policy input_dim from policy_state_dict (no Linear weights found).")

    # If policy is nn.Sequential, keys are like "3.weight", "5.weight", ...
    def layer_index(key):
        head = key.split(".")[0]
        return int(head) if head.isdigit() else 10**9

    candidates.sort(key=lambda x: layer_index(x[0]))
    return candidates[0][1]


def load_model(model_path, env_name, device):
    """Load the trained model from checkpoint."""
    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get('args', {})

    print(f"Checkpoint step: {checkpoint.get('step', 'N/A')}")
    print(f"Checkpoint avg_reward: {checkpoint.get('avg_reward', 'N/A')}")

    # Create environment to get dimensions
    env = make_env(env_name)
    act_dim = env.action_space.n

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result

    sample_obs_tensor = preprocess_obs([sample_obs])
    input_shape = sample_obs_tensor.shape[1:]  # (C, H, W)

    # Get model config from saved args
    ae_model_type = saved_args.get('ae_model_type', 'vqvae')
    codebook_size = saved_args.get('codebook_size', 16)
    embedding_dim = saved_args.get('embedding_dim', 64)
    policy_hidden = saved_args.get('policy_hidden', [256, 256])
    rl_activation = saved_args.get('rl_activation', 'relu')

    print(f"\nModel config:")
    print(f"  ae_model_type: {ae_model_type}")
    print(f"  codebook_size: {codebook_size}")
    print(f"  embedding_dim: {embedding_dim}")
    print(f"  policy_hidden: {policy_hidden}")
    print(f"  input_shape: {input_shape}")

    # Infer expected policy input dim from checkpoint (skip embedding weight)
    policy_state = checkpoint.get('policy_state_dict', {})
    expected_input_dim = infer_policy_input_dim(policy_state, codebook_size, embedding_dim)
    print(f"  Policy expects input_dim={expected_input_dim}")

    # For VQ-VAE discrete pipeline: expected_input_dim = embedding_dim * n_latents
    target_n_latents = None
    if ae_model_type == 'vqvae':
        if expected_input_dim % embedding_dim != 0:
            raise RuntimeError(
                f"Policy input_dim={expected_input_dim} not divisible by embedding_dim={embedding_dim}."
            )
        target_n_latents = expected_input_dim // embedding_dim
        target_spatial = int(math.sqrt(target_n_latents))
        print(f"  Inferred target_n_latents={target_n_latents} (approx spatial {target_spatial}x{target_spatial})")

    # Get encoder state dict if available
    ae_state_dict = checkpoint.get('ae_model_state_dict', None)

    # Create encoder with architecture inferred from state dict
    encoder = SimpleVQVAEEncoder(
        input_shape=input_shape,
        embedding_dim=embedding_dim,
        n_embeddings=codebook_size,
        state_dict=ae_state_dict,
        target_n_latents=target_n_latents
    )

    # Load encoder weights
    if ae_state_dict is not None:
        try:
            encoder.load_state_dict(ae_state_dict, strict=False)
            print("Loaded encoder weights from checkpoint")
        except Exception as e:
            print(f"Warning: Could not load encoder weights: {e}")
            print("Using randomly initialized encoder (this may cause issues)")

    encoder = encoder.to(device)
    encoder.eval()

    # Sanity check: encoder output must match policy expected input
    if ae_model_type == 'vqvae':
        got_input_dim = embedding_dim * encoder.n_latent_embeds
        if got_input_dim != expected_input_dim:
            raise RuntimeError(
                f"Encoder produces input_dim={got_input_dim} (embedding_dim={embedding_dim} * n_latents={encoder.n_latent_embeds}), "
                f"but policy expects input_dim={expected_input_dim}. "
                f"This means the inferred conv strides/pads didn't match the training architecture."
            )

    # Build policy using checkpoint-expected input dim (NOT recomputed from encoder)
    input_dim = expected_input_dim
    print(f"  Policy input_dim (from checkpoint): {input_dim}")

    if isinstance(policy_hidden, str):
        policy_hidden = eval(policy_hidden)

    policy = mlp(
        [input_dim] + list(policy_hidden) + [act_dim],
        activation=rl_activation,
        discrete_input=(ae_model_type == 'vqvae'),
        n_embeds=codebook_size,
        embed_dim=embedding_dim
    )

    # Load policy weights (now shapes should match)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    policy = policy.to(device)
    policy.eval()

    env.close()

    return encoder, policy, saved_args


def get_action(encoder, policy, obs, device, deterministic=False):
    """Get action from policy given observation."""
    obs_tensor = preprocess_obs([obs])

    with torch.no_grad():
        state = encoder.encode(obs_tensor.to(device), return_one_hot=True)
        logits = policy(state)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1).item()
        else:
            action = torch.multinomial(probs, 1).item()

    return action, probs.cpu().numpy()[0]


# ============================================================================
# Episode Runner
# ============================================================================

def run_episode(env, encoder, policy, device, max_steps=100,
                deterministic=False, delay=0.1, verbose=True, collect_frames=False):
    """Run a single episode and return statistics."""

    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    total_reward = 0
    steps = 0
    frames = []

    # MiniGrid action names
    action_names = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

    if verbose:
        print("\n" + "=" * 50)
        print("Starting Episode")
        print("=" * 50)

    for step in range(max_steps):
        # Collect frame if needed
        if collect_frames:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                pass

        # Get action
        action, probs = get_action(encoder, policy, obs, device, deterministic)

        if verbose:
            probs_str = ', '.join([f'{action_names[i]}:{p:.2f}' for i, p in enumerate(probs[:len(action_names)])])
            print(f"Step {step + 1}: action={action_names[min(action, len(action_names) - 1)]}, probs=[{probs_str}]")

        # Take step
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        total_reward += reward
        steps += 1

        if reward > 0 and verbose:
            print(f"  -> Got reward: {reward:.4f}")

        # Add delay for visualization
        if delay > 0 and not collect_frames:
            time.sleep(delay)

        if done:
            if verbose:
                print(f"\nEpisode finished!")
                print(f"  Total steps: {steps}")
                print(f"  Total reward: {total_reward:.4f}")
            break

    # Collect final frame
    if collect_frames:
        try:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        except Exception:
            pass

    return {
        'total_reward': total_reward,
        'steps': steps,
        'frames': frames
    }


def save_gif(frames, output_path, fps=5):
    """Save frames as GIF."""
    try:
        import imageio
        duration = 1000 / fps
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
        print(f"GIF saved to: {output_path}")
        return True
    except ImportError:
        print("imageio not installed. Install with: pip install imageio")
        return False


def save_video(frames, output_path, fps=5):
    """Save frames as video."""
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Video saved to: {output_path}")
        return True
    except ImportError:
        print("imageio not installed. Install with: pip install imageio imageio-ffmpeg")
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize trained MiniGrid agent')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--env_name', type=str, default='MiniGrid-Empty-5x5-v0',
                        help='MiniGrid environment name')
    parser.add_argument('--n_episodes', type=int, default=3,
                        help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Maximum steps per episode')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic (greedy) action selection')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Delay between steps (seconds)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--save_gif', type=str, default=None,
                        help='Path to save GIF (e.g., rollout.gif)')
    parser.add_argument('--save_video', type=str, default=None,
                        help='Path to save video (e.g., rollout.mp4)')
    parser.add_argument('--fps', type=int, default=5,
                        help='FPS for saved video/gif')
    parser.add_argument('--no_render', action='store_true',
                        help='Disable rendering (useful for headless environments)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce verbosity')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"Device: {args.device}")

    # Load model
    encoder, policy, saved_args = load_model(args.model_path, args.env_name, args.device)

    # Determine render mode
    collect_frames = args.save_gif or args.save_video
    if collect_frames:
        render_mode = 'rgb_array'
    elif args.no_render:
        render_mode = None
    else:
        render_mode = 'human'

    # Create environment
    print(f"\nCreating environment: {args.env_name}")
    env = make_env(args.env_name, render_mode=render_mode)

    # Run episodes
    all_rewards = []
    all_steps = []
    all_frames = []

    verbose = not args.quiet

    for ep in range(args.n_episodes):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Episode {ep + 1}/{args.n_episodes}")
            print(f"{'=' * 50}")

        result = run_episode(
            env, encoder, policy, args.device,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            delay=args.delay if not collect_frames else 0,
            verbose=verbose,
            collect_frames=collect_frames
        )

        all_rewards.append(result['total_reward'])
        all_steps.append(result['steps'])
        if result['frames']:
            all_frames.extend(result['frames'])

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Episodes: {args.n_episodes}")
    print(f"Average reward: {np.mean(all_rewards):.4f} (+/- {np.std(all_rewards):.4f})")
    print(f"Average steps: {np.mean(all_steps):.1f} (+/- {np.std(all_steps):.1f})")
    print(f"Min/Max reward: {np.min(all_rewards):.4f} / {np.max(all_rewards):.4f}")

    # Save video/gif
    if all_frames:
        if args.save_gif:
            save_gif(all_frames, args.save_gif, fps=args.fps)
        if args.save_video:
            save_video(all_frames, args.save_video, fps=args.fps)

    env.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
