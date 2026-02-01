#!/usr/bin/env python3
"""
Visualize trained RL model rollouts on MiniGrid environments.

Fixes:
1) Capture the terminal (post-step) frame:
   - render once after reset() (initial frame)
   - render after each env.step(action), so final frame shows the goal state
   => len(frames) == steps + 1

2) Avoid argparse conflicts with make_argparser():
   - add arguments only if not already defined (e.g., --stochastic)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "discrete_mbrl"))

import numpy as np
import torch
from torch.distributions import Categorical

# Try importing visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not found. Install with: pip install Pillow")

try:
    from IPython.display import HTML, display

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Import project modules
from discrete_mbrl.env_helpers import make_env, preprocess_obs
from discrete_mbrl.model_construction import construct_ae_model
from discrete_mbrl.training_helpers import freeze_model, make_argparser
from discrete_mbrl.model_free.rl_utils import interpret_layer_sizes

# Import for building policy network
from shared.models import mlp


def add_argument_if_missing(parser: argparse.ArgumentParser, *flags, **kwargs):
    """
    Add an argparse argument only if none of the option strings already exist.
    This prevents conflicts when make_argparser() already defines some flags.
    """
    for f in flags:
        if f in parser._option_string_actions:
            return
    parser.add_argument(*flags, **kwargs)


def load_model_free_checkpoint(model_path, encoder, args, device):
    """Load the saved model-free checkpoint and reconstruct policy/critic"""

    # Handle PyTorch 2.6+ security changes
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        print(f"Note: Loading with weights_only=False due to: {type(e).__name__}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    print(f"Checkpoint keys: {checkpoint.keys()}")
    print(f"Model info: {checkpoint.get('model_info', 'N/A')}")

    # Get dimensions
    if args.ae_model_type == "vqvae":
        input_dim = args.embedding_dim * encoder.n_latent_embeds
    else:
        input_dim = encoder.latent_dim

    # Get action dimension from environment
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_dim = env.action_space.n
    env.close()

    # Parse hidden sizes
    policy_hidden = interpret_layer_sizes(getattr(args, "policy_hidden", [256, 256]))
    critic_hidden = interpret_layer_sizes(getattr(args, "critic_hidden", [256, 256]))

    # Build MLP kwargs
    mlp_kwargs = {
        "activation": getattr(args, "rl_activation", "relu"),
        "discrete_input": args.ae_model_type == "vqvae",
    }
    if args.ae_model_type == "vqvae":
        mlp_kwargs["n_embeds"] = args.codebook_size
        mlp_kwargs["embed_dim"] = args.embedding_dim

    # Reconstruct policy and critic
    policy = mlp([input_dim] + policy_hidden + [act_dim], **mlp_kwargs)
    critic = mlp([input_dim] + critic_hidden + [1], **mlp_kwargs)

    # Load weights
    policy.load_state_dict(checkpoint["policy_state_dict"])
    critic.load_state_dict(checkpoint["critic_state_dict"])

    policy = policy.to(device).eval()
    critic = critic.to(device).eval()

    return policy, critic


def get_action(policy, state, device, deterministic=True):
    """Get action from policy"""
    with torch.no_grad():
        state_tensor = state.unsqueeze(0).to(device)
        logits = policy(state_tensor)

        if deterministic:
            return logits.argmax(dim=-1).item()
        dist = Categorical(logits=logits)
        return dist.sample().item()


def render_frame(env):
    """Best-effort render into an RGB array frame."""
    try:
        render_env = env
        while hasattr(render_env, "env"):
            render_env = render_env.env

        if hasattr(render_env, "get_full_render"):
            return render_env.get_full_render(highlight=True, tile_size=32)
        if hasattr(render_env, "render"):
            return render_env.render()
    except Exception:
        return None
    return None


def run_episode(env, encoder, policy, device, max_steps=1000, deterministic=True, debug_done=False):
    """
    Run a single episode and collect frames.

    Frames policy:
    - frames[0] is the initial state right after reset()
    - after each action: step(), then capture the post-step state (including terminal)
    So len(frames) == steps + 1
    """
    frames = []
    rewards = []
    actions = []

    # Reset
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
    else:
        obs = reset_result
        info = {}

    # Initial frame
    f0 = render_frame(env)
    if f0 is not None:
        frames.append(f0)

    done = False
    step = 0
    total_reward = 0.0

    while (not done) and (step < max_steps):
        # Encode observation
        obs_tensor = preprocess_obs([obs])
        with torch.no_grad():
            state = encoder.encode(obs_tensor.to(device), return_one_hot=True).squeeze(0)

        # Action
        action = get_action(policy, state, device, deterministic=deterministic)
        actions.append(action)

        # Step
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = bool(terminated or truncated)
            if debug_done and done:
                print(f"[DONE] terminated={terminated} truncated={truncated} reward={reward} step={step+1}")
        else:
            obs, reward, done, info = step_result

        rewards.append(float(reward))
        total_reward += float(reward)
        step += 1

        # Post-step frame (includes terminal frame)
        f = render_frame(env)
        if f is not None:
            frames.append(f)

    return frames, rewards, actions, total_reward, step


def create_video(frames, save_path, fps=5):
    """Create GIF from frames"""
    if not frames:
        print("No frames to save!")
        return

    if not HAS_PIL:
        print("PIL not available, cannot save GIF")
        return

    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        save_path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / max(1, fps)),
        loop=0,
    )
    print(f"Saved GIF to: {save_path}")


def display_frames_interactive(frames, rewards, delay=0.2):
    """Display frames interactively one by one (frames are steps+1)."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available")
        return
    if not frames:
        print("No frames to display!")
        return

    # Align rewards to frames (frames = steps+1)
    cumulative_rewards = np.concatenate([[0.0], np.cumsum(rewards)]) if rewards else np.array([0.0])
    if len(cumulative_rewards) != len(frames):
        cumulative_rewards = cumulative_rewards[: len(frames)]

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i, frame in enumerate(frames):
        ax1.clear()
        ax1.imshow(frame)
        ax1.axis("off")
        ax1.set_title(f"Frame {i}/{len(frames)-1} (After step {max(0, i)})")

        ax2.clear()
        ax2.plot(cumulative_rewards[: i + 1], linewidth=2)
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Cumulative Reward")
        ax2.set_title(f"Total Reward: {cumulative_rewards[i]:.4f}")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.pause(delay)

    plt.ioff()
    plt.show()


def main():
    parser = make_argparser()

    # Add ONLY if missing to avoid conflicts with make_argparser()
    add_argument_if_missing(parser, "--model_path", type=str, required=True,
                            help="Path to the saved model checkpoint")
    add_argument_if_missing(parser, "--n_episodes", type=int, default=3,
                            help="Number of episodes to visualize")
    add_argument_if_missing(parser, "--save_gifs", action="store_true",
                            help="Save episodes as GIF files")
    add_argument_if_missing(parser, "--output_dir", type=str, default="./rollout_videos",
                            help="Directory to save GIFs")
    add_argument_if_missing(parser, "--fps", type=int, default=5,
                            help="Frames per second for visualization")
    add_argument_if_missing(parser, "--delay", type=float, default=0.3,
                            help="Delay between frames in interactive mode")
    add_argument_if_missing(parser, "--max_steps", type=int, default=200,
                            help="Maximum steps per episode")
    add_argument_if_missing(parser, "--policy_hidden", nargs="*", default=[256, 256])
    add_argument_if_missing(parser, "--critic_hidden", nargs="*", default=[256, 256])
    add_argument_if_missing(parser, "--rl_activation", default="relu")

    # This one was conflicting for you; now it will only be added if absent.
    add_argument_if_missing(parser, "--stochastic", action="store_true",
                            help="Use stochastic actions instead of argmax")

    # Debug-only flag (safe add)
    add_argument_if_missing(parser, "--debug_done", action="store_true",
                            help="Print termination/truncation info when done")

    args = parser.parse_args()

    print("=" * 60)
    print("Model Rollout Visualization")
    print("=" * 60)
    print(f"Environment: {args.env_name}")
    print(f"Model path: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Episodes: {args.n_episodes}")

    if not os.path.exists(args.model_path):
        print(f"\nError: Model not found at {args.model_path}")
        print("\nLooking for models in ./models/:")
        if os.path.exists("./models"):
            for root, _, files in os.walk("./models"):
                for f in files:
                    if f.endswith(".pt") or f.endswith(".zip"):
                        print(f"  {os.path.join(root, f)}")
        return

    print("\nCreating environment...")
    env = make_env(args.env_name, max_steps=args.env_max_steps)

    # Sample observation for AE construction
    reset_result = env.reset()
    sample_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    sample_obs = preprocess_obs([sample_obs])

    print(f"Observation shape: {sample_obs.shape[1:]}")
    print(f"Action space: {env.action_space}")

    print("\nLoading encoder...")
    encoder, _ = construct_ae_model(sample_obs.shape[1:], args, load=True)
    encoder = encoder.to(args.device)
    freeze_model(encoder)
    encoder.eval()
    print(f"Encoder type: {encoder.encoder_type}")

    print("\nLoading policy...")
    policy, critic = load_model_free_checkpoint(args.model_path, encoder, args, args.device)
    print("Policy loaded successfully!")

    if getattr(args, "save_gifs", False):
        os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("Running Episodes")
    print("=" * 60)

    all_rewards = []
    all_lengths = []

    deterministic = not bool(getattr(args, "stochastic", False))

    for ep in range(args.n_episodes):
        print(f"\n--- Episode {ep + 1}/{args.n_episodes} ---")

        frames, rewards, actions, total_reward, steps = run_episode(
            env,
            encoder,
            policy,
            args.device,
            max_steps=args.max_steps,
            deterministic=deterministic,
            debug_done=bool(getattr(args, "debug_done", False)),
        )

        all_rewards.append(total_reward)
        all_lengths.append(steps)

        print(f"Total Reward: {total_reward:.4f}")
        print(f"Episode Length (steps): {steps}")
        print(f"Frames captured: {len(frames)} (should be steps+1)")

        if frames:
            if getattr(args, "save_gifs", False):
                gif_path = os.path.join(args.output_dir, f"episode_{ep + 1}.gif")
                create_video(frames, gif_path, fps=args.fps)

            if HAS_MATPLOTLIB:
                print(f"\nDisplaying episode {ep + 1}...")
                print("(Close the window to continue to next episode)")
                display_frames_interactive(frames, rewards, delay=args.delay)
        else:
            print("No frames captured for this episode")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes: {args.n_episodes}")
    print(f"Mean Reward: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"Mean Length: {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")

    if getattr(args, "save_gifs", False):
        print(f"\nGIFs saved to: {args.output_dir}/")

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
