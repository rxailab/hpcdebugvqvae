#!/usr/bin/env python3
"""
Encoder Evaluation Script - Compatible with shared argument parser
Evaluates and visualizes encoder reconstructions (robust to CHW/HWC + VQ index formats)

Key fixes vs your version:
- Robust CHW/HWC detection in obs_to_numpy()
- Robust VQ index extraction that handles tensors/tuples and multiple shapes
- Uses encoder(obs_tensor) for reconstruction when possible (matches training forward)
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .env_helpers import make_env
from .model_construction import construct_ae_model
from .training_helpers import make_argparser, process_args
from .data_logging import init_experiment


def preprocess_obs(obs):
    """Convert observation to (1,C,H,W) float tensor in [0,1] if possible."""
    if isinstance(obs, np.ndarray):
        obs_t = torch.from_numpy(obs).float()
    elif isinstance(obs, torch.Tensor):
        obs_t = obs.float()
    else:
        raise TypeError(f"Unsupported obs type: {type(obs)}")

    # Add batch dim
    if obs_t.ndim == 3:
        obs_t = obs_t.unsqueeze(0)

    # Try to ensure channel-first (B,C,H,W) if it looks like HWC
    # Heuristic: if last dim is 1/3/4 and second dim isn't, assume BHWC -> BCHW
    if obs_t.ndim == 4:
        # obs_t shape could be (B,H,W,C) or (B,C,H,W)
        if obs_t.shape[-1] in (1, 3, 4) and obs_t.shape[1] not in (1, 3, 4):
            obs_t = obs_t.permute(0, 3, 1, 2).contiguous()

    return obs_t


def obs_to_numpy(obs):
    """Convert observation/reconstruction to HWC numpy float in [0,1] for visualization."""
    if isinstance(obs, torch.Tensor):
        x = obs.detach().cpu().numpy()
    else:
        x = np.asarray(obs)

    # Remove batch if present
    if x.ndim == 4:
        x = x[0]

    # Now x is 3D or 2D
    if x.ndim == 3:
        # Detect CHW vs HWC deterministically
        if x.shape[-1] in (1, 3, 4):
            # already HWC
            pass
        elif x.shape[0] in (1, 3, 4):
            # CHW -> HWC
            x = np.transpose(x, (1, 2, 0))
        else:
            # ambiguous; default assume HWC
            pass

    x = np.clip(x, 0.0, 1.0)

    # If grayscale 2D -> RGB
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)

    # If has alpha channel, drop it for display
    if x.ndim == 3 and x.shape[-1] == 4:
        x = x[..., :3]

    return x


def extract_quantized_indices(encoder, obs_tensor):
    """
    Try hard to extract VQ code indices in a robust way.
    Returns a 2D numpy int array if possible, else None.

    Works for:
    - encode() returning indices (LongTensor) shaped [B,L] or [B,H,W]
    - encode() returning tuples/lists containing indices
    - some implementations returning (z_q, indices, ...)
    """
    if not hasattr(encoder, "quantizer"):
        return None

    with torch.no_grad():
        enc = encoder.encode(obs_tensor)

        # If encode returns (something, indices, ...)
        idx = None
        if isinstance(enc, (tuple, list)):
            # Prefer LongTensor candidates
            long_candidates = [
                t for t in enc
                if torch.is_tensor(t) and t.dtype in (torch.long, torch.int64)
            ]
            if len(long_candidates) > 0:
                idx = long_candidates[0]
            else:
                # Fallback: first tensor element
                for t in enc:
                    if torch.is_tensor(t):
                        idx = t
                        break
        else:
            idx = enc if torch.is_tensor(enc) else None

        if idx is None or (not torch.is_tensor(idx)):
            return None

        idx0 = idx[0].detach().cpu()  # remove batch

        # Already (H,W)
        if idx0.ndim == 2:
            return idx0.numpy().astype(int)

        # Flattened (L,)
        if idx0.ndim == 1:
            L = idx0.numel()
            side = int(np.sqrt(L))
            if side * side == L:
                return idx0.view(side, side).numpy().astype(int)
            if L == 64:
                return idx0.view(8, 8).numpy().astype(int)
            # Can't safely reshape
            return None

        # Sometimes comes as (H,W,?) or (C,H,W) etc — too ambiguous
        return None


def safe_forward_recon(encoder, obs_tensor):
    """
    Prefer encoder(obs_tensor) to match training forward.
    Falls back to encode+decode if needed.
    Returns recon tensor (B,C,H,W) on the same device as encoder.
    """
    with torch.no_grad():
        try:
            out = encoder(obs_tensor)
            if isinstance(out, (tuple, list)):
                recon = out[0]
            else:
                recon = out
            if torch.is_tensor(recon):
                return recon
        except Exception:
            pass

        # Fallback
        encoded = encoder.encode(obs_tensor)
        recon = encoder.decode(encoded)
        return recon


def evaluate_encoder(args):
    print(f"\n{'=' * 60}")
    print("Encoder Evaluation")
    print(f"{'=' * 60}")
    print(f"Environment: {args.env_name}")
    print(f"Model Type: {args.ae_model_type}")
    print(f"Device: {args.device}")
    print(f"{'=' * 60}\n")

    env = make_env(args.env_name)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    print(f"Raw observation shape: {np.array(obs).shape}")

    print("Loading encoder from checkpoint...")
    encoder = construct_ae_model(np.array(obs).shape, args)[0].to(args.device).eval()
    print(f"✓ Encoder loaded: {type(encoder).__name__}")

    if hasattr(encoder, "quantizer"):
        n_emb = getattr(encoder, "n_embeddings", None)
        if n_emb is not None:
            print(f"✓ VQ-VAE detected with {n_emb} codebook entries")
        else:
            print("✓ VQ-VAE detected (quantizer present)")

    print(f"\nCollecting {args.steps} environment samples...")
    observations = []
    reconstructions = []
    indices_list = []

    # Step through env and collect samples
    for step in range(args.steps):
        if step > 0:
            action = env.action_space.sample()
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, _, done, _ = step_result
            else:
                obs, _, terminated, truncated, _ = step_result
                done = terminated or truncated

            if done:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

        obs_tensor = preprocess_obs(np.array(obs)).to(args.device)

        # Recon via forward (preferred)
        recon = safe_forward_recon(encoder, obs_tensor)

        observations.append(np.array(obs))
        reconstructions.append(recon.detach().cpu().numpy()[0])

        # VQ indices (if possible)
        idx = extract_quantized_indices(encoder, obs_tensor)
        if idx is not None:
            indices_list.append(idx)

    env.close()
    print(f"✓ Collected {len(observations)} samples")

    # Reconstruction error (display-space)
    mse_errors = []
    for orig, recon in zip(observations, reconstructions):
        orig_np = obs_to_numpy(orig)
        recon_np = obs_to_numpy(recon)
        mse = float(np.mean((orig_np - recon_np) ** 2))
        mse_errors.append(mse)

    avg_mse = float(np.mean(mse_errors)) if len(mse_errors) > 0 else float("nan")
    print(f"\nAverage MSE (display-space): {avg_mse:.6f}")

    # Visualization
    print("\nCreating visualization...")
    n_display = min(args.ncols, len(observations))
    show_indices = len(indices_list) > 0
    n_rows = 3 if show_indices else 2

    fig = plt.figure(figsize=(3 * n_display, 3 * n_rows))
    gs = GridSpec(n_rows, n_display, figure=fig, hspace=0.25, wspace=0.15)

    for i in range(n_display):
        # Original
        ax_orig = fig.add_subplot(gs[0, i])
        ax_orig.imshow(obs_to_numpy(observations[i]))
        ax_orig.set_title(f"Original {i + 1}")
        ax_orig.axis("off")

        # Reconstruction
        ax_recon = fig.add_subplot(gs[1, i])
        ax_recon.imshow(obs_to_numpy(reconstructions[i]))
        ax_recon.set_title(f"Recon (MSE: {mse_errors[i]:.4f})")
        ax_recon.axis("off")

        # Indices (if available)
        if show_indices:
            ax_idx = fig.add_subplot(gs[2, i])
            if i < len(indices_list):
                im = ax_idx.imshow(indices_list[i], cmap="tab20", interpolation="nearest")
                ax_idx.set_title("Indices")
                ax_idx.axis("off")
                if i == 0:
                    plt.colorbar(im, ax=ax_idx, fraction=0.046, pad=0.04)
            else:
                ax_idx.set_title("Indices (N/A)")
                ax_idx.axis("off")

    plt.suptitle(
        f"{args.ae_model_type} Evaluation - {args.env_name}\nAvg MSE: {avg_mse:.6f}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if args.save_path:
        plt.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Saved to {args.save_path}")

    if args.show:
        print("✓ Displaying visualization (close window to exit)")
        plt.show()
    else:
        plt.close()

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"{'=' * 60}\n")


def main():
    # Use shared argument parser (same as ModelDebugGUI)
    parser = make_argparser()

    # Add evaluation-specific arguments
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to sample")
    parser.add_argument("--ncols", type=int, default=5, help="Number of columns in visualization")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save visualization")
    parser.add_argument("--show", action="store_true", help="Show visualization window")

    args = parser.parse_args()
    args = process_args(args)

    # Disable logging for evaluation
    args.wandb = False
    args.comet_ml = False
    args = init_experiment("eval_encoder", args)

    try:
        evaluate_encoder(args)
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
