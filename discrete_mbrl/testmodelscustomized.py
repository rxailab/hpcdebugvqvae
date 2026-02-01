#!/usr/bin/env python3
"""
Simplified MiniGrid Model Debug Tool - WITH HEATMAP - FIXED FORMAT
+ Prints AE checkpoint / weights names used (best-effort via args + model metadata)
"""

import sys
import os
import numpy as np
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_helpers import make_env
from model_construction import construct_ae_model, construct_trans_model
from training_helpers import make_argparser, process_args
from data_logging import init_experiment


class ModelDebugGUI:
    def __init__(self, args):
        self.args = args
        self.real_obs = None
        self.model_obs = None
        self.model_state = None
        self.quantized_indices = None
        self.step_count = 0
        print(f"Loading models for {args.env_name} ({args.ae_model_type} + {args.trans_model_type})")

        self.setup_models()
        self.setup_environment()
        self.setup_gui()
        self.reset_environment()

    # ---------------------------
    # NEW: checkpoint printing
    # ---------------------------
    def _looks_like_ckpt_key(self, k: str) -> bool:
        k = k.lower()
        return any(s in k for s in ["ckpt", "checkpoint", "resume", "pretrain", "weights", "model_path", "load", "path"])

    def _looks_like_path(self, v) -> bool:
        if not isinstance(v, str):
            return False
        v2 = v.lower()
        return any(v2.endswith(ext) for ext in [".pt", ".pth", ".ckpt", ".bin"]) or ("/" in v2) or ("\\" in v2)

    def _shorten_path(self, p: str, keep: int = 2) -> str:
        """Return a shortened path for printing while still being informative."""
        if not isinstance(p, str):
            return str(p)
        p = p.strip()
        if len(p) == 0:
            return p
        parts = p.replace("\\", "/").split("/")
        if len(parts) <= (2 * keep + 1):
            return p
        return "/".join(parts[:keep]) + "/.../" + "/".join(parts[-keep:])

    def print_checkpoint_info(self):
        """
        Print any args fields that look like checkpoint paths,
        and any model-attached ckpt metadata if available.
        """
        print("\n=== Checkpoint / Weights Info (best-effort) ===")

        # 1) Print args.* that likely contain checkpoint paths
        arg_items = sorted(vars(self.args).items(), key=lambda x: x[0])
        hits = []
        for k, v in arg_items:
            if self._looks_like_ckpt_key(k) and (self._looks_like_path(v) or isinstance(v, (str, int, float, bool))):
                hits.append((k, v))

        if hits:
            print("Args fields that look checkpoint-related:")
            for k, v in hits:
                if isinstance(v, str):
                    print(f"  - args.{k} = {v}  (short: {self._shorten_path(v)})")
                else:
                    print(f"  - args.{k} = {v}")
        else:
            print("No obvious checkpoint-related fields found in args.")

        # 2) Print possible model-attached checkpoint metadata (common attribute names)
        for name, model in [
            ("encoder/AE", getattr(self, "encoder", None)),
            ("transition", getattr(self, "transition", None)),
        ]:
            if model is None:
                continue

            print(f"\nModel metadata hints for {name}:")
            found_any = False
            for attr in [
                "ckpt_path",
                "checkpoint_path",
                "weights_path",
                "load_path",
                "_ckpt_path",
                "_checkpoint_path",
                "pretrained_path",
            ]:
                if hasattr(model, attr):
                    val = getattr(model, attr)
                    if isinstance(val, str):
                        print(f"  - {attr} = {val}  (short: {self._shorten_path(val)})")
                    else:
                        print(f"  - {attr} = {val}")
                    found_any = True
            if not found_any:
                print("  (no common ckpt-path attributes found)")

        print("=== End Checkpoint Info ===\n")

    def setup_models(self):
        """Load encoder and transition models."""
        # Get observation shape
        temp_env = make_env(self.args.env_name)
        obs = temp_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        temp_env.close()

        # Load models
        self.encoder = construct_ae_model(obs.shape, self.args)[0].to(self.args.device).eval()
        self.transition = construct_trans_model(
            self.encoder,
            self.args,
            make_env(self.args.env_name).action_space
        )[0].to(self.args.device).eval()

        print(f"✓ Models loaded: {type(self.encoder).__name__} + {type(self.transition).__name__}")

        # NEW: print checkpoint names / paths used (best-effort)
        self.print_checkpoint_info()

        # Get quantizer info for heatmap
        if hasattr(self.encoder, 'quantizer'):
            print(f"✓ Quantizer found: {self.encoder.n_embeddings} codebook entries")
            self.has_quantizer = True
            self.codebook_size = self.encoder.n_embeddings
        else:
            print("⚠ No quantizer found - heatmap will show encoded values")
            self.has_quantizer = False
            self.codebook_size = 256  # Default for visualization

    def setup_environment(self):
        """Setup real environment."""
        self.env = make_env(self.args.env_name)

    def setup_gui(self):
        """Create GUI interface."""
        self.root = tk.Tk()
        self.root.title(f"Model Debug: {self.args.env_name}")
        self.root.geometry("900x600")

        # Main frame
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill='both', expand=True)

        # Title
        ttk.Label(
            main,
            text=f"Model Debug: {self.args.ae_model_type} + {self.args.trans_model_type}",
            font=("Arial", 12, "bold")
        ).pack(pady=(0, 10))

        # Images frame - now with 3 panels
        img_frame = ttk.Frame(main)
        img_frame.pack(pady=10)

        # Real observation
        real_frame = ttk.LabelFrame(img_frame, text="Real", padding="5")
        real_frame.pack(side='left', padx=5)
        self.real_canvas = tk.Canvas(real_frame, width=150, height=150, bg="white")
        self.real_canvas.pack()

        # Predicted observation
        pred_frame = ttk.LabelFrame(img_frame, text="Predicted", padding="5")
        pred_frame.pack(side='left', padx=5)
        self.pred_canvas = tk.Canvas(pred_frame, width=150, height=150, bg="white")
        self.pred_canvas.pack()

        # Quantized embedding heatmap
        heatmap_frame = ttk.LabelFrame(img_frame, text="Quantized Indices", padding="5")
        heatmap_frame.pack(side='right', padx=5)

        # Create matplotlib figure for heatmap with fixed layout
        self.fig, self.ax = plt.subplots(figsize=(3, 3))
        self.ax.set_title("Codebook Indices", fontsize=10)
        self.colorbar = None  # Initialize colorbar as None

        # Set fixed position for the axes to prevent shrinking
        self.ax.set_position([0.1, 0.1, 0.65, 0.8])  # [left, bottom, width, height]

        # Embed matplotlib in tkinter
        self.heatmap_canvas = FigureCanvasTkAgg(self.fig, heatmap_frame)
        self.heatmap_canvas.get_tk_widget().pack()

        # Info frame
        info_frame = ttk.Frame(main)
        info_frame.pack(pady=10)

        info_left = ttk.Frame(info_frame)
        info_left.pack(side='left', padx=20)
        self.step_label = ttk.Label(info_left, text="Steps: 0")
        self.step_label.pack()
        self.indices_label = ttk.Label(info_left, text="Unique indices: 0")
        self.indices_label.pack()

        info_right = ttk.Frame(info_frame)
        info_right.pack(side='right', padx=20)
        if self.has_quantizer:
            self.codebook_label = ttk.Label(info_right, text=f"Codebook size: {self.codebook_size}")
            self.codebook_label.pack()

        # Controls
        controls = ttk.LabelFrame(main, text="Controls", padding="10")
        controls.pack(pady=10, fill='x')
        ttk.Label(controls, text="↑=Forward, ←=Left, →=Right, R=Reset, Q=Quit").pack()

        # Key bindings
        self.action_map = {'Up': 2, 'Left': 0, 'Right': 1}
        self.root.bind('<KeyPress>', self.on_key_press)
        self.root.focus_set()

    def preprocess_obs(self, obs):
        """Convert observation to tensor."""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()
        if len(obs.shape) == 3:
            obs = obs.unsqueeze(0)
        return obs

    def extract_quantized_indices(self, obs_tensor):
        """Extract quantized indices for heatmap visualization."""
        try:
            with torch.no_grad():
                if hasattr(self.encoder, 'quantizer'):
                    # Get the quantized indices from the VQ-VAE's encode method
                    encoded_output = self.encoder.encode(obs_tensor)

                    if isinstance(encoded_output, torch.Tensor):
                        indices = encoded_output[0].cpu().numpy()  # Remove batch dimension

                        # Handle the flattened 64-element array -> reshape to 8x8 grid
                        if indices.ndim == 1 and len(indices) == 64:
                            return indices.reshape(8, 8).astype(int)
                        elif indices.ndim == 2 and indices.shape == (8, 8):
                            return indices.astype(int)
                        else:
                            # For other shapes, try to make the best 2D representation
                            if indices.ndim == 1:
                                side_len = int(np.sqrt(len(indices)))
                                if side_len * side_len == len(indices):
                                    return indices.reshape(side_len, side_len).astype(int)
                                else:
                                    # Can't make square, pad to nearest square
                                    next_square = int(np.ceil(np.sqrt(len(indices)))) ** 2
                                    padded = np.pad(indices, (0, next_square - len(indices)), mode='constant')
                                    side_len = int(np.sqrt(next_square))
                                    return padded.reshape(side_len, side_len).astype(int)
                            return indices.astype(int)

                else:
                    # For non-VQ models, show the encoded representation
                    encoded = self.encoder.encode(obs_tensor)
                    if len(encoded.shape) > 2:
                        encoded = encoded[0].mean(dim=0).cpu().numpy()
                        encoded = ((encoded - encoded.min()) / (encoded.max() - encoded.min()) * 255).astype(int)
                        return encoded
                    else:
                        encoded = encoded[0].cpu().numpy()
                        side_len = int(np.sqrt(len(encoded)))
                        if side_len * side_len == len(encoded):
                            encoded = encoded.reshape(side_len, side_len)
                            encoded = ((encoded - encoded.min()) / (encoded.max() - encoded.min()) * 255).astype(int)
                            return encoded
                        else:
                            return np.zeros((8, 8), dtype=int)

                return np.zeros((8, 8), dtype=int)

        except Exception as e:
            print(f"Error extracting indices: {e}")
            return np.zeros((8, 8), dtype=int)

    def create_heatmap(self, indices):
        """Create heatmap visualization of quantized indices with overlaid text."""
        try:
            # Clear previous content
            for img in self.ax.get_images():
                img.remove()
            for text in self.ax.texts:
                text.remove()

            if indices is None:
                indices = np.zeros((8, 8), dtype=int)

            # Create heatmap
            if self.has_quantizer:
                im = self.ax.imshow(
                    indices,
                    cmap='tab20',
                    vmin=0,
                    vmax=self.codebook_size - 1,
                    interpolation='nearest',
                    aspect='equal'
                )
                self.ax.set_title(
                    f"Codebook Indices\n({indices.shape[0]}×{indices.shape[1]} grid)",
                    fontsize=10
                )

                # IMPROVED: Add text annotations with dynamic font sizing
                h, w = indices.shape

                # Calculate appropriate font size based on grid dimensions
                # Larger grids get smaller fonts
                base_fontsize = max(6, min(10, 80 // max(h, w)))

                # Only show text for reasonable grid sizes (adjust threshold as needed)
                if h <= 16 and w <= 16:  # Increased from 8x8 to 16x16
                    for i in range(h):
                        for j in range(w):
                            idx_val = indices[i, j]

                            # Better color contrast: use luminance calculation
                            # Get the color from the colormap for this value
                            norm_val = idx_val / (self.codebook_size - 1) if self.codebook_size > 1 else 0
                            rgba = cm.get_cmap('tab20')(norm_val % 1.0)  # Wrap for tab20

                            # Calculate luminance (perceived brightness)
                            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                            text_color = "black" if luminance > 0.5 else "white"

                            self.ax.text(
                                j, i, str(idx_val),
                                ha="center", va="center",
                                color=text_color,
                                fontsize=base_fontsize,
                                weight='bold',
                                # Add outline for better readability
                                bbox=dict(
                                    boxstyle='round,pad=0.1',
                                    facecolor='none',
                                    edgecolor='none',
                                    alpha=0
                                )
                            )
            else:
                # For non-quantized models
                im = self.ax.imshow(indices, cmap='viridis', interpolation='nearest', aspect='equal')
                self.ax.set_title(
                    f"Encoded Values\n({indices.shape[0]}×{indices.shape[1]} grid)",
                    fontsize=10
                )

            # Remove axes ticks
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            # Handle colorbar
            if self.colorbar is None:
                try:
                    cax = self.fig.add_axes([0.78, 0.1, 0.03, 0.8])
                    self.colorbar = self.fig.colorbar(im, cax=cax)
                except Exception as cb_error:
                    print(f"Colorbar creation error: {cb_error}")
            else:
                try:
                    self.colorbar.update_normal(im)
                except Exception:
                    try:
                        self.colorbar.remove()
                        cax = self.fig.add_axes([0.78, 0.1, 0.03, 0.8])
                        self.colorbar = self.fig.colorbar(im, cax=cax)
                    except Exception:
                        pass

            try:
                self.heatmap_canvas.draw()
            except Exception as draw_error:
                print(f"Canvas draw error: {draw_error}")

            return indices

        except Exception as e:
            print(f"Heatmap creation error: {e}")
            return indices

    def obs_to_image(self, obs, size=(150, 150)):
        """Convert observation to PIL image."""
        try:
            if isinstance(obs, torch.Tensor):
                obs = obs.detach().cpu().numpy()

            if len(obs.shape) == 4:
                obs = obs[0]
            if len(obs.shape) == 3 and obs.shape[0] <= 3:
                obs = obs.transpose(1, 2, 0)

            obs = np.clip(obs, 0, 1)

            if len(obs.shape) == 2:
                obs = np.stack([obs] * 3, axis=-1)

            img_array = (obs * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            return img.resize(size, Image.NEAREST)

        except Exception as e:
            print(f"Image conversion error: {e}")
            # Return red error image
            error_img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            error_img[:, :, 0] = 255
            return Image.fromarray(error_img)

    def convert_state_for_transition(self, encoded_state):
        """Convert encoded state for transition model input."""
        if self.args.trans_model_type in ['discrete', 'shared_vq', 'universal_vq']:
            # For discrete models: indices -> embeddings -> flatten
            if hasattr(self.encoder, 'quantizer') and hasattr(self.encoder.quantizer, 'embedding'):
                embeddings = self.encoder.quantizer.embedding(encoded_state.long())
                return embeddings.reshape(embeddings.shape[0], -1)
            else:
                # Fallback: just flatten the indices
                return encoded_state.reshape(encoded_state.shape[0], -1)

        # For continuous models: just reshape/flatten
        return encoded_state.reshape(encoded_state.shape[0], -1)

    def convert_state_for_decoding(self, trans_output):
        """Convert transition output back for decoding."""
        if self.args.trans_model_type in ['discrete', 'shared_vq', 'universal_vq']:
            # For discrete models: embeddings -> indices
            if hasattr(self.encoder, 'quantizer') and len(trans_output.shape) > 2:
                batch_size = trans_output.shape[0]
                embed_dim = self.encoder.quantizer.embedding.weight.shape[1]
                n_embeds = trans_output.shape[1] // embed_dim
                reshaped = trans_output.reshape(batch_size, n_embeds, embed_dim)

                if hasattr(self.encoder.quantizer, 'forward'):
                    _, indices, _ = self.encoder.quantizer(reshaped)
                    return indices
                return reshaped
        return trans_output

    def reset_environment(self):
        """Reset environment and sync model state."""
        print("Resetting environment...")

        # Reset real environment
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self.real_obs, _ = reset_result
        else:
            self.real_obs = reset_result

        # Sync model state and extract quantized indices
        with torch.no_grad():
            obs_tensor = self.preprocess_obs(self.real_obs).to(self.args.device)

            # Extract quantized indices for heatmap
            self.quantized_indices = self.extract_quantized_indices(obs_tensor)

            # Get model state and prediction
            self.model_state = self.encoder.encode(obs_tensor)
            model_obs_tensor = self.encoder.decode(self.model_state)
            self.model_obs = model_obs_tensor.cpu().numpy()[0]

        self.step_count = 0
        self.update_display()
        print("✓ Reset complete")

    def step_environment(self, action):
        """Take a step in both real and model environments."""
        try:
            # Step real environment
            step_result = self.env.step(action)
            if len(step_result) == 4:
                self.real_obs, reward, done, info = step_result
            else:
                self.real_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # Step model prediction
            with torch.no_grad():
                # Convert current state for transition model
                current_state = self.convert_state_for_transition(self.model_state)

                # Create action tensor
                action_tensor = torch.tensor([action], dtype=torch.long).to(self.args.device)

                # Predict next state
                trans_output = self.transition(current_state, action_tensor)
                if isinstance(trans_output, tuple):
                    next_state = trans_output[0]
                else:
                    next_state = trans_output

                # Convert back for decoding and storage
                next_state_for_decode = self.convert_state_for_decoding(next_state)

                # Update model state and observation
                if self.args.trans_model_type in ['discrete', 'shared_vq', 'universal_vq']:
                    self.model_state = next_state_for_decode
                else:
                    self.model_state = next_state

                pred_obs_tensor = self.encoder.decode(next_state_for_decode)
                self.model_obs = pred_obs_tensor.cpu().numpy()[0]

                # Update quantized indices for predicted state
                pred_obs_tensor_input = torch.from_numpy(self.model_obs).unsqueeze(0).to(self.args.device)
                self.quantized_indices = self.extract_quantized_indices(pred_obs_tensor_input)

            self.step_count += 1
            self.update_display()

            if done:
                print(f"Episode done after {self.step_count} steps")
                self.root.after(2000, self.reset_environment)

        except Exception as e:
            print(f"Step error: {e}")
            # Continue with real environment only
            self.step_count += 1
            self.update_display()

    def update_display(self):
        """Update GUI images and heatmap."""
        try:
            # Update real image
            if self.real_obs is not None:
                real_img = self.obs_to_image(self.real_obs)
                self.real_photo = ImageTk.PhotoImage(real_img)
                self.real_canvas.delete("all")
                self.real_canvas.create_image(75, 75, image=self.real_photo)

            # Update predicted image
            if self.model_obs is not None:
                pred_img = self.obs_to_image(self.model_obs)
                self.pred_photo = ImageTk.PhotoImage(pred_img)
                self.pred_canvas.delete("all")
                self.pred_canvas.create_image(75, 75, image=self.pred_photo)

            # Update heatmap
            if self.quantized_indices is not None:
                self.create_heatmap(self.quantized_indices)

                # Update info labels
                unique_indices = len(np.unique(self.quantized_indices))
                self.indices_label.config(text=f"Unique indices: {unique_indices}")

            # Update step counter
            self.step_label.config(text=f"Steps: {self.step_count}")

        except Exception as e:
            print(f"Display update error: {e}")

    def on_key_press(self, event):
        """Handle keyboard input."""
        key = event.keysym

        if key.lower() == 'q':
            self.quit()
        elif key.lower() == 'r':
            self.reset_environment()
        elif key in self.action_map:
            self.step_environment(self.action_map[key])

    def quit(self):
        """Clean up and quit."""
        try:
            self.env.close()
        except Exception:
            pass

        # Clean up matplotlib resources
        try:
            if hasattr(self, 'colorbar') and self.colorbar is not None:
                self.colorbar.remove()
        except Exception:
            pass

        try:
            plt.close(self.fig)  # Close matplotlib figure
        except Exception:
            pass

        self.root.quit()
        self.root.destroy()

    def run(self):
        """Start the GUI."""
        print("Starting GUI... Use arrow keys to control, R to reset, Q to quit")
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()


def main():
    # Parse arguments using the same system as other scripts
    parser = make_argparser()
    args = parser.parse_args()
    args = process_args(args)

    # Disable logging
    args.wandb = False
    args.comet_ml = False
    args = init_experiment('debug', args)

    print(f"Environment: {args.env_name}")
    print(f"Models: {args.ae_model_type} + {args.trans_model_type}")
    print(f"Device: {args.device}")

    try:
        app = ModelDebugGUI(args)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
