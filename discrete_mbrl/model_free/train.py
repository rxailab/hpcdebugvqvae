from collections import defaultdict
import os
import sys

import numpy as np
import torch
import torch.optim as optim

from ..env_helpers import preprocess_obs

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

from torch.distributions import Categorical
from tqdm import tqdm

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from .data import SizedReplayBuffer as ReplayBuffer
from .ppo import ortho_init, PPOTrainer
from ..data_logging import *
try:
    from shared.models import *
    from shared.trainers import *
except ModuleNotFoundError:
    from ..shared.models import *
    from ..shared.trainers import *
from ..env_helpers import *
try:
    from .training_helpers import *
except ModuleNotFoundError:
    try:
        from ..training_helpers import *
    except ModuleNotFoundError:
        from training_helpers import *
from ..model_construction import *
from .rl_utils import *


def save_best_model(ae_model, policy, critic, optimizer, step, args, avg_reward, suffix="best"):
    """
    Save a checkpoint to a FIXED filename (overwrites).
    Produces:
      ./models/<env_name>/best_model.pt
      ./models/<env_name>/final_model.pt
    """
    checkpoint = {
        'step': int(step),
        'avg_reward': float(avg_reward),
        'ae_model_state_dict': ae_model.state_dict(),
        'policy_state_dict': policy.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'model_info': {
            'ae_params': sum(p.numel() for p in ae_model.parameters()),
            'policy_params': sum(p.numel() for p in policy.parameters()),
            'critic_params': sum(p.numel() for p in critic.parameters()),
        }
    }

    save_dir = f"./models/{args.env_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_path = f"{save_dir}/{suffix}_model.pt"  # fixed name
    torch.save(checkpoint, save_path)
    print(f"Saved {suffix} model to: {save_path}  (avg_reward={avg_reward:.4f})")
    return save_path


def _maybe_compile(model, name="model"):
    """Safely apply torch.compile where supported. Skip on Windows to avoid TritonMissing."""
    if sys.platform.startswith("win"):
        print(f"Skipping torch.compile for {name} on Windows")
        return model

    if os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1":
        print(f"torch.compile disabled via TORCHDYNAMO_DISABLE for {name}")
        return model

    try:
        return torch.compile(model)
    except Exception as e:
        print(f"torch.compile skipped for {name}: {e}")
        return model


def _freeze_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_params(model: torch.nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def train(args, encoder_model=None):
    env = make_env(args.env_name, max_steps=args.env_max_steps)
    act_space = env.action_space
    act_dim = act_space.n

    # Sample obs for model construction
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        sample_obs, _ = reset_result
    else:
        sample_obs = reset_result
    sample_obs = preprocess_obs([sample_obs])

    # Load / construct encoder
    if encoder_model is None:
        ae_model, ae_trainer = construct_ae_model(
            sample_obs.shape[1:], args, latent_activation=True, load=args.load
        )
        if ae_trainer is not None:
            ae_trainer.log_freq = -1
    else:
        ae_model = encoder_model
        ae_trainer = None

    ae_model = _maybe_compile(ae_model, "ae_model")

    # Decide whether the encoder will be trained at all in this run
    # - e2e_loss: PPO gradients flow through encoder
    # - ae_recon_loss: AE trainer updates encoder/decoder
    train_encoder = bool(args.e2e_loss) or (bool(getattr(args, "ae_recon_loss", False)) and ae_trainer is not None)

    # IMPORTANT: if not training encoder, keep it frozen AND in eval mode always
    ae_model = ae_model.to(args.device)
    if train_encoder:
        _unfreeze_params(ae_model)
        ae_model.train()
    else:
        _freeze_params(ae_model)
        ae_model.eval()

    print("Loaded encoder")

    update_params(args)

    mlp_kwargs = {
        'activation': args.rl_activation,
        'discrete_input': args.ae_model_type == 'vqvae',
    }

    if args.ae_model_type == 'vqvae':
        mlp_kwargs['n_embeds'] = args.codebook_size
        mlp_kwargs['embed_dim'] = args.embedding_dim
        input_dim = args.embedding_dim * ae_model.n_latent_embeds
    else:
        input_dim = ae_model.latent_dim

    policy = mlp([input_dim] + args.policy_hidden + [act_dim], **mlp_kwargs).to(args.device)
    critic = mlp([input_dim] + args.critic_hidden + [1], **mlp_kwargs).to(args.device)

    policy = _maybe_compile(policy, "policy")
    critic = _maybe_compile(critic, "critic")

    if args.ortho_init:
        ortho_init(ae_model, policy, critic)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'AE Model Params: {count_params(ae_model)}')
    print(f'Policy Params: {count_params(policy)}')
    print(f'Critic Params: {count_params(critic)}')

    # Optimizer (only include encoder params if e2e_loss)
    all_params = list(policy.parameters()) + list(critic.parameters())
    if args.e2e_loss:
        all_params += list(ae_model.parameters())
    optimizer = optim.Adam(all_params, lr=args.learning_rate, eps=1e-5)

    ppo = PPOTrainer(
        env, policy, critic, ae_model, optimizer,
        ppo_iters=args.ppo_iters,
        ppo_clip=args.ppo_clip,
        minibatch_size=args.ppo_batch_size,
        value_coef=args.ppo_value_coef,
        entropy_coef=args.ppo_entropy_coef,
        gae_lambda=args.ppo_gae_lambda,
        norm_advantages=args.ppo_norm_advantages,
        max_grad_norm=args.ppo_max_grad_norm,
        e2e_loss=args.e2e_loss,
    )

    replay_buffer = ReplayBuffer(args.replay_size) if args.ae_er_train else None

    run_stats = defaultdict(list)
    ep_info = defaultdict(list)

    # Best tracking
    best_avg_reward = float('-inf')
    recent_rewards = []
    reward_window = 10

    # Global episode stats (run_stats is reset at log_freq)
    all_episode_rewards = []
    all_episode_lengths = []

    # Rollout init
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        curr_obs, _ = reset_result
    else:
        curr_obs = reset_result
    curr_obs = torch.from_numpy(curr_obs).float()

    ep_rewards = []
    n_batches = int(np.ceil(args.mf_steps / args.batch_size))
    step = 0

    for _batch in tqdm(range(n_batches)):
        # During rollout, ALWAYS eval encoder unless training is explicitly needed.
        # (Even if training encoder, rollout is normally eval to avoid BN/dropout noise.)
        ae_model.eval()
        policy.eval()

        batch_data = {k: [] for k in ['obs', 'states', 'next_obs', 'rewards', 'acts', 'gammas']}

        for _ in range(args.batch_size):
            with torch.no_grad():
                env_change = (
                    isinstance(args.env_change_freq, int)
                    and args.env_change_freq > 0
                    and (step + 1) % args.env_change_freq == 0
                )

                model_device = next(ae_model.parameters()).device
                state = ae_model.encode(
                    curr_obs.unsqueeze(0).to(model_device),
                    return_one_hot=True
                )

                act_logits = policy(state)
                act_dist = Categorical(logits=act_logits)

                # keep action tensor shape [1] for PPO gather()
                act_tensor = act_dist.sample().cpu()     # shape: [1]
                act_int = int(act_tensor.item())         # python int for env.step()

            batch_data['obs'].append(curr_obs)
            batch_data['states'].append(state.squeeze(0))
            batch_data['acts'].append(act_tensor)

            # Step env
            step_result = env.step(act_int)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, info = step_result

            done = done or env_change

            next_obs = torch.from_numpy(next_obs).float()
            ep_rewards.append(reward)

            batch_data['next_obs'].append(next_obs)
            batch_data['rewards'].append(torch.tensor(reward).float())
            batch_data['gammas'].append(torch.tensor(args.gamma * (1 - done)).float())

            # Crafter achievement logging
            if 'achievements' in info:
                for k, v in info['achievements'].items():
                    run_stats[f'achievement/{k}'].append(v)
                    ep_info[f'achievement/{k}'].append(v)

            # Replay buffer
            if replay_buffer is not None:
                replay_buffer.add_step(
                    curr_obs, act_tensor, next_obs,
                    batch_data['rewards'][-1], batch_data['gammas'][-1]
                )

            if done:
                if replay_buffer is not None:
                    replay_buffer.add_step(
                        next_obs, act_tensor, next_obs,
                        batch_data['rewards'][-1], batch_data['gammas'][-1]
                    )

                if env_change or args.env_change_freq == 'episode':
                    if args.env_change_type == 'random':
                        env.seeds = [np.random.randint(0, 1000000)]
                    elif args.env_change_type == 'next':
                        env.seeds = [env.seeds[0] + 1]
                    else:
                        raise ValueError(f'Invalid env change type: {args.env_change_type}')

                reset_result = env.reset()
                if isinstance(reset_result, tuple):
                    curr_obs, _ = reset_result
                else:
                    curr_obs = reset_result
                curr_obs = torch.from_numpy(curr_obs).float()

                episode_reward = float(np.sum(ep_rewards))
                episode_length = int(len(ep_rewards))

                all_episode_rewards.append(episode_reward)
                all_episode_lengths.append(episode_length)

                recent_rewards.append(episode_reward)
                if len(recent_rewards) > reward_window:
                    recent_rewards.pop(0)

                current_avg_reward = float(np.mean(recent_rewards))

                # Save best only when (a) window is full AND (b) avg_reward > 0
                # This avoids spamming saves at 0.0 and stabilizes metric.
                if (len(recent_rewards) >= reward_window) and (current_avg_reward > best_avg_reward) and (current_avg_reward > 0.0):
                    best_avg_reward = current_avg_reward
                    print(f"New best average reward: {best_avg_reward:.4f} (over {len(recent_rewards)} episodes)")
                    if args.save:
                        save_best_model(
                            ae_model, policy, critic, optimizer,
                            step, args, best_avg_reward, suffix="best"
                        )

                run_stats['ep_length'].append(episode_length)
                run_stats['ep_reward'].append(episode_reward)

                if 'crafter' in args.env_name.lower():
                    achievement_keys = [k for k in ep_info.keys() if 'achievement' in k]
                    percents = np.array([np.mean(ep_info[k]) * 100 for k in achievement_keys])
                    score = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
                    run_stats['achievement/score'].append(score)

                #print('\n--- Episode Stats ---')
                #print(f'Reward: {episode_reward:.4f}')
                #print(f'Length: {episode_length}')
                #print(f'Avg Reward (last {len(recent_rewards)}): {current_avg_reward:.4f}')
                #print(f'Best Avg Reward: {best_avg_reward:.4f}')

                ep_rewards = []
                ep_info = defaultdict(list)
            else:
                curr_obs = next_obs

            update_stats(run_stats, {'reward': reward})

            # Logging
            if step > 0 and step % args.log_freq == 0:
                if 'crafter' in args.env_name.lower():
                    achievement_keys = [k for k in run_stats.keys() if 'achievement' in k]
                    percents = np.array([np.mean(run_stats[k]) * 100 for k in achievement_keys])
                    score = np.exp(np.nanmean(np.log(1 + percents), -1)) - 1
                    run_stats['achievement/score'].append(score)

                log_stats(run_stats, step, args)
                run_stats = defaultdict(list)

                # Recon images
                if step % (args.log_freq * args.checkpoint_freq) == 0:
                    recons = sample_recon_imgs(ae_model, batch_data['obs'], env_name=args.env_name)
                    log_images({'img_recon': recons}, args, step=step)

            step += 1

        # === Updates ===
        policy.train()

        # Only switch encoder to train if we are actually training it
        if train_encoder:
            ae_model.train()
        else:
            ae_model.eval()

        ae_model.to(args.device)
        policy.to(args.device)

        batch_data = {k: torch.stack(v).to(args.device) for k, v in batch_data.items()}

        # PPO updates
        if step >= args.rl_start_step:
            loss_dict = ppo.train(batch_data)
            for k, v in loss_dict.items():
                run_stats[k].append(v.item())

        # AE recon updates (ONLY if enabled AND trainer exists)
        if getattr(args, "ae_recon_loss", False) and ae_trainer is not None:
            for _ in range(args.n_ae_updates):
                if args.ae_er_train:
                    sampled = replay_buffer.sample(args.ae_batch_size or args.batch_size)
                    batch_obs = sampled[0]
                    batch_next_obs = sampled[2]
                else:
                    batch_obs = batch_data['obs']
                    batch_next_obs = batch_data['next_obs']

                loss_dict, ae_stats = ae_trainer.train((batch_obs, None, batch_next_obs))
                for k, v in {**loss_dict, **ae_stats}.items():
                    run_stats[k].append(v.item())

        # If encoder is meant to be frozen, ensure it stays frozen (guardrail)
        if not train_encoder:
            _freeze_params(ae_model)
            ae_model.eval()

    # Save final checkpoint
    if args.save:
        if len(all_episode_rewards) > 0:
            final_avg_reward = float(np.mean(all_episode_rewards[-reward_window:]))
            overall_avg_reward = float(np.mean(all_episode_rewards))
            # if best never improved beyond -inf (e.g., no rewards), set to best observed anyway
            best_report = best_avg_reward if best_avg_reward != float('-inf') else 0.0

            print(f"Training Complete!")
            print(f"   Final {reward_window}-episode average: {final_avg_reward:.4f}")
            print(f"   Overall average reward: {overall_avg_reward:.4f}")
            print(f"   Best rolling average: {best_report:.4f}")

            save_best_model(ae_model, policy, critic, optimizer, step, args, final_avg_reward, suffix="final")
        else:
            print("No episodes completed, saving final model anyway")
            save_best_model(ae_model, policy, critic, optimizer, step, args, 0.0, suffix="final")

    return policy, critic


if __name__ == '__main__':
    mf_arg_parser = make_mf_arg_parser()
    args = get_args(mf_arg_parser)

    if hasattr(args.env_change_freq, "isdecimal") and args.env_change_freq.isdecimal():
        args.env_change_freq = int(args.env_change_freq)

    args = init_experiment('discrete-mbrl-model-free', args)

    if args.wandb:
        args.update({'policy_hidden': interpret_layer_sizes(args.policy_hidden)}, allow_val_change=True)
        args.update({'critic_hidden': interpret_layer_sizes(args.critic_hidden)}, allow_val_change=True)
    else:
        args.policy_hidden = interpret_layer_sizes(args.policy_hidden)
        args.critic_hidden = interpret_layer_sizes(args.critic_hidden)

    policy, critic = train(args)


