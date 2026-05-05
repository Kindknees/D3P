"""
Evaluate D3P-fine-tuned diffusion policy with a TD-residual adaptive replanner.

The diffusion policy still produces 4-action chunks via D3P's dynamic-stride
adaptor. Instead of executing all 4 (open-loop) or replanning every env-step
(closed-loop), we maintain a per-env action queue and resample the chunk only
when one of the following fires:

  (a) the chunk is exhausted (cursor == horizon_steps),
  (b) the env terminated/truncated,
  (c) the TD residual δ_t = r_t + γ V(s_{t+1}) − V(s_t) is far enough below
      its running pooled EMA — i.e. the world transitioned much worse than
      the critic predicted.

Reports success rate, episode reward, average dynamic NFE per env-step
(amortized — only steps that resampled pay), replan rate, and trigger rate.
"""

import os
import numpy as np
import torch
import logging

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.eval.eval_agent import EvalAgent
from d3p_utils.D3PAdaptor import D3PAdaptor


class EvalD3PReplanAgent(EvalAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # --- D3P adaptor (same shape as training) ---
        d3p_cfg = cfg.get("d3p", {})
        adaptor_mlp_dims = d3p_cfg.get("mlp_dims", [256, 512, 1024, 512, 256])
        stride = cfg.denoising_steps // cfg.ft_denoising_steps
        self.adaptor = D3PAdaptor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            output_mean=stride,
            seq_len=cfg.cond_steps,
            chunk_size=cfg.horizon_steps,
            mlp_dims=adaptor_mlp_dims,
        ).to(self.device)

        adaptor_path = cfg.get("adaptor_path", None)
        assert adaptor_path is not None and os.path.isfile(adaptor_path), (
            f"adaptor_path must point to a saved adaptor_*.pt; got {adaptor_path}"
        )
        payload = torch.load(adaptor_path, map_location=self.device, weights_only=True)
        self.adaptor.load_state_dict(payload["adaptor"])
        self.adaptor.eval()
        log.info(f"Loaded D3P adaptor from {adaptor_path}")

        # --- Replan trigger config ---
        replan_cfg = cfg.replan
        self.tau = float(replan_cfg.tau)                  # z-score threshold (downside)
        self.ema_alpha = float(replan_cfg.ema_alpha)      # EMA decay
        self.warmup_steps = int(replan_cfg.warmup_steps)  # samples before trigger is allowed
        self.gamma = float(replan_cfg.gamma)              # discount used in TD residual
        self.chunk_len = int(replan_cfg.get("max_chunk_steps", cfg.horizon_steps))
        assert self.chunk_len <= cfg.horizon_steps, (
            "max_chunk_steps cannot exceed horizon_steps (chunk length)"
        )
        # Multi-step wrapper must be configured to step one action at a time
        # so we can read the per-action critic signal.
        assert cfg.act_steps == 1, (
            "Replan agent requires act_steps=1 at the wrapper level so the env "
            "advances one action per step; got act_steps="
            f"{cfg.act_steps}"
        )

    def run(self):
        timer = Timer()
        N = self.n_envs
        H = self.chunk_len
        A = self.action_dim

        options_venv = [{} for _ in range(N)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, N))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, N))

        # Replan bookkeeping
        cursor = np.zeros(N, dtype=np.int64)
        queued_chunks = np.zeros((N, H, A), dtype=np.float32)
        nfe_trajs = np.zeros((self.n_steps, N))
        replan_trajs = np.zeros((self.n_steps, N))   # 1 if env i resampled this step
        trigger_trajs = np.zeros((self.n_steps, N))  # 1 if TD trigger fired (subset of replan)
        delta_log = np.zeros((self.n_steps, N))      # for diagnostics / histograms

        # Pooled EMA of the TD residual
        ema_mu = 0.0
        ema_var = 1.0
        n_seen = 0

        if self.save_full_observations:
            obs_full_trajs = np.empty((0, N, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )

        for step in range(self.n_steps):
            if step % 20 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            # ---- 1. Sample fresh chunks for envs whose cursor is at 0 ----
            need_chunk = cursor == 0
            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples, _, _, _, _, stp_t = self.model.forward_d3p(
                    cond=cond,
                    adaptor=self.adaptor,
                    deterministic=True,
                )
                v_t = self.model.critic(cond).cpu().numpy().flatten()

            new_chunks = samples.trajectories.cpu().numpy()  # (N, horizon_steps, A)
            stp_np = stp_t.cpu().numpy()
            queued_chunks[need_chunk] = new_chunks[need_chunk, :H]
            nfe_trajs[step] = np.where(need_chunk, stp_np, 0.0)
            replan_trajs[step] = need_chunk.astype(np.float32)

            # ---- 2. Pop one action per env from the queue ----
            action_per_env = queued_chunks[np.arange(N), cursor]      # (N, A)
            action_venv = action_per_env[:, None, :]                  # (N, 1, A)

            # ---- 3. Step env, observe r, s' ----
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            done_venv = terminated_venv | truncated_venv

            # ---- 4. V(s_{t+1}) and TD residual ----
            with torch.no_grad():
                cond_next = {
                    "state": torch.from_numpy(obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                v_tp1 = self.model.critic(cond_next).cpu().numpy().flatten()

            nonterminal = (~done_venv).astype(np.float32)
            delta = reward_venv + self.gamma * v_tp1 * nonterminal - v_t
            delta_log[step] = delta

            # ---- 5. Score against OLD pooled EMA, then update ----
            sigma = max(np.sqrt(ema_var), 1e-6)
            z = (delta - ema_mu) / sigma
            valid = ~done_venv
            trigger = (
                (z < -self.tau)
                & valid
                & (n_seen >= self.warmup_steps)
            )
            trigger_trajs[step] = trigger.astype(np.float32)

            # Sequentially update the pooled EMA with each valid delta this step
            for d in delta[valid]:
                ema_mu = (1.0 - self.ema_alpha) * ema_mu + self.ema_alpha * d
                ema_var = (
                    (1.0 - self.ema_alpha) * ema_var
                    + self.ema_alpha * (d - ema_mu) ** 2
                )
                n_seen += 1

            # ---- 6. Update cursor for next env-step ----
            next_cursor = cursor + 1
            forced = (next_cursor >= H) | done_venv | trigger
            next_cursor[forced] = 0
            cursor = next_cursor

            # ---- 7. Standard rollout bookkeeping ----
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            if self.save_full_observations:
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )
            prev_obs_venv = obs_venv

        # ---- Episode aggregation (mirrors EvalDiffusionAgent) ----
        episodes_start_end = []
        for env_ind in range(N):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episode_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                # act_steps == 1 here so divisor is 1 (kept for parity).
                episode_best_reward = np.array(
                    [np.max(r) / self.act_steps for r in reward_trajs_split]
                )
            avg_episode_reward = float(np.mean(episode_reward))
            avg_best_reward = float(np.mean(episode_best_reward))
            success_rate = float(
                np.mean(episode_best_reward >= self.best_reward_threshold_for_success)
            )
        else:
            num_episode_finished = 0
            avg_episode_reward = 0.0
            avg_best_reward = 0.0
            success_rate = 0.0
            log.info("[WARNING] No episode completed within the iteration!")

        if self.traj_plotter is not None:
            self.traj_plotter(
                obs_full_trajs=obs_full_trajs,
                n_render=self.n_render,
                max_episode_steps=self.max_episode_steps,
                render_dir=self.render_dir,
                itr=0,
            )

        avg_nfe = float(nfe_trajs.mean())
        replan_rate = float(replan_trajs.mean())
        trigger_rate = float(trigger_trajs.mean())
        nfe_hist, nfe_edges = np.histogram(
            nfe_trajs.ravel(),
            bins=np.arange(0, self.model.ddim_steps + 2) - 0.5,
        )
        delta_hist, delta_edges = np.histogram(delta_log.ravel(), bins=51)

        time = timer()
        log.info(
            f"eval: ep {num_episode_finished:4d} | success {success_rate:6.4f} "
            f"| reward {avg_episode_reward:8.4f} (best/step {avg_best_reward:8.4f}) "
            f"| avg NFE {avg_nfe:6.3f} | replan {replan_rate:5.3f} | trigger {trigger_rate:5.3f} "
            f"| EMA δ μ={ema_mu:+.4f} σ={np.sqrt(ema_var):.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            avg_nfe=avg_nfe,
            replan_rate=replan_rate,
            trigger_rate=trigger_rate,
            ema_mu=ema_mu,
            ema_var=ema_var,
            nfe_hist=nfe_hist,
            nfe_edges=nfe_edges,
            delta_hist=delta_hist,
            delta_edges=delta_edges,
            tau=self.tau,
            time=time,
        )
