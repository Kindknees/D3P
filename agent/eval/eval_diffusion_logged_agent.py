"""
Behavior-preserving diffusion policy eval with adaptive-execution CSV logs.
"""

import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from agent.eval.eval_agent import EvalAgent
from util.timer import Timer
from utils.execution_logger import ExecutionLogger

log = logging.getLogger(__name__)


class LoggedEvalDiffusionAgent(EvalAgent):
    """Fixed-chunk eval path that writes Iteration 1 execution logs."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def run(self):
        timer = Timer()
        execution_logger = self._make_execution_logger()
        nfe_per_sample = self._nfe_per_sample()

        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )

        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        prev_obs_venv = self.reset_env_all(options_venv=options_venv)
        firsts_trajs[0] = 1
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        if self.save_full_observations:
            obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
            obs_full_trajs = np.vstack(
                (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
            )

        for step in range(self.n_steps):
            if step % 10 == 0:
                print(f"Processed step {step} of {self.n_steps}")

            with torch.no_grad():
                cond = {
                    "state": torch.from_numpy(prev_obs_venv["state"])
                    .float()
                    .to(self.device)
                }
                samples = self.model(cond=cond, deterministic=True)
                output_venv = samples.trajectories.cpu().numpy()
            action_venv = output_venv[:, : self.act_steps]
            self._log_fixed_chunk_events(
                execution_logger, step, action_venv, nfe_per_sample
            )

            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            done_venv = terminated_venv | truncated_venv
            reward_trajs[step] = reward_venv
            firsts_trajs[step + 1] = done_venv
            self._log_fixed_chunk_steps(
                execution_logger,
                step,
                action_venv,
                reward_venv,
                done_venv,
                nfe_per_sample,
            )
            if self.save_full_observations:
                obs_full_venv = np.array(
                    [info["full_obs"]["state"] for info in info_venv]
                )
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                )

            prev_obs_venv = obs_venv

        episodes_start_end = []
        for env_ind in range(self.n_envs):
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
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            self._log_episode_summaries(
                execution_logger,
                reward_trajs_split,
                nfe_per_sample,
            )
        else:
            episode_reward = np.array([])
            num_episode_finished = 0
            avg_episode_reward = 0
            avg_best_reward = 0
            success_rate = 0
            log.info("[WARNING] No episode completed within the iteration!")

        if self.traj_plotter is not None:
            self.traj_plotter(
                obs_full_trajs=obs_full_trajs,
                n_render=self.n_render,
                max_episode_steps=self.max_episode_steps,
                render_dir=self.render_dir,
                itr=0,
            )

        time = timer()
        execution_logger.close()
        log.info(
            f"eval: num episode {num_episode_finished:4d} | success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
        )
        np.savez(
            self.result_path,
            num_episode=num_episode_finished,
            eval_success_rate=success_rate,
            eval_episode_reward=avg_episode_reward,
            eval_best_reward=avg_best_reward,
            time=time,
        )

    def _make_execution_logger(self):
        logging_cfg = self.cfg.get("execution_logging", {})
        output_dir = logging_cfg.get(
            "output_dir", os.path.join(self.logdir, "execution_logs")
        )
        return ExecutionLogger(
            output_dir=output_dir,
            config_resolved=OmegaConf.to_yaml(self.cfg, resolve=True),
        )

    def _nfe_per_sample(self):
        if getattr(self.model, "use_ddim", False):
            return len(self.model.ddim_t)
        return self.model.denoising_steps

    def _log_fixed_chunk_events(
        self, execution_logger, step, action_venv, nfe_per_sample
    ):
        for env_ind in range(self.n_envs):
            first_action = action_venv[env_ind, 0]
            execution_logger.log_event(
                task=self.env_name,
                seed=self.seed + env_ind,
                episode_id=env_ind,
                timestep=step * self.act_steps,
                method="fixed_chunk",
                event_type="buffer_empty",
                phase=0,
                plan_age=0,
                remaining_buffer_length=0,
                td_error="",
                likelihood_score="",
                criticality_score="",
                old_first_action_l2="",
                new_first_action_l2=np.linalg.norm(first_action),
                action_jump=0,
                buffer_jump=0,
                repair_noise_ratio="",
                repair_anchor_rho="",
                num_candidates=1,
                selected_candidate_score="",
                nfe_event=nfe_per_sample,
                accepted=True,
                fallback_reason="",
            )

    def _log_fixed_chunk_steps(
        self,
        execution_logger,
        step,
        action_venv,
        reward_venv,
        done_venv,
        nfe_per_sample,
    ):
        for env_ind in range(self.n_envs):
            execution_logger.log_step(
                task=self.env_name,
                seed=self.seed + env_ind,
                episode_id=env_ind,
                timestep=step * self.act_steps,
                method="fixed_chunk",
                phase=0,
                plan_age=0,
                remaining_buffer_length=0,
                commitment_horizon=self.act_steps,
                committed_steps_remaining=0,
                td_error="",
                likelihood_score="",
                criticality_score="",
                executed_action_l2=np.linalg.norm(action_venv[env_ind]),
                reward=reward_venv[env_ind],
                done=done_venv[env_ind],
                success=reward_venv[env_ind]
                >= self.best_reward_threshold_for_success,
                nfe_this_step=nfe_per_sample,
                wall_time_ms="",
            )

    def _log_episode_summaries(
        self, execution_logger, reward_trajs_split, nfe_per_sample
    ):
        for episode_id, reward_traj in enumerate(reward_trajs_split):
            episode_return = np.sum(reward_traj)
            if self.furniture_sparse_reward:
                episode_best_reward = episode_return
            else:
                episode_best_reward = np.max(reward_traj) / self.act_steps
            num_chunks = len(reward_traj)
            total_nfe = num_chunks * nfe_per_sample
            execution_logger.log_episode_summary(
                task=self.env_name,
                seed=self.seed,
                episode_id=episode_id,
                method="fixed_chunk",
                success=episode_best_reward
                >= self.best_reward_threshold_for_success,
                episode_return=episode_return,
                num_repairs=0,
                num_resets=num_chunks,
                num_random_resets=0,
                num_buffer_empty_resets=num_chunks,
                num_discard_nonempty_buffer=0,
                total_nfe=total_nfe,
                nfe_per_action=total_nfe / max(1, num_chunks * self.act_steps),
                mean_action_jump=0,
                mean_buffer_jump=0,
                wall_time_total_ms="",
            )
