import gym
import numpy as np


class ActionPerturbWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        noise_std=0.0,
        noise_prob=1.0,
        start_step=0,
        end_step=None,
        seed=None,
    ):
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.noise_prob = float(noise_prob)
        self.start_step = int(start_step)
        self.end_step = None if end_step is None else int(end_step)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0

    def reset(self, *args, **kwargs):
        self.step_count = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        self.step_count += 1
        if self._should_perturb():
            action = np.asarray(action, dtype=np.float32)
            noise = self.rng.normal(0.0, self.noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def _should_perturb(self):
        if self.noise_std <= 0.0:
            return False
        if self.step_count < self.start_step:
            return False
        if self.end_step is not None and self.step_count > self.end_step:
            return False
        return self.rng.random() < self.noise_prob
