from pathlib import Path

import numpy as np

from adaptive_diffusion.vector_env_adapter import SingleVectorEnvAdapter


def test_adaptive_eval_configs_exist_for_matrix_a_tasks():
    for task in ("lift", "can", "square"):
        path = Path(f"cfg/robomimic/eval/{task}/eval_adaptive_commitment_mlp.yaml")
        text = path.read_text()
        assert "EvalAdaptiveCommitmentAgent" in text
        assert "act_steps: 1" in text
        assert "n_envs: 1" in text
        assert "horizon_steps: 4" in text


def test_single_vector_env_adapter_steps_single_action_as_one_step_chunk():
    class FakeVectorEnv:
        def __init__(self):
            self.last_action = None

        def reset_arg(self, options_list):
            assert options_list == [{"seed": 3}]
            return {"state": np.array([[[0.0, 1.0]]], dtype=np.float32)}

        def step(self, action):
            self.last_action = action
            return (
                {"state": np.array([[[2.0, 3.0]]], dtype=np.float32)},
                np.array([1.0]),
                np.array([False]),
                np.array([True]),
                [{"td_error": 0.1}],
            )

    fake = FakeVectorEnv()
    adapter = SingleVectorEnvAdapter(fake)

    obs = adapter.reset(seed=3)
    next_obs, reward, done, info = adapter.step(np.array([0.5, -0.5]))

    assert obs["state"].shape == (1, 2)
    assert fake.last_action.shape == (1, 1, 2)
    assert next_obs["state"].shape == (1, 2)
    assert reward == 1.0
    assert done
    assert info["td_error"] == 0.1
