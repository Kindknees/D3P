import numpy as np

from adaptive_diffusion.diffusion_policy_adapter import DiffusionPolicyAdapter
from adaptive_diffusion.executor import _parse_uncertainty_result


class DummyPolicy(DiffusionPolicyAdapter):
    def __init__(self):
        self.calls = 0
        self.device = "cpu"

    def sample_action_chunk(self, obs):
        value = float(self.calls)
        self.calls += 1
        return np.full((4, 2), value, dtype=np.float32)


def test_action_disagreement_uses_multiple_stochastic_chunks():
    policy = DummyPolicy()

    value = policy.action_disagreement(obs=np.zeros(3), num_samples=2)

    assert np.isclose(value, np.sqrt(0.5))
    assert policy.calls == 2


def test_parse_uncertainty_result_counts_optional_nfe():
    assert _parse_uncertainty_result(None) == (None, 0)
    assert _parse_uncertainty_result(0.25) == (0.25, 0)
    assert _parse_uncertainty_result((0.25, 40)) == (0.25, 40)
