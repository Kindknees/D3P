"""Debug-only adaptive commitment eval agent with synthetic triggers."""

from __future__ import annotations

from agent.eval.eval_adaptive_commitment_agent import EvalAdaptiveCommitmentAgent


class EvalAdaptiveCommitmentDebugAgent(EvalAdaptiveCommitmentAgent):
    """Enable fixed-schedule triggers to smoke-test repair on real envs."""

    def _make_uncertainty_fn(self, method_cfg):
        uncertainty = method_cfg.get("uncertainty", {})
        source = uncertainty.get("source", "none")
        if source != "fixed_schedule":
            return super()._make_uncertainty_fn(method_cfg)

        trigger_steps = set(int(step) for step in uncertainty.get("trigger_steps", []))
        trigger_value = float(uncertainty.get("trigger_value", 1.0))

        def uncertainty_fn(obs, info, step):
            return trigger_value if step in trigger_steps else 0.0

        return uncertainty_fn
