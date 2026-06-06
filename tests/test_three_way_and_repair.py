import numpy as np

from adaptive_diffusion.controllers.three_way import (
    CONTINUE,
    REPAIR,
    RESET,
    BinaryResetController,
    PhaseAwareHysteresisResetController,
    RepairOnlyController,
    ExecutionState,
    ThreeWayControllerConfig,
    ThreeWayHeuristicController,
)
from adaptive_diffusion.repair import (
    RepairConfig,
    finalize_repair_candidate,
    prepare_repair_initialization,
    repair_noise_step,
)


def test_three_way_controller_orders_reset_repair_continue():
    controller = ThreeWayHeuristicController(
        ThreeWayControllerConfig(
            repair_td_threshold=8,
            high_td_threshold=9,
            medium_td_threshold=5,
        )
    )

    assert (
        controller.decide(
            ExecutionState(
                remaining_buffer_length=0,
                plan_age=0,
                committed_steps_remaining=0,
                td_error=10,
            )
        ).action
        == RESET
    )
    assert (
        controller.decide(
            ExecutionState(
                remaining_buffer_length=2,
                plan_age=1,
                committed_steps_remaining=1,
                td_error=8,
            )
        ).action
        == REPAIR
    )
    assert (
        controller.decide(
            ExecutionState(
                remaining_buffer_length=2,
                plan_age=1,
                committed_steps_remaining=1,
                td_error=1,
            )
        ).action
        == CONTINUE
    )


def test_repair_finalize_applies_anchor_clip_and_acceptance():
    residual = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    repaired = np.array([[2.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    config = RepairConfig(
        target_len=4,
        anchor_rho=0.5,
        accept_max_action_jump=2.0,
        accept_max_buffer_jump=2.0,
    )

    result = finalize_repair_candidate(residual, repaired, config)

    assert result.accepted
    assert np.allclose(result.repaired_actions[0], [1.0, 0.0])
    assert result.action_jump == 1.0


def test_prepare_repair_initialization_and_noise_step():
    residual = np.array([[1.0, 2.0]], dtype=np.float32)
    padded = prepare_repair_initialization(residual, RepairConfig(target_len=3))

    assert np.allclose(padded, [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    assert repair_noise_step(20, 0.5) == 10


def test_repair_finalize_rejects_too_small_action_jump():
    residual = np.array([[0.0, 0.0], [0.5, 0.5]], dtype=np.float32)
    repaired = residual.copy()
    config = RepairConfig(
        target_len=4,
        anchor_rho=1.0,
        accept_min_action_jump=0.1,
    )

    result = finalize_repair_candidate(residual, repaired, config)

    assert not result.accepted
    assert result.fallback_reason == "action_jump_too_small"


def test_binary_reset_controller_resets_nonempty_buffer_on_threshold():
    controller = BinaryResetController(reset_td_threshold=0.5, chunk_size=4)

    decision = controller.decide(
        ExecutionState(
            remaining_buffer_length=2,
            plan_age=1,
            committed_steps_remaining=2,
            td_error=0.6,
        )
    )

    assert decision.action == RESET
    assert decision.reason == "td_anywhere"


def test_phase_hysteresis_requires_consecutive_triggers():
    controller = PhaseAwareHysteresisResetController(
        phase_thresholds=(0.5, 0.5, 0.5, 0.5),
        min_consecutive=2,
        chunk_size=4,
    )
    state = ExecutionState(
        remaining_buffer_length=2,
        plan_age=1,
        committed_steps_remaining=2,
        td_error=0.6,
    )

    assert controller.decide(state).action == CONTINUE
    assert controller.decide(state).action == RESET


def test_repair_only_controller_triggers_repair_without_boundary_reset():
    controller = RepairOnlyController(repair_td_threshold=0.5, chunk_size=4)

    decision = controller.decide(
        ExecutionState(
            remaining_buffer_length=2,
            plan_age=1,
            committed_steps_remaining=0,
            td_error=0.6,
        )
    )

    assert decision.action == REPAIR


def test_benefit_table_controller_uses_margin_gate():
    from adaptive_diffusion.controllers.three_way import (
        BenefitTableController,
        ExecutionState,
        REPAIR,
        RESET,
        CONTINUE,
    )

    table = {
        ("square", 2, 2): (REPAIR, 16.0),
        ("square", 3, 1): (RESET, 78.0),
        ("square", 1, 3): (CONTINUE, 0.0),
    }
    controller = BenefitTableController(
        task="square",
        table=table,
        high_td_threshold=0.09,
        medium_td_threshold=0.05,
        min_benefit=5.0,
        max_interventions=None,
    )

    repair = controller.decide(
        ExecutionState(
            remaining_buffer_length=2,
            plan_age=2,
            committed_steps_remaining=2,
            td_error=0.1,
        )
    )
    assert repair.action == REPAIR
    assert repair.reason == "benefit_predictor"

    reset = controller.decide(
        ExecutionState(
            remaining_buffer_length=1,
            plan_age=3,
            committed_steps_remaining=1,
            td_error=0.1,
        )
    )
    assert reset.action == RESET
    assert reset.reason == "benefit_predictor"

    cont = controller.decide(
        ExecutionState(
            remaining_buffer_length=3,
            plan_age=1,
            committed_steps_remaining=3,
            td_error=0.1,
        )
    )
    assert cont.action == CONTINUE
    assert cont.reason == "benefit_predictor_continue"


def test_benefit_table_controller_respects_min_uncertainty():
    from adaptive_diffusion.controllers.three_way import (
        BenefitTableController,
        ExecutionState,
        REPAIR,
        CONTINUE,
    )

    controller = BenefitTableController(
        task="square",
        table={("square", 2, 2): (REPAIR, 16.0)},
        high_td_threshold=0.09,
        medium_td_threshold=0.05,
        min_benefit=5.0,
        min_uncertainty=0.2,
    )
    decision = controller.decide(
        ExecutionState(
            remaining_buffer_length=2,
            plan_age=2,
            committed_steps_remaining=2,
            td_error=0.1,
        )
    )
    assert decision.action == CONTINUE


def test_benefit_table_controller_limits_interventions():
    from adaptive_diffusion.controllers.three_way import (
        BenefitTableController,
        ExecutionState,
        REPAIR,
        CONTINUE,
    )

    controller = BenefitTableController(
        task="square",
        table={("square", 2, 2): (REPAIR, 16.0)},
        high_td_threshold=0.09,
        medium_td_threshold=0.05,
        min_benefit=5.0,
        max_interventions=1,
    )
    state = ExecutionState(
        remaining_buffer_length=2,
        plan_age=2,
        committed_steps_remaining=2,
        td_error=0.1,
    )
    assert controller.decide(state).action == REPAIR
    assert controller.decide(state).action == CONTINUE


def test_criticality_aware_controller_sets_sample_nfe():
    from adaptive_diffusion.controllers.three_way import (
        CriticalityAwareThreeWayController,
        ExecutionState,
        ThreeWayControllerConfig,
    )

    controller = CriticalityAwareThreeWayController(
        ThreeWayControllerConfig(
            repair_td_threshold=10.0,
            high_td_threshold=0.5,
            medium_td_threshold=0.1,
        ),
        low_nfe=5,
        high_nfe=20,
        criticality_threshold=0.3,
    )
    low = controller.decide(
        ExecutionState(0, plan_age=4, committed_steps_remaining=0, td_error=0.1)
    )
    high = controller.decide(
        ExecutionState(0, plan_age=4, committed_steps_remaining=0, td_error=0.4)
    )
    assert low.sample_nfe == 5
    assert high.sample_nfe == 20
