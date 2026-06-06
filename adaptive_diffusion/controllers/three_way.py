"""Three-way continue / repair / reset controller logic."""

from __future__ import annotations

from dataclasses import dataclass

from adaptive_diffusion.controllers.commitment import choose_td_commitment_horizon


CONTINUE = "continue"
REPAIR = "repair"
RESET = "reset"


@dataclass(frozen=True)
class ControllerDecision:
    action: str
    reason: str
    commitment_horizon: int | None = None
    sample_nfe: int | None = None


@dataclass(frozen=True)
class ExecutionState:
    remaining_buffer_length: int
    plan_age: int
    committed_steps_remaining: int
    td_error: float | None = None
    catastrophic: bool = False
    likelihood_score: float | None = None

    @property
    def phase(self) -> int:
        if self.plan_age < 0:
            raise ValueError("plan_age must be non-negative.")
        return self.plan_age


@dataclass(frozen=True)
class ThreeWayControllerConfig:
    repair_td_threshold: float
    high_td_threshold: float
    medium_td_threshold: float
    horizons: tuple[int, int, int] = (1, 2, 4)
    catastrophic_reset_enabled: bool = False


class ThreeWayHeuristicController:
    """Heuristic controller for adaptive commitment plus residual repair."""

    def __init__(self, config: ThreeWayControllerConfig) -> None:
        self.config = config

    def choose_new_commitment(self, td_error: float | None) -> int:
        if td_error is None:
            return self.config.horizons[-1]
        return choose_td_commitment_horizon(
            td_error=td_error,
            high_threshold=self.config.high_td_threshold,
            medium_threshold=self.config.medium_td_threshold,
            horizons=self.config.horizons,
        )

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length < 0:
            raise ValueError("remaining_buffer_length must be non-negative.")
        if state.committed_steps_remaining < 0:
            raise ValueError("committed_steps_remaining must be non-negative.")

        if state.remaining_buffer_length == 0:
            horizon = self.choose_new_commitment(state.td_error)
            return ControllerDecision(
                action=RESET,
                reason="buffer_empty",
                commitment_horizon=horizon,
            )

        if self.config.catastrophic_reset_enabled and state.catastrophic:
            horizon = self.choose_new_commitment(state.td_error)
            return ControllerDecision(
                action=RESET,
                reason="catastrophic",
                commitment_horizon=horizon,
            )

        if (
            state.td_error is not None
            and state.td_error >= self.config.repair_td_threshold
        ):
            return ControllerDecision(action=REPAIR, reason="td_or_likelihood")

        if state.committed_steps_remaining == 0:
            horizon = self.choose_new_commitment(state.td_error)
            return ControllerDecision(
                action=RESET,
                reason="commitment_boundary",
                commitment_horizon=horizon,
            )

        return ControllerDecision(action=CONTINUE, reason="within_commitment")


class AdaptiveCommitmentController:
    """Boundary-only adaptive commitment controller without repair."""

    def __init__(
        self,
        high_td_threshold: float,
        medium_td_threshold: float,
        horizons: tuple[int, int, int] = (1, 2, 4),
    ) -> None:
        self.high_td_threshold = high_td_threshold
        self.medium_td_threshold = medium_td_threshold
        self.horizons = horizons

    def choose_new_commitment(self, td_error: float | None) -> int:
        if td_error is None:
            return self.horizons[-1]
        return choose_td_commitment_horizon(
            td_error=td_error,
            high_threshold=self.high_td_threshold,
            medium_threshold=self.medium_td_threshold,
            horizons=self.horizons,
        )

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length == 0:
            horizon = self.choose_new_commitment(state.td_error)
            return ControllerDecision(RESET, "buffer_empty", horizon)
        if state.committed_steps_remaining == 0:
            horizon = self.choose_new_commitment(state.td_error)
            return ControllerDecision(RESET, "commitment_boundary", horizon)
        return ControllerDecision(CONTINUE, "within_commitment")


class BinaryResetController:
    """Phase-blind continue/reset baseline."""

    def __init__(self, reset_td_threshold: float, chunk_size: int = 4) -> None:
        self.reset_td_threshold = reset_td_threshold
        self.chunk_size = chunk_size

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length == 0:
            return ControllerDecision(RESET, "buffer_empty", self.chunk_size)
        if state.td_error is not None and state.td_error >= self.reset_td_threshold:
            return ControllerDecision(RESET, "td_anywhere", self.chunk_size)
        return ControllerDecision(CONTINUE, "within_chunk")


class BoundaryOnlyResetController:
    """Reset only at buffer or commitment boundaries."""

    def __init__(self, high_td_threshold: float, medium_td_threshold: float, horizons=(1, 2, 4)) -> None:
        self.high_td_threshold = high_td_threshold
        self.medium_td_threshold = medium_td_threshold
        self.horizons = tuple(horizons)

    def choose_new_commitment(self, td_error: float | None) -> int:
        if td_error is None:
            return self.horizons[-1]
        return choose_td_commitment_horizon(
            td_error=td_error,
            high_threshold=self.high_td_threshold,
            medium_threshold=self.medium_td_threshold,
            horizons=self.horizons,
        )

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length == 0:
            return ControllerDecision(
                RESET,
                "buffer_empty",
                self.choose_new_commitment(state.td_error),
            )
        if state.committed_steps_remaining == 0:
            return ControllerDecision(
                RESET,
                "commitment_boundary",
                self.choose_new_commitment(state.td_error),
            )
        return ControllerDecision(CONTINUE, "within_boundary_commitment")


class PhaseAwareHysteresisResetController:
    """Phase-aware reset baseline with consecutive-trigger hysteresis."""

    def __init__(
        self,
        phase_thresholds: tuple[float, ...],
        min_consecutive: int = 2,
        chunk_size: int = 4,
    ) -> None:
        if min_consecutive <= 0:
            raise ValueError("min_consecutive must be positive.")
        if not phase_thresholds:
            raise ValueError("phase_thresholds must not be empty.")
        self.phase_thresholds = tuple(float(value) for value in phase_thresholds)
        self.min_consecutive = min_consecutive
        self.chunk_size = chunk_size
        self.consecutive_count = 0

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length == 0:
            self.consecutive_count = 0
            return ControllerDecision(RESET, "buffer_empty", self.chunk_size)
        phase = state.plan_age % len(self.phase_thresholds)
        threshold = self.phase_thresholds[phase]
        if state.td_error is not None and state.td_error >= threshold:
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
        if self.consecutive_count >= self.min_consecutive:
            self.consecutive_count = 0
            return ControllerDecision(RESET, "phase_hysteresis", self.chunk_size)
        return ControllerDecision(CONTINUE, "within_phase_hysteresis")


class RepairOnlyController:
    """Residual-buffer repair baseline without adaptive commitment boundaries."""

    def __init__(self, repair_td_threshold: float, chunk_size: int = 4) -> None:
        self.repair_td_threshold = repair_td_threshold
        self.chunk_size = chunk_size

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length == 0:
            return ControllerDecision(RESET, "buffer_empty", self.chunk_size)
        if state.td_error is not None and state.td_error >= self.repair_td_threshold:
            return ControllerDecision(REPAIR, "td_repair")
        return ControllerDecision(CONTINUE, "within_chunk")


class BenefitTableController:
    """Lookup-table intervention-benefit controller from branch return data."""

    def __init__(
        self,
        task: str,
        table: dict[tuple[str, int, int], tuple[str, float]],
        high_td_threshold: float,
        medium_td_threshold: float,
        horizons: tuple[int, int, int] = (1, 2, 4),
        min_benefit: float = 0.0,
        min_uncertainty: float = 0.0,
        allow_reset: bool = True,
        allow_repair: bool = True,
        max_interventions: int | None = 1,
    ) -> None:
        self.task = task
        self.table = table
        self.high_td_threshold = high_td_threshold
        self.medium_td_threshold = medium_td_threshold
        self.horizons = horizons
        self.min_benefit = float(min_benefit)
        self.min_uncertainty = float(min_uncertainty)
        self.allow_reset = allow_reset
        self.allow_repair = allow_repair
        self.max_interventions = max_interventions
        self.num_interventions = 0

    def choose_new_commitment(self, td_error: float | None) -> int:
        if td_error is None:
            return self.horizons[-1]
        return choose_td_commitment_horizon(
            td_error=td_error,
            high_threshold=self.high_td_threshold,
            medium_threshold=self.medium_td_threshold,
            horizons=self.horizons,
        )

    def decide(self, state: ExecutionState) -> ControllerDecision:
        if state.remaining_buffer_length == 0:
            return ControllerDecision(
                RESET,
                "buffer_empty",
                self.choose_new_commitment(state.td_error),
            )

        phase = state.plan_age % self.horizons[-1]
        action, benefit = self.table.get(
            (self.task, phase, state.remaining_buffer_length),
            (CONTINUE, 0.0),
        )
        uncertainty = 0.0 if state.td_error is None else state.td_error
        can_intervene = (
            self.max_interventions is None
            or self.num_interventions < self.max_interventions
        )
        if can_intervene and benefit >= self.min_benefit and uncertainty >= self.min_uncertainty:
            if action == REPAIR and self.allow_repair:
                self.num_interventions += 1
                return ControllerDecision(REPAIR, "benefit_predictor")
            if action == RESET and self.allow_reset:
                self.num_interventions += 1
                return ControllerDecision(
                    RESET,
                    "benefit_predictor",
                    self.choose_new_commitment(state.td_error),
                )

        if state.committed_steps_remaining == 0:
            return ControllerDecision(
                RESET,
                "commitment_boundary",
                self.choose_new_commitment(state.td_error),
            )
        return ControllerDecision(CONTINUE, "benefit_predictor_continue")


class CriticalityAwareThreeWayController(ThreeWayHeuristicController):
    """Three-way controller that allocates diffusion sample NFE by criticality."""

    def __init__(
        self,
        config: ThreeWayControllerConfig,
        low_nfe: int,
        high_nfe: int,
        criticality_threshold: float,
    ) -> None:
        super().__init__(config)
        if low_nfe <= 0 or high_nfe <= 0:
            raise ValueError("criticality NFE values must be positive.")
        self.low_nfe = int(low_nfe)
        self.high_nfe = int(high_nfe)
        self.criticality_threshold = float(criticality_threshold)

    def choose_sample_nfe(self, td_error: float | None) -> int:
        if td_error is not None and td_error >= self.criticality_threshold:
            return self.high_nfe
        return self.low_nfe

    def decide(self, state: ExecutionState) -> ControllerDecision:
        decision = super().decide(state)
        if decision.action == RESET:
            return ControllerDecision(
                decision.action,
                decision.reason,
                decision.commitment_horizon,
                self.choose_sample_nfe(state.td_error),
            )
        return decision
