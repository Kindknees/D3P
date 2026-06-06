"""Heuristic commitment-horizon selection."""


def choose_td_commitment_horizon(
    td_error,
    high_threshold,
    medium_threshold,
    horizons=(1, 2, 4),
):
    """Map TD-error risk to a commitment horizon."""

    if len(horizons) != 3:
        raise ValueError("horizons must contain [high_risk, medium_risk, low_risk].")
    high_risk_horizon, medium_risk_horizon, low_risk_horizon = horizons
    if td_error >= high_threshold:
        return high_risk_horizon
    if td_error >= medium_threshold:
        return medium_risk_horizon
    return low_risk_horizon
