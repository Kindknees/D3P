import csv
import subprocess
import sys


def test_analyze_adaptive_commitment_detects_positive_signal(tmp_path):
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    path = log_dir / "episode_summary.csv"
    fields = [
        "task",
        "seed",
        "episode_id",
        "method",
        "success",
        "episode_return",
        "nfe_per_action",
        "mean_action_jump",
        "mean_buffer_jump",
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "task": "lift",
                "seed": 0,
                "episode_id": 0,
                "method": "fixed_chunk",
                "success": 0,
                "episode_return": 1,
                "nfe_per_action": 5,
                "mean_action_jump": 0,
                "mean_buffer_jump": 0,
            }
        )
        writer.writerow(
            {
                "task": "lift",
                "seed": 0,
                "episode_id": 1,
                "method": "adaptive_commitment_with_repair",
                "success": 1,
                "episode_return": 2,
                "nfe_per_action": 5,
                "mean_action_jump": 0,
                "mean_buffer_jump": 0,
            }
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/analyze_adaptive_commitment.py",
            str(tmp_path),
            "--fallback-baseline",
            "fixed_chunk",
        ],
        check=True,
    )

    positives = tmp_path / "analysis" / "positive_signals.csv"
    assert "adaptive_commitment_with_repair" in positives.read_text()
