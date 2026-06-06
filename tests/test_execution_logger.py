import csv

from utils.execution_logger import (
    EPISODE_SUMMARY_FIELDS,
    EVENT_LOG_FIELDS,
    STEP_LOG_FIELDS,
    ExecutionLogger,
)


def test_execution_logger_writes_required_files_and_headers(tmp_path):
    with ExecutionLogger(tmp_path, config_resolved={"method": "fixed_chunk"}) as logger:
        logger.log_step(task="lift", seed=0, done=False)
        logger.log_event(task="lift", seed=0, event_type="reset", accepted=True)
        logger.log_episode_summary(task="lift", seed=0, success=True)
        logger.flush()

    assert (tmp_path / "config_resolved.yaml").read_text() == "method: fixed_chunk\n"
    assert _header(tmp_path / "step_logs.csv") == STEP_LOG_FIELDS
    assert _header(tmp_path / "event_logs.csv") == EVENT_LOG_FIELDS
    assert _header(tmp_path / "episode_summary.csv") == EPISODE_SUMMARY_FIELDS

    step_rows = _rows(tmp_path / "step_logs.csv")
    event_rows = _rows(tmp_path / "event_logs.csv")
    summary_rows = _rows(tmp_path / "episode_summary.csv")
    assert step_rows[0]["done"] == "0"
    assert event_rows[0]["accepted"] == "1"
    assert summary_rows[0]["success"] == "1"


def _header(path):
    with path.open(newline="") as file_obj:
        return next(csv.reader(file_obj))


def _rows(path):
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))
