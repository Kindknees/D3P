"""Train a small intervention-benefit classifier from branch logs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


ACTIONS = ["continue", "repair_050", "reset"]
TASKS = ["lift", "can", "square"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("branch_logs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[4])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--leave-one-seed-out", action="store_true")
    args = parser.parse_args()

    rows = _read_rows(Path(args.branch_logs))
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.branch_logs).parent / "benefit_predictor"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.leave_one_seed_out:
        _run_leave_one_seed_out(rows, output_dir, args)
        return

    metrics, model, test_rows, x_test = _train_split(rows, set(args.test_seeds), args)
    metrics["test_seeds"] = args.test_seeds
    metrics["actions"] = ACTIONS
    metrics["feature_names"] = _feature_names()

    with (output_dir / "metrics.json").open("w") as file_obj:
        json.dump(metrics, file_obj, indent=2, sort_keys=True)
    _write_predictions(output_dir / "predictions.csv", model, test_rows, x_test)
    torch.save(model.state_dict(), output_dir / "model.pt")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _run_leave_one_seed_out(rows: list[dict[str, str]], output_dir: Path, args) -> None:
    seeds = sorted({int(row["seed"]) for row in rows})
    metrics_by_seed = []
    for seed in seeds:
        metrics, _, _, _ = _train_split(rows, {seed}, args)
        metrics["test_seed"] = seed
        metrics_by_seed.append(metrics)

    fields = [
        "test_seed",
        "train_accuracy",
        "test_accuracy",
        "macro_f1",
        "majority_baseline_accuracy",
        "num_train",
        "num_test",
    ]
    with (output_dir / "lopo_metrics.csv").open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for metrics in metrics_by_seed:
            writer.writerow({field: metrics[field] for field in fields})

    summary = {
        "num_splits": len(metrics_by_seed),
        "mean_test_accuracy": float(np.mean([m["test_accuracy"] for m in metrics_by_seed])),
        "mean_macro_f1": float(np.mean([m["macro_f1"] for m in metrics_by_seed])),
        "mean_majority_baseline_accuracy": float(
            np.mean([m["majority_baseline_accuracy"] for m in metrics_by_seed])
        ),
        "splits": metrics_by_seed,
        "actions": ACTIONS,
        "feature_names": _feature_names(),
    }
    with (output_dir / "lopo_summary.json").open("w") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _train_split(rows: list[dict[str, str]], test_seeds: set[int], args):
    train_rows = [row for row in rows if int(row["seed"]) not in test_seeds]
    test_rows = [row for row in rows if int(row["seed"]) in test_seeds]
    if not train_rows or not test_rows:
        raise SystemExit("Need non-empty train and test rows.")

    x_train, y_train = _features_and_labels(train_rows)
    x_test, y_test = _features_and_labels(test_rows)
    torch.manual_seed(0)
    model = torch.nn.Linear(x_train.shape[1], len(ACTIONS))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    x_train_t = torch.from_numpy(x_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    for _ in range(args.epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    metrics = _evaluate(model, x_train, y_train, x_test, y_test, train_rows, test_rows)
    metrics["majority_baseline_accuracy"] = _majority_baseline_accuracy(train_rows, test_rows)
    metrics["num_rows"] = len(rows)
    metrics["num_train"] = len(train_rows)
    metrics["num_test"] = len(test_rows)
    return metrics, model, test_rows, x_test


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _features_and_labels(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([_features(row) for row in rows], dtype=np.float32)
    labels = np.array([ACTIONS.index(row["best_action"]) for row in rows], dtype=np.int64)
    return features, labels


def _features(row: dict[str, str]) -> list[float]:
    task = row["task"]
    one_hot = [1.0 if task == name else 0.0 for name in TASKS]
    event_step = float(row["event_step"])
    phase = float(row["phase"])
    remaining = float(row["remaining_buffer_length"])
    uncertainty = float(row["uncertainty"])
    return one_hot + [
        event_step / 4.0,
        phase / 4.0,
        remaining / 4.0,
        uncertainty,
        float(row.get("residual_first_l2", 0.0)),
        float(row.get("residual_mean_l2", 0.0)),
        float(row.get("repair_action_jump", 0.0)),
        float(row.get("repair_buffer_jump", 0.0)),
    ]


def _feature_names() -> list[str]:
    return [f"task_{task}" for task in TASKS] + [
        "event_step_over_4",
        "phase_over_4",
        "remaining_over_4",
        "uncertainty",
        "residual_first_l2",
        "residual_mean_l2",
        "repair_action_jump",
        "repair_buffer_jump",
    ]


def _evaluate(model, x_train, y_train, x_test, y_test, train_rows, test_rows):
    with torch.no_grad():
        train_pred = model(torch.from_numpy(x_train).float()).argmax(dim=1).numpy()
        test_pred = model(torch.from_numpy(x_test).float()).argmax(dim=1).numpy()
    return {
        "train_accuracy": float(np.mean(train_pred == y_train)),
        "test_accuracy": float(np.mean(test_pred == y_test)),
        "macro_f1": _macro_f1(y_test, test_pred),
        "train_label_counts": _label_counts(train_rows),
        "test_label_counts": _label_counts(test_rows),
        "test_prediction_counts": _prediction_counts(test_pred),
    }


def _macro_f1(y_true, y_pred) -> float:
    scores = []
    for label in range(len(ACTIONS)):
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0




def _majority_baseline_accuracy(train_rows, test_rows) -> float:
    counts = _label_counts(train_rows)
    majority = max(counts, key=counts.get)
    return float(np.mean([row["best_action"] == majority for row in test_rows]))


def _label_counts(rows):
    counts = {action: 0 for action in ACTIONS}
    for row in rows:
        counts[row["best_action"]] += 1
    return counts


def _prediction_counts(pred):
    counts = {action: 0 for action in ACTIONS}
    for index in pred:
        counts[ACTIONS[int(index)]] += 1
    return counts


def _write_predictions(path: Path, model, rows, features) -> None:
    with torch.no_grad():
        pred = model(torch.from_numpy(features).float()).argmax(dim=1).numpy()
    fields = [
        "task",
        "seed",
        "event_step",
        "best_action",
        "predicted_action",
        "G_continue",
        "G_repair_050",
        "G_reset",
    ]
    with path.open("w", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row, index in zip(rows, pred):
            out = {field: row.get(field, "") for field in fields}
            out["predicted_action"] = ACTIONS[int(index)]
            writer.writerow(out)


if __name__ == "__main__":
    main()
