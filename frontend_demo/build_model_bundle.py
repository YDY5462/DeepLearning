"""Build a browser-friendly JS bundle from real model outputs.

Usage:
  python frontend_demo/build_model_bundle.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_scalar(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip())


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    final15 = repo_root / "final result" / "15"
    saved_runs = repo_root / "saved_runs"

    pred_path = final15 / "predictions.csv"
    true_path = final15 / "Y_test_original.csv"

    pred = np.loadtxt(pred_path, delimiter=",").round().astype(int)
    true = np.loadtxt(true_path, delimiter=",").round().astype(int)

    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")

    sample_count, station_count = map(int, pred.shape)

    provenance_rows = []
    provenance_path = saved_runs / "baseline_15min_provenance.csv"
    with provenance_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            provenance_rows.append(
                {
                    "file": row["file"],
                    "metricValue": float(row["metric_value"]),
                    "gitBlobId": row["git_blob_id"],
                    "sha256": row["sha256"],
                }
            )

    latest_metrics = {}
    latest_metrics_path = saved_runs / "latest_metrics_summary.csv"
    if latest_metrics_path.exists():
        with latest_metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row:
                latest_metrics = {
                    "epoch": int(float(row["epoch"])),
                    "RMSE": float(row["RMSE"]),
                    "R2": float(row["R2"]),
                    "MAE": float(row["MAE"]),
                    "WMAPE": float(row["WMAPE"]),
                    "AverageTrainTime": float(row["Average_train_time"]),
                }

    official_merge_metrics = {
        "RMSE": read_scalar(final15 / "RMSE_merge.txt"),
        "R2": read_scalar(final15 / "R2_merge.txt"),
        "MAE": read_scalar(final15 / "MAE_merge.txt"),
        "WMAPE": read_scalar(final15 / "WMAPE_merge.txt"),
    }

    err = pred - true
    recomputed_metrics = {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": float(1.0 - np.sum(err**2) / np.sum((true - np.mean(true)) ** 2)),
        "MAE": float(np.mean(np.abs(err))),
        "WMAPE": float(np.sum(np.abs(err)) / np.sum(np.abs(true))),
    }

    station_mae = np.mean(np.abs(err), axis=0)
    station_mean_true = np.mean(true, axis=0)
    station_mape = np.divide(
        station_mae,
        np.maximum(station_mean_true, 1e-6),
        out=np.zeros_like(station_mae, dtype=np.float64),
        where=np.maximum(station_mean_true, 1e-6) > 0,
    )

    bundle = {
        "schemaVersion": 1,
        "dataset": "final_result_15min",
        "timeGranularityMin": 15,
        "stationCount": station_count,
        "sampleCount": sample_count,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceFiles": {
            "prediction": "final result/15/predictions.csv",
            "groundTruth": "final result/15/Y_test_original.csv",
            "metrics": {
                "RMSE": "final result/15/RMSE_merge.txt",
                "R2": "final result/15/R2_merge.txt",
                "MAE": "final result/15/MAE_merge.txt",
                "WMAPE": "final result/15/WMAPE_merge.txt",
            },
        },
        "officialMergeMetrics": official_merge_metrics,
        "latestRunMetrics": latest_metrics,
        "recomputedMetrics": recomputed_metrics,
        "provenance": provenance_rows,
        "stationSummary": {
            "meanMAE": station_mae.round(3).tolist(),
            "meanMAPE": station_mape.round(6).tolist(),
        },
        "predMatrix": pred.tolist(),
        "trueMatrix": true.tolist(),
    }

    output_path = Path(__file__).resolve().parent / "model_bundle.js"
    output_path.write_text(
        "window.ModelBundle = " + json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Generated: {output_path}")
    print(f"Shape: samples={sample_count}, stations={station_count}")


if __name__ == "__main__":
    main()
