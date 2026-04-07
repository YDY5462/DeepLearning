"""
Video feature pipeline for passenger-flow modeling.

This module converts raw station videos into a station-time feature matrix:
    shape = (num_stations, num_time_slots)

Output can be consumed by load_data.py as:
    data/videodata/video_<TG>min.csv
"""

import argparse
import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PipelineConfig:
    num_stations: int = 276
    time_granularity_min: int = 15
    sample_fps: float = 1.0
    frame_width: int = 320
    frame_height: int = 180


class MotionDensityExtractor:
    """Extracts a simple crowd proxy by motion density."""

    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def _sample_interval(self, source_fps: float) -> int:
        if source_fps <= 0:
            return 1
        return max(1, int(round(source_fps / self.cfg.sample_fps)))

    def extract_slot_scores(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = self._sample_interval(fps)
        slot_seconds = self.cfg.time_granularity_min * 60

        prev_gray = None
        frame_idx = 0
        slot_sum = {}
        slot_cnt = {}

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            frame = cv2.resize(frame, (self.cfg.frame_width, self.cfg.frame_height))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, fg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_density = float(np.count_nonzero(fg)) / float(fg.size)

                timestamp_sec = frame_idx / fps if fps > 0 else 0.0
                slot_idx = int(timestamp_sec // slot_seconds)
                slot_sum[slot_idx] = slot_sum.get(slot_idx, 0.0) + motion_density
                slot_cnt[slot_idx] = slot_cnt.get(slot_idx, 0) + 1

            prev_gray = gray
            frame_idx += 1

        cap.release()

        if not slot_sum:
            return {}

        scores = {}
        for k in slot_sum:
            scores[k] = slot_sum[k] / max(slot_cnt[k], 1)
        return scores


def read_station_video_map(csv_path: str):
    """
    CSV format:
    station_id,video_path

    station_id can be 0-based or 1-based.
    """
    mappings = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = int(row["station_id"])
            video_path = row["video_path"].strip()
            mappings.append((station_id, video_path))
    return mappings


def normalize_per_station(matrix: np.ndarray):
    out = matrix.astype(np.float32).copy()
    for i in range(out.shape[0]):
        row = out[i]
        max_v = np.max(row)
        min_v = np.min(row)
        if max_v == min_v:
            out[i] = 0.0
        else:
            out[i] = (row - min_v) / (max_v - min_v)
    return out


def build_video_feature_matrix(map_csv: str, output_csv: str, total_time_slots: int, config: PipelineConfig):
    mappings = read_station_video_map(map_csv)
    extractor = MotionDensityExtractor(config)

    matrix = np.zeros((config.num_stations, total_time_slots), dtype=np.float32)

    # detect 1-based station ids
    ids = [sid for sid, _ in mappings]
    one_based = min(ids) >= 1

    for sid, video_path in mappings:
        station_idx = sid - 1 if one_based else sid
        if station_idx < 0 or station_idx >= config.num_stations:
            print(f"Skip invalid station id: {sid}")
            continue

        try:
            scores = extractor.extract_slot_scores(video_path)
        except Exception as e:
            print(f"Skip station {sid}, extract failed: {e}")
            continue

        for slot_idx, score in scores.items():
            if 0 <= slot_idx < total_time_slots:
                matrix[station_idx, slot_idx] = score

        print(f"Station {sid}: extracted {len(scores)} slots")

    matrix = normalize_per_station(matrix)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savetxt(output_csv, matrix, delimiter=",", fmt="%.6f")
    print(f"Saved video feature matrix: {output_csv}, shape={matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Build station-time video feature matrix.")
    parser.add_argument("--map_csv", type=str, required=True, help="station-video mapping CSV")
    parser.add_argument("--output_csv", type=str, default="data/videodata/video_15min.csv")
    parser.add_argument("--total_time_slots", type=int, required=True)
    parser.add_argument("--time_granularity_min", type=int, default=15)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--num_stations", type=int, default=276)
    args = parser.parse_args()

    cfg = PipelineConfig(
        num_stations=args.num_stations,
        time_granularity_min=args.time_granularity_min,
        sample_fps=args.sample_fps,
    )

    build_video_feature_matrix(
        map_csv=args.map_csv,
        output_csv=args.output_csv,
        total_time_slots=args.total_time_slots,
        config=cfg,
    )


if __name__ == "__main__":
    main()
