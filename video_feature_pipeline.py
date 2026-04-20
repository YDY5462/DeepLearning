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
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


@dataclass
class PipelineConfig:
    num_stations: int = 276
    time_granularity_min: int = 15
    sample_fps: float = 1.0
    frame_width: int = 320
    frame_height: int = 180
    backend: str = "motion"  # motion | multifeature | hybrid | proxy
    slot_agg: str = "mean"  # mean | max | median
    fill_missing: str = "zero"  # zero | ffill | interp
    hybrid_motion_weight: float = 0.6


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) is required for motion/multifeature/hybrid backends. "
            "Install opencv-python or use --backend proxy."
        )


def aggregate(values: List[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "max":
        return float(np.max(values))
    if mode == "median":
        return float(np.median(values))
    return float(np.mean(values))


class BaseSlotExtractor:
    def __init__(self, config: PipelineConfig):
        self.cfg = config

    def _sample_interval(self, source_fps: float) -> int:
        if source_fps <= 0:
            return 1
        return max(1, int(round(source_fps / self.cfg.sample_fps)))

    def _iter_sampled_frames(self, video_path: str) -> Iterable[Tuple[int, np.ndarray, np.ndarray]]:
        _require_cv2()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        sample_interval = self._sample_interval(fps)
        slot_seconds = self.cfg.time_granularity_min * 60

        frame_idx = 0
        try:
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

                timestamp_sec = frame_idx / fps if fps > 0 else 0.0
                slot_idx = int(timestamp_sec // slot_seconds)
                yield slot_idx, frame, gray

                frame_idx += 1
        finally:
            cap.release()

    def extract_slot_scores(self, video_path: str) -> Dict[int, float]:
        raise NotImplementedError


class MotionDensityExtractor(BaseSlotExtractor):
    """Extracts a crowd proxy by motion density (frame difference)."""

    def extract_slot_scores(self, video_path: str) -> Dict[int, float]:
        slot_values: Dict[int, List[float]] = {}
        prev_gray = None
        for slot_idx, _, gray in self._iter_sampled_frames(video_path):
            if prev_gray is None:
                prev_gray = gray
                continue

            diff = cv2.absdiff(gray, prev_gray)
            _, fg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            motion_density = float(np.count_nonzero(fg)) / float(fg.size)
            slot_values.setdefault(slot_idx, []).append(motion_density)
            prev_gray = gray

        return {k: aggregate(v, self.cfg.slot_agg) for k, v in slot_values.items()}


class MultiFeatureDensityExtractor(BaseSlotExtractor):
    """
    Extracts a robust crowd proxy from appearance cues:
    edge density + texture complexity + color variance + foreground ratio.
    """

    @staticmethod
    def _frame_density(frame_bgr: np.ndarray, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges) / 255.0)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        texture_complexity = float(np.var(laplacian) / 10000.0)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        color_variance = float(np.mean([np.var(hsv[:, :, i]) for i in range(3)]) / 10000.0)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        foreground_ratio = float(np.mean(thresh) / 255.0)

        features = np.array(
            [edge_density, texture_complexity, color_variance, foreground_ratio],
            dtype=np.float32,
        )
        weights = np.array([0.35, 0.30, 0.20, 0.15], dtype=np.float32)
        score = float(np.dot(features, weights))
        return float(np.clip(score, 0.0, 1.0))

    def extract_slot_scores(self, video_path: str) -> Dict[int, float]:
        slot_values: Dict[int, List[float]] = {}
        for slot_idx, frame_bgr, gray in self._iter_sampled_frames(video_path):
            score = self._frame_density(frame_bgr, gray)
            slot_values.setdefault(slot_idx, []).append(score)
        return {k: aggregate(v, self.cfg.slot_agg) for k, v in slot_values.items()}


class HybridDensityExtractor(BaseSlotExtractor):
    """Combines motion and appearance features to reduce single-method bias."""

    def extract_slot_scores(self, video_path: str) -> Dict[int, float]:
        slot_motion: Dict[int, List[float]] = {}
        slot_appearance: Dict[int, List[float]] = {}
        prev_gray = None
        motion_w = float(np.clip(self.cfg.hybrid_motion_weight, 0.0, 1.0))
        app_w = 1.0 - motion_w

        for slot_idx, frame_bgr, gray in self._iter_sampled_frames(video_path):
            appearance = MultiFeatureDensityExtractor._frame_density(frame_bgr, gray)
            slot_appearance.setdefault(slot_idx, []).append(appearance)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                _, fg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion = float(np.count_nonzero(fg)) / float(fg.size)
                slot_motion.setdefault(slot_idx, []).append(motion)
            prev_gray = gray

        all_slots = set(slot_motion.keys()) | set(slot_appearance.keys())
        scores: Dict[int, float] = {}
        for slot in all_slots:
            motion_score = aggregate(slot_motion.get(slot, []), self.cfg.slot_agg)
            appearance_score = aggregate(slot_appearance.get(slot, []), self.cfg.slot_agg)
            scores[slot] = motion_w * motion_score + app_w * appearance_score
        return scores


def create_extractor(config: PipelineConfig) -> BaseSlotExtractor:
    if config.backend == "proxy":
        raise ValueError("Proxy backend does not use frame extractor.")
    if config.backend == "multifeature":
        return MultiFeatureDensityExtractor(config)
    if config.backend == "hybrid":
        return HybridDensityExtractor(config)
    return MotionDensityExtractor(config)


def read_station_video_map(csv_path: str) -> List[Tuple[int, str]]:
    """
    CSV format:
    station_id,video_path

    station_id can be 0-based or 1-based.
    """
    mappings: List[Tuple[int, str]] = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station_id = int(row["station_id"])
            video_path = row["video_path"].strip()
            mappings.append((station_id, video_path))
    return mappings


def discover_station_videos(video_dir: str) -> List[Tuple[int, str]]:
    """
    Auto-discover files like station_000.mp4, station-001.avi, ...
    """
    mappings: List[Tuple[int, str]] = []
    if not video_dir or not os.path.isdir(video_dir):
        return mappings

    pattern = re.compile(r"station[_-]?(\d+)", flags=re.IGNORECASE)
    valid_ext = {".mp4", ".avi", ".mov", ".mkv"}
    for name in sorted(os.listdir(video_dir)):
        _, ext = os.path.splitext(name)
        if ext.lower() not in valid_ext:
            continue
        m = pattern.search(name)
        if not m:
            continue
        sid = int(m.group(1))
        mappings.append((sid, os.path.join(video_dir, name)))
    return mappings


def _maybe_one_based(station_ids: List[int], num_stations: int) -> bool:
    if not station_ids:
        return False
    if 0 in station_ids:
        return False
    return min(station_ids) >= 1 and max(station_ids) <= num_stations


def fill_missing_per_station(matrix: np.ndarray, observed: np.ndarray, mode: str) -> np.ndarray:
    if mode == "zero":
        return matrix

    out = matrix.astype(np.float32).copy()
    for i in range(out.shape[0]):
        idx = np.where(observed[i])[0]
        if len(idx) == 0:
            continue

        values = out[i, idx]
        if mode == "interp":
            out[i] = np.interp(np.arange(out.shape[1]), idx, values).astype(np.float32)
            continue

        # forward-fill with first observed value as left boundary
        filled = np.zeros(out.shape[1], dtype=np.float32)
        first_idx = int(idx[0])
        first_val = float(values[0])
        filled[: first_idx + 1] = first_val
        ptr = 1
        last_val = first_val
        for t in range(first_idx + 1, out.shape[1]):
            if ptr < len(idx) and t == int(idx[ptr]):
                last_val = float(values[ptr])
                ptr += 1
            filled[t] = last_val
        out[i] = filled
    return out


def normalize_per_station(matrix: np.ndarray) -> np.ndarray:
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


def _safe_row_minmax(row: np.ndarray) -> np.ndarray:
    max_v = float(np.max(row))
    min_v = float(np.min(row))
    if max_v == min_v:
        return np.zeros_like(row, dtype=np.float32)
    return ((row - min_v) / (max_v - min_v)).astype(np.float32)


def _shift_right(arr: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return arr.copy()
    out = np.empty_like(arr, dtype=np.float32)
    out[:steps] = float(arr[0])
    out[steps:] = arr[:-steps]
    return out


def _read_csv_matrix(path: str, dtype=float) -> np.ndarray:
    rows: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if not line:
                continue
            rows.append([dtype(x) for x in line])
    return np.asarray(rows)


def _ffill_1d(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32).copy()
    valid = np.where(~np.isnan(out))[0]
    if len(valid) == 0:
        return np.zeros_like(out, dtype=np.float32)
    first_idx = int(valid[0])
    out[: first_idx + 1] = out[first_idx]
    last = float(out[first_idx])
    for i in range(first_idx + 1, len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = float(out[i])
    return out


def build_video_proxy_matrix(
    output_csv: str,
    total_time_slots: int,
    config: PipelineConfig,
    proxy_tg: int = 15,
    noise_std: float = 0.03,
    delay_steps: int = 1,
    missing_rate: float = 0.1,
    seed: int = 42,
    weights: Tuple[float, float, float] = (0.3, 0.5, 0.2),
) -> None:
    inflow_path = os.path.join("data", "inflowdata", f"in_{proxy_tg}min.csv")
    outflow_path = os.path.join("data", "outflowdata", f"out_{proxy_tg}min.csv")
    if not os.path.exists(inflow_path) or not os.path.exists(outflow_path):
        raise FileNotFoundError(
            f"Missing AFC source files for proxy: {inflow_path} / {outflow_path}"
        )

    inflow = _read_csv_matrix(inflow_path, dtype=float)
    outflow = _read_csv_matrix(outflow_path, dtype=float)
    if inflow.shape != outflow.shape:
        raise ValueError(f"AFC inflow/outflow shape mismatch: {inflow.shape} vs {outflow.shape}")

    n_stations = min(config.num_stations, inflow.shape[0])
    n_slots = min(total_time_slots, inflow.shape[1])
    matrix = np.zeros((config.num_stations, total_time_slots), dtype=np.float32)

    w1, w2, w3 = weights
    rng = np.random.default_rng(seed)

    for s in range(n_stations):
        in_row = inflow[s, :n_slots].astype(np.float32)
        out_row = outflow[s, :n_slots].astype(np.float32)
        net = in_row - out_row

        backlog = np.zeros_like(net, dtype=np.float32)
        for t in range(1, len(net)):
            backlog[t] = max(0.0, backlog[t - 1] + float(net[t]))
        delta = np.diff(net, prepend=net[0]).astype(np.float32)

        net_n = _safe_row_minmax(net)
        backlog_n = _safe_row_minmax(backlog)
        delta_n = _safe_row_minmax(delta)

        proxy = (
            w1 * _shift_right(net_n, delay_steps)
            + w2 * _shift_right(backlog_n, delay_steps + 1)
            + w3 * _shift_right(delta_n, delay_steps)
        )

        if noise_std > 0:
            proxy = proxy + rng.normal(0.0, noise_std, size=proxy.shape).astype(np.float32)

        proxy = np.clip(proxy, 0.0, 1.0)

        if missing_rate > 0:
            miss_mask = rng.random(proxy.shape[0]) < missing_rate
            proxy = proxy.astype(np.float32)
            proxy[miss_mask] = np.nan
            if config.fill_missing == "interp":
                valid = np.where(~np.isnan(proxy))[0]
                if len(valid) > 1:
                    proxy = np.interp(np.arange(proxy.shape[0]), valid, proxy[valid]).astype(np.float32)
                else:
                    proxy = _ffill_1d(proxy)
            elif config.fill_missing == "ffill":
                proxy = _ffill_1d(proxy)
            else:
                proxy = np.nan_to_num(proxy, nan=0.0).astype(np.float32)

        proxy = _safe_row_minmax(proxy.astype(np.float32))
        matrix[s, :n_slots] = proxy

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savetxt(output_csv, matrix, delimiter=",", fmt="%.6f")
    print(
        "Saved proxy video feature matrix: "
        f"{output_csv}, shape={matrix.shape}, "
        f"weights={weights}, noise_std={noise_std}, delay={delay_steps}, missing_rate={missing_rate}"
    )


def build_video_feature_matrix(
    output_csv: str,
    total_time_slots: int,
    config: PipelineConfig,
    map_csv: str = "",
    video_dir: str = "",
) -> None:
    mappings = read_station_video_map(map_csv) if map_csv else discover_station_videos(video_dir)
    if not mappings:
        raise ValueError("No station-video mappings found. Provide --map_csv or a valid --video_dir.")

    extractor = create_extractor(config)
    matrix = np.zeros((config.num_stations, total_time_slots), dtype=np.float32)
    observed = np.zeros((config.num_stations, total_time_slots), dtype=bool)

    ids = [sid for sid, _ in mappings]
    one_based = _maybe_one_based(ids, config.num_stations)

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

        filled_count = 0
        for slot_idx, score in scores.items():
            if 0 <= slot_idx < total_time_slots:
                matrix[station_idx, slot_idx] = float(score)
                observed[station_idx, slot_idx] = True
                filled_count += 1
        print(f"Station {sid}: extracted {filled_count} slots")

    matrix = fill_missing_per_station(matrix, observed, config.fill_missing)
    matrix = normalize_per_station(matrix)

    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savetxt(output_csv, matrix, delimiter=",", fmt="%.6f")
    print(f"Saved video feature matrix: {output_csv}, shape={matrix.shape}")


def main():
    parser = argparse.ArgumentParser(description="Build station-time video feature matrix.")
    parser.add_argument("--map_csv", type=str, default="", help="station-video mapping CSV")
    parser.add_argument("--video_dir", type=str, default="", help="auto-discover station videos from directory")
    parser.add_argument("--output_csv", type=str, default="data/videodata/video_15min.csv")
    parser.add_argument("--total_time_slots", type=int, required=True)
    parser.add_argument("--time_granularity_min", type=int, default=15)
    parser.add_argument("--sample_fps", type=float, default=1.0)
    parser.add_argument("--num_stations", type=int, default=276)
    parser.add_argument("--backend", type=str, choices=["motion", "multifeature", "hybrid", "proxy"], default="motion")
    parser.add_argument("--slot_agg", type=str, choices=["mean", "max", "median"], default="mean")
    parser.add_argument("--fill_missing", type=str, choices=["zero", "ffill", "interp"], default="zero")
    parser.add_argument("--hybrid_motion_weight", type=float, default=0.6)
    parser.add_argument("--proxy_tg", type=int, default=15, help="AFC source time granularity for proxy backend")
    parser.add_argument("--proxy_noise_std", type=float, default=0.03)
    parser.add_argument("--proxy_delay_steps", type=int, default=1)
    parser.add_argument("--proxy_missing_rate", type=float, default=0.1)
    parser.add_argument("--proxy_seed", type=int, default=42)
    parser.add_argument(
        "--proxy_weights",
        type=str,
        default="0.3,0.5,0.2",
        help="comma-separated weights for net, backlog, delta",
    )
    args = parser.parse_args()

    if args.backend != "proxy" and not args.map_csv and not args.video_dir:
        raise ValueError("Provide at least one input source: --map_csv or --video_dir")

    cfg = PipelineConfig(
        num_stations=args.num_stations,
        time_granularity_min=args.time_granularity_min,
        sample_fps=args.sample_fps,
        backend=args.backend,
        slot_agg=args.slot_agg,
        fill_missing=args.fill_missing,
        hybrid_motion_weight=args.hybrid_motion_weight,
    )

    if args.backend == "proxy":
        parts = [x.strip() for x in args.proxy_weights.split(",") if x.strip()]
        if len(parts) != 3:
            raise ValueError("--proxy_weights must provide exactly 3 numbers, e.g. 0.3,0.5,0.2")
        weights = (float(parts[0]), float(parts[1]), float(parts[2]))
        build_video_proxy_matrix(
            output_csv=args.output_csv,
            total_time_slots=args.total_time_slots,
            config=cfg,
            proxy_tg=args.proxy_tg,
            noise_std=args.proxy_noise_std,
            delay_steps=args.proxy_delay_steps,
            missing_rate=args.proxy_missing_rate,
            seed=args.proxy_seed,
            weights=weights,
        )
    else:
        build_video_feature_matrix(
            map_csv=args.map_csv,
            video_dir=args.video_dir,
            output_csv=args.output_csv,
            total_time_slots=args.total_time_slots,
            config=cfg,
        )


if __name__ == "__main__":
    main()
