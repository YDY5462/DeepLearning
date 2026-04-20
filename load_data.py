import csv
import os
from math import sqrt

import numpy as np


NUM_STATIONS = 276


def _safe_minmax_norm(data: np.ndarray):
    max_v = float(np.max(data))
    min_v = float(np.min(data))
    if max_v == min_v:
        return np.zeros_like(data, dtype=np.float32), max_v, min_v
    norm = (data - min_v) / (max_v - min_v)
    return norm.astype(np.float32), max_v, min_v


def _read_csv_matrix(path: str, dtype=float) -> np.ndarray:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=",")
        for line in reader:
            if not line:
                continue
            rows.append([dtype(x) for x in line])
    return np.asarray(rows)


def _build_three_pattern_features(data_norm, time_lag, start_idx, end_idx, tg_day, tg_week):
    # Output shape: (samples, stations, 3 * (time_lag - 1))
    x = []
    for idx in range(start_idx, end_idx):
        station_features = []
        for s in range(NUM_STATIONS):
            weekly = data_norm[s, idx - tg_week: idx + time_lag - 1 - tg_week].tolist()
            daily = data_norm[s, idx - tg_day: idx + time_lag - 1 - tg_day].tolist()
            recent = data_norm[s, idx: idx + time_lag - 1].tolist()
            station_features.append(weekly + daily + recent)
        x.append(station_features)
    return np.asarray(x, dtype=np.float32)


def _build_y(data_norm, time_lag, start_idx, end_idx):
    y = []
    for idx in range(start_idx, end_idx):
        y.append(data_norm[:, idx + time_lag - 1])
    return np.asarray(y, dtype=np.float32)


def _build_y_original(data_raw, time_lag, start_idx, end_idx):
    y = []
    for idx in range(start_idx, end_idx):
        y.append(data_raw[:, idx + time_lag - 1])
    return np.asarray(y)


def _graph_norm_adj(adjacency: np.ndarray):
    eye = np.eye(NUM_STATIONS)
    a_hat = adjacency + eye
    d_hat = np.sum(a_hat, axis=0)
    d_hat_sqrt = np.diag([sqrt(x) for x in d_hat])
    d_hat_sqrt_inv = np.linalg.inv(d_hat_sqrt)
    return d_hat_sqrt_inv @ a_hat @ d_hat_sqrt_inv


def _build_graph_features(data_norm, d_a_final, time_lag, start_idx, end_idx):
    x = []
    for idx in range(start_idx, end_idx):
        station_features = []
        for s in range(NUM_STATIONS):
            station_features.append(data_norm[s, idx: idx + time_lag - 1])
        station_features = np.asarray(station_features, dtype=np.float32)
        x.append(d_a_final @ station_features)
    return np.asarray(x, dtype=np.float32)


def _build_weather_features(data, time_lag, start_idx, end_idx):
    # Output shape: (samples, indicator_num, time_lag - 1)
    x = []
    for idx in range(start_idx, end_idx):
        features = []
        for i in range(data.shape[0]):
            features.append(data[i, idx: idx + time_lag - 1])
        x.append(features)
    return np.asarray(x, dtype=np.float32)


def _load_video_matrix(tg, time_steps):
    candidates = [
        os.path.join("data", "videodata", f"video_{tg}min.csv"),
        os.path.join("data", "video_features", f"video_{tg}min.csv"),
    ]

    for path in candidates:
        if not os.path.exists(path):
            continue

        raw = _read_csv_matrix(path, dtype=float)

        # Accept either (stations, timesteps) or (timesteps, stations)
        if raw.shape[0] == NUM_STATIONS and raw.shape[1] == time_steps:
            video = raw
        elif raw.shape[1] == NUM_STATIONS and raw.shape[0] == time_steps:
            video = raw.T
        else:
            raise ValueError(
                f"Video feature shape mismatch in {path}, got {raw.shape}, "
                f"expected ({NUM_STATIONS}, {time_steps}) or ({time_steps}, {NUM_STATIONS})"
            )

        print(f"Loaded video feature matrix from: {path}, shape={video.shape}")
        return video, True

    print("Video feature matrix not found, video branch will be disabled.")
    return None, False


def Get_All_Data(
    TG,
    time_lag,
    TG_in_one_day,
    forecast_day_number,
    TG_in_one_week,
    force_disable_video=False,
):
    metro_enter = _read_csv_matrix(os.path.join("data", "inflowdata", f"in_{TG}min.csv"), dtype=int)
    metro_exit = _read_csv_matrix(os.path.join("data", "outflowdata", f"out_{TG}min.csv"), dtype=int)

    if metro_enter.shape[0] != NUM_STATIONS or metro_exit.shape[0] != NUM_STATIONS:
        raise ValueError(
            f"Station number mismatch. Expected {NUM_STATIONS}, "
            f"got inflow={metro_enter.shape[0]}, outflow={metro_exit.shape[0]}"
        )

    time_steps = metro_enter.shape[1]
    train_start = TG_in_one_week
    train_end = time_steps - time_lag + 1 - TG_in_one_day * forecast_day_number
    test_start = time_steps - TG_in_one_day * forecast_day_number
    test_end = time_steps - time_lag + 1

    # Inflow + label
    metro_enter_norm, a, b = _safe_minmax_norm(metro_enter)
    X_train_1 = _build_three_pattern_features(
        metro_enter_norm, time_lag, train_start, train_end, TG_in_one_day, TG_in_one_week
    )
    X_test_1 = _build_three_pattern_features(
        metro_enter_norm, time_lag, test_start, test_end, TG_in_one_day, TG_in_one_week
    )
    Y_train = _build_y(metro_enter_norm, time_lag, train_start, train_end)
    Y_test = _build_y(metro_enter_norm, time_lag, test_start, test_end)
    Y_test_original = _build_y_original(metro_enter, time_lag, test_start, test_end)

    print("Inflow train/test:", X_train_1.shape, X_test_1.shape)
    print("Label train/test:", Y_train.shape, Y_test.shape, "original:", Y_test_original.shape)

    # Outflow
    metro_exit_norm, _, _ = _safe_minmax_norm(metro_exit)
    X_train_2 = _build_three_pattern_features(
        metro_exit_norm, time_lag, train_start, train_end, TG_in_one_day, TG_in_one_week
    )
    X_test_2 = _build_three_pattern_features(
        metro_exit_norm, time_lag, test_start, test_end, TG_in_one_day, TG_in_one_week
    )
    print("Outflow train/test:", X_train_2.shape, X_test_2.shape)

    # Graph topology features
    adjacency = _read_csv_matrix("adjacency.csv", dtype=float)
    d_a_final = _graph_norm_adj(adjacency)
    X_train_3 = _build_graph_features(metro_enter_norm, d_a_final, time_lag, train_start, train_end)
    X_test_3 = _build_graph_features(metro_enter_norm, d_a_final, time_lag, test_start, test_end)
    print("Graph train/test:", X_train_3.shape, X_test_3.shape)

    # Weather features
    weather = _read_csv_matrix(os.path.join("data", "meteorology", f"{TG} min after normolization.csv"), dtype=float)
    X_train_4 = _build_weather_features(weather, time_lag, train_start, train_end)
    X_test_4 = _build_weather_features(weather, time_lag, test_start, test_end)
    print("Weather train/test:", X_train_4.shape, X_test_4.shape)

    # Video features (optional branch)
    if force_disable_video:
        print("Video branch is force-disabled by parameter.")
        has_video_data = False
        X_train_5 = None
        X_test_5 = None
    else:
        video_matrix, has_video_data = _load_video_matrix(TG, time_steps)
        if has_video_data:
            video_norm, _, _ = _safe_minmax_norm(video_matrix)
            X_train_5 = _build_three_pattern_features(
                video_norm, time_lag, train_start, train_end, TG_in_one_day, TG_in_one_week
            )
            X_test_5 = _build_three_pattern_features(
                video_norm, time_lag, test_start, test_end, TG_in_one_day, TG_in_one_week
            )
            print("Video train/test:", X_train_5.shape, X_test_5.shape)
        else:
            X_train_5 = None
            X_test_5 = None

    return (
        X_train_1,
        Y_train,
        X_test_1,
        Y_test,
        Y_test_original,
        a,
        b,
        X_train_2,
        X_test_2,
        X_train_3,
        X_test_3,
        X_train_4,
        X_test_4,
        X_train_5,
        X_test_5,
        has_video_data,
    )
