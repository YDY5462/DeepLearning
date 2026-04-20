"""
Realtime video feature service skeleton.

Purpose:
1) Read live video stream / local video file.
2) Extract realtime crowd-related features.
3) Expose `start / stop / latest` methods for frontend backend integration.

Notes:
- This module is for realtime inference/demo, not offline training CSV generation.
- It intentionally keeps logic separate from video_feature_pipeline.py.
"""

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception:
    FastAPI = None  # type: ignore
    HTTPException = Exception  # type: ignore
    BaseModel = object  # type: ignore


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError("OpenCV (cv2) is required. Please install opencv-python.")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(v: Union[int, float]) -> float:
    return float(np.clip(v, 0.0, 1.0))


@dataclass
class RealtimeConfig:
    source: str = "0"  # camera index like "0" or video file path
    backend: str = "hybrid"  # motion | multifeature | hybrid
    sample_fps: float = 4.0
    frame_width: int = 640
    frame_height: int = 360
    history_seconds: int = 60
    crowd_capacity_hint: int = 320
    hybrid_motion_weight: float = 0.6
    enable_yolo: bool = False


class RealtimeVideoService:
    """
    Realtime video analyzer skeleton.

    Usage:
        svc = RealtimeVideoService(config)
        svc.start()
        data = svc.latest()
        svc.stop()
    """

    def __init__(self, config: RealtimeConfig):
        self.cfg = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._cap = None
        self._prev_gray = None
        self._frame_id = 0
        self._history: Deque[Tuple[float, float]] = deque()  # (ts, density)
        self._latest: Dict[str, Any] = {
            "timestamp": _now_iso(),
            "running": False,
            "frame_id": 0,
            "backend": self.cfg.backend,
            "source": self.cfg.source,
            "people_count": 0,
            "density": 0.0,
            "motion_score": 0.0,
            "appearance_score": 0.0,
            "change_rate": 0.0,
            "crowd_level": "low",
            "latency_ms": 0,
            "note": "service initialized",
        }

    @staticmethod
    def _parse_source(src: str) -> Union[int, str]:
        s = str(src).strip()
        return int(s) if s.isdigit() else s

    @staticmethod
    def _appearance_score(frame_bgr: np.ndarray, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges) / 255.0)

        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        texture = float(np.var(laplacian) / 10000.0)

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        color_variance = float(np.mean([np.var(hsv[:, :, i]) for i in range(3)]) / 10000.0)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        fg_ratio = float(np.mean(thresh) / 255.0)

        features = np.array([edge_density, texture, color_variance, fg_ratio], dtype=np.float32)
        weights = np.array([0.35, 0.30, 0.20, 0.15], dtype=np.float32)
        return _safe_float(float(np.dot(features, weights)))

    @staticmethod
    def _motion_score(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
        if prev_gray is None:
            return 0.0
        diff = cv2.absdiff(gray, prev_gray)
        _, fg = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        return _safe_float(float(np.count_nonzero(fg)) / float(fg.size))

    @staticmethod
    def _crowd_level(density: float) -> str:
        if density >= 0.75:
            return "high"
        if density >= 0.55:
            return "mid_high"
        if density >= 0.35:
            return "mid"
        return "low"

    def _estimate_people_count(self, density: float, frame_shape: Tuple[int, ...]) -> int:
        """
        Heuristic count estimation when no detector model is attached.
        Replace with YOLO count if available.
        """
        if self.cfg.enable_yolo:
            count = self._run_yolo_count_placeholder(frame_shape)
            if count is not None:
                return int(max(0, count))
        return int(round(np.clip(density, 0.0, 1.0) * self.cfg.crowd_capacity_hint))

    def _run_yolo_count_placeholder(self, _frame_shape: Tuple[int, ...]) -> Optional[int]:
        """
        TODO:
        1) Load YOLO model once in __init__.
        2) Run detection per frame in worker loop.
        3) Count class='person' with confidence threshold.
        """
        return None

    def _compute_change_rate(self, now_ts: float, density: float) -> float:
        self._history.append((now_ts, density))
        horizon = max(5, int(self.cfg.history_seconds))
        while self._history and (now_ts - self._history[0][0]) > horizon:
            self._history.popleft()

        if len(self._history) < 2:
            return 0.0
        old = self._history[0][1]
        if old <= 1e-6:
            return 0.0
        return float((density - old) / old)

    def _open_capture(self) -> Any:
        _require_cv2()
        source = self._parse_source(self.cfg.source)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.cfg.source}")
        return cap

    def _process_one_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        start = time.time()
        frame = cv2.resize(frame_bgr, (self.cfg.frame_width, self.cfg.frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        motion = self._motion_score(self._prev_gray, gray)
        appearance = self._appearance_score(frame, gray)

        backend = self.cfg.backend
        if backend == "motion":
            density = motion
        elif backend == "multifeature":
            density = appearance
        else:
            m_w = float(np.clip(self.cfg.hybrid_motion_weight, 0.0, 1.0))
            density = m_w * motion + (1.0 - m_w) * appearance

        now_ts = time.time()
        change_rate = self._compute_change_rate(now_ts, density)
        people_count = self._estimate_people_count(density, frame.shape)
        latency_ms = int((time.time() - start) * 1000)

        self._prev_gray = gray
        self._frame_id += 1

        return {
            "timestamp": _now_iso(),
            "running": True,
            "frame_id": self._frame_id,
            "backend": self.cfg.backend,
            "source": self.cfg.source,
            "people_count": people_count,
            "density": round(float(density), 4),
            "motion_score": round(float(motion), 4),
            "appearance_score": round(float(appearance), 4),
            "change_rate": round(float(change_rate), 4),
            "crowd_level": self._crowd_level(float(density)),
            "latency_ms": latency_ms,
            "note": "ok",
        }

    def _loop(self) -> None:
        try:
            self._cap = self._open_capture()
            source_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if source_fps <= 0:
                source_fps = 25.0
            sample_interval = max(1, int(round(source_fps / max(self.cfg.sample_fps, 0.5))))

            frame_idx = 0
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    with self._lock:
                        self._latest.update(
                            {
                                "timestamp": _now_iso(),
                                "running": False,
                                "note": "stream ended or read failed",
                            }
                        )
                    break

                if frame_idx % sample_interval == 0:
                    data = self._process_one_frame(frame)
                    with self._lock:
                        self._latest = data
                frame_idx += 1

        except Exception as exc:
            with self._lock:
                self._latest.update(
                    {
                        "timestamp": _now_iso(),
                        "running": False,
                        "note": f"error: {exc}",
                    }
                )
        finally:
            self._running = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    def start(self) -> Dict[str, Any]:
        if self._running:
            return {"ok": True, "message": "already running"}
        self._running = True
        self._prev_gray = None
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return {"ok": True, "message": "started"}

    def stop(self) -> Dict[str, Any]:
        if not self._running:
            return {"ok": True, "message": "already stopped"}
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        with self._lock:
            self._latest["running"] = False
            self._latest["note"] = "stopped by user"
            self._latest["timestamp"] = _now_iso()
        return {"ok": True, "message": "stopped"}

    def latest(self) -> Dict[str, Any]:
        with self._lock:
            out = dict(self._latest)
        out["running"] = bool(self._running)
        return out

    def update_config(self, **kwargs: Any) -> Dict[str, Any]:
        if self._running:
            return {"ok": False, "message": "stop service before updating config"}
        for k, v in kwargs.items():
            if hasattr(self.cfg, k) and v is not None:
                setattr(self.cfg, k, v)
        with self._lock:
            self._latest["backend"] = self.cfg.backend
            self._latest["source"] = self.cfg.source
        return {"ok": True, "message": "config updated"}


# ------------------------------
# FastAPI wrapper (optional)
# ------------------------------
class StartBody(BaseModel):  # type: ignore[misc,valid-type]
    source: Optional[str] = None
    backend: Optional[str] = None
    sample_fps: Optional[float] = None


def create_fastapi_app(service: RealtimeVideoService) -> Any:
    if FastAPI is None:
        raise ImportError("fastapi is not installed. Please install fastapi and uvicorn.")

    app = FastAPI(title="Realtime Video Service", version="0.1.0")

    @app.get("/video/status")
    def video_status() -> Dict[str, Any]:
        return service.latest()

    @app.get("/video/latest")
    def video_latest() -> Dict[str, Any]:
        return service.latest()

    @app.post("/video/start")
    def video_start(body: StartBody) -> Dict[str, Any]:
        if body.source or body.backend or body.sample_fps is not None:
            cfg_update = {
                "source": body.source if body.source else service.cfg.source,
                "backend": body.backend if body.backend else service.cfg.backend,
                "sample_fps": body.sample_fps if body.sample_fps is not None else service.cfg.sample_fps,
            }
            result = service.update_config(**cfg_update)
            if not result.get("ok"):
                raise HTTPException(status_code=400, detail=result.get("message"))
        return service.start()

    @app.post("/video/stop")
    def video_stop() -> Dict[str, Any]:
        return service.stop()

    return app


def _run_cli(service: RealtimeVideoService, print_every_sec: float = 1.0) -> None:
    print("Starting realtime service in CLI mode ...")
    print("Press Ctrl+C to stop.")
    service.start()
    try:
        while True:
            data = service.latest()
            print(
                f"[{data['timestamp']}] "
                f"people={data['people_count']}, density={data['density']}, "
                f"change={data['change_rate']}, level={data['crowd_level']}, "
                f"latency={data['latency_ms']}ms, note={data['note']}"
            )
            time.sleep(max(0.2, float(print_every_sec)))
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        print("Realtime service stopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime video service skeleton")
    parser.add_argument("--source", type=str, default="0", help="camera index or video file path")
    parser.add_argument("--backend", type=str, choices=["motion", "multifeature", "hybrid"], default="hybrid")
    parser.add_argument("--sample_fps", type=float, default=4.0)
    parser.add_argument("--frame_width", type=int, default=640)
    parser.add_argument("--frame_height", type=int, default=360)
    parser.add_argument("--history_seconds", type=int, default=60)
    parser.add_argument("--crowd_capacity_hint", type=int, default=320)
    parser.add_argument("--hybrid_motion_weight", type=float, default=0.6)
    parser.add_argument("--enable_yolo", action="store_true")
    parser.add_argument("--mode", type=str, choices=["cli", "api"], default="cli")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    cfg = RealtimeConfig(
        source=args.source,
        backend=args.backend,
        sample_fps=args.sample_fps,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        history_seconds=args.history_seconds,
        crowd_capacity_hint=args.crowd_capacity_hint,
        hybrid_motion_weight=args.hybrid_motion_weight,
        enable_yolo=args.enable_yolo,
    )
    service = RealtimeVideoService(cfg)

    if args.mode == "api":
        try:
            import uvicorn
        except ImportError as exc:
            raise ImportError("uvicorn is required for --mode api") from exc
        app = create_fastapi_app(service)
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        _run_cli(service)


if __name__ == "__main__":
    main()
