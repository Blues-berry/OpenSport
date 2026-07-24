"""Local web app that connects IMU uploads to the trained exercise model."""

from __future__ import annotations

import json
import pickle
import tempfile
import csv
import io
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from activity_taxonomy import ACTION_NAMES_ZH
from extract_features import windows_for_file
from replay_activity import analyze_temporary
from train_logistic import predict_probability
from workout_store import WorkoutStore


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opensport.models.registry import ModelRegistry  # noqa: E402

WEB_ROOT = HERE / "web"
MODEL_PATH = PROJECT_ROOT / "imu_output" / "headphone_all" / "model" / "l2_logistic_model.pkl"
MODEL_REGISTRY = ModelRegistry(PROJECT_ROOT / "imu_output" / "models")


def resolve_model_path(kind: str) -> Path | None:
    registered = MODEL_REGISTRY.resolve(kind)
    if registered:
        return registered.path
    fallbacks = (
        (
            PROJECT_ROOT / "imu_output" / "activity_registry" / "champion" / "activity_model.pkl",
            PROJECT_ROOT / "imu_output" / "activity_multiclass_20260724" / "model" / "activity_model.pkl",
            PROJECT_ROOT / "imu_output" / "demo_activity" / "model" / "activity_model.pkl",
        )
        if kind == "activity"
        else (
            PROJECT_ROOT / "imu_output" / "head_posture" / "model" / "head_posture_model.pkl",
        )
    )
    return next((path for path in fallbacks if path.exists()), None)


ACTIVITY_MODEL_PATH = resolve_model_path("activity")
HEAD_POSTURE_MODEL_PATH = resolve_model_path("posture")
LIVE_CSV_PATH = HERE.parent / "imu_output" / "live_imu.csv"
LIVE_STATUS_PATH = HERE.parent / "imu_output" / "live_status.json"
ACTIVITY_STATUS_PATH = HERE.parent / "imu_output" / "activity_live_status.json"
POSTURE_COMMAND_PATH = HERE.parent / "imu_output" / "posture_command.json"
WORKOUT_DATABASE_PATH = HERE.parent / "imu_output" / "workouts.sqlite3"
RECEIVER_SCRIPT = HERE / "realtime_activity_ble.py"
RECEIVER_LOCK = threading.Lock()
RECEIVER_PROCESS: subprocess.Popen | None = None


def receiver_control(action: str | None = None) -> dict:
    """Start or stop the hidden BLE receiver used by the test frontend."""
    global RECEIVER_PROCESS
    with RECEIVER_LOCK:
        running = RECEIVER_PROCESS is not None and RECEIVER_PROCESS.poll() is None
        if action == "start" and not running:
            if not RECEIVER_SCRIPT.exists():
                raise FileNotFoundError(f"BLE receiver not found: {RECEIVER_SCRIPT}")
            if ACTIVITY_MODEL_PATH is None:
                raise FileNotFoundError("未注册可用的动作模型")
            if HEAD_POSTURE_MODEL_PATH is None:
                raise FileNotFoundError("未注册可用的头部姿态模型")
            address = "F6:B1:93:B5:2B:23"
            RECEIVER_PROCESS = subprocess.Popen(
                [
                    sys.executable,
                    str(RECEIVER_SCRIPT),
                    "--address",
                    address,
                    "--model",
                    str(ACTIVITY_MODEL_PATH),
                    "--database",
                    str(WORKOUT_DATABASE_PATH),
                    "--posture-model",
                    str(HEAD_POSTURE_MODEL_PATH),
                    "--posture-command",
                    str(POSTURE_COMMAND_PATH),
                    "--status",
                    str(ACTIVITY_STATUS_PATH),
                ],
                cwd=str(RECEIVER_SCRIPT.parent), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            running = True
        elif action == "stop" and running:
            RECEIVER_PROCESS.terminate()
            try:
                RECEIVER_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                RECEIVER_PROCESS.kill()
                RECEIVER_PROCESS.wait(timeout=2)
            RECEIVER_PROCESS = None
            running = False
        return {"running": running, "pid": RECEIVER_PROCESS.pid if running else None}


def load_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"未找到模型文件：{MODEL_PATH}")
    with MODEL_PATH.open("rb") as handle:
        return pickle.load(handle)


MODEL: dict | None = None


def legacy_model() -> dict:
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


def merged_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    start, end = sorted(intervals)[0]
    for next_start, next_end in sorted(intervals)[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def analyze_upload(filename: str, content: str) -> dict:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".txt"}:
        raise ValueError("请上传耳机导出的 CSV 或 TXT 数据文件")
    with tempfile.NamedTemporaryFile("w", suffix=suffix, encoding="utf-8-sig", delete=False) as temp:
        temp.write(content)
        source = Path(temp.name)
    if ACTIVITY_MODEL_PATH is not None and ACTIVITY_MODEL_PATH.exists():
        try:
            result = analyze_temporary(source, ACTIVITY_MODEL_PATH)
            last = result.get("last_inference") or {}
            sessions = result.get("sessions", [])
            return {
                **result,
                "windows": int(result.get("inference_windows", 0)),
                "exercise_probability": round(float(last.get("motion_probability", 0)) * 100),
                "exercise_duration_seconds": int(result.get("total_workout_seconds", 0)),
                "status": "已检测到训练" if sessions else "本次未检测到训练",
                "experimental": bool(last.get("experimental", True)),
                "warning": last.get("warning"),
                "posture": {
                    "state": "运动分析已完成",
                    "detail": "动作模型使用六轴 IMU；静态姿态评价由独立功能处理",
                    "angle_degrees": None,
                    "mood": "neutral",
                },
            }
        finally:
            source.unlink(missing_ok=True)
    try:
        rows = windows_for_file(source, window_s=2.0, overlap=0.5)
    finally:
        source.unlink(missing_ok=True)
    if not rows:
        raise ValueError("文件时长不足，无法生成 2 秒分析窗口")

    windows = pd.DataFrame(rows)
    model_payload = legacy_model()
    feature_names = model_payload["features"]
    probabilities = predict_probability(
        model_payload["model"], windows.reindex(columns=feature_names)
    )
    threshold = float(model_payload["threshold"])
    windows["probability"] = probabilities
    windows["is_exercise"] = probabilities >= threshold

    angle_column = next((column for column in ("angle_y_mean", "angle_x_mean", "angle_z_mean") if column in windows), None)
    angle = float(np.nanmedian(windows[angle_column])) if angle_column and windows[angle_column].notna().any() else None
    motion_probability = float(np.mean(probabilities))
    if angle is None:
        posture = {"state": "无法判断", "detail": "文件缺少有效姿态角数据", "angle_degrees": None, "mood": "neutral"}
    elif motion_probability >= threshold:
        posture = {"state": "运动中", "detail": "运动识别模型正在检测动态动作，暂停静态姿态评价", "angle_degrees": round(angle, 1), "mood": "neutral"}
    elif abs(angle) <= 15:
        posture = {"state": "坐姿很好", "detail": "相对耳机自然基线稳定", "angle_degrees": round(angle, 1), "mood": "good"}
    elif abs(angle) <= 30:
        posture = {"state": "姿态一般", "detail": "可轻微调整头部与肩颈位置", "angle_degrees": round(angle, 1), "mood": "neutral"}
    else:
        posture = {"state": "需要注意姿态", "detail": "姿态角度偏离较大，建议稍作舒展", "angle_degrees": round(angle, 1), "mood": "attention"}

    active = windows[windows["is_exercise"]]
    duration_seconds = merged_duration(list(zip(active["window_start_s"], active["window_end_s"])))
    sessions: list[dict] = []
    current: list[dict] = []
    for row in active.sort_values("window_start_s").to_dict("records"):
        if current and row["window_start_s"] - current[-1]["window_end_s"] > 2.5:
            sessions.append(current)
            current = []
        current.append(row)
    if current:
        sessions.append(current)

    summarized = []
    for session in sessions[-3:][::-1]:
        start = float(session[0]["window_start_s"])
        end = float(session[-1]["window_end_s"])
        summarized.append(
            {
                "start_seconds": round(start),
                "duration_seconds": round(end - start),
                "confidence": round(float(np.mean([item["probability"] for item in session])) * 100),
            }
        )
    return {
        "filename": filename,
        "windows": int(len(windows)),
        "exercise_probability": round(float(np.mean(probabilities)) * 100),
        "exercise_duration_seconds": round(duration_seconds),
        "session_count": len(sessions),
        "sessions": summarized,
        "status": "已检测到训练" if duration_seconds else "本次未检测到训练",
        "posture": posture,
    }


def live_payload() -> dict:
    """Read the append-only stream produced by the WitMotion live receivers."""
    if ACTIVITY_STATUS_PATH.exists():
        try:
            status = json.loads(ACTIVITY_STATUS_PATH.read_text(encoding="utf-8"))
            result = status.get("last_result") or {}
            strategy = result.get("strategy") or {}
            posture = status.get("last_posture_result") or {}
            sample = status.get("last_sample") or {}
            orientation = posture.get("orientation") or {}
            updated = float(status.get("updated_at") or time.time())
            action_id = result.get("action", "unknown_motion")
            return {
                "state": status.get("state", "waiting"),
                "streams": [
                    {
                        "device": status.get("device", "IMU"),
                        "state": status.get("state", "waiting"),
                        "label": ACTION_NAMES_ZH.get(action_id, "等待窗口"),
                        "action_id": action_id,
                        "exercise_probability": round(float(result.get("motion_probability", 0)) * 100),
                        "action_probability": round(float(result.get("action_probability", 0)) * 100),
                        "workout_state": strategy.get("state", "idle"),
                        "sets": int(strategy.get("sets_in_session", 0)),
                        "signal_quality": result.get("signal_quality", "waiting"),
                        "experimental": bool(result.get("experimental", True)),
                        "warning": result.get("warning"),
                        "gyro_x_dps": round(float(sample.get("gx_dps") or 0), 1),
                        "gyro_y_dps": round(float(sample.get("gy_dps") or 0), 1),
                        "gyro_z_dps": round(float(sample.get("gz_dps") or 0), 1),
                        "roll_degrees": round(float(orientation.get("roll_degrees") or 0), 1),
                        "pitch_degrees": round(float(orientation.get("pitch_degrees") or 0), 1),
                        "yaw_degrees": round(float(orientation.get("yaw_degrees") or 0), 1),
                        "sample_rate_hz": 50.0,
                        "posture": posture,
                        "last_sample_age_s": round(max(0.0, time.time() - updated), 1),
                    }
                ] if (result or posture or sample) else [],
            }
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if not LIVE_CSV_PATH.exists() or LIVE_CSV_PATH.stat().st_size == 0:
        return {"state": "waiting", "streams": []}
    try:
        with LIVE_CSV_PATH.open("rb") as handle:
            header = handle.readline().decode("utf-8-sig")
            offset = max(handle.tell(), handle.seek(0, 2) - 524288)
            handle.seek(offset)
            tail = handle.read().decode("utf-8", errors="ignore")
        if offset > len(header.encode("utf-8-sig")):
            tail = tail.split("\n", 1)[-1]
        rows = list(csv.DictReader(io.StringIO(header + tail)))
        latest: dict[str, dict] = {}
        for row in rows:
            if row.get("device"):
                latest[row["device"]] = row
        status = {}
        if LIVE_STATUS_PATH.exists():
            status = json.loads(LIVE_STATUS_PATH.read_text(encoding="utf-8")).get("devices", {})
        streams = []
        for device, row in latest.items():
            device_status = status.get(device, {}) if isinstance(status, dict) else {}
            probability = float(row.get("exercise_probability") or 0)
            sample_age = round(time.time() - float(row.get("timestamp_unix_s") or time.time()), 1)
            stream_state = device_status.get("state", "live") if sample_age <= 3 else "stale"
            streams.append({
                "device": device,
                "state": stream_state,
                "label": row.get("inference_label") or "等待窗口",
                "exercise_probability": round(probability * 100),
                "gyro_x_dps": round(float(row.get("gx_dps") or 0), 1),
                "gyro_y_dps": round(float(row.get("gy_dps") or 0), 1),
                "gyro_z_dps": round(float(row.get("gz_dps") or 0), 1),
                "roll_degrees": round(float(row.get("roll_deg") or 0), 1),
                "pitch_degrees": round(float(row.get("pitch_deg") or 0), 1),
                "yaw_degrees": round(float(row.get("yaw_deg") or 0), 1),
                "sample_rate_hz": round(float(row.get("target_rate_hz") or row.get("source_rate_hz") or 0), 1),
                "last_sample_age_s": sample_age,
            })
        return {"state": "live" if streams else "waiting", "streams": streams}
    except (OSError, ValueError, csv.Error):
        return {"state": "waiting", "streams": []}


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/posture/calibrate":
            try:
                payload = {
                    "action": "calibrate",
                    "request_id": str(time.time_ns()),
                    "requested_at": time.time(),
                }
                temporary = POSTURE_COMMAND_PATH.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                temporary.replace(POSTURE_COMMAND_PATH)
                body = json.dumps({"accepted": True, **payload}, ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.ACCEPTED)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError as error:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return
        if path == "/api/control":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                action = str(payload.get("action", ""))
                if action not in {"start", "stop"}:
                    raise ValueError("action must be start or stop")
                body = json.dumps(receiver_control(action), ensure_ascii=False).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if path != "/api/analyze":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 20 * 1024 * 1024:
                raise ValueError("文件为空或超过 20 MB 限制")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = analyze_upload(str(payload["filename"]), str(payload["content"]))
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (KeyError, ValueError, UnicodeDecodeError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception as error:  # Keep model/data parsing details in the terminal.
            self.log_error("inference failed: %s", error)
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "推理失败，请确认文件为耳机 IMU 导出数据")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/models":
            payload = MODEL_REGISTRY.status()
            payload["legacy_activity_fallback"] = {
                "available": ACTIVITY_MODEL_PATH is not None,
                "path": str(ACTIVITY_MODEL_PATH) if ACTIVITY_MODEL_PATH else None,
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/control":
            body = json.dumps(receiver_control(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/live":
            body = json.dumps(live_payload(), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/daily":
            body = json.dumps(
                WorkoutStore(WORKOUT_DATABASE_PATH).daily_summary(),
                ensure_ascii=False,
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), AppHandler)
    print("IMU app is running at http://127.0.0.1:8000")
    server.serve_forever()
