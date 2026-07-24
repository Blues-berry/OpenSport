"""SQLite event store and frontend-friendly daily workout summaries."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from activity_taxonomy import ACTION_NAMES_ZH, CARDIO_ACTIONS


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL,
    active_seconds REAL NOT NULL DEFAULT 0,
    set_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS activity_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    action TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_timestamp REAL NOT NULL,
    end_timestamp REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    confidence REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE TABLE IF NOT EXISTS strategy_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE,
    session_id TEXT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_start ON sessions(start_timestamp);
CREATE INDEX IF NOT EXISTS idx_segments_session ON activity_segments(session_id, start_timestamp);
"""


class WorkoutStore:
    def __init__(self, path: Path | str):
        self._memory = str(path) == ":memory:"
        self.path = Path(path) if not self._memory else Path(":memory:")
        self._memory_connection: sqlite3.Connection | None = None
        if self._memory:
            self._memory_connection = sqlite3.connect(":memory:")
            self._memory_connection.row_factory = sqlite3.Row
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(strategy_events)"
                ).fetchall()
            }
            if "event_id" not in columns:
                connection.execute(
                    "ALTER TABLE strategy_events ADD COLUMN event_id TEXT"
                )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS "
                "idx_strategy_event_id ON strategy_events(event_id)"
            )

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return self._memory_connection
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def apply_events(self, events: list[dict]) -> None:
        if not events:
            return
        with self._connect() as connection:
            for event in events:
                event_json = json.dumps(
                    event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                event_id = str(event.get("event_id") or hashlib.sha256(
                    event_json.encode("utf-8")
                ).hexdigest())
                before = connection.total_changes
                connection.execute(
                    """
                    INSERT OR IGNORE INTO strategy_events(
                        event_id,session_id,timestamp,event_type,payload_json
                    ) VALUES(?,?,?,?,?)
                    """,
                    (
                        event_id,
                        event.get("session_id"),
                        float(event["timestamp"]),
                        event["type"],
                        event_json,
                    ),
                )
                if connection.total_changes == before:
                    continue
                if event["type"] == "workout_started":
                    connection.execute(
                        "INSERT OR IGNORE INTO sessions(session_id,start_timestamp) VALUES(?,?)",
                        (event["session_id"], float(event["timestamp"])),
                    )
                elif event["type"] in {
                    "set_ended",
                    "cardio_ended",
                    "activity_ended",
                }:
                    kind = (
                        "cardio"
                        if event["action"] in CARDIO_ACTIONS
                        else (
                            "strength"
                            if event["type"] == "set_ended"
                            else "other"
                        )
                    )
                    connection.execute(
                        """
                        INSERT INTO activity_segments(
                            session_id,action,kind,start_timestamp,end_timestamp,duration_seconds,confidence
                        ) VALUES(?,?,?,?,?,?,?)
                        """,
                        (
                            event["session_id"],
                            event["action"],
                            kind,
                            float(event["start_timestamp"]),
                            float(event.get("end_timestamp", event["timestamp"])),
                            float(event["duration_seconds"]),
                            float(event["confidence"]),
                        ),
                    )
                elif event["type"] == "workout_ended":
                    connection.execute(
                        "UPDATE sessions SET end_timestamp=?,active_seconds=?,set_count=? WHERE session_id=?",
                        (
                            float(
                                event.get(
                                    "end_timestamp", event["timestamp"]
                                )
                            ),
                            float(event["active_seconds"]),
                            int(event["sets"]),
                            event["session_id"],
                        ),
                    )

    def daily_summary(self, date: str | None = None, now: float | None = None) -> dict:
        current = datetime.fromtimestamp(now or time.time())
        target = datetime.strptime(date, "%Y-%m-%d") if date else current
        start = target.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end = target.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp()
        with self._connect() as connection:
            sessions = connection.execute(
                "SELECT * FROM sessions WHERE start_timestamp BETWEEN ? AND ? ORDER BY start_timestamp",
                (start, end),
            ).fetchall()
            output = []
            total = 0.0
            active_total = 0.0
            for session in sessions:
                segments = connection.execute(
                    "SELECT * FROM activity_segments WHERE session_id=? ORDER BY start_timestamp",
                    (session["session_id"],),
                ).fetchall()
                if not segments:
                    continue
                effective_end = float(session["end_timestamp"] or segments[-1]["end_timestamp"])
                duration = max(0.0, effective_end - float(session["start_timestamp"]))
                total += duration
                active_total += sum(float(segment["duration_seconds"]) for segment in segments)
                activities: list[dict] = []
                for segment in segments:
                    action = segment["action"]
                    if activities and activities[-1]["action_id"] == action:
                        item = activities[-1]
                    else:
                        item = {
                            "action_id": action,
                            "action": ACTION_NAMES_ZH.get(action, action),
                            "kind": segment["kind"],
                            "sets": (
                                1 if segment["kind"] == "strength" else 0
                            ),
                            "duration_seconds": 0,
                            "start": datetime.fromtimestamp(segment["start_timestamp"]).strftime("%H:%M"),
                        }
                        activities.append(item)
                    if segment["kind"] in {"cardio", "other"}:
                        item["duration_seconds"] += round(float(segment["duration_seconds"]))
                    elif item is activities[-1] and item["sets"] and item.get("_last_id") is not None:
                        item["sets"] += 1
                    item["_last_id"] = segment["id"]
                for item in activities:
                    item.pop("_last_id", None)
                output.append(
                    {
                        "session_id": session["session_id"],
                        "start": datetime.fromtimestamp(session["start_timestamp"]).strftime("%H:%M"),
                        "end": datetime.fromtimestamp(effective_end).strftime("%H:%M"),
                        "duration_seconds": round(duration),
                        "active_seconds": round(sum(float(segment["duration_seconds"]) for segment in segments)),
                        "activities": activities,
                    }
                )
        return {
            "date": target.strftime("%Y-%m-%d"),
            "total_workout_seconds": round(total),
            "active_seconds": round(active_total),
            "session_count": len(output),
            "sessions": output,
        }
