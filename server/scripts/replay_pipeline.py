"""
Offline replay script for Auto Tester backend.

Purpose
-------
Run an existing drive's RAW files through the backend pipeline again without Flutter,
without authentication, and without MongoDB:

    sensors JSON/CSV -> sensor_sync/load CSV -> YOLO/Kalman/VisionFilter -> vector_builder -> M8 LSTM -> exports

Outputs
-------
The script writes a replay export folder containing:

    1_raw_sensors.csv
    1b_video_events.json
    yolo_annotated.mp4
    2_feature_vector.csv
    3_model_thought.json
    4_replay_summary.json

How to run from the server root
-------------------------------
    python scripts/replay_pipeline.py \
        --test-id TEST_1777763460177_REPLAY \
        --video path/to/drive.mp4 \
        --sensors path/to/sensors.json

You can also provide a directory and let the script find one video + one JSON:
    python scripts/replay_pipeline.py --input-dir path/to/raw_drive_folder

To rerun only the M8/scoring stage from an existing feature vector:
    python scripts/replay_pipeline.py \
        --test-id TEST_1777763460177_VECTOR_ONLY \
        --feature-vector analysis_exports/TEST_1777763460177/2_feature_vector.csv
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------------
# Make sure imports work when the script is executed from either:
#   1) server root:      python scripts/replay_pipeline.py ...
#   2) scripts folder:   python replay_pipeline.py ...
# -----------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
SERVER_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else Path.cwd()
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))
os.chdir(SERVER_ROOT)


from app.services.sensor_sync import process_sensor_json  # noqa: E402
from app.services.vision_service import analyze_video_for_server  # noqa: E402
from app.services.vector_builder import build_feature_vector  # noqa: E402
from app.routes import test_routes  # noqa: E402


MODEL_FEATURES = [
    "speed_kmh",
    "jerk",
    "car_distance",
    "car_ttc",
    "sign_type",
    "sign_distance",
    "sign_ttc",
]


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
SENSOR_EXTENSIONS = {".json", ".csv"}


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _print_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _ensure_file(path: Optional[str], label: str) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{label} file not found: {p}")
    return p


def _find_one_file(input_dir: Path, extensions: set[str], label: str) -> Path:
    matches = [
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    if not matches:
        raise FileNotFoundError(f"No {label} file found under: {input_dir}")
    if len(matches) > 1:
        print(f"⚠️ Found multiple {label} files. Using the largest/newest-looking one:")
        for m in matches[:10]:
            print(f"   - {m}")
        # For videos, largest is usually the real drive. For JSONs, choose largest too.
        matches.sort(key=lambda p: p.stat().st_size, reverse=True)
    return matches[0]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_sensor_dataframe(sensors_path: Path) -> pd.DataFrame:
    """Load either raw Flutter JSON or an already-processed 1_raw_sensors.csv."""
    if sensors_path.suffix.lower() == ".csv":
        df = pd.read_csv(sensors_path)

        required = {"time_seconds", "speed_kmh", "jerk", "lat", "lon"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                f"Sensor CSV is missing columns {missing}. "
                "Expected a processed 1_raw_sensors.csv export."
            )
        return df

    return process_sensor_json(str(sensors_path))


def _load_ai_models_or_fail() -> None:
    """Use the backend's actual loader, but fail loudly if it silently logs an error."""
    test_routes.load_ai_models()

    if test_routes.global_scaler is None:
        raise RuntimeError(
            "global_scaler was not loaded. Check app/ai_models/global_scaler.pkl "
            "and make sure you run this script from the server root."
        )
    if test_routes.global_lstm_model is None:
        raise RuntimeError(
            "global_lstm_model was not loaded. Check app/ai_models/final_model.keras "
            "and TensorFlow/Keras installation."
        )


# -----------------------------------------------------------------------------
# Positive action detection — mirrors the patched backend logic.
# -----------------------------------------------------------------------------
def detect_correct_stops(combined_df: pd.DataFrame) -> List[Dict[str, Any]]:
    positive_actions: List[Dict[str, Any]] = []

    SIGN_PROXIMITY_M = 8.0
    STOP_SPEED_KMH = 1.0
    STOP_DURATION_S = 0.5
    GRACE_AFTER_SIGN_LOST_S = 1.5

    active_start_idx: Optional[int] = None
    last_sign_seen_idx: Optional[int] = None
    stop_start_idx: Optional[int] = None
    correct_stop_emitted = False

    for i in range(len(combined_df)):
        row = combined_df.iloc[i]
        st = int(_safe_float(row.get("sign_type"), 0))
        sd = _safe_float(row.get("sign_distance"), 99.0)
        spd = _safe_float(row.get("speed_kmh"), 0.0)
        current_ts = _safe_float(row.get("time_seconds"), i / 10.0)

        # CorrectStop is valid only for STOP signs, not yield/no_entry.
        raw_in_stop_zone = (st == 1 and sd < SIGN_PROXIMITY_M)
        in_stop_zone = raw_in_stop_zone

        if raw_in_stop_zone:
            if active_start_idx is None:
                active_start_idx = i
                stop_start_idx = None
                correct_stop_emitted = False
            last_sign_seen_idx = i
        elif active_start_idx is not None and last_sign_seen_idx is not None:
            last_seen_ts = _safe_float(combined_df.iloc[last_sign_seen_idx].get("time_seconds"), current_ts)
            if (current_ts - last_seen_ts) <= GRACE_AFTER_SIGN_LOST_S:
                in_stop_zone = True
            else:
                active_start_idx = None
                last_sign_seen_idx = None
                stop_start_idx = None
                correct_stop_emitted = False
                continue
        else:
            continue

        if not in_stop_zone or active_start_idx is None:
            continue

        if spd < STOP_SPEED_KMH:
            if stop_start_idx is None:
                stop_start_idx = i

            stop_start_ts = _safe_float(combined_df.iloc[stop_start_idx].get("time_seconds"), current_ts)
            stopped_duration_s = current_ts - stop_start_ts + 0.1

            if stopped_duration_s >= STOP_DURATION_S and not correct_stop_emitted:
                positive_actions.append({
                    "timestamp_sec": round(stop_start_ts, 2),
                    "type": "CorrectStop",
                    "sign_code": 1,
                    "approach_distance_m": round(
                        _safe_float(combined_df.iloc[active_start_idx].get("sign_distance"), 99.0),
                        1,
                    ),
                })
                correct_stop_emitted = True
        else:
            # Moving again before STOP_DURATION_S resets only the stop timer.
            stop_start_idx = None

    return positive_actions


# -----------------------------------------------------------------------------
# M8 + scoring replay — mirrors the patched /evaluate logic.
# -----------------------------------------------------------------------------
def run_m8_and_score(combined_df: pd.DataFrame, test_id: str) -> Dict[str, Any]:
    _load_ai_models_or_fail()

    for col in MODEL_FEATURES:
        if col not in combined_df.columns:
            raise ValueError(f"Missing required model feature column: {col}")

    X_raw = combined_df[MODEL_FEATURES].values
    X_scaled = test_routes.global_scaler.transform(X_raw)

    WINDOW_SIZE = 30
    X_windows = np.array([
        X_scaled[i:i + WINDOW_SIZE]
        for i in range(len(X_scaled) - WINDOW_SIZE + 1)
    ])

    decision_log: List[Dict[str, Any]] = []
    xai_data: Dict[str, Any] = {}
    action_sequences: List[Dict[str, Any]] = []
    positive_actions: List[Dict[str, Any]] = detect_correct_stops(combined_df)
    final_grade = 100
    total_violation_events = 0
    violations_detected: List[int] = []

    if len(X_windows) == 0:
        return {
            "test_id": test_id,
            "exported_at": datetime.now().isoformat(),
            "grade": final_grade,
            "decision_log": decision_log,
            "xai_explanations": xai_data,
            "action_sequences": action_sequences,
            "positive_actions": positive_actions,
            "violations_codes": violations_detected,
            "violation_events_count": total_violation_events,
            "windows_analyzed": 0,
            "note": "Not enough rows for a 30-frame M8 window.",
        }

    _print_step(f"Running M8 on {len(X_windows)} windows...")

    BATCH_SIZE = 32
    num_batches = max(1, (len(X_windows) + BATCH_SIZE - 1) // BATCH_SIZE)
    preds_list: List[np.ndarray] = []
    attn_list: List[np.ndarray] = []

    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(X_windows))
        batch = X_windows[start:end]
        preds, attn = test_routes.global_lstm_model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
        preds_list.append(preds)
        attn_list.append(attn)
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            _print_step(f"M8 batch {batch_idx + 1}/{num_batches}")

    predictions = np.concatenate(preds_list, axis=0)
    attention_weights = np.concatenate(attn_list, axis=0)

    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    # Mode smoothing, same as backend.
    smoothed: List[int] = []
    sw = 5
    for i in range(len(predicted_classes)):
        end = min(i + sw, len(predicted_classes))
        votes = predicted_classes[i:end]
        if len(votes) == 0:
            smoothed.append(int(predicted_classes[i]))
            continue
        mode_, _ = stats.mode(votes, keepdims=False)
        smoothed.append(int(mode_))

    MERGE_GAP_S = 1.5
    MIN_CONF = 0.55
    unique_violation_types = set()
    last_event: Dict[int, Dict[str, Any]] = {}

    for i, pc in enumerate(smoothed):
        ts = _safe_float(combined_df.iloc[i].get("time_seconds"), 0.0) if i < len(combined_df) else 0.0
        conf = float(confidences[i])

        decision_log.append({
            "timestamp_sec": round(ts, 2),
            "predicted_class": int(pc),
            "raw_prediction": int(predicted_classes[i]),
            "confidence": round(conf, 3),
            "all_probabilities": [round(float(p), 3) for p in predictions[i]],
        })

        if pc == 0 or conf < MIN_CONF:
            continue

        pc_int = int(pc)
        unique_violation_types.add(pc_int)
        prev = last_event.get(pc_int)

        if prev is None or (ts - prev["end_t"]) > MERGE_GAP_S:
            total_violation_events += 1
            evt_idx = total_violation_events
            last_event[pc_int] = {
                "end_t": ts,
                "event_idx": evt_idx,
                "confidence_max": conf,
            }
            peak_frame = int(np.argmax(attention_weights[i]))
            sample_idx = min(i + peak_frame, len(combined_df) - 1)
            xai_data[f"event_{evt_idx}_class_{pc_int}"] = {
                "timestamp_sec": _safe_float(combined_df.iloc[sample_idx].get("time_seconds"), 0.0),
                "violation_code": pc_int,
                "decisive_frame_in_window": peak_frame,
                "attention_score": float(np.max(attention_weights[i])),
                "attention_array": attention_weights[i].tolist(),
            }
        else:
            prev["end_t"] = ts
            if conf > prev["confidence_max"]:
                prev["confidence_max"] = conf
                peak_frame = int(np.argmax(attention_weights[i]))
                sample_idx = min(i + peak_frame, len(combined_df) - 1)
                evt_idx = prev["event_idx"]
                xai_data[f"event_{evt_idx}_class_{pc_int}"] = {
                    "timestamp_sec": _safe_float(combined_df.iloc[sample_idx].get("time_seconds"), 0.0),
                    "violation_code": pc_int,
                    "decisive_frame_in_window": peak_frame,
                    "attention_score": float(np.max(attention_weights[i])),
                    "attention_array": attention_weights[i].tolist(),
                }

    violations_detected = sorted(int(v) for v in unique_violation_types)

    # Inject positive actions into decision_log, same as backend.
    for pa in positive_actions:
        decision_log.append({
            "timestamp_sec": pa["timestamp_sec"],
            "predicted_class": 0,
            "event_type": "POSITIVE_ACTION",
            "subtype": pa["type"],
            "sign_code": pa["sign_code"],
            "approach_distance_m": pa["approach_distance_m"],
        })
    decision_log.sort(key=lambda x: x["timestamp_sec"])

    # Action sequences, same as backend.
    i = 0
    while i < len(smoothed) - 1:
        if smoothed[i] != 0:
            seq = [int(smoothed[i])]
            seq_start = _safe_float(combined_df.iloc[i].get("time_seconds"), 0.0) if i < len(combined_df) else 0.0
            j = i + 1
            last_t = seq_start
            while j < len(smoothed):
                t = _safe_float(combined_df.iloc[j].get("time_seconds"), last_t) if j < len(combined_df) else last_t
                if t - seq_start > 5.0:
                    break
                if smoothed[j] != 0 and smoothed[j] != seq[-1]:
                    seq.append(int(smoothed[j]))
                last_t = t
                j += 1
            if len(seq) >= 2:
                action_sequences.append({
                    "start_time_sec": round(seq_start, 2),
                    "sequence_codes": seq,
                    "duration_sec": round(last_t - seq_start, 2),
                })
            i = j
        else:
            i += 1

    POINTS_PER_VIOLATION = 5
    POINTS_PER_POSITIVE = 2
    MAX_POSITIVE_BONUS = 6

    base_grade = max(0, 100 - (total_violation_events * POINTS_PER_VIOLATION))
    positive_bonus = min(len(positive_actions) * POINTS_PER_POSITIVE, MAX_POSITIVE_BONUS)
    final_grade = base_grade if total_violation_events > 0 else min(100, base_grade + positive_bonus)

    return {
        "test_id": test_id,
        "exported_at": datetime.now().isoformat(),
        "grade": final_grade,
        "decision_log": decision_log,
        "xai_explanations": xai_data,
        "action_sequences": action_sequences,
        "positive_actions": positive_actions,
        "violations_codes": violations_detected,
        "violation_events_count": total_violation_events,
        "windows_analyzed": int(len(X_windows)),
    }


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------
def summarize_feature_vector(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}

    if "sign_type" in df.columns:
        counts = df["sign_type"].value_counts(dropna=False).to_dict()
        out["sign_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["stop_rows"] = int((df["sign_type"] == 1).sum())
        out["yield_rows"] = int((df["sign_type"] == 2).sum())
        out["no_entry_rows"] = int((df["sign_type"] == 3).sum())

    if "sign_distance" in df.columns:
        active = df[df["sign_distance"] < 99.0]
        out["active_sign_rows"] = int(len(active))
        if not active.empty:
            out["min_sign_distance"] = round(float(active["sign_distance"].min()), 2)
            out["max_active_sign_distance"] = round(float(active["sign_distance"].max()), 2)

    if "car_distance" in df.columns:
        active_car = df[df["car_distance"] < 99.0]
        out["active_car_rows"] = int(len(active_car))
        if not active_car.empty:
            out["min_car_distance"] = round(float(active_car["car_distance"].min()), 2)

    return out


def build_summary(test_id: str, vector_df: pd.DataFrame, video_events: Optional[List[Dict[str, Any]]], model_out: Dict[str, Any]) -> Dict[str, Any]:
    video_event_counts: Dict[str, int] = {}
    if video_events:
        for event in video_events:
            et = str(event.get("type", "unknown"))
            video_event_counts[et] = video_event_counts.get(et, 0) + 1

    return {
        "test_id": test_id,
        "created_at": datetime.now().isoformat(),
        "video_events_count": len(video_events or []),
        "video_event_type_counts": video_event_counts,
        "feature_vector": summarize_feature_vector(vector_df),
        "model": {
            "grade": model_out.get("grade"),
            "violation_events_count": model_out.get("violation_events_count"),
            "violations_codes": model_out.get("violations_codes"),
            "positive_actions_count": len(model_out.get("positive_actions", [])),
            "positive_actions": model_out.get("positive_actions", []),
            "windows_analyzed": model_out.get("windows_analyzed"),
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print(f"Test ID: {summary['test_id']}")
    print(f"Video events: {summary['video_events_count']}")
    print(f"Video event types: {summary['video_event_type_counts']}")

    fv = summary["feature_vector"]
    print(f"Vector rows: {fv.get('rows')}")
    print(f"Sign counts: {fv.get('sign_counts')}")
    print(f"STOP rows: {fv.get('stop_rows', 0)}")
    print(f"YIELD rows: {fv.get('yield_rows', 0)}")
    print(f"NO_ENTRY rows: {fv.get('no_entry_rows', 0)}")
    print(f"Active sign rows: {fv.get('active_sign_rows', 0)}")

    model = summary["model"]
    print(f"Grade: {model.get('grade')}")
    print(f"Violation events: {model.get('violation_events_count')}")
    print(f"Violation codes: {model.get('violations_codes')}")
    print(f"Positive actions: {model.get('positive_actions_count')}")
    for pa in model.get("positive_actions", []):
        print(f"  + {pa}")
    print("=" * 80 + "\n")


# -----------------------------------------------------------------------------
# Main replay modes
# -----------------------------------------------------------------------------
async def replay_full_pipeline(
    test_id: str,
    video_path: Path,
    sensors_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_step(f"Processing sensors: {sensors_path}")
    sensor_df = load_sensor_dataframe(sensors_path)
    sensor_csv = output_dir / "1_raw_sensors.csv"
    sensor_df.to_csv(sensor_csv, index=False)
    _print_step(f"Saved {sensor_csv}")

    _print_step(f"Running YOLO/Kalman/VisionFilter on video: {video_path}")
    annotated_video_path = output_dir / "yolo_annotated.mp4"
    video_events = await analyze_video_for_server(
        str(video_path),
        sensor_df,
        test_id=test_id,
        output_video_path=str(annotated_video_path),
    )
    _print_step(f"Saved annotated YOLO video: {annotated_video_path}")
    video_events_json = output_dir / "1b_video_events.json"
    with video_events_json.open("w", encoding="utf-8") as f:
        json.dump(video_events, f, ensure_ascii=False, indent=2, default=_json_default)
    _print_step(f"Saved {video_events_json} ({len(video_events)} events)")

    _print_step("Building feature vector...")
    combined_df = build_feature_vector(sensor_df, video_events)
    vector_csv = output_dir / "2_feature_vector.csv"
    combined_df.to_csv(vector_csv, index=False)
    _print_step(f"Saved {vector_csv}")

    _print_step("Running M8 + scoring...")
    model_out = run_m8_and_score(combined_df, test_id)
    model_json = output_dir / "3_model_thought.json"
    with model_json.open("w", encoding="utf-8") as f:
        json.dump(model_out, f, ensure_ascii=False, indent=4, default=_json_default)
    _print_step(f"Saved {model_json}")

    summary = build_summary(test_id, combined_df, video_events, model_out)
    summary_json = output_dir / "4_replay_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4, default=_json_default)
    _print_step(f"Saved {summary_json}")

    return summary


async def replay_from_feature_vector(
    test_id: str,
    feature_vector_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_step(f"Loading existing feature vector: {feature_vector_path}")
    combined_df = pd.read_csv(feature_vector_path)
    vector_csv = output_dir / "2_feature_vector.csv"
    combined_df.to_csv(vector_csv, index=False)

    _print_step("Running M8 + scoring only...")
    model_out = run_m8_and_score(combined_df, test_id)
    model_json = output_dir / "3_model_thought.json"
    with model_json.open("w", encoding="utf-8") as f:
        json.dump(model_out, f, ensure_ascii=False, indent=4, default=_json_default)

    summary = build_summary(test_id, combined_df, video_events=None, model_out=model_out)
    summary_json = output_dir / "4_replay_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4, default=_json_default)

    return summary


async def replay_from_sensors_and_video_events(
    test_id: str,
    sensors_path: Path,
    video_events_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_step(f"Processing sensors: {sensors_path}")
    sensor_df = load_sensor_dataframe(sensors_path)
    sensor_csv = output_dir / "1_raw_sensors.csv"
    sensor_df.to_csv(sensor_csv, index=False)

    _print_step(f"Loading existing video events: {video_events_path}")
    with video_events_path.open("r", encoding="utf-8") as f:
        video_events = json.load(f)
    copied_events_path = output_dir / "1b_video_events.json"
    with copied_events_path.open("w", encoding="utf-8") as f:
        json.dump(video_events, f, ensure_ascii=False, indent=2, default=_json_default)

    _print_step("Building feature vector...")
    combined_df = build_feature_vector(sensor_df, video_events)
    vector_csv = output_dir / "2_feature_vector.csv"
    combined_df.to_csv(vector_csv, index=False)

    _print_step("Running M8 + scoring...")
    model_out = run_m8_and_score(combined_df, test_id)
    model_json = output_dir / "3_model_thought.json"
    with model_json.open("w", encoding="utf-8") as f:
        json.dump(model_out, f, ensure_ascii=False, indent=4, default=_json_default)

    summary = build_summary(test_id, combined_df, video_events, model_out)
    summary_json = output_dir / "4_replay_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4, default=_json_default)

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay raw Auto Tester drive data through the backend pipeline offline."
    )

    parser.add_argument("--test-id", default=None, help="Replay test ID. Default: REPLAY_<timestamp>")
    parser.add_argument("--input-dir", default=None, help="Folder containing one video file and one sensor JSON.")
    parser.add_argument("--video", default=None, help="Raw drive video path: mp4/mov/avi/mkv/m4v")
    parser.add_argument("--sensors", default=None, help="Raw Flutter sensor JSON path or processed 1_raw_sensors.csv")
    parser.add_argument("--video-events", default=None, help="Existing 1b_video_events.json to skip YOLO and rebuild vector/model")
    parser.add_argument("--feature-vector", default=None, help="Existing 2_feature_vector.csv to rerun only M8/scoring")
    parser.add_argument("--output-dir", default=None, help="Output folder. Default: analysis_exports/<test-id>")

    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    test_id = args.test_id or f"REPLAY_{int(time.time())}"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else SERVER_ROOT / "analysis_exports" / test_id

    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else None

    feature_vector_path = _ensure_file(args.feature_vector, "feature vector")
    video_events_path = _ensure_file(args.video_events, "video events")

    if feature_vector_path is not None:
        summary = await replay_from_feature_vector(test_id, feature_vector_path, output_dir)
        print_summary(summary)
        return

    video_path = _ensure_file(args.video, "video")
    sensors_path = _ensure_file(args.sensors, "sensors")

    if input_dir is not None:
        if not input_dir.exists() or not input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {input_dir}")
        if video_path is None and video_events_path is None:
            video_path = _find_one_file(input_dir, VIDEO_EXTENSIONS, "video")
        if sensors_path is None:
            sensors_path = _find_one_file(input_dir, SENSOR_EXTENSIONS, "sensor JSON/CSV")

    if sensors_path is not None and video_events_path is not None:
        summary = await replay_from_sensors_and_video_events(test_id, sensors_path, video_events_path, output_dir)
        print_summary(summary)
        return

    if video_path is None or sensors_path is None:
        raise ValueError(
            "You must provide either:\n"
            "  1) --video and --sensors for full pipeline replay, or\n"
            "  2) --input-dir containing one video and one sensor JSON/CSV, or\n"
            "  3) --feature-vector to rerun only M8/scoring, or\n"
            "  4) --sensors and --video-events to skip YOLO."
        )

    summary = await replay_full_pipeline(test_id, video_path, sensors_path, output_dir)
    print_summary(summary)


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        raise SystemExit(130)
    except Exception as e:
        print(f"\n❌ Replay failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
