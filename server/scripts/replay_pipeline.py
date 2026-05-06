"""
Offline replay script for Auto Tester backend — M12 version.

Purpose
-------
Run an existing drive's RAW files through the backend pipeline again without
Flutter, without authentication, and without MongoDB:

    sensors JSON/CSV -> sensor_sync/load CSV -> YOLO/Kalman/VisionFilter -> vector_builder -> M12 LSTM + episode resolver -> exports

Outputs
-------
The script writes a replay export folder containing:

    1_raw_sensors.csv
    1b_video_events.json
    input_video.mp4
    yolo_annotated.mp4
    2_feature_vector.csv
    3_model_thought.json
    4_replay_summary.json

How to run from the server root
-------------------------------
Full replay:
    python scripts/replay_pipeline.py \
        --test-id TEST_ONLYSTOP_M12_REPLAY \
        --video path/to/drive.mp4 \
        --sensors path/to/sensors.json

Provide a directory and let the script find one video + one sensor file:
    python scripts/replay_pipeline.py --input-dir path/to/raw_drive_folder

Skip YOLO and rebuild vector/model from existing sensors + video events:
    python scripts/replay_pipeline.py \
        --test-id TEST_ONLYSTOP_M12_FROM_EVENTS \
        --sensors analysis_exports/OLD/1_raw_sensors.csv \
        --video-events analysis_exports/OLD/1b_video_events.json

Rerun only M12/scoring from an existing feature vector:
    python scripts/replay_pipeline.py \
        --test-id TEST_ONLYSTOP_M12_VECTOR_ONLY \
        --feature-vector analysis_exports/OLD/2_feature_vector.csv

Requirements
------------
This script delegates the actual model scoring to app.routes.test_routes.
Therefore app/routes/test_routes.py must already be updated to M12 and expose:

    load_ai_models()
    run_m12_episode_scoring(combined_df, test_id, test_export_path)

Expected artifacts in app/ai_models:

    m12_model.keras
    m12_scaler.pkl
    m12_class_map.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Make sure imports work when executed from either:
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


M12_FALLBACK_FEATURES = [
    "time_seconds",
    "speed_kmh",
    "jerk",
    "car_distance",
    "car_ttc",
    "car_relative_x",
    "car_relative_y",
    "car_motion_x",
    "car_motion_y",
    "car_static_score",
    "sign_type",
    "sign_distance",
    "sign_ttc",
    "memory_sign_type",
    "memory_sign_distance",
    "time_since_sign_seen_sec",
    "meters_since_sign_seen",
    "last_seen_sign_distance",
    "memory_sign_ttc",
]

MODEL_ID = getattr(test_routes, "MODEL_ID", "M12")
MODEL_FEATURES = list(getattr(test_routes, "MODEL_FEATURES", M12_FALLBACK_FEATURES))
MODEL_WINDOW_SIZE = int(getattr(test_routes, "MODEL_WINDOW_SIZE", 30))
MODEL_CLASS_NAMES = getattr(test_routes, "MODEL_CLASS_NAMES", {
    0: "Normal Driving",
    1: "Tailgating",
    2: "Running Stop",
    3: "Failure to Yield",
    4: "No Entry Violation",
    5: "Correct Stop",
    6: "Correct Yield",
})
VIOLATION_CLASSES = set(getattr(test_routes, "VIOLATION_CLASSES", {1, 2, 3, 4}))
IGNORED_WARNING_CLASSES = set(getattr(test_routes, "IGNORED_WARNING_CLASSES", {1}))
FAIL_CLASSES = set(getattr(test_routes, "FAIL_CLASSES", {2, 3, 4}))
POSITIVE_ACTION_CLASSES = set(getattr(test_routes, "POSITIVE_ACTION_CLASSES", {5, 6}))
POSITIVE_ACTION_TYPES = getattr(test_routes, "POSITIVE_ACTION_TYPES", {
    5: "CorrectStop",
    6: "CorrectYield",
})

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
        print(f"⚠️ Found multiple {label} files. Using the largest one:")
        for m in matches[:10]:
            print(f"   - {m}")
        matches.sort(key=lambda p: p.stat().st_size, reverse=True)

    return matches[0]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
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
    """Use backend's real loader, but fail loudly if artifacts did not load."""
    test_routes.load_ai_models()

    if getattr(test_routes, "global_scaler", None) is None:
        raise RuntimeError(
            f"global_scaler was not loaded. Check app/ai_models/{MODEL_ID.lower()}_scaler.pkl "
            "and make sure you run this script from the server root."
        )

    if getattr(test_routes, "global_lstm_model", None) is None:
        raise RuntimeError(
            f"global_lstm_model was not loaded. Check app/ai_models/{MODEL_ID.lower()}_model.keras "
            "and TensorFlow/Keras installation."
        )


def _load_thought_json_if_exists(output_dir: Path) -> Dict[str, Any]:
    thought_path = output_dir / "3_model_thought.json"
    if not thought_path.exists():
        return {}
    with thought_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_thought_json(output_dir: Path, payload: Dict[str, Any]) -> None:
    thought_path = output_dir / "3_model_thought.json"
    with thought_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4, default=_json_default)


# -----------------------------------------------------------------------------
# M12 scoring replay — delegates to app.routes.test_routes
# -----------------------------------------------------------------------------
def run_model_and_score(
    combined_df: pd.DataFrame,
    test_id: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run M12 model + episode resolver using the same code path as /evaluate."""
    _load_ai_models_or_fail()

    if not hasattr(test_routes, "run_m12_episode_scoring"):
        raise RuntimeError(
            "app.routes.test_routes does not expose run_m12_episode_scoring(). "
            "Replace app/routes/test_routes.py with the M12 version before using this replay script."
        )

    missing = [col for col in MODEL_FEATURES if col not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing required {MODEL_ID} feature columns: {missing}")

    _print_step(f"Running {MODEL_ID} + episode scoring...")

    score = test_routes.run_m12_episode_scoring(
        combined_df=combined_df,
        test_id=test_id,
        test_export_path=str(output_dir),
    )

    # test_routes.run_m12_episode_scoring writes the detailed 3_model_thought.json.
    # Load it and merge the compact return fields so summary always has everything.
    thought_payload = _load_thought_json_if_exists(output_dir)
    model_out: Dict[str, Any] = dict(thought_payload) if thought_payload else {}

    model_out.setdefault("test_id", test_id)
    model_out.setdefault("exported_at", datetime.now().isoformat())
    model_out.setdefault("model_id", MODEL_ID)
    model_out.setdefault("feature_names", MODEL_FEATURES)
    model_out.setdefault("class_names", {str(k): v for k, v in MODEL_CLASS_NAMES.items()})
    model_out.setdefault("violation_classes", sorted(VIOLATION_CLASSES))
    model_out.setdefault("positive_action_classes", sorted(POSITIVE_ACTION_CLASSES))
    model_out.setdefault("fail_classes", sorted(FAIL_CLASSES))
    model_out.setdefault("ignored_warning_classes", sorted(IGNORED_WARNING_CLASSES))

    # Normalize aliases from the score dict.
    model_out["result"] = score.get("result", model_out.get("result", "PASS"))
    model_out["passed"] = score.get("passed", model_out.get("passed", True))
    model_out["grade"] = score.get("grade", model_out.get("grade", 100))
    model_out["mistakes_count"] = score.get("mistakes_count", model_out.get("mistakes_count", 0))
    model_out["mistake_codes"] = score.get(
        "violations_detected",
        model_out.get("mistake_codes", model_out.get("violations_codes", [])),
    )
    model_out["violations_codes"] = model_out.get("mistake_codes", [])
    model_out["violation_events_count"] = model_out.get("mistakes_count", 0)
    model_out["ignored_warning_codes"] = score.get(
        "ignored_warning_codes",
        model_out.get("ignored_warning_codes", []),
    )
    model_out["ignored_warning_events"] = score.get(
        "ignored_warning_events",
        model_out.get("ignored_warning_events", []),
    )
    model_out["ignored_warning_events_count"] = len(model_out.get("ignored_warning_events", []))
    model_out["decision_log"] = score.get("decision_log", model_out.get("decision_log", []))
    model_out["xai_explanations"] = score.get(
        "xai_data",
        model_out.get("xai_explanations", {}),
    )
    model_out["action_sequences"] = score.get(
        "action_sequences",
        model_out.get("action_sequences", []),
    )
    model_out["positive_actions"] = score.get(
        "positive_actions",
        model_out.get("positive_actions", []),
    )
    model_out["windows_analyzed"] = score.get(
        "windows_analyzed",
        model_out.get("windows_analyzed", 0),
    )
    model_out.setdefault(
        "episode_policy",
        "M12 sign episode resolver: CorrectStop suppresses early RunningStop in same STOP episode.",
    )

    _write_thought_json(output_dir, model_out)
    return model_out


# -----------------------------------------------------------------------------
# Summary helpers
# -----------------------------------------------------------------------------
def summarize_feature_vector(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(len(df))}

    if "sign_type" in df.columns:
        sign_type = pd.to_numeric(df["sign_type"], errors="coerce").fillna(0).round().astype(int)
        counts = sign_type.value_counts(dropna=False).to_dict()
        out["sign_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["stop_rows"] = int((sign_type == 1).sum())
        out["yield_rows"] = int((sign_type == 2).sum())
        out["no_entry_rows"] = int((sign_type == 3).sum())

    if "memory_sign_type" in df.columns:
        memory_sign_type = pd.to_numeric(df["memory_sign_type"], errors="coerce").fillna(0).round().astype(int)
        counts = memory_sign_type.value_counts(dropna=False).to_dict()
        out["memory_sign_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["memory_stop_rows"] = int((memory_sign_type == 1).sum())
        out["memory_yield_rows"] = int((memory_sign_type == 2).sum())
        out["memory_no_entry_rows"] = int((memory_sign_type == 3).sum())

    if "sign_distance" in df.columns:
        sign_distance = pd.to_numeric(df["sign_distance"], errors="coerce").fillna(99.0)
        active = df[sign_distance < 99.0]
        out["active_sign_rows"] = int(len(active))
        if not active.empty:
            out["min_sign_distance"] = round(float(sign_distance[sign_distance < 99.0].min()), 2)
            out["max_active_sign_distance"] = round(float(sign_distance[sign_distance < 99.0].max()), 2)

    if "memory_sign_distance" in df.columns:
        memory_sign_distance = pd.to_numeric(df["memory_sign_distance"], errors="coerce").fillna(99.0)
        active_memory = df[memory_sign_distance < 99.0]
        out["active_memory_sign_rows"] = int(len(active_memory))
        if not active_memory.empty:
            out["min_memory_sign_distance"] = round(float(memory_sign_distance[memory_sign_distance < 99.0].min()), 2)
            out["max_active_memory_sign_distance"] = round(float(memory_sign_distance[memory_sign_distance < 99.0].max()), 2)

    if "car_distance" in df.columns:
        car_distance = pd.to_numeric(df["car_distance"], errors="coerce").fillna(99.0)
        active_car = df[car_distance < 99.0]
        out["active_car_rows"] = int(len(active_car))
        if not active_car.empty:
            out["min_car_distance"] = round(float(car_distance[car_distance < 99.0].min()), 2)

    return out


def build_summary(
    test_id: str,
    vector_df: pd.DataFrame,
    video_events: Optional[List[Dict[str, Any]]],
    model_out: Dict[str, Any],
) -> Dict[str, Any]:
    video_event_counts: Dict[str, int] = {}
    if video_events:
        for event in video_events:
            et = str(event.get("type", "unknown"))
            video_event_counts[et] = video_event_counts.get(et, 0) + 1

    positive_actions = model_out.get("positive_actions", [])
    ignored_warning_events = model_out.get("ignored_warning_events", [])

    mistakes_count = model_out.get(
        "mistakes_count",
        model_out.get("violation_events_count", 0),
    )
    mistake_codes = model_out.get(
        "mistake_codes",
        model_out.get("violations_codes", []),
    )

    return {
        "test_id": test_id,
        "created_at": datetime.now().isoformat(),
        "model_id": model_out.get("model_id", MODEL_ID),
        "video_events_count": len(video_events or []),
        "video_event_type_counts": video_event_counts,
        "feature_vector": summarize_feature_vector(vector_df),
        "model": {
            "result": model_out.get("result"),
            "passed": model_out.get("passed"),
            "mistakes_count": mistakes_count,
            "mistake_codes": mistake_codes,
            "grade": model_out.get("grade"),
            "violation_events_count": model_out.get("violation_events_count", mistakes_count),
            "violations_codes": model_out.get("violations_codes", mistake_codes),
            "ignored_warning_codes": model_out.get("ignored_warning_codes", []),
            "ignored_warning_events_count": len(ignored_warning_events),
            "ignored_warning_events": ignored_warning_events,
            "positive_actions_count": len(positive_actions),
            "positive_actions": positive_actions,
            "windows_analyzed": model_out.get("windows_analyzed"),
            "episode_policy": model_out.get("episode_policy"),
        },
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("REPLAY SUMMARY")
    print("=" * 80)
    print(f"Test ID: {summary['test_id']}")
    print(f"Model: {summary.get('model_id', MODEL_ID)}")
    print(f"Video events: {summary['video_events_count']}")
    print(f"Video event types: {summary['video_event_type_counts']}")

    fv = summary["feature_vector"]
    print(f"Vector rows: {fv.get('rows')}")
    print(f"Sign counts: {fv.get('sign_counts')}")
    print(f"Memory sign counts: {fv.get('memory_sign_counts')}")
    print(f"STOP rows: {fv.get('stop_rows', 0)}")
    print(f"STOP memory rows: {fv.get('memory_stop_rows', 0)}")
    print(f"YIELD rows: {fv.get('yield_rows', 0)}")
    print(f"YIELD memory rows: {fv.get('memory_yield_rows', 0)}")
    print(f"NO_ENTRY rows: {fv.get('no_entry_rows', 0)}")
    print(f"NO_ENTRY memory rows: {fv.get('memory_no_entry_rows', 0)}")
    print(f"Active sign rows: {fv.get('active_sign_rows', 0)}")
    print(f"Active memory sign rows: {fv.get('active_memory_sign_rows', 0)}")

    model = summary["model"]
    print(f"Result: {model.get('result')}")
    print(f"Mistakes: {model.get('mistakes_count')}")
    print(f"Mistake codes: {model.get('mistake_codes')}")
    print(f"Ignored warning codes: {model.get('ignored_warning_codes')}")
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

    try:
        saved_input_video = output_dir / "input_video.mp4"
        shutil.copyfile(video_path, saved_input_video)
        _print_step(f"Saved input video copy: {saved_input_video}")
    except Exception as e:
        print(f"⚠️ Could not save input video copy: {e}")

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

    _print_step("Building M12 feature vector...")
    combined_df = build_feature_vector(sensor_df, video_events)
    vector_csv = output_dir / "2_feature_vector.csv"
    combined_df.to_csv(vector_csv, index=False)
    _print_step(f"Saved {vector_csv}")

    model_out = run_model_and_score(combined_df, test_id, output_dir)
    _print_step(f"Saved {output_dir / '3_model_thought.json'}")

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
    _print_step(f"Saved {vector_csv}")

    model_out = run_model_and_score(combined_df, test_id, output_dir)
    _print_step(f"Saved {output_dir / '3_model_thought.json'}")

    summary = build_summary(test_id, combined_df, video_events=None, model_out=model_out)
    summary_json = output_dir / "4_replay_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4, default=_json_default)
    _print_step(f"Saved {summary_json}")

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
    _print_step(f"Saved {sensor_csv}")

    _print_step(f"Loading existing video events: {video_events_path}")
    with video_events_path.open("r", encoding="utf-8") as f:
        video_events = json.load(f)

    copied_events_path = output_dir / "1b_video_events.json"
    with copied_events_path.open("w", encoding="utf-8") as f:
        json.dump(video_events, f, ensure_ascii=False, indent=2, default=_json_default)
    _print_step(f"Saved {copied_events_path}")

    _print_step("Building M12 feature vector...")
    combined_df = build_feature_vector(sensor_df, video_events)
    vector_csv = output_dir / "2_feature_vector.csv"
    combined_df.to_csv(vector_csv, index=False)
    _print_step(f"Saved {vector_csv}")

    model_out = run_model_and_score(combined_df, test_id, output_dir)
    _print_step(f"Saved {output_dir / '3_model_thought.json'}")

    summary = build_summary(test_id, combined_df, video_events, model_out)
    summary_json = output_dir / "4_replay_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4, default=_json_default)
    _print_step(f"Saved {summary_json}")

    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay raw Auto Tester drive data through the M12 backend pipeline offline."
    )

    parser.add_argument("--test-id", default=None, help="Replay test ID. Default: REPLAY_<timestamp>")
    parser.add_argument("--input-dir", default=None, help="Folder containing one video file and one sensor JSON/CSV.")
    parser.add_argument("--video", default=None, help="Raw drive video path: mp4/mov/avi/mkv/m4v")
    parser.add_argument("--sensors", default=None, help="Raw Flutter sensor JSON path or processed 1_raw_sensors.csv")
    parser.add_argument("--video-events", default=None, help="Existing 1b_video_events.json to skip YOLO and rebuild vector/model")
    parser.add_argument("--feature-vector", default=None, help="Existing 2_feature_vector.csv to rerun only M12/scoring")
    parser.add_argument("--output-dir", default=None, help="Output folder. Default: analysis_exports/<test-id>")

    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()

    test_id = args.test_id or f"REPLAY_{int(time.time())}"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else SERVER_ROOT / "analysis_exports" / test_id
    )

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
        summary = await replay_from_sensors_and_video_events(
            test_id,
            sensors_path,
            video_events_path,
            output_dir,
        )
        print_summary(summary)
        return

    if video_path is None or sensors_path is None:
        raise ValueError(
            "You must provide either:\n"
            "  1) --video and --sensors for full pipeline replay, or\n"
            "  2) --input-dir containing one video and one sensor JSON/CSV, or\n"
            "  3) --feature-vector to rerun only M12/scoring, or\n"
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
