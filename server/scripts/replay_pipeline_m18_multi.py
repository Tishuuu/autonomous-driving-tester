"""
M18 multi-episode replay pipeline for Auto Tester.

Supported modes from server root:

1) Re-score an existing vector:
   python scripts\replay_pipeline.py ^
     --test-id TEST_1778085692205_M14 ^
     --feature-vector analysis_exports\TEST_1778085692205_M13_REPLAY_SMOOTH\2_feature_vector.csv ^
     --output-dir analysis_exports\TEST_1778085692205_M14

2) Rebuild vector from processed sensors + existing video events:
   python scripts\replay_pipeline.py ^
     --test-id TEST_1778085692205_M14 ^
     --sensors analysis_exports\TEST_1778085692205\1_raw_sensors.csv ^
     --video-events analysis_exports\TEST_1778085692205\1b_video_events.json ^
     --output-dir analysis_exports\TEST_1778085692205_M14

This script expects app/routes/test_routes.py to be the M18 multi-episode version.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
SERVER_ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else Path.cwd()
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))
os.chdir(SERVER_ROOT)

from app.services.sensor_sync import process_sensor_json  # noqa: E402
from app.services.vector_builder import build_feature_vector  # noqa: E402
from app.routes import test_routes  # noqa: E402


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return str(obj)


def _print_step(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _ensure_file(path: Optional[str], label: str) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"{label} file not found: {p}")
    return p


def load_sensor_dataframe(sensors_path: Path) -> pd.DataFrame:
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


def vector_summary(df: pd.DataFrame) -> dict:
    sign_type = pd.to_numeric(df.get("sign_type", pd.Series([], dtype=float)), errors="coerce").fillna(0).astype(int)
    mem_type = pd.to_numeric(df.get("memory_sign_type", pd.Series([], dtype=float)), errors="coerce").fillna(0).astype(int)

    def count(series, value):
        return int((series == value).sum())

    out = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sign_counts": {str(k): int(v) for k, v in sign_type.value_counts().sort_index().items()},
        "memory_sign_counts": {str(k): int(v) for k, v in mem_type.value_counts().sort_index().items()},
        "stop_rows": count(sign_type, 1),
        "yield_rows": count(sign_type, 2),
        "no_entry_rows": count(sign_type, 3),
        "memory_stop_rows": count(mem_type, 1),
        "memory_yield_rows": count(mem_type, 2),
        "memory_no_entry_rows": count(mem_type, 3),
        "active_sign_rows": int((sign_type > 0).sum()),
        "active_memory_sign_rows": int((mem_type > 0).sum()),
    }
    return out


def build_summary(test_id: str, combined_df: pd.DataFrame, score: dict) -> dict:
    return {
        "test_id": test_id,
        "created_at": datetime.now().isoformat(),
        "model_id": getattr(test_routes, "MODEL_ID", "M18"),
        "model_type": getattr(test_routes, "MODEL_TYPE", "episode_level_scenario_bilstm"),
        "feature_vector": vector_summary(combined_df),
        "model": {
            "result": score["result"],
            "passed": bool(score["passed"]),
            "grade": int(score["grade"]),
            "mistakes_count": int(score["mistakes_count"]),
            "mistake_codes": score["violations_detected"],
            "violation_events_count": int(score["mistakes_count"]),
            "ignored_warning_codes": score["ignored_warning_codes"],
            "ignored_warning_events_count": len(score["ignored_warning_events"]),
            "positive_actions_count": len(score["positive_actions"]),
            "positive_actions": score["positive_actions"],
            "windows_analyzed": int(score["windows_analyzed"]),
            "episode_count": int(score.get("episode_count", 1)),
            "episodes": score.get("aggregation", {}).get("episodes", []),
        },
    }


def run_replay(args) -> dict:
    test_id = args.test_id or f"M18_MULTI_REPLAY_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir or Path("analysis_exports") / test_id).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_vector_path = _ensure_file(args.feature_vector, "feature-vector")
    sensors_path = _ensure_file(args.sensors, "sensors")
    video_events_path = _ensure_file(args.video_events, "video-events")

    if feature_vector_path is not None:
        _print_step(f"Loading feature vector: {feature_vector_path}")
        combined_df = pd.read_csv(feature_vector_path)
        combined_df.to_csv(out_dir / "2_feature_vector.csv", index=False)
    else:
        if sensors_path is None or video_events_path is None:
            raise ValueError("Provide --feature-vector OR both --sensors and --video-events")

        _print_step(f"Loading sensors: {sensors_path}")
        sensor_df = load_sensor_dataframe(sensors_path)
        sensor_df.to_csv(out_dir / "1_raw_sensors.csv", index=False)

        _print_step(f"Loading video events: {video_events_path}")
        with video_events_path.open("r", encoding="utf-8") as f:
            video_events = json.load(f)
        with (out_dir / "1b_video_events.json").open("w", encoding="utf-8") as f:
            json.dump(video_events, f, ensure_ascii=False, indent=2)

        _print_step("Building M18 feature vector...")
        combined_df = build_feature_vector(sensor_df, video_events)
        combined_df.to_csv(out_dir / "2_feature_vector.csv", index=False)

    _print_step("Loading M18 artifacts...")
    test_routes.load_ai_models()

    _print_step("Scoring with M18 multi-episode wrapper...")
    score = test_routes.run_m14_multilabel_scoring(combined_df, test_id, str(out_dir))

    summary = build_summary(test_id, combined_df, score)
    with (out_dir / "4_replay_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    print("=" * 80)
    print("M18 MULTI-EPISODE REPLAY SUMMARY")
    print("=" * 80)
    print(f"Test ID: {test_id}")
    print(f"Model: {getattr(test_routes, 'MODEL_ID', 'M14')}")
    print(f"Vector rows: {len(combined_df)}")
    print(f"Vector cols: {len(combined_df.columns)}")
    print(f"Windows analyzed: {score['windows_analyzed']}")
    print(f"Episode count: {score.get('episode_count', 1)}")
    for ep in score.get("aggregation", {}).get("episodes", []):
        print(
            f"  EP{ep.get('episode_index')}: "
            f"{ep.get('start_time_sec')}s-{ep.get('end_time_sec')}s "
            f"signs={ep.get('sign_codes')} result={ep.get('result')} "
            f"mistakes={ep.get('mistakes_count')} positives={ep.get('positive_actions_count')}"
        )
    print(f"Result: {score['result']}")
    print(f"Mistakes: {score['mistakes_count']}")
    print(f"Mistake codes: {score['violations_detected']}")
    print(f"Ignored warning codes: {score['ignored_warning_codes']}")
    print(f"Positive actions: {len(score['positive_actions'])}")
    for action in score["positive_actions"]:
        print(f"  + {action}")
    print(f"Wrote: {out_dir / '3_model_thought.json'}")
    print(f"Wrote: {out_dir / '4_replay_summary.json'}")
    print("=" * 80)

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-id", type=str, default=None)
    parser.add_argument("--feature-vector", type=str, default=None)
    parser.add_argument("--sensors", type=str, default=None)
    parser.add_argument("--video-events", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    run_replay(args)


if __name__ == "__main__":
    main()
