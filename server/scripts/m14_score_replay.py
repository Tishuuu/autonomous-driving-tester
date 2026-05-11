"""
M14 offline scorer for Auto Tester replay exports.

Usage from server root:

    python scripts\m14_score_replay.py ^
      --export-dir analysis_exports\TEST_1778085692205_M13_REPLAY_SMOOTH

or:

    python scripts\m14_score_replay.py ^
      --vector analysis_exports\TEST_1778085692205_M13_REPLAY_SMOOTH\2_feature_vector.csv ^
      --out-dir analysis_exports\TEST_1778085692205_M13_REPLAY_SMOOTH ^
      --test-id TEST_1778085692205_M14_SCORE

Expected artifacts in app/ai_models:
    m14_model.keras
    m14_scaler.pkl
    m14_class_map.json
    m14_thresholds.json
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------------------------------
# Custom layer required to load M14.
# -----------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable(package="AutonomousDrivingTester")
class TemporalAttention(layers.Layer):
    def __init__(self, units=64, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        self.W = self.add_weight(
            name="W",
            shape=(d_in, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(self.units, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, H):
        e = tf.tanh(tf.tensordot(H, self.W, axes=1) + self.b)
        e = tf.tensordot(e, self.v, axes=1)
        e = tf.tensordot(e, self.v, axes=1)
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e, axis=-1)
        context = tf.reduce_sum(H * tf.expand_dims(alpha, -1), axis=1)
        if self.return_attention:
            return context, alpha
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units, "return_attention": self.return_attention})
        return cfg


# -----------------------------------------------------------------------------
# M14 constants
# -----------------------------------------------------------------------------
MODEL_ID = "M14"
SENTINEL = 99.0

OUTPUT_NAMES_DEFAULT = [
    "Tailgating",
    "RunningStop",
    "FailureToYield",
    "NoEntryViolation",
    "CorrectStop",
    "CorrectYield",
]

FAIL_OUTPUTS = {"RunningStop", "FailureToYield", "NoEntryViolation"}
IGNORED_WARNING_OUTPUTS = {"Tailgating"}
POSITIVE_OUTPUTS = {"CorrectStop", "CorrectYield"}

DEFAULT_THRESHOLDS = {
    "Tailgating": 0.17,
    "RunningStop": 0.40,
    "FailureToYield": 0.15,
    "NoEntryViolation": 0.39,
    "CorrectStop": 0.15,
    "CorrectYield": 0.22,
}

SIGN_MAPPING = {
    "none": 0,
    "stop_sign": 1,
    "yield_sign": 2,
    "no_entry": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}


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


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_m14_artifacts(models_dir: Path):
    model_path = models_dir / "m14_model.keras"
    scaler_path = models_dir / "m14_scaler.pkl"
    class_map_path = models_dir / "m14_class_map.json"
    thresholds_path = models_dir / "m14_thresholds.json"

    if not model_path.exists():
        raise FileNotFoundError(f"M14 model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"M14 scaler not found: {scaler_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"M14 class map not found: {class_map_path}")

    class_map = _load_json(class_map_path)

    if thresholds_path.exists():
        th_payload = _load_json(thresholds_path)
        thresholds = th_payload.get("thresholds", {})
    else:
        thresholds = class_map.get("thresholds", {})

    if not thresholds:
        thresholds = DEFAULT_THRESHOLDS.copy()

    feature_names = class_map.get("feature_names")
    if not feature_names:
        raise ValueError("m14_class_map.json missing feature_names")

    output_names = class_map.get("output_names") or OUTPUT_NAMES_DEFAULT
    window_size = int(class_map.get("n_timesteps", 60))

    scaler = joblib.load(scaler_path)

    model = keras.models.load_model(
        model_path,
        custom_objects={
            "TemporalAttention": TemporalAttention,
            "AutonomousDrivingTester>TemporalAttention": TemporalAttention,
        },
        compile=False,
    )

    threshold_array = np.asarray(
        [float(thresholds.get(name, DEFAULT_THRESHOLDS.get(name, 0.5))) for name in output_names],
        dtype=np.float32,
    )

    return model, scaler, class_map, feature_names, output_names, threshold_array, window_size


def prepare_m14_feature_frame(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    x = df.copy()

    for col in ["sign_type", "memory_sign_type"]:
        if col in x.columns and x[col].dtype == object:
            x[col] = x[col].astype(str).str.strip().str.lower().map(SIGN_MAPPING).fillna(0).astype(int)

    missing = [c for c in feature_names if c not in x.columns]
    if missing:
        raise ValueError(f"Vector missing M14 feature columns: {missing}")

    x = x[feature_names].copy()

    for col in feature_names:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.ffill(inplace=True)
    x.bfill(inplace=True)

    distance_like = [c for c in feature_names if "distance" in c or "ttc" in c]
    for c in distance_like:
        x[c] = x[c].fillna(SENTINEL).clip(0.0, SENTINEL)

    if "speed_kmh" in x:
        x["speed_kmh"] = x["speed_kmh"].fillna(0.0).clip(0.0, 130.0)
    if "jerk" in x:
        x["jerk"] = x["jerk"].fillna(0.0).clip(-30.0, 30.0)

    for c in ["sign_type", "memory_sign_type"]:
        if c in x:
            x[c] = x[c].fillna(0).round().clip(0, 3)

    for c in ["car_relative_x"]:
        if c in x:
            x[c] = x[c].fillna(0.0).clip(-1.0, 1.0)
    for c in ["car_relative_y"]:
        if c in x:
            x[c] = x[c].fillna(0.0).clip(0.0, 1.0)
    for c in ["car_motion_x", "car_motion_y"]:
        if c in x:
            x[c] = x[c].fillna(0.0).clip(-2.0, 2.0)
    if "car_static_score" in x:
        x["car_static_score"] = x["car_static_score"].fillna(0.0).clip(0.0, 1.0)

    if "gps_accuracy" in x:
        x["gps_accuracy"] = x["gps_accuracy"].fillna(50.0).clip(0.0, 100.0)
    if "gps_age_ms" in x:
        x["gps_age_ms"] = x["gps_age_ms"].fillna(5000.0).clip(0.0, 10000.0)
    if "gps_update_count_delta" in x:
        x["gps_update_count_delta"] = x["gps_update_count_delta"].fillna(0.0).clip(0.0, 5.0)
    for c in ["gps_dx_m", "gps_dy_m"]:
        if c in x:
            x[c] = x[c].fillna(0.0).clip(-20.0, 20.0)
    if "gps_step_distance_m" in x:
        x["gps_step_distance_m"] = x["gps_step_distance_m"].fillna(0.0).clip(0.0, 20.0)
    if "gps_speed_kmh" in x:
        x["gps_speed_kmh"] = x["gps_speed_kmh"].fillna(0.0).clip(0.0, 130.0)

    if "heading_sin" in x:
        x["heading_sin"] = x["heading_sin"].fillna(0.0).clip(-1.0, 1.0)
    if "heading_cos" in x:
        x["heading_cos"] = x["heading_cos"].fillna(1.0).clip(-1.0, 1.0)
    if "heading_delta_deg" in x:
        x["heading_delta_deg"] = x["heading_delta_deg"].fillna(0.0).clip(-45.0, 45.0)
    if "turn_rate_deg_s" in x:
        x["turn_rate_deg_s"] = x["turn_rate_deg_s"].fillna(0.0).clip(-180.0, 180.0)

    for c in ["accel_x", "accel_y"]:
        if c in x:
            x[c] = x[c].fillna(0.0).clip(-12.0, 12.0)
    if "accel_z" in x:
        x["accel_z"] = x["accel_z"].fillna(9.81).clip(6.0, 14.0)

    return x.astype(np.float32)


def build_windows(x_scaled: np.ndarray, window_size: int) -> np.ndarray:
    if len(x_scaled) < window_size:
        return np.empty((0, window_size, x_scaled.shape[1]), dtype=np.float32)
    return np.stack(
        [x_scaled[i:i + window_size] for i in range(len(x_scaled) - window_size + 1)]
    ).astype(np.float32)


def merge_event(
    events: List[Dict[str, Any]],
    last_by_name: Dict[str, Dict[str, Any]],
    output_name: str,
    event_type: str,
    timestamp_sec: float,
    confidence: float,
    merge_gap_s: float,
    extra: Optional[Dict[str, Any]] = None,
):
    prev = last_by_name.get(output_name)
    if prev is None or (timestamp_sec - prev["end_t"]) > merge_gap_s:
        item = {
            "timestamp_sec": round(float(timestamp_sec), 2),
            "type": output_name,
            "output": output_name,
            "event_type": event_type,
            "confidence": round(float(confidence), 4),
        }
        if extra:
            item.update(extra)
        events.append(item)
        last_by_name[output_name] = {
            "end_t": float(timestamp_sec),
            "event_idx": len(events) - 1,
            "confidence_max": float(confidence),
        }
    else:
        prev["end_t"] = float(timestamp_sec)
        if confidence > prev["confidence_max"]:
            prev["confidence_max"] = float(confidence)
            events[prev["event_idx"]].update({
                "timestamp_sec": round(float(timestamp_sec), 2),
                "confidence": round(float(confidence), 4),
            })
            if extra:
                events[prev["event_idx"]].update(extra)


def vector_summary(df: pd.DataFrame) -> Dict[str, Any]:
    sign_type = pd.to_numeric(df.get("sign_type", pd.Series([], dtype=float)), errors="coerce").fillna(0).astype(int)
    mem_type = pd.to_numeric(df.get("memory_sign_type", pd.Series([], dtype=float)), errors="coerce").fillna(0).astype(int)

    def count_type(series, value):
        return int((series == value).sum())

    active_car_rows = int((pd.to_numeric(df.get("car_distance", 99), errors="coerce").fillna(99) < 99).sum())
    min_car_distance = None
    if "car_distance" in df.columns:
        car_dist = pd.to_numeric(df["car_distance"], errors="coerce")
        valid = car_dist[car_dist < 99]
        if len(valid):
            min_car_distance = float(valid.min())

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "sign_counts": {str(k): int(v) for k, v in sign_type.value_counts().sort_index().items()},
        "memory_sign_counts": {str(k): int(v) for k, v in mem_type.value_counts().sort_index().items()},
        "stop_rows": count_type(sign_type, 1),
        "yield_rows": count_type(sign_type, 2),
        "no_entry_rows": count_type(sign_type, 3),
        "memory_stop_rows": count_type(mem_type, 1),
        "memory_yield_rows": count_type(mem_type, 2),
        "memory_no_entry_rows": count_type(mem_type, 3),
        "active_sign_rows": int((sign_type > 0).sum()),
        "active_memory_sign_rows": int((mem_type > 0).sum()),
        "active_car_rows": active_car_rows,
        "min_car_distance": min_car_distance,
    }
    return summary


def score_m14_vector(
    vector_path: Path,
    out_dir: Path,
    models_dir: Path,
    test_id: Optional[str] = None,
    merge_gap_s: float = 2.0,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    model, scaler, class_map, feature_names, output_names, thresholds, window_size = load_m14_artifacts(models_dir)

    combined_df = pd.read_csv(vector_path)
    if test_id is None:
        test_id = out_dir.name

    X_df = prepare_m14_feature_frame(combined_df, feature_names)
    X_raw = X_df.to_numpy(dtype=np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    X_windows = build_windows(X_scaled, window_size)

    decision_log: List[Dict[str, Any]] = []
    violation_events: List[Dict[str, Any]] = []
    ignored_warning_events: List[Dict[str, Any]] = []
    positive_actions: List[Dict[str, Any]] = []

    last_violation: Dict[str, Dict[str, Any]] = {}
    last_warning: Dict[str, Dict[str, Any]] = {}
    last_positive: Dict[str, Dict[str, Any]] = {}

    if len(X_windows) == 0:
        probs = np.empty((0, len(output_names)), dtype=np.float32)
        preds = np.empty((0, len(output_names)), dtype=np.int32)
    else:
        probs = model.predict(X_windows, batch_size=128, verbose=0).astype(np.float32)
        preds = (probs >= thresholds[None, :]).astype(np.int32)

    for i in range(len(preds)):
        end_idx = min(i + window_size - 1, len(combined_df) - 1)
        ts = float(combined_df.iloc[end_idx].get("time_seconds", i * 0.1))

        active = [output_names[j] for j, v in enumerate(preds[i]) if int(v) == 1]
        decision_log.append({
            "window_index": int(i),
            "timestamp_sec": round(ts, 2),
            "active_outputs": active if active else ["Normal/all_zero"],
            "probabilities": {output_names[j]: round(float(probs[i, j]), 4) for j in range(len(output_names))},
            "thresholds": {output_names[j]: round(float(thresholds[j]), 4) for j in range(len(output_names))},
            "predictions": {output_names[j]: int(preds[i, j]) for j in range(len(output_names))},
        })

        for j, name in enumerate(output_names):
            if int(preds[i, j]) != 1:
                continue

            conf = float(probs[i, j])
            extra = {
                "output_index": int(j),
                "window_index": int(i),
            }

            if name in IGNORED_WARNING_OUTPUTS:
                merge_event(
                    ignored_warning_events,
                    last_warning,
                    name,
                    "IGNORED_WARNING",
                    ts,
                    conf,
                    merge_gap_s,
                    extra,
                )
            elif name in FAIL_OUTPUTS:
                merge_event(
                    violation_events,
                    last_violation,
                    name,
                    "VIOLATION",
                    ts,
                    conf,
                    merge_gap_s,
                    extra,
                )
            elif name in POSITIVE_OUTPUTS:
                positive_type = {
                    "CorrectStop": "CorrectStop",
                    "CorrectYield": "CorrectYield",
                }.get(name, name)
                merge_event(
                    positive_actions,
                    last_positive,
                    positive_type,
                    "POSITIVE_ACTION",
                    ts,
                    conf,
                    merge_gap_s,
                    extra,
                )

    fail_outputs = sorted({e["output"] for e in violation_events})
    ignored_warning_outputs = sorted({e["output"] for e in ignored_warning_events})

    passed = len(fail_outputs) == 0
    result = "PASS" if passed else "FAIL"
    grade = 100 if passed else 0

    thought = {
        "test_id": test_id,
        "exported_at": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "model_type": "multi_label_sigmoid",
        "feature_names": feature_names,
        "window_size": int(window_size),
        "output_names": output_names,
        "thresholds": {output_names[i]: float(thresholds[i]) for i in range(len(output_names))},
        "result": result,
        "passed": bool(passed),
        "grade": int(grade),
        "violation_events_count": int(len(violation_events)),
        "violations_outputs": fail_outputs,
        "ignored_warning_events_count": int(len(ignored_warning_events)),
        "ignored_warning_outputs": ignored_warning_outputs,
        "ignored_warning_events": ignored_warning_events,
        "positive_actions_count": int(len(positive_actions)),
        "positive_actions": positive_actions,
        "violation_events": violation_events,
        "decision_log": decision_log,
        "episode_policy": (
            "M14 multi-label sigmoid: outputs are independent. "
            "Normal Driving is all outputs below threshold. "
            "STOP_OK can emit CorrectStop + CorrectYield. "
            "STOP_BAD_YIELD can emit CorrectStop + FailureToYield."
        ),
        "windows_analyzed": int(len(X_windows)),
    }

    summary = {
        "test_id": test_id,
        "created_at": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "model_type": "multi_label_sigmoid",
        "feature_vector": vector_summary(combined_df),
        "model": {
            "result": result,
            "passed": bool(passed),
            "grade": int(grade),
            "violation_events_count": int(len(violation_events)),
            "violations_outputs": fail_outputs,
            "ignored_warning_events_count": int(len(ignored_warning_events)),
            "ignored_warning_outputs": ignored_warning_outputs,
            "positive_actions_count": int(len(positive_actions)),
            "positive_actions": positive_actions,
            "windows_analyzed": int(len(X_windows)),
            "thresholds": {output_names[i]: float(thresholds[i]) for i in range(len(output_names))},
        },
    }

    thought_path = out_dir / "3_model_thought_m14.json"
    summary_path = out_dir / "4_replay_summary_m14.json"

    with thought_path.open("w", encoding="utf-8") as f:
        json.dump(thought, f, ensure_ascii=False, indent=2, default=_json_default)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)

    print("=" * 80)
    print("M14 REPLAY SCORE SUMMARY")
    print("=" * 80)
    print(f"Test ID: {test_id}")
    print(f"Vector: {vector_path}")
    print(f"Rows: {len(combined_df)}")
    print(f"Windows analyzed: {len(X_windows)}")
    print(f"Result: {result}")
    print(f"Violations: {fail_outputs}")
    print(f"Positive actions: {[p['type'] for p in positive_actions]}")
    print(f"Ignored warnings: {ignored_warning_outputs}")
    print(f"Wrote: {thought_path}")
    print(f"Wrote: {summary_path}")
    print("=" * 80)

    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-dir", type=str, default=None, help="Replay export folder containing 2_feature_vector.csv")
    parser.add_argument("--vector", type=str, default=None, help="Path to 2_feature_vector.csv")
    parser.add_argument("--out-dir", type=str, default=None, help="Output folder. Defaults to export-dir or vector parent.")
    parser.add_argument("--models-dir", type=str, default=None, help="Defaults to app/ai_models under server root")
    parser.add_argument("--test-id", type=str, default=None)
    parser.add_argument("--merge-gap-s", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()

    cwd = Path.cwd()

    if args.export_dir:
        export_dir = Path(args.export_dir).expanduser().resolve()
        vector_path = export_dir / "2_feature_vector.csv"
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else export_dir
    elif args.vector:
        vector_path = Path(args.vector).expanduser().resolve()
        out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else vector_path.parent
    else:
        raise ValueError("Provide either --export-dir or --vector")

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector CSV not found: {vector_path}")

    if args.models_dir:
        models_dir = Path(args.models_dir).expanduser().resolve()
    else:
        models_dir = cwd / "app" / "ai_models"

    score_m14_vector(
        vector_path=vector_path,
        out_dir=out_dir,
        models_dir=models_dir,
        test_id=args.test_id,
        merge_gap_s=float(args.merge_gap_s),
    )


if __name__ == "__main__":
    main()
