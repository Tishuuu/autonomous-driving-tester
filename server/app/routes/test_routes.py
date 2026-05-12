import os
import uuid
import asyncio
import json
import hashlib
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib

from app.core.database import db
from app.utils.logger import log
from app.services.sensor_sync import process_sensor_json
from app.services.vision_service import analyze_video_for_server
from app.services.vector_builder import build_feature_vector
from app.services.student_success_predictor import get_student_predictor
from app.models.student_model import TestSaveRequest
from app.routes.auth_routes import get_current_tester


router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPORT_BASE_DIR = os.path.join(BASE_DIR, "analysis_exports")

PROGRESS_STORE: dict = {}


def update_progress(test_id: str, percent: int, message: str):
    PROGRESS_STORE[test_id] = {
        "percent": max(0, min(100, int(percent))),
        "message": message,
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@router.get("/progress/{test_id}")
async def get_progress(test_id: str):
    return PROGRESS_STORE.get(test_id, {"percent": 0, "message": "Waiting..."})


# ==========================================================================
# M16 model config
# ==========================================================================
MODEL_ID = "M16"
MODEL_WINDOW_SIZE = 60
MODEL_WINDOW_STRIDE = 15
MODEL_TYPE = "hierarchical_multi_head_lstm"
SENTINEL = 99.0

MODEL_FEATURES = [
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

    "gps_accuracy",
    "gps_age_ms",
    "gps_update_count_delta",

    "gps_dx_m",
    "gps_dy_m",
    "gps_step_distance_m",
    "gps_speed_kmh",

    "heading_sin",
    "heading_cos",
    "heading_delta_deg",
    "turn_rate_deg_s",

    "accel_x",
    "accel_y",
    "accel_z",
]

FAMILY_NONE = 0
FAMILY_STOP = 1
FAMILY_YIELD = 2
FAMILY_NOENTRY = 3
FAMILY_NAMES = ["None", "Stop", "Yield", "NoEntry"]

STOP_OK = 0
STOP_BAD_YIELD = 1
RUNNING_STOP = 2
STOP_NAMES = ["StopOk", "StopBadYield", "RunningStop"]

YIELD_OK = 0
YIELD_BAD = 1
YIELD_NAMES = ["YieldOk", "YieldBad"]

NOENTRY_NOEVENT = 0
NOENTRY_VIOLATION = 1
NOENTRY_NAMES = ["NoEvent", "Violation"]

TAILGATING_THRESHOLD = 0.5
AGG_CLUSTER_MIN_WINDOWS = 2
AGG_HIGH_CONF_THRESHOLD = 0.90

# Legacy IDs kept stable for Flutter / DB compatibility.
LEGACY_CLASS_NAMES = {
    0: "Normal Driving",
    1: "Tailgating",
    2: "Running Stop",
    3: "Failure to Yield",
    4: "No Entry Violation",
    5: "Correct Stop",
    6: "Correct Yield",
}

LEGACY_EVENT_CODES = {
    "Tailgating": 1,
    "RunningStop": 2,
    "FailureToYield": 3,
    "NoEntryViolation": 4,
    "CorrectStop": 5,
    "CorrectYield": 6,
}

LEGACY_EVENT_LABELS = {
    "Tailgating": "Tailgating",
    "RunningStop": "Running Stop",
    "FailureToYield": "Failure to Yield",
    "NoEntryViolation": "No Entry Violation",
    "CorrectStop": "Correct Stop",
    "CorrectYield": "Correct Yield",
}

EVENT_BUCKET = {
    "Tailgating": "warning",
    "RunningStop": "violation",
    "FailureToYield": "violation",
    "NoEntryViolation": "violation",
    "CorrectStop": "positive",
    "CorrectYield": "positive",
}

# Kept for older code paths / frontend expectations.
MODEL_CLASS_NAMES = LEGACY_CLASS_NAMES
VIOLATION_CLASSES = {1, 2, 3, 4}
IGNORED_WARNING_CLASSES = {1}
FAIL_CLASSES = {2, 3, 4}
POSITIVE_ACTION_CLASSES = {5, 6}
POSITIVE_ACTION_TYPES = {
    5: "CorrectStop",
    6: "CorrectYield",
}

SIGN_MAPPING = {
    "": 0,
    "nan": 0,
    "none": 0,
    "stop": 1,
    "stop_sign": 1,
    "yield": 2,
    "yield_sign": 2,
    "no_entry": 3,
    "noentry": 3,
    "no entry": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}


# ==========================================================================
# Keras custom layers used by m16_model.keras
# ==========================================================================
@tf.keras.utils.register_keras_serializable(package="M16", name="AttentionPool")
class AttentionPool(layers.Layer):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = int(dim)
        self.score = layers.Dense(self.dim, activation="tanh")
        self.v = layers.Dense(1, activation=None)

    def build(self, input_shape):
        self.score.build(input_shape)
        score_out_shape = list(input_shape)
        score_out_shape[-1] = self.dim
        self.v.build(tuple(score_out_shape))
        super().build(input_shape)

    def call(self, x, mask=None):
        scores = self.v(self.score(x))
        alpha = tf.nn.softmax(scores, axis=1)
        context = tf.reduce_sum(x * alpha, axis=1)
        return context, alpha

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"dim": self.dim})
        return cfg


@tf.keras.utils.register_keras_serializable(package="M16", name="StreamSlice")
class StreamSlice(layers.Layer):
    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = [int(i) for i in indices]

    def call(self, x):
        return tf.gather(x, self.indices, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"indices": list(self.indices)})
        return cfg


@tf.keras.utils.register_keras_serializable(package="M16", name="SoftmaxStopGrad")
class SoftmaxStopGrad(layers.Layer):
    def call(self, x):
        return tf.stop_gradient(tf.nn.softmax(x))

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "AttentionPool": AttentionPool,
    "StreamSlice": StreamSlice,
    "SoftmaxStopGrad": SoftmaxStopGrad,
    "M16>AttentionPool": AttentionPool,
    "M16>StreamSlice": StreamSlice,
    "M16>SoftmaxStopGrad": SoftmaxStopGrad,
}


# ==========================================================================
# Model loading
# ==========================================================================
global_scaler = None
global_lstm_model = None
global_model_config = None


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dict_values_by_numeric_key(payload: dict, fallback: list[str]) -> list[str]:
    if not isinstance(payload, dict):
        return fallback
    out = []
    for i in range(len(fallback)):
        out.append(payload.get(str(i), fallback[i]))
    return out


def _load_model_class_map(class_map_path: str) -> dict:
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"{MODEL_ID} class map not found: {class_map_path}")

    payload = _load_json(class_map_path)

    feature_names = payload.get("feature_names", [])
    if not feature_names:
        raise ValueError(f"{MODEL_ID} class map is missing feature_names")
    if feature_names != MODEL_FEATURES:
        raise ValueError(
            f"{MODEL_ID} feature schema mismatch.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Got:      {feature_names}"
        )

    model_type = payload.get("model_type", MODEL_TYPE)
    if model_type != MODEL_TYPE:
        raise ValueError(f"Expected {MODEL_TYPE} class map, got: {model_type}")

    family = _dict_values_by_numeric_key(payload.get("family", {}), FAMILY_NAMES)
    stop = _dict_values_by_numeric_key(payload.get("stop_behavior", {}), STOP_NAMES)
    yld = _dict_values_by_numeric_key(payload.get("yield_behavior", {}), YIELD_NAMES)
    noentry = _dict_values_by_numeric_key(payload.get("noentry_behavior", {}), NOENTRY_NAMES)

    if family != FAMILY_NAMES:
        raise ValueError(f"M16 family schema mismatch. Expected {FAMILY_NAMES}, got {family}")
    if stop != STOP_NAMES:
        raise ValueError(f"M16 stop schema mismatch. Expected {STOP_NAMES}, got {stop}")
    if yld != YIELD_NAMES:
        raise ValueError(f"M16 yield schema mismatch. Expected {YIELD_NAMES}, got {yld}")
    if noentry != NOENTRY_NAMES:
        raise ValueError(f"M16 noentry schema mismatch. Expected {NOENTRY_NAMES}, got {noentry}")

    agg = payload.get("aggregation", {}) if isinstance(payload.get("aggregation", {}), dict) else {}
    global MODEL_WINDOW_STRIDE, AGG_CLUSTER_MIN_WINDOWS, AGG_HIGH_CONF_THRESHOLD, TAILGATING_THRESHOLD
    MODEL_WINDOW_STRIDE = int(agg.get("window_stride", MODEL_WINDOW_STRIDE))
    AGG_CLUSTER_MIN_WINDOWS = int(agg.get("cluster_min_windows", AGG_CLUSTER_MIN_WINDOWS))
    AGG_HIGH_CONF_THRESHOLD = float(agg.get("high_conf_threshold", AGG_HIGH_CONF_THRESHOLD))
    TAILGATING_THRESHOLD = float(payload.get("tailgating_threshold", TAILGATING_THRESHOLD))

    return payload


def load_ai_models():
    """Load M16 scaler + hierarchical multi-head LSTM once and fail loudly if an artifact is missing."""
    global global_scaler, global_lstm_model, global_model_config

    if global_scaler is not None and global_lstm_model is not None:
        return

    log.info("Loading M16 scaler + hierarchical multi-head LSTM model...")

    try:
        models_dir = os.path.join(BASE_DIR, "app", "ai_models")

        scaler_path = os.path.join(models_dir, "m16_scaler.pkl")
        model_path = os.path.join(models_dir, "m16_model.keras")
        class_map_path = os.path.join(models_dir, "m16_class_map.json")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"M16 scaler not found: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"M16 model not found: {model_path}")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"M16 class map not found: {class_map_path}")

        global_model_config = _load_model_class_map(class_map_path)
        global_scaler = joblib.load(scaler_path)
        global_lstm_model = keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )

        log.info(
            "M16 Model + Scaler ready. Outputs: family, stop, yield_, noentry, tailgating."
        )

    except Exception as e:
        global_scaler = None
        global_lstm_model = None
        global_model_config = None
        log.error(f"Failed to load M16 model artifacts: {e}")
        raise RuntimeError(f"AI model loading failed: {e}")


# ==========================================================================
# M16 feature prep
# ==========================================================================
def _default_for_feature(col: str) -> float:
    if col == "time_seconds":
        return 0.0
    if col in {"sign_type", "memory_sign_type"}:
        return 0.0
    if col in {
        "car_distance",
        "car_ttc",
        "sign_distance",
        "sign_ttc",
        "memory_sign_distance",
        "time_since_sign_seen_sec",
        "meters_since_sign_seen",
        "last_seen_sign_distance",
        "memory_sign_ttc",
    }:
        return SENTINEL
    if col == "gps_accuracy":
        return 50.0
    if col == "gps_age_ms":
        return 5000.0
    if col == "heading_cos":
        return 1.0
    if col == "accel_z":
        return 9.81
    return 0.0


def prepare_m16_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the 60x33 M16 model frame.

    Missing columns are filled with safe defaults so old replay vectors can still
    be scored. This does not add conclusions or traffic-rule gates.
    """
    if combined_df is None or len(combined_df) == 0:
        raise ValueError("combined_df is empty")

    x = pd.DataFrame(index=combined_df.index)

    for col in MODEL_FEATURES:
        if col in combined_df.columns:
            x[col] = combined_df[col]
        elif col == "time_seconds":
            x[col] = np.arange(len(combined_df), dtype=np.float32) * 0.1
        else:
            x[col] = _default_for_feature(col)

    for col in ["sign_type", "memory_sign_type"]:
        if x[col].dtype == object:
            x[col] = (
                x[col]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(SIGN_MAPPING)
                .fillna(0)
                .astype(int)
            )

    for col in MODEL_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    x.ffill(inplace=True)
    x.bfill(inplace=True)

    for col in MODEL_FEATURES:
        x[col] = x[col].fillna(_default_for_feature(col))

    distance_like = [c for c in MODEL_FEATURES if "distance" in c or "ttc" in c]
    for col in distance_like:
        x[col] = x[col].clip(0.0, SENTINEL)

    x["time_seconds"] = x["time_seconds"].clip(0.0, 100000.0)
    x["speed_kmh"] = x["speed_kmh"].clip(0.0, 130.0)
    x["jerk"] = x["jerk"].clip(-30.0, 30.0)

    x["sign_type"] = x["sign_type"].round().clip(0, 3)
    x["memory_sign_type"] = x["memory_sign_type"].round().clip(0, 3)

    x["car_relative_x"] = x["car_relative_x"].clip(-1.0, 1.0)
    x["car_relative_y"] = x["car_relative_y"].clip(0.0, 1.0)
    x["car_motion_x"] = x["car_motion_x"].clip(-2.0, 2.0)
    x["car_motion_y"] = x["car_motion_y"].clip(-2.0, 2.0)
    x["car_static_score"] = x["car_static_score"].clip(0.0, 1.0)

    x["gps_accuracy"] = x["gps_accuracy"].clip(0.0, 999.0)
    x["gps_age_ms"] = x["gps_age_ms"].clip(0.0, 999999.0)
    x["gps_update_count_delta"] = x["gps_update_count_delta"].clip(0.0, 10.0)
    x["gps_dx_m"] = x["gps_dx_m"].clip(-20.0, 20.0)
    x["gps_dy_m"] = x["gps_dy_m"].clip(-20.0, 20.0)
    x["gps_step_distance_m"] = x["gps_step_distance_m"].clip(0.0, 20.0)
    x["gps_speed_kmh"] = x["gps_speed_kmh"].clip(0.0, 130.0)

    x["heading_sin"] = x["heading_sin"].clip(-1.0, 1.0)
    x["heading_cos"] = x["heading_cos"].clip(-1.0, 1.0)
    x["heading_delta_deg"] = x["heading_delta_deg"].clip(-45.0, 45.0)
    x["turn_rate_deg_s"] = x["turn_rate_deg_s"].clip(-180.0, 180.0)

    x["accel_x"] = x["accel_x"].clip(-12.0, 12.0)
    x["accel_y"] = x["accel_y"].clip(-12.0, 12.0)
    x["accel_z"] = x["accel_z"].clip(6.0, 14.0)

    return x[MODEL_FEATURES].astype(np.float32)


# Backward-compatible aliases for old imports/scripts.
def prepare_m15_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_m16_feature_frame(combined_df)


def prepare_m14_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_m16_feature_frame(combined_df)


# ==========================================================================
# M16 scoring helpers
# ==========================================================================
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


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


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_model_outputs(raw: Any) -> Dict[str, np.ndarray]:
    if isinstance(raw, dict):
        out = dict(raw)
    elif isinstance(raw, (list, tuple)):
        names = list(getattr(global_lstm_model, "output_names", []))
        out = {}
        for name, arr in zip(names, raw):
            if name in {"family", "family_logits"}:
                out["family"] = arr
            elif name in {"stop", "stop_logits"}:
                out["stop"] = arr
            elif name in {"yield_", "yield_logits"}:
                out["yield_"] = arr
            elif name in {"noentry", "noentry_logits"}:
                out["noentry"] = arr
            elif name in {"tailgating", "tg_logits"}:
                out["tailgating"] = arr
        # Fallback to the known construction order.
        if len(out) < 5 and len(raw) == 5:
            out = {
                "family": raw[0],
                "stop": raw[1],
                "yield_": raw[2],
                "noentry": raw[3],
                "tailgating": raw[4],
            }
    else:
        raise ValueError(f"Unsupported M16 model output type: {type(raw)}")

    required = ["family", "stop", "yield_", "noentry", "tailgating"]
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"M16 model output missing keys: {missing}; got {list(out.keys())}")

    return {k: np.asarray(out[k]) for k in required}


def translate_to_legacy_events(family_idx: int, behavior_idx: int | None, tailgating_bool: bool) -> list[str]:
    """Pure model-output -> legacy events. No sign_type checks."""
    events: list[str] = []

    if family_idx == FAMILY_STOP:
        if behavior_idx == STOP_OK:
            events += ["CorrectStop", "CorrectYield"]
        elif behavior_idx == STOP_BAD_YIELD:
            events += ["CorrectStop", "FailureToYield"]
        elif behavior_idx == RUNNING_STOP:
            events += ["RunningStop"]

    elif family_idx == FAMILY_YIELD:
        if behavior_idx == YIELD_OK:
            events += ["CorrectYield"]
        elif behavior_idx == YIELD_BAD:
            events += ["FailureToYield"]

    elif family_idx == FAMILY_NOENTRY:
        if behavior_idx == NOENTRY_VIOLATION:
            events += ["NoEntryViolation"]
        # NoEntry.NoEvent emits no positive action and no violation.

    if tailgating_bool:
        events.append("Tailgating")

    return events


def _row_context(combined_df: pd.DataFrame, sample_idx: int) -> dict:
    row = combined_df.iloc[min(max(0, sample_idx), len(combined_df) - 1)]
    sign_code = int(round(_safe_float(row.get("sign_type"), 0)))
    memory_sign_code = int(round(_safe_float(row.get("memory_sign_type"), 0)))
    active_sign_code = sign_code if sign_code != 0 else memory_sign_code

    sign_distance = _safe_float(row.get("sign_distance"), SENTINEL)
    memory_sign_distance = _safe_float(row.get("memory_sign_distance"), SENTINEL)
    active_sign_distance = sign_distance if sign_code != 0 else memory_sign_distance

    return {
        "timestamp_sec": round(_safe_float(row.get("time_seconds"), 0.0), 2),
        "sign_code": int(active_sign_code),
        "sign_distance_m": round(float(active_sign_distance), 2),
        "speed_kmh": round(_safe_float(row.get("speed_kmh"), 0.0), 2),
        "car_distance_m": round(_safe_float(row.get("car_distance"), SENTINEL), 2),
        "heading_delta_deg": round(_safe_float(row.get("heading_delta_deg"), 0.0), 3),
        "turn_rate_deg_s": round(_safe_float(row.get("turn_rate_deg_s"), 0.0), 3),
    }


def _event_payload(event_name: str, window_pred: dict, ctx: dict) -> dict:
    legacy_code = int(LEGACY_EVENT_CODES[event_name])
    legacy_label = LEGACY_CLASS_NAMES.get(legacy_code, LEGACY_EVENT_LABELS.get(event_name, event_name))
    return {
        "timestamp_sec": ctx["timestamp_sec"],
        "type": event_name,
        "event_key": f"{EVENT_BUCKET[event_name]}:{legacy_code}:{event_name}",
        "event_type": (
            "POSITIVE_ACTION" if EVENT_BUCKET[event_name] == "positive"
            else "IGNORED_WARNING" if EVENT_BUCKET[event_name] == "warning"
            else "VIOLATION"
        ),
        "confidence": round(float(window_pred.get("event_confidence", window_pred.get("behavior_prob", 0.0))), 4),
        "family_id": int(window_pred["family"]),
        "family_name": FAMILY_NAMES[int(window_pred["family"])],
        "behavior_id": None if window_pred.get("behavior") is None else int(window_pred["behavior"]),
        "behavior_name": window_pred.get("behavior_name"),
        "window_index": int(window_pred["window_index"]),
        "window_start": int(window_pred["window_start"]),
        "class_id": legacy_code,
        "class_label": legacy_label,
        **ctx,
    }


def _window_events(window_preds: list[dict]) -> list[set[str]]:
    out = []
    for wp in window_preds:
        out.append(set(translate_to_legacy_events(
            int(wp["family"]),
            wp["behavior"] if wp["behavior"] is not None else -1,
            float(wp["tg_prob"]) >= TAILGATING_THRESHOLD,
        )))
    return out


def aggregate_window_predictions(window_preds: list[dict]) -> dict:
    """
    Aggregate model-selected window events into a test-level result.

    NoEntry temporal resolution:
    early NoEntry.Violation windows are suppressed when later model-selected
    NoEntry.NoEvent resolves the cluster. This is model-output aggregation only;
    it does not inspect sign_type and it is not a backend sign gate.
    """
    positive_set = {"CorrectStop", "CorrectYield"}
    mistake_set = {"RunningStop", "FailureToYield", "NoEntryViolation"}
    warning_set = {"Tailgating"}

    per_window_events = _window_events(window_preds)

    def _consecutive_signal(event_name: str) -> bool:
        for wp, events in zip(window_preds, per_window_events):
            if event_name in events:
                prob = float(wp["tg_prob"]) if event_name == "Tailgating" else float(wp["behavior_prob"])
                if prob >= AGG_HIGH_CONF_THRESHOLD:
                    return True

        run = 0
        for events in per_window_events:
            if event_name in events:
                run += 1
                if run >= AGG_CLUSTER_MIN_WINDOWS:
                    return True
            else:
                run = 0
        return False

    def _noentry_violation_signal() -> bool:
        ne = [(i, wp) for i, wp in enumerate(window_preds) if int(wp["family"]) == FAMILY_NOENTRY]
        if not ne:
            return False

        viol = [(i, float(wp["behavior_prob"])) for i, wp in ne if wp["behavior"] == NOENTRY_VIOLATION]
        noev = [(i, float(wp["behavior_prob"])) for i, wp in ne if wp["behavior"] == NOENTRY_NOEVENT]
        if not viol:
            return False

        viol_count = len(viol)
        noev_count = len(noev)
        max_viol = max((p for _, p in viol), default=0.0)
        max_noev = max((p for _, p in noev), default=0.0)
        first_viol_i = min(i for i, _ in viol)
        last_ne_behavior = ne[-1][1]["behavior"]
        late_noev = [(i, p) for i, p in noev if i > first_viol_i]

        if late_noev and last_ne_behavior == NOENTRY_NOEVENT:
            late_noev_max = max(p for _, p in late_noev)
            if noev_count >= viol_count:
                return False
            if noev_count >= 2 and late_noev_max >= 0.80:
                return False
            if late_noev_max >= 0.95 and max_noev >= max_viol * 0.90:
                return False

        if max_viol >= AGG_HIGH_CONF_THRESHOLD:
            return True

        run = 0
        for _, wp in ne:
            if wp["behavior"] == NOENTRY_VIOLATION:
                run += 1
                if run >= AGG_CLUSTER_MIN_WINDOWS:
                    return True
            else:
                run = 0
        return False

    emitted = set()
    for event_name in positive_set | warning_set | {"RunningStop", "FailureToYield"}:
        if _consecutive_signal(event_name):
            emitted.add(event_name)

    if _noentry_violation_signal():
        emitted.add("NoEntryViolation")

    family_hist = {name: 0 for name in FAMILY_NAMES}
    noentry_hist = {"NoEvent": 0, "Violation": 0}
    for wp in window_preds:
        family_hist[FAMILY_NAMES[int(wp["family"])]] += 1
        if int(wp["family"]) == FAMILY_NOENTRY:
            if wp["behavior"] == NOENTRY_NOEVENT:
                noentry_hist["NoEvent"] += 1
            elif wp["behavior"] == NOENTRY_VIOLATION:
                noentry_hist["Violation"] += 1

    mistakes = sorted(emitted & mistake_set)
    positives = sorted(emitted & positive_set)
    warnings = sorted(emitted & warning_set)

    return {
        "events": sorted(emitted),
        "positive_events": positives,
        "positive_actions": len(positives),
        "mistakes": mistakes,
        "mistake_codes": [LEGACY_EVENT_CODES[m] for m in mistakes],
        "warnings": warnings,
        "warning_codes": [LEGACY_EVENT_CODES[w] for w in warnings],
        "n_windows": len(window_preds),
        "family_hist": family_hist,
        "noentry_hist": noentry_hist,
    }


def _best_window_for_event(event_name: str, window_preds: list[dict], combined_df: pd.DataFrame) -> tuple[dict, dict] | None:
    candidates = []
    for wp in window_preds:
        events = translate_to_legacy_events(
            int(wp["family"]),
            wp["behavior"] if wp["behavior"] is not None else -1,
            float(wp["tg_prob"]) >= TAILGATING_THRESHOLD,
        )
        if event_name in events:
            if event_name == "Tailgating":
                conf = float(wp["tg_prob"])
            elif event_name == "NoEntryViolation":
                conf = float(wp.get("noentry_probs", [0, 0])[NOENTRY_VIOLATION])
            else:
                conf = float(wp["behavior_prob"])
            c = dict(wp)
            c["event_confidence"] = conf
            candidates.append(c)

    if not candidates:
        return None

    best = max(candidates, key=lambda x: float(x.get("event_confidence", 0.0)))
    ctx = _row_context(combined_df, int(best["window_end"]))
    return best, ctx


# ==========================================================================
# M16 scoring
# ==========================================================================
def run_m16_hierarchical_scoring(
    combined_df: pd.DataFrame,
    test_id: str,
    test_export_path: str,
) -> dict:
    """
    M16 scoring:
    - Hierarchical multi-head LSTM.
    - Family head selects None/Stop/Yield/NoEntry.
    - Selected family determines which behavior head is translated.
    - No sign-type resolver/gate is applied after the model.
    - Aggregation is based only on model-selected family/behavior predictions.
    """
    if global_scaler is None or global_lstm_model is None:
        load_ai_models()

    X_model_df = prepare_m16_feature_frame(combined_df)
    X_raw = X_model_df.to_numpy(dtype=np.float32)

    decision_log = []
    violation_events = []
    ignored_warning_events = []
    positive_actions = []
    xai_data = {}
    action_sequences = []

    if len(X_raw) < MODEL_WINDOW_SIZE:
        return {
            "result": "PASS",
            "passed": True,
            "grade": 100,
            "mistakes_count": 0,
            "violations_detected": [],
            "ignored_warning_codes": [],
            "ignored_warning_events": [],
            "decision_log": [],
            "xai_data": {},
            "action_sequences": [],
            "positive_actions": [],
            "windows_analyzed": 0,
            "violation_events": [],
        }

    X_scaled = global_scaler.transform(X_raw).astype(np.float32)

    window_starts = list(range(0, len(X_scaled) - MODEL_WINDOW_SIZE + 1, max(1, int(MODEL_WINDOW_STRIDE))))
    X_windows = np.stack([
        X_scaled[i:i + MODEL_WINDOW_SIZE]
        for i in window_starts
    ]).astype(np.float32)

    update_progress(test_id, 85, "Running M16 hierarchical multi-head LSTM...")

    batch_size = 64
    num_batches = max(1, (len(X_windows) + batch_size - 1) // batch_size)
    raw_parts: Dict[str, list[np.ndarray]] = {
        "family": [],
        "stop": [],
        "yield_": [],
        "noentry": [],
        "tailgating": [],
    }

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(X_windows))
        batch = X_windows[start:end]
        raw = _normalize_model_outputs(global_lstm_model.predict(batch, batch_size=batch_size, verbose=0))
        for k in raw_parts:
            raw_parts[k].append(raw[k])

        pct = 85 + int(10 * (batch_idx + 1) / num_batches)
        update_progress(test_id, pct, f"M16 batch {batch_idx + 1}/{num_batches}")

    raw_out = {k: np.concatenate(v, axis=0) for k, v in raw_parts.items()}

    family_probs = _softmax_np(raw_out["family"])
    stop_probs = _softmax_np(raw_out["stop"])
    yield_probs = _softmax_np(raw_out["yield_"])
    noentry_probs = _softmax_np(raw_out["noentry"])
    tailgating_probs = _sigmoid_np(raw_out["tailgating"].reshape(-1))

    family_pred = family_probs.argmax(axis=1).astype(np.int32)
    stop_pred = stop_probs.argmax(axis=1).astype(np.int32)
    yield_pred = yield_probs.argmax(axis=1).astype(np.int32)
    noentry_pred = noentry_probs.argmax(axis=1).astype(np.int32)

    window_preds = []

    for wi, start_idx in enumerate(window_starts):
        end_idx = min(start_idx + MODEL_WINDOW_SIZE - 1, len(combined_df) - 1)
        ctx = _row_context(combined_df, end_idx)

        fam = int(family_pred[wi])
        behavior = None
        behavior_name = None
        behavior_prob = 0.0

        if fam == FAMILY_STOP:
            behavior = int(stop_pred[wi])
            behavior_name = STOP_NAMES[behavior]
            behavior_prob = float(stop_probs[wi, behavior])
        elif fam == FAMILY_YIELD:
            behavior = int(yield_pred[wi])
            behavior_name = YIELD_NAMES[behavior]
            behavior_prob = float(yield_probs[wi, behavior])
        elif fam == FAMILY_NOENTRY:
            behavior = int(noentry_pred[wi])
            behavior_name = NOENTRY_NAMES[behavior]
            behavior_prob = float(noentry_probs[wi, behavior])

        wp = {
            "window_index": int(wi),
            "window_start": int(start_idx),
            "window_end": int(end_idx),
            "timestamp_sec": ctx["timestamp_sec"],
            "family": fam,
            "family_name": FAMILY_NAMES[fam],
            "behavior": behavior,
            "behavior_name": behavior_name,
            "behavior_prob": behavior_prob,
            "tg_prob": float(tailgating_probs[wi]),
            "family_probs": family_probs[wi].tolist(),
            "stop_probs": stop_probs[wi].tolist(),
            "yield_probs": yield_probs[wi].tolist(),
            "noentry_probs": noentry_probs[wi].tolist(),
            "context": ctx,
        }
        window_preds.append(wp)

        decision_log.append({
            "window_index": int(wi),
            "window_start": int(start_idx),
            "window_end": int(end_idx),
            "timestamp_sec": ctx["timestamp_sec"],
            "family_id": fam,
            "family_name": FAMILY_NAMES[fam],
            "behavior_id": behavior,
            "behavior_name": behavior_name,
            "behavior_prob": round(float(behavior_prob), 4),
            "tailgating_prob": round(float(tailgating_probs[wi]), 4),
            "family_probs": {FAMILY_NAMES[j]: round(float(family_probs[wi, j]), 4) for j in range(len(FAMILY_NAMES))},
            "stop_probs": {STOP_NAMES[j]: round(float(stop_probs[wi, j]), 4) for j in range(len(STOP_NAMES))},
            "yield_probs": {YIELD_NAMES[j]: round(float(yield_probs[wi, j]), 4) for j in range(len(YIELD_NAMES))},
            "noentry_probs": {NOENTRY_NAMES[j]: round(float(noentry_probs[wi, j]), 4) for j in range(len(NOENTRY_NAMES))},
            "context": ctx,
        })

    agg = aggregate_window_predictions(window_preds)

    for event_name in agg["events"]:
        best = _best_window_for_event(event_name, window_preds, combined_df)
        if best is None:
            continue
        wp, ctx = best
        payload = _event_payload(event_name, wp, ctx)

        bucket = EVENT_BUCKET[event_name]
        if bucket == "positive":
            positive_actions.append(payload)
        elif bucket == "warning":
            ignored_warning_events.append(payload)
        elif bucket == "violation":
            violation_events.append(payload)

    violation_codes = sorted({int(v["class_id"]) for v in violation_events})
    ignored_warning_codes = sorted({int(v["class_id"]) for v in ignored_warning_events})

    mistakes_count = len(violation_events)
    result = "FAIL" if mistakes_count > 0 else "PASS"
    passed = result == "PASS"
    final_grade = 100 if passed else 0

    thought_payload = {
        "test_id": test_id,
        "exported_at": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "model_type": MODEL_TYPE,
        "feature_names": MODEL_FEATURES,
        "family_names": FAMILY_NAMES,
        "stop_names": STOP_NAMES,
        "yield_names": YIELD_NAMES,
        "noentry_names": NOENTRY_NAMES,
        "legacy_class_names": {str(k): v for k, v in LEGACY_CLASS_NAMES.items()},
        "result": result,
        "passed": passed,
        "mistakes_count": mistakes_count,
        "mistake_codes": violation_codes,
        "ignored_warning_codes": ignored_warning_codes,
        "ignored_warning_events_count": len(ignored_warning_events),
        "ignored_warning_events": ignored_warning_events,
        "grade": final_grade,
        "decision_log": decision_log,
        "window_predictions": window_preds,
        "aggregation": agg,
        "xai_explanations": xai_data,
        "action_sequences": action_sequences,
        "positive_actions": positive_actions,
        "positive_actions_count": len(positive_actions),
        "violation_events": violation_events,
        "violation_events_count": len(violation_events),
        "windows_analyzed": len(X_windows),
        "window_stride": MODEL_WINDOW_STRIDE,
        "episode_policy": (
            "M16 hierarchical multi-head LSTM. The model chooses family and behavior heads. "
            "No sign-type resolver/gate is applied after the model. "
            "Backend only aggregates model-selected outputs and translates to legacy events. "
            "NoEntry.NoEvent emits no positive action."
        ),
    }

    with open(os.path.join(test_export_path, "3_model_thought.json"), "w", encoding="utf-8") as f:
        json.dump(thought_payload, f, ensure_ascii=False, indent=4, default=_json_default)

    return {
        "result": result,
        "passed": passed,
        "grade": final_grade,
        "mistakes_count": mistakes_count,
        "violations_detected": violation_codes,
        "ignored_warning_codes": ignored_warning_codes,
        "ignored_warning_events": ignored_warning_events,
        "decision_log": decision_log,
        "xai_data": xai_data,
        "action_sequences": action_sequences,
        "positive_actions": positive_actions,
        "windows_analyzed": len(X_windows),
        "violation_events": violation_events,
        "aggregation": agg,
    }


# Backward-compatible aliases for scripts that call old names.
def run_m15_composite_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m16_hierarchical_scoring(combined_df, test_id, test_export_path)


def run_m14_multilabel_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m16_hierarchical_scoring(combined_df, test_id, test_export_path)


def run_m12_episode_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m16_hierarchical_scoring(combined_df, test_id, test_export_path)


# ==========================================================================
# /evaluate — async, JWT-auth, SHA256-validated
# ==========================================================================
@router.post("/evaluate")
async def evaluate_test(
    test_id: str = Form(...),
    video: UploadFile = File(...),
    sensors: UploadFile = File(...),
    student_id: str = Form("pending"),
    video_sha256: str = Form(None),
    sensors_sha256: str = Form(None),
    tester: dict = Depends(get_current_tester),
):
    try:
        load_ai_models()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    test_export_path = os.path.join(EXPORT_BASE_DIR, test_id)
    os.makedirs(test_export_path, exist_ok=True)

    temp_id = str(uuid.uuid4())
    temp_vid = f"temp_{temp_id}.mp4"
    temp_json = f"temp_{temp_id}.json"

    try:
        update_progress(test_id, 0, "Receiving files...")

        with open(temp_vid, "wb") as buf:
            buf.write(video.file.read())

        with open(temp_json, "wb") as buf:
            buf.write(sensors.file.read())

        if video_sha256:
            actual = _sha256_file(temp_vid)
            if actual.lower() != video_sha256.lower():
                raise HTTPException(
                    status_code=422,
                    detail=f"Video integrity failed: expected {video_sha256[:8]}, got {actual[:8]}",
                )

        if sensors_sha256:
            actual = _sha256_file(temp_json)
            if actual.lower() != sensors_sha256.lower():
                raise HTTPException(
                    status_code=422,
                    detail="Sensor JSON integrity failed",
                )

        log.info(f"✅ Integrity verified for {test_id}")

        saved_input_video = os.path.join(test_export_path, "input_video.mp4")
        try:
            shutil.copyfile(temp_vid, saved_input_video)
            log.info(f"🎞️ Saved original input video: {saved_input_video}")
        except Exception as e:
            log.warning(f"⚠️ Could not save original input video copy: {e}")

        update_progress(test_id, 5, "Processing sensor data...")
        sensor_df = await asyncio.to_thread(process_sensor_json, temp_json)
        sensor_df.to_csv(
            os.path.join(test_export_path, "1_raw_sensors.csv"),
            index=False,
        )

        update_progress(test_id, 10, "Running YOLO detection...")
        annotated_video_path = os.path.join(test_export_path, "yolo_annotated.mp4")

        video_events = await analyze_video_for_server(
            temp_vid,
            sensor_df,
            test_id,
            output_video_path=annotated_video_path,
        )

        with open(os.path.join(test_export_path, "1b_video_events.json"), "w", encoding="utf-8") as f:
            json.dump(video_events, f, ensure_ascii=False, indent=2)

        update_progress(test_id, 80, "Building feature vector...")
        combined_df = build_feature_vector(sensor_df, video_events)
        combined_df.to_csv(
            os.path.join(test_export_path, "2_feature_vector.csv"),
            index=False,
        )

        score = await asyncio.to_thread(
            run_m16_hierarchical_scoring,
            combined_df,
            test_id,
            test_export_path,
        )

        decision_log = score["decision_log"]
        xai_data = score["xai_data"]
        action_sequences = score["action_sequences"]
        positive_actions = score["positive_actions"]

        result = score["result"]
        passed = score["passed"]
        final_grade = score["grade"]

        total_violation_events = score["mistakes_count"]
        violations_detected = score["violations_detected"]

        ignored_warning_codes = score["ignored_warning_codes"]
        ignored_warning_events = score["ignored_warning_events"]

        windows_analyzed = score["windows_analyzed"]

        update_progress(test_id, 100, "Complete!")

        log.info(
            f"🎯 Done. Result: {result} | Mistakes: {total_violation_events} | "
            f"IgnoredWarnings: {len(ignored_warning_events)} | "
            f"PositiveActions: {len(positive_actions)} | Tester: {tester['email']}"
        )

        return {
            "status": "success",
            "test_id": test_id,
            "model_id": MODEL_ID,
            "model_type": MODEL_TYPE,
            "student_id": student_id,
            "tester_email": tester["email"],
            "result": result,
            "passed": passed,
            "mistakes_count": total_violation_events,
            "mistake_codes": violations_detected,
            "grade": final_grade,
            "violations_codes": violations_detected,
            "violation_events_count": total_violation_events,
            "ignored_warning_codes": ignored_warning_codes,
            "ignored_warning_events_count": len(ignored_warning_events),
            "ignored_warning_events": ignored_warning_events,
            "windows_analyzed": windows_analyzed,
            "xai_explanations": xai_data,
            "decision_log": decision_log,
            "action_sequences": action_sequences,
            "positive_actions": positive_actions,
            "exported_video_path": os.path.join(test_export_path, "input_video.mp4"),
            "annotated_video_path": os.path.join(test_export_path, "yolo_annotated.mp4"),
        }

    except HTTPException:
        raise

    except asyncio.CancelledError:
        log.warning(f"⚠️ Test {test_id} cancelled")
        raise

    except Exception as e:
        log.error(f"❌ Error processing test {test_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for f in [temp_vid, temp_json]:
            for attempt in range(5):
                if not os.path.exists(f):
                    break
                try:
                    os.remove(f)
                    break
                except PermissionError:
                    if attempt < 4:
                        await asyncio.sleep(0.5)
                except Exception as e:
                    log.warning(f"⚠️ Cleanup error on {f}: {e}")
                    break

        if test_id in PROGRESS_STORE:
            try:
                del PROGRESS_STORE[test_id]
            except Exception:
                pass


# ==========================================================================
# /save — JWT-protected
# ==========================================================================
@router.post("/save")
async def save_test(
    payload: TestSaveRequest,
    tester: dict = Depends(get_current_tester),
):
    try:
        tester_email = tester["email"]

        student = await db.db["students"].find_one({
            "student_id": payload.student_id,
            "tester_email": tester_email,
        })

        if not student:
            raise HTTPException(
                status_code=404,
                detail="Student not found for this tester",
            )

        if payload.test_id:
            existing = await db.db["tests"].find_one({
                "test_id": payload.test_id,
                "tester_email": tester_email,
            })

            if existing:
                return {
                    "status": "already_saved",
                    "saved_test_id": str(existing["_id"]),
                    "student_name": existing.get("student_name", student["name"]),
                    "result": existing.get(
                        "result",
                        "PASS" if existing.get("grade", payload.grade) >= 80 else "FAIL",
                    ),
                    "passed": existing.get(
                        "passed",
                        existing.get("grade", payload.grade) >= 80,
                    ),
                    "mistakes_count": existing.get(
                        "mistakes_count",
                        existing.get("violation_events_count", 0),
                    ),
                    "grade": existing.get("grade", payload.grade),
                }

        record = payload.model_dump()
        record["tester_email"] = tester_email
        record["student_name"] = student["name"]
        record["saved_at"] = datetime.now()

        if record.get("test_date") is None:
            record["test_date"] = datetime.now()

        if record.get("result") not in ("PASS", "FAIL"):
            record["result"] = "PASS" if record.get("grade", 0) >= 80 else "FAIL"

        record["passed"] = bool(record.get("passed", record["result"] == "PASS"))
        record["status"] = "passed" if record["passed"] else "failed"

        result = await db.db["tests"].insert_one(record)

        log.info(
            f"💾 Test saved | {student['name']} ({payload.student_id}) "
            f"| result: {record['result']} | mistakes: {record.get('mistakes_count', 0)} "
            f"| by: {tester_email}"
        )

        return {
            "status": "success",
            "saved_test_id": str(result.inserted_id),
            "student_name": student["name"],
            "result": record["result"],
            "passed": record["passed"],
            "mistakes_count": record.get("mistakes_count", 0),
            "grade": payload.grade,
        }

    except HTTPException:
        raise

    except Exception as e:
        log.error(f"❌ Failed to save test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================================
# Tester history routes — JWT only, /me/* preferred
# ==========================================================================
@router.get("/student/{student_id}")
async def get_student_tests(
    student_id: str,
    tester: dict = Depends(get_current_tester),
):
    cursor = db.db["tests"].find({
        "student_id": student_id,
        "tester_email": tester["email"],
    }).sort("saved_at", -1)

    tests = []

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        for k in ("saved_at", "test_date"):
            if k in doc and hasattr(doc[k], "isoformat"):
                doc[k] = doc[k].isoformat()
        tests.append(doc)

    return tests


@router.get("/me/tests")
async def get_my_tests(tester: dict = Depends(get_current_tester)):
    cursor = db.db["tests"].find({
        "tester_email": tester["email"],
    }).sort("saved_at", -1)

    tests = []

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        for k in ("saved_at", "test_date"):
            if k in doc and hasattr(doc[k], "isoformat"):
                doc[k] = doc[k].isoformat()
        tests.append(doc)

    return tests


@router.get("/tester/{tester_email}")
async def get_tester_tests_legacy(
    tester_email: str,
    tester: dict = Depends(get_current_tester),
):
    return await get_my_tests(tester)


@router.get("/detail/{test_object_id}")
async def get_test_detail(
    test_object_id: str,
    tester: dict = Depends(get_current_tester),
):
    from bson import ObjectId

    try:
        oid = ObjectId(test_object_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid test ID")

    doc = await db.db["tests"].find_one({
        "_id": oid,
        "tester_email": tester["email"],
    })

    if not doc:
        raise HTTPException(status_code=404, detail="Test not found")

    doc["_id"] = str(doc["_id"])

    for k in ("saved_at", "test_date"):
        if k in doc and hasattr(doc[k], "isoformat"):
            doc[k] = doc[k].isoformat()

    return doc


# ===========================================================================
# Student Success Predictions
# ===========================================================================
# This is the SECOND model area: student-history success prediction.
# It is independent from the driving-recognition model, YOLO, vector_builder,
# replay, and M18/M18_6 artifacts.
#
# Expected future artifacts in app/ai_models/:
#   student_success_model.keras
#   student_success_scaler.pkl
#   student_success_class_map.json
#
# Until these artifacts exist, the API returns predicted_success_rate=None
# instead of using manual IF / weighted-average logic.
# ===========================================================================


async def _load_student_tests_for_prediction(student_id: str, tester_email: str) -> List[Dict[str, Any]]:
    cursor = db.db["tests"].find({
        "student_id": student_id,
        "tester_email": tester_email,
    }).sort("saved_at", 1)

    tests: List[Dict[str, Any]] = []
    async for doc in cursor:
        tests.append(doc)
    return tests


@router.get("/prediction/{student_id}")
async def predict_student_success(
    student_id: str,
    tester: dict = Depends(get_current_tester),
):
    student = await db.db["students"].find_one({
        "student_id": student_id,
        "tester_email": tester["email"],
    })

    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    tests = await _load_student_tests_for_prediction(student_id, tester["email"])
    predictor = get_student_predictor(BASE_DIR)

    return predictor.predict_payload(
        student_id=student_id,
        student_name=student.get("name", "Unknown"),
        tests=tests,
    )


@router.get("/predictions")
async def predict_all_students(tester: dict = Depends(get_current_tester)):
    cursor = db.db["students"].find({
        "tester_email": tester["email"],
    })

    predictor = get_student_predictor(BASE_DIR)
    out: List[Dict[str, Any]] = []

    async for student in cursor:
        sid = student["student_id"]
        tests = await _load_student_tests_for_prediction(sid, tester["email"])
        out.append(
            predictor.predict_payload(
                student_id=sid,
                student_name=student.get("name", "Unknown"),
                tests=tests,
            )
        )

    # Students with no model/no prediction go last. Among predicted students,
    # lower readiness appears first so the teacher sees who needs attention.
    out.sort(key=lambda p: (
        p.get("predicted_success_rate") is None,
        p.get("predicted_success_rate") if p.get("predicted_success_rate") is not None else 101,
        p.get("student_name", ""),
    ))

    return out


@router.get("/predictions/{tester_email}")
async def predict_all_students_legacy(
    tester_email: str,
    tester: dict = Depends(get_current_tester),
):
    # Kept only for old clients. The authenticated token is the source of truth.
    return await predict_all_students(tester)
