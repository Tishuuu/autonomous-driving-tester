# FINAL MERGED ROUTER: M18/M18_6 driving model + Student Success Predictor v2.3 endpoints.
import os
import uuid
import asyncio
import json
import pickle
import hashlib
import shutil
from datetime import datetime
from typing import Any, Dict, List, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
# M18 episode-level scenario model config
# ==========================================================================
MODEL_ID = "M18"
MODEL_ACCEPTED_IDS = {
    "M18",
    "M18.0", "M18.1", "M18.2", "M18.3", "M18.4", "M18.5", "M18.6",
    "M18_1", "M18_2", "M18_3", "M18_4", "M18_5", "M18_6",
}
MODEL_FAMILY = "episode_level_scenario_bilstm"
MODEL_TYPE = MODEL_FAMILY
MODEL_MAX_LEN = 256
PAD_VALUE = -1000000.0
SCALE_CLIP_VALUE = 12.0
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

SCENARIO_NAMES = [
    "NORMAL",
    "STOP_CORRECT",
    "RUNNING_STOP",
    "STOP_THEN_BAD_YIELD",
    "YIELD_CORRECT",
    "YIELD_FAILURE",
    "NOENTRY_VIOLATION",
]

DEFAULT_SCENARIO_TO_EVENTS = {
    "NORMAL": [],
    "STOP_CORRECT": ["CorrectStop", "CorrectYield"],
    "RUNNING_STOP": ["RunningStop"],
    "STOP_THEN_BAD_YIELD": ["CorrectStop", "FailureToYield"],
    "YIELD_CORRECT": ["CorrectYield"],
    "YIELD_FAILURE": ["FailureToYield"],
    "NOENTRY_VIOLATION": ["NoEntryViolation"],
}

NORMAL_SUBTYPE_NAMES = [
    "NORMAL_NO_SIGN",
    "NORMAL_HANDLED_YIELD",
    "NORMAL_HANDLED_NOENTRY",
]

ATOMIC_AUX_HEAD_NAMES = [
    "speed_reached_zero",
    "stopped_at_sign",
    "relevant_threat_present",
    "yielded_to_threat",
    "continued_past_sign",
]

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
    "do_not_enter": 3,
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
}

POSITIVE_EVENTS = {"CorrectStop", "CorrectYield"}
MISTAKE_EVENTS = {"RunningStop", "FailureToYield", "NoEntryViolation"}
WARN_EVENTS = {"Tailgating"}


# ==========================================================================
# Keras custom layer used by m18_model.keras
# ==========================================================================
@tf.keras.utils.register_keras_serializable(package="m18", name="MaskedAttentionPool")
class MaskedAttentionPool(layers.Layer):
    """Attention pooling with explicit Keras mask support."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        scores = self.score_dense(inputs)
        scores = tf.squeeze(scores, axis=-1)
        if mask is not None:
            mask_f = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask_f) * tf.constant(-1e9, dtype=scores.dtype)
        weights = tf.nn.softmax(scores, axis=-1)
        weights = tf.expand_dims(weights, axis=-1)
        return tf.reduce_sum(inputs * weights, axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "MaskedAttentionPool": MaskedAttentionPool,
    "m18>MaskedAttentionPool": MaskedAttentionPool,
    "m18.MaskedAttentionPool": MaskedAttentionPool,
    "M18>MaskedAttentionPool": MaskedAttentionPool,
    "M18.MaskedAttentionPool": MaskedAttentionPool,
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


def _load_scaler(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return joblib.load(path)


def _load_model_class_map(class_map_path: str) -> dict:
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"{MODEL_ID} class map not found: {class_map_path}")

    payload = _load_json(class_map_path)

    model_id = str(payload.get("model_id", MODEL_ID)).strip()
    accepted_ids_norm = {x.upper() for x in MODEL_ACCEPTED_IDS}
    if model_id.upper() not in accepted_ids_norm:
        raise ValueError(
            f"Expected one of {sorted(MODEL_ACCEPTED_IDS)} class map, got model_id={model_id}"
        )

    model_family = payload.get("model_family", payload.get("model_type", MODEL_FAMILY))
    if model_family != MODEL_FAMILY:
        raise ValueError(f"Expected {MODEL_FAMILY} class map, got: {model_family}")

    feature_names = payload.get("feature_names", [])
    if feature_names != MODEL_FEATURES:
        raise ValueError(
            f"{MODEL_ID} feature schema mismatch.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Got:      {feature_names}"
        )

    n_features = int(payload.get("n_features", len(feature_names)))
    if n_features != len(MODEL_FEATURES):
        raise ValueError(f"Expected {len(MODEL_FEATURES)} features, got {n_features}")

    max_len = int(payload.get("max_len", MODEL_MAX_LEN))
    if max_len != MODEL_MAX_LEN:
        raise ValueError(f"Expected max_len {MODEL_MAX_LEN}, got {max_len}")

    pad_value = float(payload.get("pad_value", PAD_VALUE))
    if abs(pad_value - PAD_VALUE) > 1e-6:
        raise ValueError(f"Expected pad_value {PAD_VALUE}, got {pad_value}")

    scenario_names = payload.get("scenario_names", SCENARIO_NAMES)
    if scenario_names != SCENARIO_NAMES:
        raise ValueError(f"Scenario schema mismatch. Expected {SCENARIO_NAMES}, got {scenario_names}")

    scenario_to_events = payload.get("scenario_to_events", DEFAULT_SCENARIO_TO_EVENTS)
    missing = [s for s in SCENARIO_NAMES if s not in scenario_to_events]
    if missing:
        raise ValueError(f"scenario_to_events missing scenarios: {missing}")

    decision_output = payload.get("inference_decision_output", "scenario_probs")
    if decision_output != "scenario_probs":
        raise ValueError(f"Expected inference_decision_output=scenario_probs, got {decision_output}")

    return payload


def load_ai_models():
    """Load M18 scaler + episode-level scenario BiLSTM once and fail loudly if missing."""
    global global_scaler, global_lstm_model, global_model_config

    if global_scaler is not None and global_lstm_model is not None:
        return

    log.info("Loading M18 episode-level scenario model...")

    try:
        models_dir = os.path.join(BASE_DIR, "app", "ai_models")

        scaler_path = os.path.join(models_dir, "m18_scaler.pkl")
        model_path = os.path.join(models_dir, "m18_model.keras")
        class_map_path = os.path.join(models_dir, "m18_class_map.json")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"M18 scaler not found: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"M18 model not found: {model_path}")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"M18 class map not found: {class_map_path}")

        global_model_config = _load_model_class_map(class_map_path)
        global_scaler = _load_scaler(scaler_path)
        global_lstm_model = keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )

        log.info(
            f"M18 Model + Scaler ready. Outputs: {getattr(global_lstm_model, 'output_names', [])}"
        )

    except Exception as e:
        global_scaler = None
        global_lstm_model = None
        global_model_config = None
        log.error(f"Failed to load M18 model artifacts: {e}")
        raise RuntimeError(f"AI model loading failed: {e}")


# ==========================================================================
# M18 feature prep
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
    if col == "heading_cos":
        return 1.0
    # M18 is trained with accel_z around 0, not gravity around 9.81.
    return 0.0


def prepare_m18_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the 33-feature frame expected by M18.

    This function only normalizes schema/ranges. It does not perform traffic-rule
    decisions. Decision logic remains strictly scenario_probs -> scenario_to_events.
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
    x["accel_z"] = x["accel_z"].clip(-0.2, 0.2)

    return x[MODEL_FEATURES].astype(np.float32)


# Backward-compatible aliases for old imports/scripts.
def prepare_m17_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_m18_feature_frame(combined_df)


def prepare_m16_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_m18_feature_frame(combined_df)


def prepare_m15_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_m18_feature_frame(combined_df)


def prepare_m14_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    return prepare_m18_feature_frame(combined_df)


def sanitize_m18_model_features(X: np.ndarray) -> np.ndarray:
    """Apply the same non-semantic GPS metadata sanitization used during M18 training."""
    Xs = np.asarray(X, dtype=np.float32).copy()
    idx = {name: i for i, name in enumerate(MODEL_FEATURES)}
    for name in ("gps_accuracy", "gps_age_ms", "gps_update_count_delta"):
        Xs[..., idx[name]] = 0.0
    return Xs


def _pad_scale_episode(X_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    """Pad/crop full episode to 256x33 and scale only real rows."""
    if global_scaler is None:
        load_ai_models()

    real_len = int(min(len(X_raw), MODEL_MAX_LEN))
    was_cropped = bool(len(X_raw) > MODEL_MAX_LEN)

    Xp = np.full((MODEL_MAX_LEN, len(MODEL_FEATURES)), PAD_VALUE, dtype=np.float32)
    mask = np.zeros((MODEL_MAX_LEN,), dtype=np.float32)

    if real_len > 0:
        Xp[:real_len] = X_raw[:real_len]
        mask[:real_len] = 1.0

    Xs = sanitize_m18_model_features(Xp)
    real = mask.astype(bool)
    if np.any(real):
        scaled = global_scaler.transform(Xs[real]).astype(np.float32)
        scaled = np.clip(scaled, -SCALE_CLIP_VALUE, SCALE_CLIP_VALUE).astype(np.float32)
        Xs[real] = scaled
    Xs[~real] = PAD_VALUE

    return Xs[None, ...].astype(np.float32), mask, real_len, was_cropped


# ==========================================================================
# M18 scoring helpers
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


def _normalize_m18_outputs(raw: Any) -> Dict[str, np.ndarray]:
    if isinstance(raw, dict):
        out = dict(raw)
    elif isinstance(raw, (list, tuple)):
        names = list(getattr(global_lstm_model, "output_names", []))
        out = {name: arr for name, arr in zip(names, raw)}
    else:
        raise ValueError(f"Unsupported M18 model output type: {type(raw)}")

    normalized: Dict[str, np.ndarray] = {}
    known = {"scenario_probs", "normal_subtype_probs", *ATOMIC_AUX_HEAD_NAMES}
    for key, value in out.items():
        base = str(key).split("/")[-1].split(":")[0]
        if base in known:
            normalized[base] = np.asarray(value)
        else:
            normalized[str(key)] = np.asarray(value)

    if "scenario_probs" not in normalized:
        raise ValueError(f"M18 model output missing scenario_probs; got {list(out.keys())}")

    return normalized


def _row_context(combined_df: pd.DataFrame, sample_idx: int) -> dict:
    if len(combined_df) == 0:
        sample_idx = 0
        row = pd.Series(dtype=float)
    else:
        row = combined_df.iloc[min(max(0, int(sample_idx)), len(combined_df) - 1)]

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


def _representative_context_index(features_df: pd.DataFrame) -> int:
    """Choose a reporting row. This is only for payload context, not for decisions."""
    if features_df is None or len(features_df) == 0:
        return 0

    sign_active = (features_df["sign_type"].to_numpy() > 0) | (features_df["memory_sign_type"].to_numpy() > 0)
    idxs = np.where(sign_active)[0]
    if len(idxs) > 0:
        # Prefer the row where active sign/memory distance is closest.
        dist = np.minimum(
            features_df["sign_distance"].to_numpy(dtype=np.float32),
            features_df["memory_sign_distance"].to_numpy(dtype=np.float32),
        )
        local = int(idxs[np.argmin(dist[idxs])])
        return local

    return int(min(len(features_df) - 1, max(0, len(features_df) // 2)))


def _event_payload(event_name: str, source: dict, ctx: dict) -> dict:
    legacy_code = int(LEGACY_EVENT_CODES[event_name])
    legacy_label = LEGACY_CLASS_NAMES.get(legacy_code, LEGACY_EVENT_LABELS.get(event_name, event_name))
    bucket = EVENT_BUCKET[event_name]
    return {
        "timestamp_sec": ctx["timestamp_sec"],
        "type": event_name,
        "event_key": f"{bucket}:{legacy_code}:{event_name}",
        "event_type": (
            "POSITIVE_ACTION" if bucket == "positive"
            else "IGNORED_WARNING" if bucket == "warning"
            else "VIOLATION"
        ),
        "confidence": round(float(source.get("event_confidence", 0.0)), 4),
        "scenario_id": int(source.get("scenario_id", 0)),
        "scenario_name": str(source.get("scenario_name", "NORMAL")),
        "family_id": None,
        "family_name": None,
        "behavior_id": None,
        "behavior_name": event_name,
        "window_index": None,
        "window_start": 0,
        "window_end": int(source.get("episode_len", 0)),
        "cluster_index": None,
        "class_id": legacy_code,
        "class_label": legacy_label,
        **ctx,
    }


def _scenario_to_events(scenario_name: str) -> List[str]:
    cfg = global_model_config or {}
    scenario_to_events = cfg.get("scenario_to_events", DEFAULT_SCENARIO_TO_EVENTS)
    return list(scenario_to_events.get(scenario_name, []))


def _outputs_debug(outputs: Dict[str, np.ndarray], scenario_names: List[str]) -> dict:
    debug: Dict[str, Any] = {}
    if "normal_subtype_probs" in outputs:
        normal_probs = np.asarray(outputs["normal_subtype_probs"])[0].reshape(-1)
        debug["normal_subtype_probs"] = {
            NORMAL_SUBTYPE_NAMES[i] if i < len(NORMAL_SUBTYPE_NAMES) else f"subtype_{i}": round(float(normal_probs[i]), 4)
            for i in range(len(normal_probs))
        }

    atomic_summary = {}
    for head in ATOMIC_AUX_HEAD_NAMES:
        if head not in outputs:
            continue
        arr = np.asarray(outputs[head])[0]
        arr = arr.reshape(arr.shape[0], -1)[:, 0]
        atomic_summary[head] = {
            "max": round(float(np.max(arr)), 4) if len(arr) else 0.0,
            "mean": round(float(np.mean(arr)), 4) if len(arr) else 0.0,
        }
    if atomic_summary:
        debug["atomic_aux_summary_training_debug_only"] = atomic_summary

    return debug


# ==========================================================================
# M18 scoring
# ==========================================================================
def run_m18_episode_scoring(
    combined_df: pd.DataFrame,
    test_id: str,
    test_export_path: str,
) -> dict:
    """
    M18 scoring:
    - Full episode is padded/cropped to 256x33.
    - Same training preprocessing is applied: GPS metadata sanitization, scaler, z-clip.
    - Model predicts scenario_probs.
    - Backend maps argmax scenario to fixed legacy events.

    No raw speed/sign/distance rules, no atomic overrides, no clusters, no thresholds.
    """
    if global_scaler is None or global_lstm_model is None:
        load_ai_models()

    X_model_df = prepare_m18_feature_frame(combined_df)
    X_raw = X_model_df.to_numpy(dtype=np.float32)
    X_scaled, mask, real_len, was_cropped = _pad_scale_episode(X_raw)

    update_progress(test_id, 85, "Running M18 episode-level scenario model...")
    raw = global_lstm_model.predict(X_scaled, batch_size=1, verbose=0)
    outputs = _normalize_m18_outputs(raw)

    scenario_names = list((global_model_config or {}).get("scenario_names", SCENARIO_NAMES))
    scenario_probs_arr = np.asarray(outputs["scenario_probs"])[0].reshape(-1)
    scenario_idx = int(np.argmax(scenario_probs_arr))
    scenario_name = scenario_names[scenario_idx]
    scenario_conf = float(scenario_probs_arr[scenario_idx])
    events = _scenario_to_events(scenario_name)

    rep_idx = _representative_context_index(X_model_df.iloc[:real_len] if real_len > 0 else X_model_df)
    ctx = _row_context(X_model_df, rep_idx)

    source = {
        "event_confidence": scenario_conf,
        "scenario_id": scenario_idx,
        "scenario_name": scenario_name,
        "episode_len": real_len,
    }

    violation_events = []
    ignored_warning_events = []
    positive_actions = []

    for event_name in events:
        payload = _event_payload(event_name, source, ctx)
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

    decision_log = [{
        "episode_index": 0,
        "timestamp_sec": ctx["timestamp_sec"],
        "scenario_id": scenario_idx,
        "scenario_name": scenario_name,
        "scenario_confidence": round(scenario_conf, 4),
        "scenario_probs": {
            scenario_names[i]: round(float(scenario_probs_arr[i]), 4)
            for i in range(len(scenario_probs_arr))
        },
        "events": list(events),
        "context": ctx,
        "input_rows": int(len(X_raw)),
        "model_rows": int(real_len),
        "max_len": MODEL_MAX_LEN,
        "was_cropped": bool(was_cropped),
    }]

    debug = _outputs_debug(outputs, scenario_names)
    if debug:
        decision_log[0].update(debug)

    aggregation = {
        "scenario_id": scenario_idx,
        "scenario_name": scenario_name,
        "scenario_confidence": scenario_conf,
        "scenario_probs": {
            scenario_names[i]: float(scenario_probs_arr[i])
            for i in range(len(scenario_probs_arr))
        },
        "events": list(events),
        "scenario_to_events": (global_model_config or {}).get("scenario_to_events", DEFAULT_SCENARIO_TO_EVENTS),
        "input_rows": int(len(X_raw)),
        "model_rows": int(real_len),
        "max_len": MODEL_MAX_LEN,
        "was_cropped": bool(was_cropped),
        "policy": "M18 episode-level scenario argmax only. No raw runtime traffic-rule gates.",
    }
    aggregation.update(debug)

    thought_payload = {
        "test_id": test_id,
        "exported_at": datetime.now().isoformat(),
        "model_id": (global_model_config or {}).get("model_id", MODEL_ID),
        "model_type": MODEL_TYPE,
        "model_family": MODEL_FAMILY,
        "code_version": (global_model_config or {}).get("code_version"),
        "generator_version": (global_model_config or {}).get("generator_version"),
        "feature_names": MODEL_FEATURES,
        "scenario_names": scenario_names,
        "scenario_to_events": (global_model_config or {}).get("scenario_to_events", DEFAULT_SCENARIO_TO_EVENTS),
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
        "window_predictions": [],
        "aggregation": aggregation,
        "xai_explanations": {},
        "action_sequences": [],
        "positive_actions": positive_actions,
        "positive_actions_count": len(positive_actions),
        "violation_events": violation_events,
        "violation_events_count": len(violation_events),
        "windows_analyzed": 0,
        "episode_rows_analyzed": int(real_len),
        "episode_policy": (
            "M18 episode-level scenario BiLSTM. Backend uses only scenario_probs argmax "
            "and class_map scenario_to_events. Auxiliary heads are debug/training-only. "
            "No raw speed/sign/distance runtime gates are used for road-rule decisions."
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
        "xai_data": {},
        "action_sequences": [],
        "positive_actions": positive_actions,
        "windows_analyzed": 0,
        "episode_rows_analyzed": int(real_len),
        "violation_events": violation_events,
        "aggregation": aggregation,
    }


# Backward-compatible aliases for scripts that call old names.
def run_m17_atomic_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m18_episode_scoring(combined_df, test_id, test_export_path)


def run_m16_hierarchical_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m18_episode_scoring(combined_df, test_id, test_export_path)


def run_m15_composite_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m18_episode_scoring(combined_df, test_id, test_export_path)


def run_m14_multilabel_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m18_episode_scoring(combined_df, test_id, test_export_path)


def run_m12_episode_scoring(combined_df: pd.DataFrame, test_id: str, test_export_path: str) -> dict:
    return run_m18_episode_scoring(combined_df, test_id, test_export_path)


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
            run_m18_episode_scoring,
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
        episode_rows_analyzed = score.get("episode_rows_analyzed", 0)

        update_progress(test_id, 100, "Complete!")

        log.info(
            f"🎯 Done. Result: {result} | Mistakes: {total_violation_events} | "
            f"IgnoredWarnings: {len(ignored_warning_events)} | "
            f"PositiveActions: {len(positive_actions)} | Tester: {tester['email']}"
        )

        return {
            "status": "success",
            "test_id": test_id,
            "model_id": (global_model_config or {}).get("model_id", MODEL_ID),
            "model_type": MODEL_TYPE,
            "model_family": MODEL_FAMILY,
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
            "episode_rows_analyzed": episode_rows_analyzed,
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
