import os
import uuid
import asyncio
import json
import hashlib
import shutil
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
import pandas as pd
import numpy as np
from scipy import stats
import tensorflow as tf
from tensorflow import keras
from keras import layers

from app.core.database import db
from app.utils.logger import log
from app.services.sensor_sync import process_sensor_json
from app.services.vision_service import analyze_video_for_server
from app.services.vector_builder import build_feature_vector
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
# M12 model config
# ==========================================================================
MODEL_ID = "M12"
MODEL_WINDOW_SIZE = 30

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
]

MODEL_CLASS_NAMES = {
    0: "Normal Driving",
    1: "Tailgating",
    2: "Running Stop",
    3: "Failure to Yield",
    4: "No Entry Violation",
    5: "Correct Stop",
    6: "Correct Yield",
}

VIOLATION_CLASSES = {1, 2, 3, 4}

# Tailgating is detected but ignored for PASS/FAIL for now.
IGNORED_WARNING_CLASSES = {1}

FAIL_CLASSES = {2, 3, 4}

POSITIVE_ACTION_CLASSES = {5, 6}

POSITIVE_ACTION_TYPES = {
    5: "CorrectStop",
    6: "CorrectYield",
}

SIGN_NONE = 0
SIGN_STOP = 1
SIGN_YIELD = 2
SIGN_NO_ENTRY = 3


# ==========================================================================
# Keras custom layer
# ==========================================================================
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
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e, axis=-1)
        context = tf.reduce_sum(H * tf.expand_dims(alpha, -1), axis=1)
        if self.return_attention:
            return context, alpha
        return context

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "return_attention": self.return_attention,
        })
        return cfg


# ==========================================================================
# Model loading
# ==========================================================================
global_scaler = None
global_lstm_model = None
global_model_config = None


def build_attention_extractor(trained_model):
    """
    Builds a prediction + attention extractor for M12.

    M12 architecture:
      input -> BiLSTM -> TemporalAttention -> avg_pool + max_pool
            -> dense_1 -> dense_2 -> output
    """
    inp = layers.Input(
        shape=(MODEL_WINDOW_SIZE, len(MODEL_FEATURES)),
        name="input_window",
    )

    bidir_trained = trained_model.get_layer("bidir_lstm_1")
    bidir_clone = layers.Bidirectional(
        layers.LSTM(
            bidir_trained.forward_layer.units,
            return_sequences=True,
            name="lstm_clone_inner",
        ),
        name="bidir_lstm_1_clone",
    )
    H = bidir_clone(inp)

    att_trained = trained_model.get_layer("temporal_attention")
    att_clone = TemporalAttention(
        units=att_trained.units,
        return_attention=True,
        name="temporal_attention_clone",
    )
    context, alpha = att_clone(H)

    avg_pool = layers.GlobalAveragePooling1D(name="avg_pool_clone")(H)

    dense_trained = trained_model.get_layer("dense_1")
    dense_1_input_dim = int(dense_trained.get_weights()[0].shape[0])
    hidden_dim = int(H.shape[-1])

    if dense_1_input_dim == hidden_dim * 3:
        max_pool = layers.GlobalMaxPooling1D(name="max_pool_clone")(H)
        merged = layers.Concatenate(name="context_avg_max_clone")(
            [context, avg_pool, max_pool]
        )
    elif dense_1_input_dim == hidden_dim * 2:
        merged = layers.Concatenate(name="context_plus_avg_clone")(
            [context, avg_pool]
        )
    elif dense_1_input_dim == hidden_dim:
        merged = context
    else:
        raise ValueError(
            f"Unsupported attention extractor shape: dense_1 expects "
            f"{dense_1_input_dim}, BiLSTM hidden_dim is {hidden_dim}"
        )

    x = layers.Dense(
        dense_trained.units,
        activation="relu",
        name="dense_1_clone",
    )(merged)

    if "dense_2" in [layer.name for layer in trained_model.layers]:
        dense2_trained = trained_model.get_layer("dense_2")
        x = layers.Dense(
            dense2_trained.units,
            activation="relu",
            name="dense_2_clone",
        )(x)
    else:
        dense2_trained = None

    out_trained = trained_model.get_layer("output")
    out = layers.Dense(
        out_trained.units,
        activation="softmax",
        name="output_clone",
    )(x)

    extractor = keras.Model(
        inp,
        [out, alpha],
        name="attention_extractor_clone",
    )

    bidir_clone.set_weights(bidir_trained.get_weights())
    att_clone.set_weights(att_trained.get_weights())
    extractor.get_layer("dense_1_clone").set_weights(dense_trained.get_weights())

    if dense2_trained is not None:
        extractor.get_layer("dense_2_clone").set_weights(
            dense2_trained.get_weights()
        )

    extractor.get_layer("output_clone").set_weights(out_trained.get_weights())
    return extractor


def _load_model_class_map(class_map_path: str) -> dict:
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"{MODEL_ID} class map not found: {class_map_path}")

    with open(class_map_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    classes = payload.get("classes", {})
    feature_names = payload.get("feature_names", [])

    if not classes:
        raise ValueError(f"{MODEL_ID} class map is missing classes")

    if feature_names and feature_names != MODEL_FEATURES:
        raise ValueError(
            f"{MODEL_ID} feature schema mismatch.\n"
            f"Expected: {MODEL_FEATURES}\n"
            f"Got:      {feature_names}"
        )

    return payload


def prepare_m12_feature_frame(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare M12 feature frame.

    M12 was trained with:
      - 19-feature schema
      - sign_type + memory_sign_type
      - 99 sentinel for no visible object/sign distances
      - vehicle context fields
    """
    missing = [col for col in MODEL_FEATURES if col not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing required M12 feature columns: {missing}")

    x = combined_df[MODEL_FEATURES].copy()

    sign_mapping = {
        "none": 0,
        "stop_sign": 1,
        "yield_sign": 2,
        "no_entry": 3,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
    }

    for col in ["sign_type", "memory_sign_type"]:
        if x[col].dtype == object:
            x[col] = (
                x[col]
                .astype(str)
                .str.strip()
                .map(sign_mapping)
                .fillna(0)
                .astype(int)
            )

    for col in MODEL_FEATURES:
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)

    x["sign_type"] = x["sign_type"].round().clip(0, 3).astype(int)
    x["memory_sign_type"] = x["memory_sign_type"].round().clip(0, 3).astype(int)

    # Preserve M12 sentinel convention.
    no_visible_sign = x["sign_type"] == 0
    x.loc[no_visible_sign, ["sign_distance", "sign_ttc"]] = x.loc[
        no_visible_sign,
        ["sign_distance", "sign_ttc"],
    ].replace(0, 99)

    no_memory_sign = x["memory_sign_type"] == 0
    memory_cols = [
        "memory_sign_distance",
        "time_since_sign_seen_sec",
        "meters_since_sign_seen",
        "last_seen_sign_distance",
        "memory_sign_ttc",
    ]
    x.loc[no_memory_sign, memory_cols] = x.loc[
        no_memory_sign,
        memory_cols,
    ].replace(0, 99)

    x["speed_kmh"] = x["speed_kmh"].clip(0.0, 150.0)
    x["jerk"] = x["jerk"].clip(-30.0, 30.0)

    for col in [
        "car_distance",
        "car_ttc",
        "sign_distance",
        "sign_ttc",
        "memory_sign_distance",
        "time_since_sign_seen_sec",
        "meters_since_sign_seen",
        "last_seen_sign_distance",
        "memory_sign_ttc",
    ]:
        x[col] = x[col].clip(0.0, 99.0)

    x["car_relative_x"] = x["car_relative_x"].clip(-1.0, 1.0)
    x["car_relative_y"] = x["car_relative_y"].clip(0.0, 1.0)
    x["car_motion_x"] = x["car_motion_x"].clip(-2.0, 2.0)
    x["car_motion_y"] = x["car_motion_y"].clip(-2.0, 2.0)
    x["car_static_score"] = x["car_static_score"].clip(0.0, 1.0)

    return x.astype(float)


def load_ai_models():
    """Load M12 scaler + model once and fail loudly if an artifact is missing."""
    global global_scaler, global_lstm_model, global_model_config

    if global_scaler is not None and global_lstm_model is not None:
        return

    log.info("Loading M12 scaler + model...")

    try:
        import joblib

        models_dir = os.path.join(BASE_DIR, "app", "ai_models")

        scaler_path = os.path.join(models_dir, "m12_scaler.pkl")
        model_path = os.path.join(models_dir, "m12_model.keras")
        class_map_path = os.path.join(models_dir, "m12_class_map.json")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"M12 scaler not found: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"M12 model not found: {model_path}")
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"M12 class map not found: {class_map_path}")

        global_model_config = _load_model_class_map(class_map_path)
        global_scaler = joblib.load(scaler_path)

        base_model = keras.models.load_model(
            model_path,
            custom_objects={"TemporalAttention": TemporalAttention},
        )

        global_lstm_model = build_attention_extractor(base_model)

        log.info(
            "M12 Model + Scaler ready. "
            "Classes: 1=ignored warning, 2/3/4=fail, 5/6=positive actions."
        )

    except Exception as e:
        global_scaler = None
        global_lstm_model = None
        global_model_config = None
        log.error(f"Failed to load M12 model artifacts: {e}")
        raise RuntimeError(f"AI model loading failed: {e}")


# ==========================================================================
# M12 episode resolver
# ==========================================================================
def _safe_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _extract_sign_episodes(df: pd.DataFrame, sign_code: int, max_gap_rows: int = 2):
    """
    Episode = contiguous region where either visible sign or memory sign is active.
    Allows tiny gaps caused by vision/memory flicker.
    """
    sign_type = (
        pd.to_numeric(df["sign_type"], errors="coerce")
        .fillna(0)
        .round()
        .astype(int)
    )
    memory_sign_type = (
        pd.to_numeric(df["memory_sign_type"], errors="coerce")
        .fillna(0)
        .round()
        .astype(int)
    )

    active = ((sign_type == sign_code) | (memory_sign_type == sign_code)).to_numpy()

    episodes = []
    start = None
    last_active = None
    gap = 0

    for i, is_active in enumerate(active):
        if is_active:
            if start is None:
                start = i
            last_active = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap_rows:
                episodes.append((start, last_active))
                start = None
                last_active = None
                gap = 0

    if start is not None:
        episodes.append((start, last_active))

    return episodes


def _episode_window_indices(ep_start: int, ep_end: int, n_windows: int):
    """
    Window i covers rows i : i + MODEL_WINDOW_SIZE.
    Include windows that overlap the sign episode.
    """
    out = []
    for i in range(n_windows):
        w_start = i
        w_end = i + MODEL_WINDOW_SIZE - 1
        if w_start <= ep_end and w_end >= ep_start:
            out.append(i)
    return out


def _episode_stats(df: pd.DataFrame, start: int, end: int) -> dict:
    rows = df.iloc[start:end + 1]

    speed = (
        pd.to_numeric(rows["speed_kmh"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    car_distance = (
        pd.to_numeric(rows["car_distance"], errors="coerce")
        .fillna(99.0)
        .to_numpy(dtype=float)
    )
    car_static = (
        pd.to_numeric(rows["car_static_score"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    car_motion_x = (
        pd.to_numeric(rows["car_motion_x"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    car_motion_y = (
        pd.to_numeric(rows["car_motion_y"], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    stopped_mask = speed <= 2.5

    return {
        "start_time": _safe_float(rows.iloc[0].get("time_seconds")),
        "end_time": _safe_float(rows.iloc[-1].get("time_seconds")),
        "min_speed": float(np.min(speed)) if len(speed) else 99.0,
        "max_speed": float(np.max(speed)) if len(speed) else 0.0,
        "first_speed": float(speed[0]) if len(speed) else 0.0,
        "last_speed": float(speed[-1]) if len(speed) else 0.0,
        "speed_drop": float(max(0.0, speed[0] - np.min(speed))) if len(speed) else 0.0,
        "stopped_rows": int(stopped_mask.sum()),
        "has_full_stop": bool(stopped_mask.sum() >= 3),
        "min_car_distance": float(np.min(car_distance)) if len(car_distance) else 99.0,
        "has_dynamic_vehicle": bool(
            np.any(
                (car_distance < 30.0)
                & (
                    (car_static < 0.65)
                    | (np.abs(car_motion_x) > 0.03)
                    | (np.abs(car_motion_y) > 0.03)
                )
            )
        ),
    }


def _best_class_window(predictions, window_indices, class_id: int):
    if not window_indices:
        return None

    best_i = max(window_indices, key=lambda i: float(predictions[i][class_id]))
    return {
        "window_idx": int(best_i),
        "confidence": float(predictions[best_i][class_id]),
    }


def _make_xai_event(
    event_type: str,
    class_id: int,
    window_idx: int,
    sample_idx: int,
    combined_df: pd.DataFrame,
    attention_weights,
):
    peak_frame = int(np.argmax(attention_weights[window_idx]))
    row = combined_df.iloc[min(sample_idx, len(combined_df) - 1)]

    return {
        "timestamp_sec": _safe_float(row.get("time_seconds")),
        "event_type": event_type,
        "class_id": int(class_id),
        "class_label": MODEL_CLASS_NAMES.get(int(class_id), f"Class {class_id}"),
        "decisive_frame_in_window": peak_frame,
        "attention_score": float(np.max(attention_weights[window_idx])),
        "attention_array": attention_weights[window_idx].tolist(),
    }


def run_m12_episode_scoring(
    combined_df: pd.DataFrame,
    test_id: str,
    test_export_path: str,
) -> dict:
    """
    M12 scoring:
    - Run all windows.
    - Keep decision_log per window.
    - Resolve PASS/FAIL by sign episodes, not by single early windows.
    """
    X_model_df = prepare_m12_feature_frame(combined_df)
    X_raw = X_model_df.values

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
        }

    X_scaled = global_scaler.transform(X_raw)

    X_windows = np.array([
        X_scaled[i:i + MODEL_WINDOW_SIZE]
        for i in range(len(X_scaled) - MODEL_WINDOW_SIZE + 1)
    ])

    update_progress(test_id, 85, "Running M12 LSTM...")

    batch_size = 32
    num_batches = max(1, (len(X_windows) + batch_size - 1) // batch_size)

    preds_list = []
    attn_list = []

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(X_windows))
        batch = X_windows[start:end]

        preds, attn = global_lstm_model.predict(
            batch,
            batch_size=batch_size,
            verbose=0,
        )

        preds_list.append(preds)
        attn_list.append(attn)

        pct = 85 + int(10 * (batch_idx + 1) / num_batches)
        update_progress(test_id, pct, f"M12 batch {batch_idx + 1}/{num_batches}")

    predictions = np.concatenate(preds_list, axis=0)
    attention_weights = np.concatenate(attn_list, axis=0)

    predicted_classes = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    # Smooth only for reporting/warnings. Episode resolver also checks raw probabilities.
    smoothed = []
    smooth_window = 5
    for i in range(len(predicted_classes)):
        end = min(i + smooth_window, len(predicted_classes))
        votes = predicted_classes[i:end]
        if len(votes) == 0:
            smoothed.append(int(predicted_classes[i]))
            continue

        mode_, _ = stats.mode(votes, keepdims=False)
        smoothed.append(int(mode_))

    decision_log = []

    for i, pc in enumerate(smoothed):
        ts = _safe_float(combined_df.iloc[i].get("time_seconds")) if i < len(combined_df) else 0.0

        pc_int = int(pc)
        raw_pc = int(predicted_classes[i])
        raw_conf = float(confidences[i])
        conf = float(predictions[i][pc_int])

        decision_log.append({
            "timestamp_sec": round(ts, 2),
            "predicted_class": pc_int,
            "predicted_label": MODEL_CLASS_NAMES.get(pc_int, f"Class {pc_int}"),
            "raw_prediction": raw_pc,
            "raw_label": MODEL_CLASS_NAMES.get(raw_pc, f"Class {raw_pc}"),
            "confidence": round(conf, 3),
            "raw_confidence": round(raw_conf, 3),
            "all_probabilities": [round(float(p), 3) for p in predictions[i]],
        })

    update_progress(test_id, 95, "Resolving M12 sign episodes...")

    violation_events = []
    ignored_warning_events = []
    positive_actions = []
    xai_data = {}
    action_sequences = []

    # ------------------------------------------------------------
    # Tailgating warnings — ignored for PASS/FAIL
    # ------------------------------------------------------------
    merge_gap_s = 1.5
    last_tailgating = None

    for i, pc in enumerate(smoothed):
        conf = float(predictions[i][1])
        if int(pc) != 1 or conf < 0.55:
            continue

        ts = _safe_float(combined_df.iloc[i].get("time_seconds")) if i < len(combined_df) else 0.0

        if last_tailgating is None or (ts - last_tailgating["end_t"]) > merge_gap_s:
            ignored_warning_events.append({
                "timestamp_sec": round(ts, 2),
                "type": "Tailgating",
                "class_id": 1,
                "class_label": MODEL_CLASS_NAMES[1],
                "confidence": round(conf, 3),
            })
            last_tailgating = {
                "end_t": ts,
                "event_idx": len(ignored_warning_events) - 1,
                "confidence_max": conf,
            }
        else:
            last_tailgating["end_t"] = ts
            if conf > last_tailgating["confidence_max"]:
                last_tailgating["confidence_max"] = conf
                ignored_warning_events[last_tailgating["event_idx"]].update({
                    "timestamp_sec": round(ts, 2),
                    "confidence": round(conf, 3),
                })

    # ------------------------------------------------------------
    # STOP episodes
    # ------------------------------------------------------------
    for ep_start, ep_end in _extract_sign_episodes(X_model_df, SIGN_STOP):
        stats_ep = _episode_stats(X_model_df, ep_start, ep_end)
        win_idx = _episode_window_indices(ep_start, ep_end, len(predictions))

        best_running = _best_class_window(predictions, win_idx, 2)
        best_fail_yield = _best_class_window(predictions, win_idx, 3)
        best_correct_stop = _best_class_window(predictions, win_idx, 5)
        best_correct_yield = _best_class_window(predictions, win_idx, 6)

        running_conf = best_running["confidence"] if best_running else 0.0
        correct_stop_conf = best_correct_stop["confidence"] if best_correct_stop else 0.0

        # Main M12 fix:
        # early RunningStop inside the same STOP episode is suppressed if the episode
        # later contains physical full stop + strong CorrectStop.
        if stats_ep["has_full_stop"] and correct_stop_conf >= 0.75:
            sample_idx = int(
                X_model_df.iloc[ep_start:ep_end + 1]["speed_kmh"]
                .astype(float)
                .idxmin()
            )

            action = {
                "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                "type": "CorrectStop",
                "class_id": 5,
                "class_label": MODEL_CLASS_NAMES[5],
                "confidence": round(correct_stop_conf, 3),
                "sign_code": SIGN_STOP,
                "sign_distance_m": round(
                    _safe_float(combined_df.iloc[sample_idx].get("memory_sign_distance", 99.0), 99.0),
                    2,
                ),
                "speed_kmh": round(_safe_float(combined_df.iloc[sample_idx].get("speed_kmh")), 2),
            }
            positive_actions.append(action)

            xai_data[f"positive_{len(positive_actions)}_class_5"] = _make_xai_event(
                "POSITIVE_ACTION",
                5,
                best_correct_stop["window_idx"],
                sample_idx,
                combined_df,
                attention_weights,
            )

            # STOP has two obligations:
            #   1) full stop
            #   2) safe junction entry / yield behavior after the stop
            #
            # Therefore, after a confirmed CorrectStop, we still allow a second
            # positive action: CorrectYield. This is intentionally allowed even
            # when the sign is STOP, because after STOP the driver must enter the
            # junction with yield/right-of-way behavior.
            #
            # A FailureToYield after STOP requires stronger evidence and a dynamic
            # vehicle. A CorrectYield after STOP can be accepted with a lower
            # threshold because it is only a positive/explanatory action and does
            # not change PASS to FAIL.
            fail_yield_conf = best_fail_yield["confidence"] if best_fail_yield else 0.0
            correct_yield_conf = best_correct_yield["confidence"] if best_correct_yield else 0.0

            stop_after_yield_failure = (
                stats_ep["has_dynamic_vehicle"]
                and fail_yield_conf >= 0.78
                and fail_yield_conf > max(0.65, correct_yield_conf * 1.15)
            )

            stop_after_correct_yield = (
                best_correct_yield is not None
                and correct_yield_conf >= 0.45
                and not stop_after_yield_failure
                and fail_yield_conf < max(0.78, correct_yield_conf * 1.25)
            )

            if stop_after_yield_failure:
                sample_idx = min(
                    best_fail_yield["window_idx"] + MODEL_WINDOW_SIZE - 1,
                    len(combined_df) - 1,
                )

                violation_events.append({
                    "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                    "type": "Failure to Yield",
                    "class_id": 3,
                    "class_label": MODEL_CLASS_NAMES[3],
                    "confidence": round(fail_yield_conf, 3),
                })

                xai_data[f"event_{len(violation_events)}_class_3"] = _make_xai_event(
                    "VIOLATION",
                    3,
                    best_fail_yield["window_idx"],
                    sample_idx,
                    combined_df,
                    attention_weights,
                )

            elif stop_after_correct_yield:
                sample_idx = min(
                    best_correct_yield["window_idx"] + MODEL_WINDOW_SIZE - 1,
                    len(combined_df) - 1,
                )

                positive_actions.append({
                    "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                    "type": "CorrectYield",
                    "class_id": 6,
                    "class_label": MODEL_CLASS_NAMES[6],
                    "confidence": round(correct_yield_conf, 3),
                    "sign_code": SIGN_STOP,
                    "sign_distance_m": round(
                        _safe_float(combined_df.iloc[sample_idx].get("memory_sign_distance", 99.0), 99.0),
                        2,
                    ),
                    "speed_kmh": round(_safe_float(combined_df.iloc[sample_idx].get("speed_kmh")), 2),
                    "source_policy": "STOP_AFTER_CORRECT_STOP_SAFE_ENTRY",
                })

                xai_data[f"positive_{len(positive_actions)}_class_6"] = _make_xai_event(
                    "POSITIVE_ACTION",
                    6,
                    best_correct_yield["window_idx"],
                    sample_idx,
                    combined_df,
                    attention_weights,
                )

            continue

        # If no full stop happened in the whole STOP episode, RunningStop is valid.
        if not stats_ep["has_full_stop"] and running_conf >= 0.65:
            sample_idx = min(
                best_running["window_idx"] + MODEL_WINDOW_SIZE - 1,
                len(combined_df) - 1,
            )

            violation_events.append({
                "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                "type": "Running Stop",
                "class_id": 2,
                "class_label": MODEL_CLASS_NAMES[2],
                "confidence": round(running_conf, 3),
            })

            xai_data[f"event_{len(violation_events)}_class_2"] = _make_xai_event(
                "VIOLATION",
                2,
                best_running["window_idx"],
                sample_idx,
                combined_df,
                attention_weights,
            )

    # ------------------------------------------------------------
    # YIELD episodes
    # ------------------------------------------------------------
    for ep_start, ep_end in _extract_sign_episodes(X_model_df, SIGN_YIELD):
        stats_ep = _episode_stats(X_model_df, ep_start, ep_end)
        win_idx = _episode_window_indices(ep_start, ep_end, len(predictions))

        best_fail_yield = _best_class_window(predictions, win_idx, 3)
        best_correct_yield = _best_class_window(predictions, win_idx, 6)

        fail_conf = best_fail_yield["confidence"] if best_fail_yield else 0.0
        correct_conf = best_correct_yield["confidence"] if best_correct_yield else 0.0

        yield_behavior_ok = (
            stats_ep["speed_drop"] >= 3.0
            or stats_ep["min_speed"] <= 12.0
            or stats_ep["has_full_stop"]
        )

        # CorrectYield wins over early/ambiguous FailureYield.
        if yield_behavior_ok and correct_conf >= 0.72:
            sample_idx = min(
                best_correct_yield["window_idx"] + MODEL_WINDOW_SIZE - 1,
                len(combined_df) - 1,
            )

            positive_actions.append({
                "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                "type": "CorrectYield",
                "class_id": 6,
                "class_label": MODEL_CLASS_NAMES[6],
                "confidence": round(correct_conf, 3),
                "sign_code": SIGN_YIELD,
                "sign_distance_m": round(
                    _safe_float(combined_df.iloc[sample_idx].get("memory_sign_distance", 99.0), 99.0),
                    2,
                ),
                "speed_kmh": round(_safe_float(combined_df.iloc[sample_idx].get("speed_kmh")), 2),
            })

            xai_data[f"positive_{len(positive_actions)}_class_6"] = _make_xai_event(
                "POSITIVE_ACTION",
                6,
                best_correct_yield["window_idx"],
                sample_idx,
                combined_df,
                attention_weights,
            )

            continue

        if (
            stats_ep["has_dynamic_vehicle"]
            and not yield_behavior_ok
            and fail_conf >= 0.65
            and fail_conf > max(0.55, correct_conf * 1.10)
        ):
            sample_idx = min(
                best_fail_yield["window_idx"] + MODEL_WINDOW_SIZE - 1,
                len(combined_df) - 1,
            )

            violation_events.append({
                "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                "type": "Failure to Yield",
                "class_id": 3,
                "class_label": MODEL_CLASS_NAMES[3],
                "confidence": round(fail_conf, 3),
            })

            xai_data[f"event_{len(violation_events)}_class_3"] = _make_xai_event(
                "VIOLATION",
                3,
                best_fail_yield["window_idx"],
                sample_idx,
                combined_df,
                attention_weights,
            )

    # ------------------------------------------------------------
    # NO ENTRY episodes
    # ------------------------------------------------------------
    for ep_start, ep_end in _extract_sign_episodes(X_model_df, SIGN_NO_ENTRY):
        stats_ep = _episode_stats(X_model_df, ep_start, ep_end)
        win_idx = _episode_window_indices(ep_start, ep_end, len(predictions))

        best_no_entry = _best_class_window(predictions, win_idx, 4)
        no_entry_conf = best_no_entry["confidence"] if best_no_entry else 0.0

        if no_entry_conf >= 0.65 and stats_ep["max_speed"] >= 5.0:
            sample_idx = min(
                best_no_entry["window_idx"] + MODEL_WINDOW_SIZE - 1,
                len(combined_df) - 1,
            )

            violation_events.append({
                "timestamp_sec": round(_safe_float(combined_df.iloc[sample_idx].get("time_seconds")), 2),
                "type": "No Entry Violation",
                "class_id": 4,
                "class_label": MODEL_CLASS_NAMES[4],
                "confidence": round(no_entry_conf, 3),
            })

            xai_data[f"event_{len(violation_events)}_class_4"] = _make_xai_event(
                "VIOLATION",
                4,
                best_no_entry["window_idx"],
                sample_idx,
                combined_df,
                attention_weights,
            )

    # Compact markers in decision log.
    for pa in positive_actions:
        decision_log.append({
            "timestamp_sec": pa["timestamp_sec"],
            "predicted_class": pa["class_id"],
            "event_type": "POSITIVE_ACTION",
            "subtype": pa["type"],
            "confidence": pa["confidence"],
            "sign_code": pa["sign_code"],
            "sign_distance_m": pa["sign_distance_m"],
            "speed_kmh": pa["speed_kmh"],
        })

    for ve in violation_events:
        decision_log.append({
            "timestamp_sec": ve["timestamp_sec"],
            "predicted_class": ve["class_id"],
            "event_type": "VIOLATION",
            "subtype": ve["type"],
            "confidence": ve["confidence"],
        })

    decision_log.sort(key=lambda x: x["timestamp_sec"])

    violations_detected = sorted({int(v["class_id"]) for v in violation_events})
    ignored_warning_codes = sorted({1 for _ in ignored_warning_events})

    mistakes_count = len(violation_events)
    result = "FAIL" if mistakes_count > 0 else "PASS"
    passed = result == "PASS"
    final_grade = 100 if passed else 0

    thought_payload = {
        "test_id": test_id,
        "exported_at": datetime.now().isoformat(),
        "model_id": MODEL_ID,
        "feature_names": MODEL_FEATURES,
        "class_names": {str(k): v for k, v in MODEL_CLASS_NAMES.items()},
        "violation_classes": sorted(VIOLATION_CLASSES),
        "positive_action_classes": sorted(POSITIVE_ACTION_CLASSES),
        "fail_classes": sorted(FAIL_CLASSES),
        "ignored_warning_classes": sorted(IGNORED_WARNING_CLASSES),
        "result": result,
        "passed": passed,
        "mistakes_count": mistakes_count,
        "mistake_codes": violations_detected,
        "ignored_warning_codes": ignored_warning_codes,
        "ignored_warning_events_count": len(ignored_warning_events),
        "ignored_warning_events": ignored_warning_events,
        "grade": final_grade,
        "decision_log": decision_log,
        "xai_explanations": xai_data,
        "action_sequences": action_sequences,
        "positive_actions": positive_actions,
        "episode_policy": (
            "M12 sign episode resolver: CorrectStop suppresses early "
            "RunningStop in same STOP episode; STOP can also produce CorrectYield "
            "for safe post-stop junction entry."
        ),
    }

    with open(os.path.join(test_export_path, "3_model_thought.json"), "w", encoding="utf-8") as f:
        json.dump(thought_payload, f, ensure_ascii=False, indent=4)

    return {
        "result": result,
        "passed": passed,
        "grade": final_grade,
        "mistakes_count": mistakes_count,
        "violations_detected": violations_detected,
        "ignored_warning_codes": ignored_warning_codes,
        "ignored_warning_events": ignored_warning_events,
        "decision_log": decision_log,
        "xai_data": xai_data,
        "action_sequences": action_sequences,
        "positive_actions": positive_actions,
        "windows_analyzed": len(X_windows),
    }


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

        # Integrity validation BEFORE heavy work.
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

        # Sensor processing.
        update_progress(test_id, 5, "Processing sensor data...")
        sensor_df = await asyncio.to_thread(process_sensor_json, temp_json)
        sensor_df.to_csv(
            os.path.join(test_export_path, "1_raw_sensors.csv"),
            index=False,
        )

        # YOLO.
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

        # Vector build.
        update_progress(test_id, 80, "Building feature vector...")
        combined_df = build_feature_vector(sensor_df, video_events)
        combined_df.to_csv(
            os.path.join(test_export_path, "2_feature_vector.csv"),
            index=False,
        )

        # M12 LSTM + episode scoring.
        score = await asyncio.to_thread(
            run_m12_episode_scoring,
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
        # Force tester_email from token; ignore any client-supplied value.
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


# ==========================================================================
# Predictions
# ==========================================================================
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

    cursor = db.db["tests"].find({
        "student_id": student_id,
        "tester_email": tester["email"],
    }).sort("saved_at", 1)

    tests = []
    async for doc in cursor:
        tests.append(doc)

    if not tests:
        return {
            "student_id": student_id,
            "student_name": student["name"],
            "tests_count": 0,
            "predicted_success_rate": None,
            "confidence": "no_data",
            "trend": "unknown",
            "average_grade": 0,
            "last_grades": [],
            "weakest_violations": [],
            "recommendation": "Run at least one test to get predictions",
        }

    grades = [t.get("grade", 0) for t in tests]
    avg_grade = sum(grades) / len(grades)

    if len(grades) >= 3:
        recent = grades[-3:]
        older = grades[:-3] if len(grades) > 3 else grades[:-1]
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older) if older else recent_avg
        delta = recent_avg - older_avg

        if delta > 5:
            trend = "improving"
        elif delta < -5:
            trend = "declining"
        else:
            trend = "stable"
    else:
        trend = "insufficient_data"

    weights = [i + 1 for i in range(len(grades))]
    weighted_avg = sum(g * w for g, w in zip(grades, weights)) / sum(weights)

    adj = {
        "improving": 5,
        "stable": 0,
        "declining": -5,
    }.get(trend, 0)

    predicted = round(max(0, min(100, weighted_avg + adj)))

    if len(tests) >= 5:
        confidence = "high"
    elif len(tests) >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    counter = {}
    for t in tests:
        for code in t.get("violations_codes", []):
            counter[code] = counter.get(code, 0) + 1

    weakest = sorted(counter.items(), key=lambda x: -x[1])[:3]

    if predicted >= 85:
        rec = "Excellent! Ready for the official driving test."
    elif predicted >= 70:
        rec = "Good progress. Focus on consistency."
    elif predicted >= 50:
        rec = "Improvement needed. Practice common scenarios."
    else:
        rec = "Significant practice required."

    return {
        "student_id": student_id,
        "student_name": student["name"],
        "tests_count": len(tests),
        "predicted_success_rate": predicted,
        "confidence": confidence,
        "trend": trend,
        "average_grade": round(avg_grade, 1),
        "last_grades": grades[-5:],
        "weakest_violations": [
            {"code": int(c), "count": int(n)}
            for c, n in weakest
        ],
        "recommendation": rec,
    }


@router.get("/predictions")
async def predict_all_students(tester: dict = Depends(get_current_tester)):
    cursor = db.db["students"].find({
        "tester_email": tester["email"],
    })

    out = []

    async for student in cursor:
        sid = student["student_id"]

        tcursor = db.db["tests"].find({
            "student_id": sid,
            "tester_email": tester["email"],
        }).sort("saved_at", 1)

        grades = []
        counter = {}

        async for t in tcursor:
            grades.append(t.get("grade", 0))
            for code in t.get("violations_codes", []):
                counter[code] = counter.get(code, 0) + 1

        if not grades:
            out.append({
                "student_id": sid,
                "student_name": student["name"],
                "tests_count": 0,
                "predicted_success_rate": None,
                "trend": "unknown",
                "average_grade": 0,
                "last_grade": None,
                "top_violations": [],
            })
            continue

        weights = [i + 1 for i in range(len(grades))]
        weighted_avg = sum(g * w for g, w in zip(grades, weights)) / sum(weights)

        if len(grades) >= 3:
            recent_avg = sum(grades[-3:]) / 3
            older = grades[:-3] if len(grades) > 3 else grades[:-1]
            older_avg = sum(older) / len(older) if older else recent_avg
            delta = recent_avg - older_avg

            if delta > 5:
                trend = "improving"
            elif delta < -5:
                trend = "declining"
            else:
                trend = "stable"

            adj = {
                "improving": 5,
                "stable": 0,
                "declining": -5,
            }[trend]
        else:
            trend = "insufficient_data"
            adj = 0

        predicted = round(max(0, min(100, weighted_avg + adj)))
        top = sorted(counter.items(), key=lambda x: -x[1])[:2]

        out.append({
            "student_id": sid,
            "student_name": student["name"],
            "tests_count": len(grades),
            "predicted_success_rate": predicted,
            "trend": trend,
            "average_grade": round(sum(grades) / len(grades), 1),
            "last_grade": grades[-1],
            "top_violations": [
                {"code": int(c), "count": int(n)}
                for c, n in top
            ],
        })

    out.sort(key=lambda p: (
        p["predicted_success_rate"] is None,
        p["predicted_success_rate"] or 0,
    ))

    return out


@router.get("/predictions/{tester_email}")
async def predict_all_students_legacy(
    tester_email: str,
    tester: dict = Depends(get_current_tester),
):
    return await predict_all_students(tester)
