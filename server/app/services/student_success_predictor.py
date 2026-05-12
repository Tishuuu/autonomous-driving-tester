"""
Student Success Predictor v2.3
==============================

Runtime service for the Auto Tester Predictions screen.

This is the second model in the project: a student-history sequence model.
It predicts whether a student is likely to pass if they take a test now,
based on previous saved test summaries from MongoDB.

It does NOT use IF rules for prediction. Rules are used only for feature
extraction and for explaining recurring historical weaknesses.

Artifacts expected in: server/app/ai_models/
    student_success_model.keras
    student_success_scaler.pkl
    student_success_class_map.json

Supported model outputs:
    pass_probability: shape (1, 1)
    risk_probs:       shape (1, 4), optional
    profile_head:      shape (1, 8), ignored at runtime
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


MODEL_ID = "STUDENT_SUCCESS_LSTM"
MODEL_FAMILY = "student_success_sequence_bilstm"
MODEL_VERSION = "v2.3-sharp-profile-synthetic-risk"

SEQUENCE_LEN = 10
PAD_VALUE = -1000000.0

RISK_CODES = [1, 2, 3, 4]
RISK_LABELS = {
    1: "Tailgating",
    2: "Running Stop",
    3: "Failure to Yield",
    4: "No Entry Violation",
}

# v2.3: calibration is intentionally disabled at runtime.
# Small real calibration caused near-constant outputs in v2.2, so we keep raw model probability.
RUNTIME_CALIBRATION_ENABLED = False

FEATURE_NAMES = [
    "grade_norm",
    "passed_flag",
    "mistakes_norm",
    "violation_events_norm",
    "ignored_warning_events_norm",
    "windows_norm",
    "running_stop_count_norm",
    "failure_to_yield_count_norm",
    "no_entry_count_norm",
    "tailgating_warning_count_norm",
    "positive_actions_norm",
    "correct_stop_count_norm",
    "correct_yield_count_norm",
    "days_since_prev_norm",
    "test_index_norm",
    "recent_position_norm",
    "recent_pass_rate",
    "recent_mistake_rate",
    "recent_running_stop_rate",
    "recent_yield_failure_rate",
    "recent_noentry_rate",
    "recent_tailgating_rate",
    "trend_grade_norm",
    "trend_mistakes_norm",
    "streak_pass_norm",
    "streak_fail_norm",
    "cumulative_pass_rate",
    "cumulative_mistake_rate",
    "last3_pass_rate",
    "last3_mistake_rate",
    "tests_since_running_stop_norm",
    "tests_since_failure_yield_norm",
    "tests_since_no_entry_norm",
    "tests_since_tailgating_norm",
    "repeated_same_violation_norm",
    "positive_trend_norm",
    "days_avg_last3_norm",
]


@tf.keras.utils.register_keras_serializable(package="student_success", name="SequenceAttentionPool")
class SequenceAttentionPool(layers.Layer):
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
        return tf.reduce_sum(inputs * tf.expand_dims(weights, -1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        return super().get_config()


CUSTOM_OBJECTS = {
    "SequenceAttentionPool": SequenceAttentionPool,
    "student_success>SequenceAttentionPool": SequenceAttentionPool,
    "student_success.SequenceAttentionPool": SequenceAttentionPool,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None
    return None


def _test_datetime(doc: Dict[str, Any]) -> datetime:
    return (
        _parse_dt(doc.get("saved_at"))
        or _parse_dt(doc.get("test_date"))
        or _parse_dt(doc.get("start_time"))
        or datetime(1970, 1, 1)
    )


def sort_tests_chronologically(tests: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(list(tests), key=_test_datetime)


def _extract_event_type(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(
        item.get("type")
        or item.get("behavior_name")
        or item.get("class_label")
        or item.get("event_key")
        or ""
    )


def _collect_codes(doc: Dict[str, Any]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return (violation_counts, warning_counts)."""
    violation_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    warning_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for key in ("violations_codes", "violation_codes", "mistake_codes"):
        for code in _as_list(doc.get(key)):
            try:
                c = int(code)
            except Exception:
                continue
            if c in violation_counts:
                violation_counts[c] += 1

    for key in ("ignored_warning_codes", "warning_codes"):
        for code in _as_list(doc.get(key)):
            try:
                c = int(code)
            except Exception:
                continue
            if c in warning_counts:
                warning_counts[c] += 1

    for key in ("violation_events", "mistakes"):
        for item in _as_list(doc.get(key)):
            typ = _extract_event_type(item)
            if "Tailgating" in typ:
                warning_counts[1] += 1
            if "RunningStop" in typ or "Running Stop" in typ:
                violation_counts[2] += 1
            if "FailureToYield" in typ or "Failure to Yield" in typ:
                violation_counts[3] += 1
            if "NoEntry" in typ or "No Entry" in typ:
                violation_counts[4] += 1

    for key in ("ignored_warning_events", "warnings"):
        for item in _as_list(doc.get(key)):
            typ = _extract_event_type(item)
            if "Tailgating" in typ:
                warning_counts[1] += 1

    return violation_counts, warning_counts


def _count_positive_actions(doc: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return (positive_count, correct_stop_count, correct_yield_count)."""
    count = 0
    stop_count = 0
    yield_count = 0

    for key in ("positive_actions", "action_sequences"):
        for item in _as_list(doc.get(key)):
            typ = _extract_event_type(item)
            if not typ:
                continue
            if "CorrectStop" in typ or "Correct Stop" in typ:
                count += 1
                stop_count += 1
            if "CorrectYield" in typ or "Correct Yield" in typ:
                count += 1
                yield_count += 1

    # Some exports store counts directly.
    direct_count = _safe_int(doc.get("positive_actions_count"), 0)
    if direct_count > count:
        count = direct_count

    return count, stop_count, yield_count


def _basic_counts(doc: Dict[str, Any]) -> Dict[str, float]:
    violation_counts, warning_counts = _collect_codes(doc)
    positive_count, stop_count, yield_count = _count_positive_actions(doc)

    grade = _safe_float(doc.get("grade"), 100.0 if doc.get("passed") else 0.0)
    passed = bool(doc.get("passed", grade >= 80.0))
    mistakes = _safe_float(doc.get("mistakes_count"), _safe_float(doc.get("violation_events_count"), 0.0))
    violation_events = _safe_float(doc.get("violation_events_count"), mistakes)
    ignored_events = _safe_float(doc.get("ignored_warning_events_count"), sum(warning_counts.values()))
    windows = _safe_float(doc.get("windows_analyzed"), _safe_float(doc.get("episode_rows_analyzed"), 0.0))

    return {
        "grade": grade,
        "passed": 1.0 if passed else 0.0,
        "mistakes": mistakes,
        "violation_events": violation_events,
        "ignored_events": ignored_events,
        "windows": windows,
        "running_stop": float(violation_counts[2]),
        "failure_yield": float(violation_counts[3]),
        "no_entry": float(violation_counts[4]),
        "tailgating": float(warning_counts[1] + violation_counts[1]),
        "positive_count": float(positive_count),
        "correct_stop": float(stop_count),
        "correct_yield": float(yield_count),
    }


def _streak(values: List[float], target: float) -> int:
    n = 0
    for value in reversed(values):
        if float(value) == float(target):
            n += 1
        else:
            break
    return n


def _tests_since(counts_list: List[Dict[str, float]], key: str) -> int:
    if not counts_list:
        return 10
    for idx, counts in enumerate(reversed(counts_list)):
        if counts.get(key, 0.0) > 0:
            return idx
    return 10


def _slope(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float32)
    y = np.asarray(values, dtype=np.float32)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


def _risk_percentile(raw_value: float, quantiles: Any) -> int:
    try:
        vals = np.asarray(quantiles, dtype=float)
        if vals.size == 0:
            return int(round(float(raw_value) * 100.0))
        idx = int(np.searchsorted(vals, float(raw_value), side="right") - 1)
        return int(np.clip(idx, 0, 100))
    except Exception:
        return int(round(float(raw_value) * 100.0))


def test_doc_to_feature(doc: Dict[str, Any], history_before_or_including: List[Dict[str, Any]], index: int, total: int) -> np.ndarray:
    """Convert one saved test into a per-test feature vector.

    history_before_or_including must contain tests up to and including doc.
    This prevents future leakage and matches the Colab v2.3 feature schema.
    """
    counts = _basic_counts(doc)

    cur_dt = _test_datetime(doc)
    prev_doc = history_before_or_including[-2] if len(history_before_or_including) >= 2 else None
    prev_dt = _test_datetime(prev_doc) if prev_doc is not None else None
    if prev_dt is None or cur_dt.year == 1970 or prev_dt.year == 1970:
        days_since_prev = 0.0
    else:
        days_since_prev = max(0.0, min(90.0, (cur_dt - prev_dt).total_seconds() / 86400.0))

    all_counts = [_basic_counts(t) for t in history_before_or_including]
    recent_counts = all_counts[-5:]
    last3_counts = all_counts[-3:]

    recent_pass_rate = float(np.mean([c["passed"] for c in recent_counts])) if recent_counts else 0.0
    recent_mistake_rate = float(np.mean([c["mistakes"] for c in recent_counts])) if recent_counts else 0.0
    recent_running_stop_rate = float(np.mean([c["running_stop"] for c in recent_counts])) if recent_counts else 0.0
    recent_yield_failure_rate = float(np.mean([c["failure_yield"] for c in recent_counts])) if recent_counts else 0.0
    recent_noentry_rate = float(np.mean([c["no_entry"] for c in recent_counts])) if recent_counts else 0.0
    recent_tailgating_rate = float(np.mean([c["tailgating"] for c in recent_counts])) if recent_counts else 0.0

    cumulative_pass_rate = float(np.mean([c["passed"] for c in all_counts])) if all_counts else 0.0
    cumulative_mistake_rate = float(np.mean([c["mistakes"] for c in all_counts])) if all_counts else 0.0
    last3_pass_rate = float(np.mean([c["passed"] for c in last3_counts])) if last3_counts else 0.0
    last3_mistake_rate = float(np.mean([c["mistakes"] for c in last3_counts])) if last3_counts else 0.0

    if len(all_counts) >= 2:
        prev = all_counts[-2]
        trend_grade = (counts["grade"] - prev["grade"]) / 100.0
        trend_mistakes = (prev["mistakes"] - counts["mistakes"]) / 6.0  # positive means improving
    else:
        trend_grade = 0.0
        trend_mistakes = 0.0

    pass_history = [c["passed"] for c in recent_counts]
    pass_streak = _streak(pass_history, 1.0)
    fail_streak = _streak(pass_history, 0.0)

    recent_violation_rates = [
        recent_running_stop_rate,
        recent_yield_failure_rate,
        recent_noentry_rate,
        recent_tailgating_rate,
    ]
    repeated_same_violation = max(recent_violation_rates) if recent_violation_rates else 0.0
    positive_trend = _slope([c["positive_count"] for c in last3_counts]) / 3.0 if len(last3_counts) >= 2 else 0.0

    days_values: List[float] = []
    ordered = history_before_or_including
    for j in range(max(1, len(ordered) - 2), len(ordered)):
        if j <= 0:
            continue
        d0 = _test_datetime(ordered[j - 1])
        d1 = _test_datetime(ordered[j])
        if d0.year != 1970 and d1.year != 1970:
            days_values.append(max(0.0, min(90.0, (d1 - d0).total_seconds() / 86400.0)))
    days_avg_last3 = float(np.mean(days_values)) if days_values else days_since_prev

    denom = max(1, total - 1)
    recent_position = index / denom

    values = [
        np.clip(counts["grade"] / 100.0, 0.0, 1.0),
        counts["passed"],
        np.clip(counts["mistakes"] / 6.0, 0.0, 1.0),
        np.clip(counts["violation_events"] / 6.0, 0.0, 1.0),
        np.clip(counts["ignored_events"] / 6.0, 0.0, 1.0),
        np.clip(counts["windows"] / 512.0, 0.0, 1.0),
        np.clip(counts["running_stop"] / 3.0, 0.0, 1.0),
        np.clip(counts["failure_yield"] / 3.0, 0.0, 1.0),
        np.clip(counts["no_entry"] / 3.0, 0.0, 1.0),
        np.clip(counts["tailgating"] / 3.0, 0.0, 1.0),
        np.clip(counts["positive_count"] / 5.0, 0.0, 1.0),
        np.clip(counts["correct_stop"] / 3.0, 0.0, 1.0),
        np.clip(counts["correct_yield"] / 3.0, 0.0, 1.0),
        np.clip(days_since_prev / 30.0, 0.0, 3.0),
        np.clip((index + 1) / 20.0, 0.0, 1.0),
        np.clip(recent_position, 0.0, 1.0),
        np.clip(recent_pass_rate, 0.0, 1.0),
        np.clip(recent_mistake_rate / 6.0, 0.0, 1.0),
        np.clip(recent_running_stop_rate / 3.0, 0.0, 1.0),
        np.clip(recent_yield_failure_rate / 3.0, 0.0, 1.0),
        np.clip(recent_noentry_rate / 3.0, 0.0, 1.0),
        np.clip(recent_tailgating_rate / 3.0, 0.0, 1.0),
        np.clip((trend_grade + 1.0) / 2.0, 0.0, 1.0),
        np.clip((trend_mistakes + 1.0) / 2.0, 0.0, 1.0),
        np.clip(pass_streak / 5.0, 0.0, 1.0),
        np.clip(fail_streak / 5.0, 0.0, 1.0),
        np.clip(cumulative_pass_rate, 0.0, 1.0),
        np.clip(cumulative_mistake_rate / 6.0, 0.0, 1.0),
        np.clip(last3_pass_rate, 0.0, 1.0),
        np.clip(last3_mistake_rate / 6.0, 0.0, 1.0),
        np.clip(_tests_since(all_counts, "running_stop") / 10.0, 0.0, 1.0),
        np.clip(_tests_since(all_counts, "failure_yield") / 10.0, 0.0, 1.0),
        np.clip(_tests_since(all_counts, "no_entry") / 10.0, 0.0, 1.0),
        np.clip(_tests_since(all_counts, "tailgating") / 10.0, 0.0, 1.0),
        np.clip(repeated_same_violation / 3.0, 0.0, 1.0),
        np.clip((positive_trend + 1.0) / 2.0, 0.0, 1.0),
        np.clip(days_avg_last3 / 30.0, 0.0, 3.0),
    ]
    return np.asarray(values, dtype=np.float32)


def tests_to_sequence(tests: List[Dict[str, Any]], scaler: Any, sequence_len: int = SEQUENCE_LEN) -> Tuple[np.ndarray, np.ndarray, int]:
    ordered = sort_tests_chronologically(tests)
    total = len(ordered)
    selected = ordered[-sequence_len:]

    raw = []
    offset = max(0, total - len(selected))
    for local_idx, doc in enumerate(selected):
        global_idx = offset + local_idx
        history = ordered[: global_idx + 1]
        raw.append(test_doc_to_feature(doc, history, global_idx, total))

    arr = np.zeros((sequence_len, len(FEATURE_NAMES)), dtype=np.float32)
    mask = np.zeros((sequence_len,), dtype=np.float32)

    if raw:
        raw_arr = np.vstack(raw).astype(np.float32)
        scaled = scaler.transform(raw_arr).astype(np.float32)
        arr[: len(raw)] = scaled
        mask[: len(raw)] = 1.0

    arr[mask == 0] = PAD_VALUE
    return arr[None, ...], mask, total


def summarize_history(tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    ordered = sort_tests_chronologically(tests)
    counts_list = [_basic_counts(t) for t in ordered]
    grades = [c["grade"] for c in counts_list]
    pass_flags = [bool(c["passed"]) for c in counts_list]

    counter: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
    for t in ordered:
        v, w = _collect_codes(t)
        for code in counter:
            counter[code] += int(v.get(code, 0))
        counter[1] += int(w.get(1, 0))

    if len(grades) >= 3:
        recent_avg = float(np.mean(grades[-3:]))
        older = grades[:-3] if len(grades) > 3 else grades[:-1]
        older_avg = float(np.mean(older)) if older else recent_avg
        delta = recent_avg - older_avg
        if delta > 5:
            trend = "improving"
        elif delta < -5:
            trend = "declining"
        else:
            trend = "stable"
    elif len(grades) > 0:
        trend = "insufficient_data"
    else:
        trend = "unknown"

    weakest = sorted([(c, n) for c, n in counter.items() if n > 0], key=lambda kv: (-kv[1], kv[0]))[:4]
    return {
        "tests_count": len(ordered),
        "average_grade": round(float(np.mean(grades)), 1) if grades else 0,
        "last_grade": round(float(grades[-1]), 1) if grades else None,
        "last_grades": [round(float(g), 1) for g in grades[-5:]],
        "historical_pass_rate": round(100.0 * float(np.mean(pass_flags)), 1) if pass_flags else None,
        "trend": trend,
        "top_violations": [{"code": int(c), "count": int(n), "label": RISK_LABELS.get(int(c), f"Code {c}")} for c, n in weakest[:2]],
        "weakest_violations": [{"code": int(c), "count": int(n), "label": RISK_LABELS.get(int(c), f"Code {c}")} for c, n in weakest],
        "violation_counts": {str(int(c)): int(n) for c, n in counter.items()},
    }


def recommendation_from_probability(prob: Optional[float], tests_count: int, top_risk: Optional[Dict[str, Any]] = None) -> str:
    if prob is None:
        return "Run more tests to enable the student prediction model."
    pct = prob * 100.0
    risk_part = ""
    if top_risk and top_risk.get("risk") is not None:
        risk_part = f" Main risk: {top_risk.get('label')}."
    if tests_count < 2:
        return "Prediction is based on limited history. Add more saved tests for a stronger forecast." + risk_part
    if pct >= 85:
        return "High readiness signal. Keep consistency and avoid recurring mistakes." + risk_part
    if pct >= 70:
        return "Good readiness signal. Focus practice on the recurring weak scenarios." + risk_part
    if pct >= 50:
        return "Moderate readiness. Add targeted practice before the official test." + risk_part
    return "Low readiness signal. Continue structured practice before the official test." + risk_part


@dataclass
class StudentPredictor:
    models_dir: str
    model: Any = None
    scaler: Any = None
    class_map: Optional[dict] = None
    load_error: Optional[str] = None

    def artifact_paths(self) -> Dict[str, str]:
        return {
            "model": os.path.join(self.models_dir, "student_success_model.keras"),
            "scaler": os.path.join(self.models_dir, "student_success_scaler.pkl"),
            "class_map": os.path.join(self.models_dir, "student_success_class_map.json"),
        }

    def load(self) -> None:
        if self.model is not None and self.scaler is not None and self.class_map is not None:
            return

        paths = self.artifact_paths()
        try:
            for label, path in paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing student prediction artifact: {label} -> {path}")

            with open(paths["class_map"], "r", encoding="utf-8") as f:
                class_map = json.load(f)

            if class_map.get("model_id") != MODEL_ID:
                raise ValueError(f"Expected model_id={MODEL_ID}, got {class_map.get('model_id')}")
            if class_map.get("feature_names") != FEATURE_NAMES:
                raise ValueError("Student prediction feature schema mismatch")
            if int(class_map.get("sequence_len", SEQUENCE_LEN)) != SEQUENCE_LEN:
                raise ValueError("Student prediction sequence length mismatch")

            with open(paths["scaler"], "rb") as f:
                scaler = pickle.load(f)

            model = keras.models.load_model(paths["model"], custom_objects=CUSTOM_OBJECTS, compile=False, safe_mode=False)

            self.model = model
            self.scaler = scaler
            self.class_map = class_map
            self.load_error = None
        except Exception as exc:
            self.model = None
            self.scaler = None
            self.class_map = None
            self.load_error = str(exc)

    def is_ready(self) -> bool:
        self.load()
        return self.model is not None and self.scaler is not None and self.class_map is not None

    def _unpack_prediction(self, raw: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pass_prob_arr = None
        risk_arr = None

        if isinstance(raw, dict):
            pass_prob_arr = raw.get("pass_probability")
            risk_arr = raw.get("risk_probs")
        elif isinstance(raw, (list, tuple)):
            names = list(getattr(self.model, "output_names", [])) if self.model is not None else []
            out = {name: value for name, value in zip(names, raw)}
            pass_prob_arr = out.get("pass_probability", raw[0] if raw else None)
            risk_arr = out.get("risk_probs", raw[1] if len(raw) > 1 else None)
        else:
            pass_prob_arr = raw

        return np.asarray(pass_prob_arr), None if risk_arr is None else np.asarray(risk_arr)

    def predict_probability(self, tests: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
        self.load()
        if self.model is None or self.scaler is None:
            return None, {"model_status": "missing_model", "load_error": self.load_error}

        if not tests:
            return None, {"model_status": "no_history"}

        X, mask, real_count = tests_to_sequence(tests, self.scaler, SEQUENCE_LEN)
        raw = self.model.predict(X, verbose=0)
        pass_arr, risk_arr = self._unpack_prediction(raw)

        raw_prob = float(np.asarray(pass_arr).reshape(-1)[0])
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        calibration = self.class_map.get("calibration", {}) if self.class_map else {}
        prob = raw_prob
        calibration_available = bool(calibration.get("enabled"))
        if (
            RUNTIME_CALIBRATION_ENABLED
            and calibration_available
            and calibration.get("type") == "platt_logistic_regression"
        ):
            coef = _safe_float(calibration.get("coef"), 1.0)
            intercept = _safe_float(calibration.get("intercept"), 0.0)
            prob = _sigmoid(coef * raw_prob + intercept)
        prob = float(np.clip(prob, 0.0, 1.0))

        pass_threshold = float(self.class_map.get("pass_threshold", 0.5)) if self.class_map else 0.5
        debug: Dict[str, Any] = {
            "model_status": "ok",
            "model_id": MODEL_ID,
            "model_family": MODEL_FAMILY,
            "model_version": self.class_map.get("model_version", MODEL_VERSION) if self.class_map else MODEL_VERSION,
            "sequence_len": SEQUENCE_LEN,
            "history_used": int(real_count),
            "raw_pass_probability": round(raw_prob, 4),
            "calibrated": bool(RUNTIME_CALIBRATION_ENABLED and calibration_available),
            "calibration_available_but_ignored": bool(calibration_available and not RUNTIME_CALIBRATION_ENABLED),
            "pass_threshold": round(pass_threshold, 4),
            "predicted_pass_flag": bool(prob >= pass_threshold),
            "approved_for_runtime": bool(self.class_map.get("approved_for_runtime", False)) if self.class_map else False,
        }

        if risk_arr is not None:
            risks = np.asarray(risk_arr).reshape(-1).astype(float)
            risk_payload = []
            quantiles = self.class_map.get("risk_quantiles", {}) if self.class_map else {}
            risk_labels_raw = self.class_map.get("risk_labels", {}) if self.class_map else {}
            for i, code in enumerate(RISK_CODES):
                value = float(np.clip(risks[i], 0.0, 1.0)) if i < len(risks) else 0.0
                percentile = _risk_percentile(value, quantiles.get(str(code)))
                if percentile >= 75:
                    level = "high"
                elif percentile >= 50:
                    level = "medium"
                else:
                    level = "low"
                label = risk_labels_raw.get(str(code)) or RISK_LABELS.get(int(code), f"Code {code}")
                risk_payload.append({
                    "code": int(code),
                    "label": label,
                    "risk": round(value, 4),
                    "risk_percent": int(percentile),
                    "risk_level": level,
                })
            debug["risk_predictions"] = sorted(risk_payload, key=lambda x: x["risk_percent"], reverse=True)

        return prob, debug

    def predict_payload(self, student_id: str, student_name: str, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = summarize_history(tests)
        prob, debug = self.predict_probability(tests)
        predicted = None if prob is None else int(round(prob * 100.0))

        tests_count = int(summary["tests_count"])
        if prob is None:
            confidence = "no_model" if debug.get("model_status") == "missing_model" else "no_data"
        elif tests_count >= 6:
            confidence = "high"
        elif tests_count >= 3:
            confidence = "medium"
        else:
            confidence = "low"

        risk_predictions = debug.get("risk_predictions", []) if isinstance(debug, dict) else []
        historical_counts = summary.get("violation_counts", {})
        enriched_risks = []
        for item in risk_predictions:
            code = int(item["code"])
            enriched = dict(item)
            enriched["history_count"] = int(historical_counts.get(str(code), 0))
            enriched_risks.append(enriched)

        # If risk head exists, use it for weakest_violations; otherwise use historical fallback.
        weakest = enriched_risks[:4] if enriched_risks else summary["weakest_violations"]
        top_risk = weakest[0] if weakest else None

        payload = {
            "student_id": student_id,
            "student_name": student_name,
            "tests_count": tests_count,
            "predicted_success_rate": predicted,
            "confidence": confidence,
            "trend": summary["trend"],
            "average_grade": summary["average_grade"],
            "last_grade": summary["last_grade"],
            "last_grades": summary["last_grades"],
            "historical_pass_rate": summary["historical_pass_rate"],
            "top_violations": summary["top_violations"],
            "weakest_violations": weakest,
            "risk_predictions": enriched_risks,
            "recommendation": recommendation_from_probability(prob, tests_count, top_risk),
            "prediction_model": debug,
        }
        return payload


_PREDICTOR_CACHE: Dict[str, StudentPredictor] = {}


def get_student_predictor(base_dir: str) -> StudentPredictor:
    models_dir = os.path.join(base_dir, "app", "ai_models")
    predictor = _PREDICTOR_CACHE.get(models_dir)
    if predictor is None:
        predictor = StudentPredictor(models_dir=models_dir)
        _PREDICTOR_CACHE[models_dir] = predictor
    return predictor
