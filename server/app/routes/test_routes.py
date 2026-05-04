import os
import uuid
import asyncio
import json
import hashlib
import shutil
from datetime import datetime
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query
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
# Model features
# ==========================================================================
MODEL_FEATURES = ['speed_kmh', 'jerk', 'car_distance', 'car_ttc',
                  'sign_type', 'sign_distance', 'sign_ttc']


@tf.keras.utils.register_keras_serializable(package='AutonomousDrivingTester')
class TemporalAttention(layers.Layer):
    def __init__(self, units=64, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_attention = return_attention

    def build(self, input_shape):
        d_in = int(input_shape[-1])
        self.W = self.add_weight(name='W', shape=(d_in, self.units),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,),
                                 initializer='zeros', trainable=True)
        self.v = self.add_weight(name='v', shape=(self.units, 1),
                                 initializer='glorot_uniform', trainable=True)
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
        cfg.update({'units': self.units, 'return_attention': self.return_attention})
        return cfg


def build_attention_extractor(trained_model):
    inp = layers.Input(shape=(30, 7), name='input_window')
    bidir_trained = trained_model.get_layer('bidir_lstm_1')
    bidir_clone = layers.Bidirectional(
        layers.LSTM(bidir_trained.forward_layer.units, return_sequences=True))
    H = bidir_clone(inp)
    att_trained = trained_model.get_layer('temporal_attention')
    att_clone = TemporalAttention(units=att_trained.units, return_attention=True)
    context, alpha = att_clone(H)
    avg_pool = layers.GlobalAveragePooling1D()(H)
    merged = layers.Concatenate()([context, avg_pool])
    dense_trained = trained_model.get_layer('dense_1')
    out_trained = trained_model.get_layer('output')
    x = layers.Dense(dense_trained.units, activation='relu', name='dense_1_clone')(merged)
    out = layers.Dense(out_trained.units, activation='softmax', name='output_clone')(x)
    extractor = keras.Model(inp, [out, alpha])
    bidir_clone.set_weights(bidir_trained.get_weights())
    att_clone.set_weights(att_trained.get_weights())
    extractor.get_layer('dense_1_clone').set_weights(dense_trained.get_weights())
    extractor.get_layer('output_clone').set_weights(out_trained.get_weights())
    return extractor


global_scaler = None
global_lstm_model = None


def load_ai_models():
    """Load scaler + LSTM once and fail loudly if either artifact is missing."""
    global global_scaler, global_lstm_model
    if global_scaler is not None and global_lstm_model is not None:
        return

    log.info("Loading Scaler and M8 Model...")
    try:
        import joblib

        scaler_path = os.path.join(BASE_DIR, "app", "ai_models", "global_scaler.pkl")
        model_path = os.path.join(BASE_DIR, "app", "ai_models", "final_model.keras")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"LSTM model not found: {model_path}")

        global_scaler = joblib.load(scaler_path)
        base_model = keras.models.load_model(model_path)
        global_lstm_model = build_attention_extractor(base_model)
        log.info("M8 Model + Scaler ready.")
    except Exception as e:
        global_scaler = None
        global_lstm_model = None
        log.error(f"Failed to load models: {e}")
        raise RuntimeError(f"AI model loading failed: {e}")


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

        # 🆕 Integrity validation BEFORE heavy work
        if video_sha256:
            actual = _sha256_file(temp_vid)
            if actual.lower() != video_sha256.lower():
                raise HTTPException(
                    status_code=422,
                    detail=f"Video integrity failed: expected {video_sha256[:8]}, got {actual[:8]}"
                )
        if sensors_sha256:
            actual = _sha256_file(temp_json)
            if actual.lower() != sensors_sha256.lower():
                raise HTTPException(
                    status_code=422,
                    detail="Sensor JSON integrity failed",
                )
        log.info(f"✅ Integrity verified for {test_id}")

        # Keep a permanent copy of the uploaded video inside this test's export folder.
        # The temp file in the server root is still cleaned in finally.
        saved_input_video = os.path.join(test_export_path, "input_video.mp4")
        try:
            shutil.copyfile(temp_vid, saved_input_video)
            log.info(f"🎞️ Saved original input video: {saved_input_video}")
        except Exception as e:
            log.warning(f"⚠️ Could not save original input video copy: {e}")

        # Sensor processing
        update_progress(test_id, 5, "Processing sensor data...")
        sensor_df = await asyncio.to_thread(process_sensor_json, temp_json)
        sensor_df.to_csv(os.path.join(test_export_path, "1_raw_sensors.csv"), index=False)

        # 🆕 Async YOLO (singleton + semaphore inside vision_service)
        update_progress(test_id, 10, "Running YOLO detection...")
        annotated_video_path = os.path.join(test_export_path, "yolo_annotated.mp4")
        video_events = await analyze_video_for_server(
            temp_vid,
            sensor_df,
            test_id,
            output_video_path=annotated_video_path,
        )

        # Keep the post-filter YOLO/Kalman events too; this is critical for debugging.
        with open(os.path.join(test_export_path, "1b_video_events.json"), "w", encoding="utf-8") as f:
            json.dump(video_events, f, ensure_ascii=False, indent=2)

        # Vector build
        update_progress(test_id, 80, "Building feature vector...")
        combined_df = build_feature_vector(sensor_df, video_events)
        combined_df.to_csv(os.path.join(test_export_path, "2_feature_vector.csv"), index=False)

        # LSTM input prep
        X_raw = combined_df[MODEL_FEATURES].values
        X_scaled = global_scaler.transform(X_raw)
        WINDOW_SIZE = 30
        X_windows = np.array([X_scaled[i:i + WINDOW_SIZE]
                              for i in range(len(X_scaled) - WINDOW_SIZE + 1)])

        decision_log = []
        xai_data = {}
        action_sequences = []
        positive_actions = []
        final_grade = 100
        total_violation_events = 0
        violations_detected = []

        if len(X_windows) > 0:
            update_progress(test_id, 85, "Running M8 LSTM...")
            BATCH_SIZE = 32
            num_batches = max(1, (len(X_windows) + BATCH_SIZE - 1) // BATCH_SIZE)
            preds_list, attn_list = [], []

            for batch_idx in range(num_batches):
                start = batch_idx * BATCH_SIZE
                end = min(start + BATCH_SIZE, len(X_windows))
                batch = X_windows[start:end]
                preds, attn = global_lstm_model.predict(batch, batch_size=BATCH_SIZE, verbose=0)
                preds_list.append(preds)
                attn_list.append(attn)
                pct = 85 + int(10 * (batch_idx + 1) / num_batches)
                update_progress(test_id, pct, f"LSTM batch {batch_idx + 1}/{num_batches}")

            predictions = np.concatenate(preds_list, axis=0)
            attention_weights = np.concatenate(attn_list, axis=0)

            update_progress(test_id, 95, "Analyzing predictions...")
            predicted_classes = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)

            # Mode-smoothing
            smoothed = []
            sw = 5
            for i in range(len(predicted_classes)):
                end = min(i + sw, len(predicted_classes))
                votes = predicted_classes[i:end]
                if len(votes) == 0:
                    smoothed.append(int(predicted_classes[i]))
                    continue
                mode_, _ = stats.mode(votes, keepdims=False)
                smoothed.append(int(mode_))

            # ============================================================
            # Positive action detection — CorrectStop once per contiguous STOP sequence
            # ============================================================
            SIGN_PROXIMITY_M = 8.0
            STOP_SPEED_KMH = 1.0
            STOP_DURATION_S = 0.5
            GRACE_AFTER_SIGN_LOST_S = 1.5

            active_start_idx = None
            last_sign_seen_idx = None
            stop_start_idx = None
            correct_stop_emitted = False

            for i in range(len(combined_df)):
                row = combined_df.iloc[i]
                st = int(row['sign_type'])
                sd = float(row['sign_distance'])
                spd = float(row['speed_kmh'])
                current_ts = float(row['time_seconds'])

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
                    last_seen_ts = float(combined_df.iloc[last_sign_seen_idx]['time_seconds'])
                    if (current_ts - last_seen_ts) <= GRACE_AFTER_SIGN_LOST_S:
                        # YOLO sometimes loses the sign just before the vehicle fully stops.
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

                    stop_start_ts = float(combined_df.iloc[stop_start_idx]['time_seconds'])
                    stopped_duration_s = current_ts - stop_start_ts + 0.1

                    if stopped_duration_s >= STOP_DURATION_S and not correct_stop_emitted:
                        positive_actions.append({
                            "timestamp_sec": round(stop_start_ts, 2),
                            "type": "CorrectStop",
                            "sign_code": 1,
                            "approach_distance_m": round(
                                float(combined_df.iloc[active_start_idx]['sign_distance']), 1
                            ),
                        })
                        correct_stop_emitted = True
                else:
                    # Moving again before reaching STOP_DURATION_S resets only the stop timer.
                    stop_start_idx = None

            # ============================================================
            # 🆕 Temporal merging — replaces in_violation flag
            # ============================================================
            MERGE_GAP_S = 1.5
            MIN_CONF = 0.55
            unique_violation_types = set()
            last_event = {}  # class_id -> {end_t, event_idx, confidence_max}

            for i, pc in enumerate(smoothed):
                ts = float(combined_df.iloc[i]['time_seconds']) if i < len(combined_df) else 0.0
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
                        "timestamp_sec": float(combined_df.iloc[sample_idx]['time_seconds']),
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
                            "timestamp_sec": float(combined_df.iloc[sample_idx]['time_seconds']),
                            "violation_code": pc_int,
                            "decisive_frame_in_window": peak_frame,
                            "attention_score": float(np.max(attention_weights[i])),
                            "attention_array": attention_weights[i].tolist(),
                        }

            violations_detected = sorted(unique_violation_types)

            # Inject positive actions into decision_log
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

            # Action sequences (same logic as before)
            i = 0
            while i < len(smoothed) - 1:
                if smoothed[i] != 0:
                    seq = [int(smoothed[i])]
                    seq_start = float(combined_df.iloc[i]['time_seconds']) if i < len(combined_df) else 0.0
                    j = i + 1
                    last_t = seq_start
                    while j < len(smoothed):
                        t = float(combined_df.iloc[j]['time_seconds']) if j < len(combined_df) else last_t
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

            # Positive actions are useful for the timeline/report, but should not hide violations.
            # When there are no violations, keep a small capped bonus for correct behavior.
            positive_bonus = min(len(positive_actions) * POINTS_PER_POSITIVE, MAX_POSITIVE_BONUS)
            final_grade = base_grade if total_violation_events > 0 else min(100, base_grade + positive_bonus)

            # Export thought log
            with open(os.path.join(test_export_path, "3_model_thought.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "test_id": test_id,
                    "exported_at": datetime.now().isoformat(),
                    "grade": final_grade,
                    "decision_log": decision_log,
                    "xai_explanations": xai_data,
                    "action_sequences": action_sequences,
                    "positive_actions": positive_actions,
                }, f, indent=4)

        update_progress(test_id, 100, "Complete!")
        log.info(f"🎯 Done. Grade: {final_grade} | Events: {total_violation_events} | "
                 f"PositiveActions: {len(positive_actions)} | Tester: {tester['email']}")

        return {
            "status": "success",
            "test_id": test_id,
            "student_id": student_id,
            "tester_email": tester["email"],
            "grade": final_grade,
            "violations_codes": violations_detected,
            "violation_events_count": total_violation_events,
            "windows_analyzed": len(X_windows),
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
        # 🛡️ Force tester_email from token, ignore any client-supplied value
        tester_email = tester["email"]

        student = await db.db["students"].find_one({
            "student_id": payload.student_id,
            "tester_email": tester_email,
        })
        if not student:
            raise HTTPException(
                status_code=404, detail="Student not found for this tester")

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
                    "grade": existing.get("grade", payload.grade),
                }

        record = payload.model_dump()
        record["tester_email"] = tester_email  # force token email
        record["student_name"] = student["name"]
        record["saved_at"] = datetime.now()
        if record.get("test_date") is None:
            record["test_date"] = datetime.now()
        record["status"] = "passed" if payload.grade >= 80 else "failed"

        result = await db.db["tests"].insert_one(record)
        log.info(f"💾 Test saved | {student['name']} ({payload.student_id}) "
                 f"| grade: {payload.grade} | by: {tester_email}")
        return {
            "status": "success",
            "saved_test_id": str(result.inserted_id),
            "student_name": student["name"],
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
    """Replaces legacy /tester/{email} — same data, token-authoritative."""
    cursor = db.db["tests"].find({"tester_email": tester["email"]}).sort("saved_at", -1)
    tests = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        for k in ("saved_at", "test_date"):
            if k in doc and hasattr(doc[k], "isoformat"):
                doc[k] = doc[k].isoformat()
        tests.append(doc)
    return tests


# Legacy alias — ignores path param, uses token
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

    doc = await db.db["tests"].find_one({"_id": oid, "tester_email": tester["email"]})
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
        trend = "improving" if delta > 5 else "declining" if delta < -5 else "stable"
    else:
        trend = "insufficient_data"

    weights = [i + 1 for i in range(len(grades))]
    weighted_avg = sum(g * w for g, w in zip(grades, weights)) / sum(weights)
    adj = {"improving": 5, "stable": 0, "declining": -5}.get(trend, 0)
    predicted = round(max(0, min(100, weighted_avg + adj)))

    confidence = "high" if len(tests) >= 5 else "medium" if len(tests) >= 3 else "low"

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
        "weakest_violations": [{"code": int(c), "count": int(n)} for c, n in weakest],
        "recommendation": rec,
    }


@router.get("/predictions")
async def predict_all_students(tester: dict = Depends(get_current_tester)):
    """Replaces legacy /predictions/{tester_email}."""
    cursor = db.db["students"].find({"tester_email": tester["email"]})
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
            trend = "improving" if delta > 5 else "declining" if delta < -5 else "stable"
            adj = {"improving": 5, "stable": 0, "declining": -5}[trend]
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
            "top_violations": [{"code": int(c), "count": int(n)} for c, n in top],
        })

    out.sort(key=lambda p: (p["predicted_success_rate"] is None,
                            p["predicted_success_rate"] or 0))
    return out


# Legacy alias for /predictions/{tester_email}
@router.get("/predictions/{tester_email}")
async def predict_all_students_legacy(
    tester_email: str,
    tester: dict = Depends(get_current_tester),
):
    return await predict_all_students(tester)