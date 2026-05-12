import asyncio
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_UPDATE_CHECK"] = "False"
from ultralytics import YOLO  # noqa: E402

from app.services.vision_filter import VisionFilter
from app.utils.logger import log

log.info("YOLO STARTING")

# ============================================================================
# Runtime tuning
# ============================================================================
# The previous backend effectively used conf=0.60 because YOLO was called with
# conf=0.60. The current debug proved yield_sign had 0 raw detections at that
# stage, so for signs we must ask YOLO for low-confidence candidates and then
# apply our own class-specific thresholds.
YOLO_PREDICT_CONF = 0.05
YOLO_IMG_SIZE = 960
FRAME_SKIP = 1

# Production-balanced thresholds.
# Keep yield sensitive, but do not allow weak/far no_entry/stop false positives
# to flood the feature vector and LSTM.
YIELD_MIN_CONF = 0.08
STOP_MIN_CONF = 0.14
NO_ENTRY_MIN_CONF = 0.25
VEHICLE_MIN_CONF = 0.60
OTHER_MIN_CONF = 0.45

YIELD_MAX_DISTANCE_M = 35.0
STOP_MAX_DISTANCE_M = 35.0
NO_ENTRY_MAX_DISTANCE_M = 25.0
VEHICLE_MAX_DISTANCE_M = 80.0

# ============================================================================
# Singleton model + GPU semaphore
# ============================================================================
_GLOBAL_YOLO = None
_GPU_SEMAPHORE = asyncio.Semaphore(1)

_GHOST_POOL: dict = {}
_GHOST_TTL_S = 1.5
_GHOST_MAX_DIST_PX = 80


def get_yolo():
    global _GLOBAL_YOLO
    if _GLOBAL_YOLO is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "ai_models", "best (cars).pt"
        )
        _GLOBAL_YOLO = YOLO(model_path)
        _GLOBAL_YOLO.predict(
            np.zeros((640, 640, 3), dtype=np.uint8),
            conf=YOLO_PREDICT_CONF,
            imgsz=640,
            verbose=False,
        )
        log.info("🟢 YOLO singleton initialized & warmed")
        log.info(f"YOLO class names: {_GLOBAL_YOLO.names}")
    return _GLOBAL_YOLO


# ============================================================================
# Label helpers
# ============================================================================
def _norm_label(class_name: str) -> str:
    return str(class_name or "").lower().strip().replace("-", "_").replace(" ", "_")


def _is_stop_like(class_name: str) -> bool:
    c = _norm_label(class_name)
    return "stop" in c


def _is_yield_like(class_name: str) -> bool:
    c = _norm_label(class_name)
    aliases = (
        "yield", "yield_sign", "yieldsign",
        "give_way", "giveway", "giveway_sign", "give_way_sign",
        "give_priority", "priority", "right_of_way", "rightofway",
        "give_right_of_way", "give_rightofway",
        "triangle_sign", "triangular_sign",
    )
    return any(a in c for a in aliases)


def _is_no_entry_like(class_name: str) -> bool:
    c = _norm_label(class_name)
    aliases = (
        "no_entry", "noentry", "no_entry_sign", "noentry_sign",
        "do_not_enter", "donotenter", "dont_enter", "do_not_enter_sign",
        "forbidden_entry", "wrong_way",
    )
    return any(a in c for a in aliases)


def _is_vehicle_like(class_name: str) -> bool:
    c = _norm_label(class_name)
    return any(a in c for a in ("car", "vehicle", "bus", "truck"))


def _is_sign_like(class_name: str) -> bool:
    c = _norm_label(class_name)
    return _is_stop_like(c) or _is_yield_like(c) or _is_no_entry_like(c) or "sign" in c


def _clean_class(class_name: str) -> str:
    if _is_stop_like(class_name):
        return "stop_sign"
    if _is_yield_like(class_name):
        return "yield_sign"
    if _is_no_entry_like(class_name):
        return "no_entry"
    if _is_vehicle_like(class_name):
        return "car"
    return _norm_label(class_name)


def _class_min_conf(class_name: str) -> float:
    if _is_yield_like(class_name):
        return YIELD_MIN_CONF
    if _is_stop_like(class_name):
        return STOP_MIN_CONF
    if _is_no_entry_like(class_name):
        return NO_ENTRY_MIN_CONF
    if _is_vehicle_like(class_name):
        return VEHICLE_MIN_CONF
    return OTHER_MIN_CONF


def _class_physical_height_m(class_name: str) -> float:
    if _is_sign_like(class_name):
        return 0.80
    c = _norm_label(class_name)
    if "bus" in c or "truck" in c:
        return 3.00
    if _is_vehicle_like(class_name):
        return 1.55
    if "person" in c or "pedestrian" in c:
        return 1.70
    return 1.00


def _max_distance_for_class(class_name: str) -> float:
    if _is_yield_like(class_name):
        return YIELD_MAX_DISTANCE_M
    if _is_stop_like(class_name):
        return STOP_MAX_DISTANCE_M
    if _is_no_entry_like(class_name):
        return NO_ENTRY_MAX_DISTANCE_M
    c = _norm_label(class_name)
    if _is_vehicle_like(class_name) or "bus" in c or "truck" in c:
        return VEHICLE_MAX_DISTANCE_M
    if "person" in c or "pedestrian" in c:
        return 35.0
    return 45.0


# ============================================================================
# Kalman tracker
# ============================================================================
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, class_name, confidence: float = 0.0):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = None
        self.class_history = [class_name]
        self.class_name = class_name
        self.hits = 0
        self.distance_est = -1
        self.confidence = float(confidence or 0.0)

    def convert_bbox_to_z(self, bbox):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = bbox[0] + w / 2.0, bbox[1] + h / 2.0
        return np.array([x, y, w * h, w / float(h)]).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        w = np.sqrt(max(0, x[2] * x[3]))
        h = x[2] / w if w > 0 else 0
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0,
                         x[0] + w / 2.0, x[1] + h / 2.0]).reshape((1, 4))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)

    def update(self, bbox, new_class_name, confidence: float = 0.0):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(self.convert_bbox_to_z(bbox))
        if self.id is None and self.hits >= 2:
            self.id = KalmanBoxTracker.count
            KalmanBoxTracker.count += 1
        self.class_history.append(new_class_name)
        if len(self.class_history) > 10:
            self.class_history.pop(0)
        self.class_name = max(set(self.class_history), key=self.class_history.count)
        self.confidence = max(float(self.confidence or 0.0), float(confidence or 0.0))


def compute_iou(box1, box2):
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - intersection
    if union <= 0:
        return 0.0
    return intersection / float(union)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.05):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections))
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = compute_iou(det, trk)
    np.nan_to_num(iou_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches, unmatched = [], []
    for d in range(len(detections)):
        if d not in row_ind:
            unmatched.append(d)
        else:
            t = col_ind[np.where(row_ind == d)[0][0]]
            if iou_matrix[d, t] < iou_threshold:
                unmatched.append(d)
            else:
                matches.append(np.array([d, t]))
    return (np.array(matches) if matches else np.empty((0, 2), dtype=int)), np.array(unmatched)


def _deduplicate_detections(raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Class-aware NMS/dedup.

    Same class: keep highest confidence.
    Different sign classes on the same physical sign: keep highest confidence.
    Different categories: keep both.
    """
    accepted: List[Dict[str, Any]] = []
    for det in sorted(raw_detections, key=lambda d: float(d.get("conf", 0.0)), reverse=True):
        det_clean = _clean_class(det.get("cls"))
        det_is_sign = _is_sign_like(det.get("cls"))
        add = True
        for acc in accepted:
            acc_clean = _clean_class(acc.get("cls"))
            acc_is_sign = _is_sign_like(acc.get("cls"))
            iou = compute_iou(det["box"], acc["box"])
            if det_clean == acc_clean and iou > 0.15:
                add = False
                break
            if det_is_sign and acc_is_sign and iou > 0.60:
                add = False
                log.debug(
                    "⚠️ Conflicting sign duplicate dropped: "
                    f"{det.get('cls')} conf={float(det.get('conf', 0.0)):.2f} "
                    f"overlaps {acc.get('cls')} conf={float(acc.get('conf', 0.0)):.2f} "
                    f"iou={iou:.2f}"
                )
                break
        if add:
            accepted.append(det)
    return accepted


def _tracker_stable_enough(trk: KalmanBoxTracker) -> bool:
    """Decide when a tracked object may be reported to VisionFilter.

    Vehicles still need two hits. Yield/stop signs may be reported after one hit
    because they can be tiny and briefly visible. No-entry is stricter because
    low-confidence no_entry false positives were flooding the vector.
    """
    if trk.hits >= 2:
        return True
    conf = float(getattr(trk, "confidence", 0.0) or 0.0)
    dist = float(getattr(trk, "distance_est", 99.0) or 99.0)

    if _is_yield_like(trk.class_name):
        return conf >= YIELD_MIN_CONF and dist <= YIELD_MAX_DISTANCE_M
    if _is_stop_like(trk.class_name):
        return conf >= STOP_MIN_CONF and dist <= STOP_MAX_DISTANCE_M
    if _is_no_entry_like(trk.class_name):
        return conf >= 0.50 and dist <= min(NO_ENTRY_MAX_DISTANCE_M, 20.0)
    return False


# ============================================================================
# YOLO chunk runner
# ============================================================================
def _run_yolo_chunk(video_path, start_frame, end_frame, frame_skip=FRAME_SKIP) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[str, Counter]]:
    model = get_yolo()
    model_classes = model.names

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    FOCAL_LENGTH = 800
    chunk_results: Dict[int, List[Dict[str, Any]]] = {}
    debug = {
        "raw_model_counts": Counter(),
        "after_threshold_counts": Counter(),
        "after_geometry_counts": Counter(),
        "after_dedup_counts": Counter(),
    }

    for current_frame in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_skip > 1 and (current_frame - start_frame) % frame_skip != 0:
            continue

        results = model.predict(
            frame,
            conf=YOLO_PREDICT_CONF,
            imgsz=YOLO_IMG_SIZE,
            verbose=False,
        )

        raw_detections: List[Dict[str, Any]] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = model_classes[cls_id]
            clean_name = _clean_class(class_name)
            debug["raw_model_counts"][clean_name] += 1

            if conf < _class_min_conf(class_name):
                continue
            debug["after_threshold_counts"][clean_name] += 1

            box_w, box_h = x2 - x1, y2 - y1
            if (box_w * box_h) > (width * height) * 0.15:
                continue

            distance_est = -1
            if box_h > 0:
                real_height = _class_physical_height_m(class_name)
                max_distance = _max_distance_for_class(class_name)
                distance_est = (FOCAL_LENGTH * real_height) / box_h
                if distance_est > max_distance:
                    continue
            debug["after_geometry_counts"][clean_name] += 1

            raw_detections.append({
                "box": [x1, y1, x2, y2],
                "cls": class_name,
                "clean_cls": clean_name,
                "conf": conf,
                "distance_est": distance_est,
            })

        final_detections = _deduplicate_detections(raw_detections)
        for det in final_detections:
            debug["after_dedup_counts"][_clean_class(det.get("cls"))] += 1
        chunk_results[current_frame] = final_detections

    cap.release()
    return chunk_results, debug


def _merge_debug_counts(target: Dict[str, Counter], source: Dict[str, Counter]) -> None:
    for key, counter in source.items():
        target.setdefault(key, Counter()).update(counter)


def _update_video_progress(test_id, percent, message):
    if test_id is None:
        return
    try:
        from app.routes.test_routes import update_progress
        update_progress(test_id, percent, message)
    except Exception:
        pass


def _draw_debug_box(frame, obj, label_prefix=""):
    try:
        bbox = obj.get("bbox")
        if not bbox or len(bbox) != 4:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(0, x2), max(0, y2)

        class_name = obj.get("class_name") or obj.get("type") or "object"
        distance = obj.get("distance_est", obj.get("distance_meters", -1))
        track_id = obj.get("id", None)
        conf = obj.get("confidence", None)

        label_parts = []
        if label_prefix:
            label_parts.append(label_prefix)
        label_parts.append(str(class_name))
        if track_id is not None:
            label_parts.append(f"id={track_id}")
        try:
            if float(distance) > 0:
                label_parts.append(f"{float(distance):.1f}m")
        except Exception:
            pass
        try:
            if conf is not None:
                label_parts.append(f"conf={float(conf):.2f}")
        except Exception:
            pass
        label = " | ".join(label_parts)

        color = (120, 120, 120) if label_prefix == "RAW" else (0, 220, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    except Exception:
        return




# ============================================================================
# Final sign-event pruning
# ============================================================================
def _prune_isolated_final_sign_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove isolated sign ghosts after event-level deduplication.

    VisionFilter works frame-by-frame. A high-confidence false STOP can survive
    when YOLO repeatedly sees the same wrong patch and the distance estimate keeps
    shrinking. Because analyze_video_for_server is an offline pipeline, we can add
    a final sequence-level guard before vector_builder:

    - keep cars unchanged;
    - keep sign segments that produce at least 2 final events;
    - keep a single sign only if it is very close and strong;
    - drop single medium-distance sign events, which are usually YOLO ghosts.

    This is intentionally stricter for STOP than for YIELD because our validation
    data showed yield-only clips can create a single false STOP event, while real
    STOP clips produce multiple STOP events.
    """
    if not events:
        return events

    sign_types = {"stop_sign", "yield_sign", "no_entry"}
    passthrough = [e for e in events if e.get("type") not in sign_types]
    signs = [e for e in events if e.get("type") in sign_types]
    if not signs:
        return events

    def _as_float(v, default=99.0):
        try:
            if v is None:
                return default
            return float(v)
        except (TypeError, ValueError):
            return default

    def _seg_key(e):
        # Prefer tracker id. If missing, group by type only; final events are
        # already sparse after event-key deduplication.
        return (e.get("type"), e.get("id") if e.get("id") is not None else "no-id")

    grouped = {}
    for e in signs:
        grouped.setdefault(_seg_key(e), []).append(e)

    kept_signs = []
    dropped = []
    MAX_GAP_S = 3.2

    for key, group in grouped.items():
        group = sorted(group, key=lambda x: _as_float(x.get("time_sec"), 0.0))
        segments = []
        cur = []
        for e in group:
            if not cur:
                cur = [e]
                continue
            if _as_float(e.get("time_sec"), 0.0) - _as_float(cur[-1].get("time_sec"), 0.0) <= MAX_GAP_S:
                cur.append(e)
            else:
                segments.append(cur)
                cur = [e]
        if cur:
            segments.append(cur)

        for seg in segments:
            typ = seg[0].get("type")
            count = len(seg)
            min_dist = min(_as_float(e.get("distance_meters"), 99.0) for e in seg)
            max_conf = max(_as_float(e.get("confidence"), 0.0) for e in seg)
            span = _as_float(seg[-1].get("time_sec"), 0.0) - _as_float(seg[0].get("time_sec"), 0.0)

            keep = False
            if count >= 2:
                keep = True
            elif typ == "stop_sign":
                # Single STOP must be unmistakably close. This kills the false
                # STOP in the yield-only clip (~11m), while keeping emergency
                # one-hit close detections if they ever occur.
                keep = max_conf >= 0.70 and min_dist <= 7.5
            elif typ == "yield_sign":
                # YIELD is rarer/harder; keep one strong close yield, but still
                # reject far one-frame ghosts.
                keep = max_conf >= 0.60 and min_dist <= 16.0
            elif typ == "no_entry":
                keep = max_conf >= 0.65 and min_dist <= 12.0

            if keep:
                kept_signs.extend(seg)
            else:
                dropped.extend(seg)

    if dropped:
        try:
            summary = Counter(e.get("type", "unknown") for e in dropped)
            log.info(f"🚧 Final sign prune dropped isolated signs: {dict(summary)}")
        except Exception:
            pass

    combined = passthrough + kept_signs
    return sorted(combined, key=lambda x: _as_float(x.get("time_sec"), 0.0))


def _filtered_event_to_drawable(obj, raw_objects):
    out = dict(obj)
    obj_id = obj.get("id")
    if obj_id is not None:
        for raw in raw_objects:
            if raw.get("id") == obj_id:
                out["bbox"] = raw.get("bbox")
                out["class_name"] = obj.get("type", raw.get("class_name"))
                out["distance_est"] = obj.get("distance_meters", raw.get("distance_est", -1))
                out["confidence"] = obj.get("confidence", raw.get("confidence"))
                return out
    if len(raw_objects) == 1:
        raw = raw_objects[0]
        out["bbox"] = raw.get("bbox")
        out["class_name"] = obj.get("type", raw.get("class_name"))
        out["distance_est"] = obj.get("distance_meters", raw.get("distance_est", -1))
        out["confidence"] = obj.get("confidence", raw.get("confidence"))
    return out


# ============================================================================
# Main entry
# ============================================================================
async def analyze_video_for_server(
    input_path: str,
    sensor_df: pd.DataFrame,
    test_id: str = None,
    output_video_path: str = None,
) -> list:
    KalmanBoxTracker.count = 0
    _GHOST_POOL.clear()

    log.info("🎥 Starting Video Analysis Pipeline...")

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    debug_cap = None
    debug_writer = None
    if output_video_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            debug_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (video_width, video_height))
            if not debug_writer.isOpened():
                log.warning(f"⚠️ Could not open debug video writer: {output_video_path}")
                debug_writer = None
            else:
                debug_cap = cv2.VideoCapture(input_path)
                log.info(f"🎞️ YOLO annotated video will be saved to: {output_video_path}")
        except Exception as e:
            log.warning(f"⚠️ Failed to initialize debug video export: {e}")
            debug_writer = None
            debug_cap = None

    vision_filter = VisionFilter(video_width=video_width)

    CHUNK_SIZE = 60
    chunks = [(input_path, start, min(start + CHUNK_SIZE, total_frames))
              for start in range(0, total_frames, CHUNK_SIZE)]

    total_chunks = len(chunks)
    all_yolo_results: Dict[int, List[Dict[str, Any]]] = {}
    completed = 0
    start_time = time.time()
    debug_counts: Dict[str, Counter] = {
        "raw_model_counts": Counter(),
        "after_threshold_counts": Counter(),
        "after_geometry_counts": Counter(),
        "after_dedup_counts": Counter(),
    }

    async with _GPU_SEMAPHORE:
        for vp, sf, ef in chunks:
            chunk_res, chunk_debug = await asyncio.to_thread(_run_yolo_chunk, vp, sf, ef)
            all_yolo_results.update(chunk_res)
            _merge_debug_counts(debug_counts, chunk_debug)
            completed += 1

            elapsed = time.time() - start_time
            avg = elapsed / completed
            eta = int((total_chunks - completed) * avg)
            pct = 10 + int(60 * completed / total_chunks)
            _update_video_progress(test_id, pct, f"YOLO Detection... ETA: ~{eta}s")

    log.info(f"🔎 YOLO low-conf raw counts: {dict(debug_counts['raw_model_counts'])}")
    log.info(f"🔎 YOLO after threshold counts: {dict(debug_counts['after_threshold_counts'])}")
    log.info(f"🔎 YOLO after geometry counts: {dict(debug_counts['after_geometry_counts'])}")
    log.info(f"🔎 YOLO after dedup counts: {dict(debug_counts['after_dedup_counts'])}")
    log.info("✅ YOLO Done. Starting Tracking...")
    _update_video_progress(test_id, 70, "Tracking objects (Kalman)...")

    trackers: List[KalmanBoxTracker] = []
    final_video_events: List[Dict[str, Any]] = []
    reported_event_ids = set()
    tracked_counts = Counter()
    filtered_counts = Counter()

    for frame_idx in range(total_frames):
        if frame_idx % max(1, total_frames // 20) == 0:
            pct = 70 + int(10 * frame_idx / total_frames)
            _update_video_progress(test_id, pct, "Filtering detections...")

        current_time_sec = frame_idx / fps

        debug_frame = None
        if debug_cap is not None and debug_writer is not None:
            ok, frame = debug_cap.read()
            if ok:
                debug_frame = frame

        detections, det_classes, det_distances, det_confidences = [], [], [], []
        if frame_idx in all_yolo_results:
            for det in all_yolo_results[frame_idx]:
                detections.append(det["box"])
                det_classes.append(det["cls"])
                det_distances.append(det["distance_est"])
                det_confidences.append(det.get("conf", 0.0))

        trks = np.zeros((len(trackers), 4))
        for t, trk in enumerate(trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]

        if detections:
            matches, unmatched_dets = associate_detections_to_trackers(detections, trks)
            for m in matches:
                trackers[m[1]].update(detections[m[0]], det_classes[m[0]], det_confidences[m[0]])
                trackers[m[1]].distance_est = det_distances[m[0]]
            for i in unmatched_dets:
                trk = KalmanBoxTracker(detections[i], det_classes[i], det_confidences[i])
                trk.distance_est = det_distances[i]
                trackers.append(trk)

        active_trackers = []
        raw_frame_objects: List[Dict[str, Any]] = []

        for trk in trackers:
            # Vehicles need two hits. Yield/stop signs can be one-hit; no_entry is stricter.
            stable_enough = _tracker_stable_enough(trk)
            if trk.time_since_update <= 15 and stable_enough:
                box = trk.kf.x.flatten()
                w = np.sqrt(max(0, box[2] * box[3]))
                h = box[2] / w if w > 0 else 0
                bbox = [int(box[0] - w / 2), int(box[1] - h / 2),
                        int(box[0] + w / 2), int(box[1] + h / 2)]

                if trk.hits == 2:
                    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    best_gid, best_d = None, float("inf")
                    for gid, g in list(_GHOST_POOL.items()):
                        if (current_time_sec - g["last_seen_t"]) > _GHOST_TTL_S:
                            continue
                        if g["class"] != trk.class_name:
                            continue
                        gx = (g["last_box"][0] + g["last_box"][2]) / 2
                        gy = (g["last_box"][1] + g["last_box"][3]) / 2
                        d = ((cx - gx) ** 2 + (cy - gy) ** 2) ** 0.5
                        if d < _GHOST_MAX_DIST_PX and d < best_d:
                            best_d, best_gid = d, gid
                    if best_gid is not None:
                        log.debug(f"🔗 Re-stitched ghost {best_gid} (Δ={best_d:.0f}px)")
                        trk.id = best_gid
                        _GHOST_POOL.pop(best_gid, None)

                raw_obj = {
                    "id": trk.id,
                    "class_name": trk.class_name,
                    "bbox": bbox,
                    "distance_est": getattr(trk, "distance_est", -1),
                    "confidence": float(getattr(trk, "confidence", 0.0) or 0.0),
                    "time_since_update": int(getattr(trk, "time_since_update", 99) or 0),
                }
                raw_frame_objects.append(raw_obj)
                tracked_counts[_clean_class(trk.class_name)] += 1

            if trk.time_since_update <= 30:
                active_trackers.append(trk)
            elif trk.id is not None:
                last = trk.kf.x.flatten()
                w = np.sqrt(max(0, last[2] * last[3]))
                h = last[2] / w if w > 0 else 0
                _GHOST_POOL[trk.id] = {
                    "last_box": [last[0] - w / 2, last[1] - h / 2, last[0] + w / 2, last[1] + h / 2],
                    "class": trk.class_name,
                    "last_seen_t": current_time_sec,
                }

        if frame_idx % 30 == 0:
            stale = [g for g, v in _GHOST_POOL.items()
                     if (current_time_sec - v["last_seen_t"]) > _GHOST_TTL_S]
            for g in stale:
                _GHOST_POOL.pop(g, None)

        trackers = active_trackers

        if raw_frame_objects:
            closest_idx = (sensor_df["time_seconds"] - current_time_sec).abs().idxmin()
            current_lat = sensor_df.loc[closest_idx, "lat"]
            current_lon = sensor_df.loc[closest_idx, "lon"]
            current_speed_kmh = sensor_df.loc[closest_idx, "speed_kmh"]

            filtered_objects = vision_filter.filter_detections(
                current_time=current_time_sec,
                current_lat=current_lat,
                current_lon=current_lon,
                frame_detections=raw_frame_objects,
                current_speed_kmh=current_speed_kmh,
            )

            if debug_frame is not None:
                for raw_obj in raw_frame_objects:
                    _draw_debug_box(debug_frame, raw_obj, label_prefix="RAW")
                for obj in filtered_objects:
                    drawable = _filtered_event_to_drawable(obj, raw_frame_objects)
                    _draw_debug_box(debug_frame, drawable, label_prefix="USED")

            for obj in filtered_objects:
                filtered_counts[obj.get("type", "unknown")] += 1
                cx_bucket = obj["id"] if obj.get("id") is not None else 0
                event_key = f"{obj['type']}_{cx_bucket}_{int(current_time_sec) // 2}"
                if event_key not in reported_event_ids:
                    final_video_events.append(obj)
                    reported_event_ids.add(event_key)

        if debug_frame is not None and debug_writer is not None:
            cv2.putText(debug_frame, f"t={current_time_sec:.2f}s frame={frame_idx}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (255, 255, 255), 2, cv2.LINE_AA)
            debug_writer.write(debug_frame)

    if debug_writer is not None:
        debug_writer.release()
    if debug_cap is not None:
        debug_cap.release()

    final_event_counts_before_prune = Counter(e.get("type", "unknown") for e in final_video_events)
    final_video_events = _prune_isolated_final_sign_events(final_video_events)
    final_event_counts_after_prune = Counter(e.get("type", "unknown") for e in final_video_events)

    debug_export = {
        "config": {
            "YOLO_PREDICT_CONF": YOLO_PREDICT_CONF,
            "YOLO_IMG_SIZE": YOLO_IMG_SIZE,
            "FRAME_SKIP": FRAME_SKIP,
            "YIELD_MIN_CONF": YIELD_MIN_CONF,
            "STOP_MIN_CONF": STOP_MIN_CONF,
            "NO_ENTRY_MIN_CONF": NO_ENTRY_MIN_CONF,
            "VEHICLE_MIN_CONF": VEHICLE_MIN_CONF,
            "OTHER_MIN_CONF": OTHER_MIN_CONF,
            "YIELD_MAX_DISTANCE_M": YIELD_MAX_DISTANCE_M,
            "STOP_MAX_DISTANCE_M": STOP_MAX_DISTANCE_M,
            "NO_ENTRY_MAX_DISTANCE_M": NO_ENTRY_MAX_DISTANCE_M,
            "VEHICLE_MAX_DISTANCE_M": VEHICLE_MAX_DISTANCE_M,
        },
        "raw_model_counts": dict(debug_counts["raw_model_counts"]),
        "after_threshold_counts": dict(debug_counts["after_threshold_counts"]),
        "after_geometry_counts": dict(debug_counts["after_geometry_counts"]),
        "after_dedup_counts": dict(debug_counts["after_dedup_counts"]),
        "tracked_counts": dict(tracked_counts),
        "filtered_counts": dict(filtered_counts),
        "final_event_counts_before_prune": dict(final_event_counts_before_prune),
        "final_event_counts_after_prune": dict(final_event_counts_after_prune),
    }

    if output_video_path:
        try:
            debug_path = os.path.join(os.path.dirname(os.path.abspath(output_video_path)), "0_yolo_debug_counts.json")
            with open(debug_path, "w", encoding="utf-8") as f:
                json.dump(debug_export, f, ensure_ascii=False, indent=2)
            log.info(f"🔎 Saved YOLO debug counts: {debug_path}")
        except Exception as e:
            log.warning(f"⚠️ Failed to save YOLO debug counts: {e}")

    log.info(f"🔎 Tracked counts: {dict(tracked_counts)}")
    log.info(f"🔎 Filtered counts: {dict(filtered_counts)}")
    log.info(f"🔎 Final event counts before prune: {dict(final_event_counts_before_prune)}")
    log.info(f"🔎 Final event counts after prune: {dict(final_event_counts_after_prune)}")
    log.info(f"🏁 Pipeline Complete. Events: {len(final_video_events)}")
    if output_video_path:
        log.info(f"🎞️ Saved YOLO annotated video: {output_video_path}")
    return final_video_events
