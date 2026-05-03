import cv2
import numpy as np
import asyncio
import time
import os
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import pandas as pd

os.environ['YOLO_OFFLINE'] = 'True'
os.environ['YOLO_UPDATE_CHECK'] = 'False'
from ultralytics import YOLO
from app.utils.logger import log
from app.services.vision_filter import VisionFilter

log.info("YOLO STARTING")

# ==========================================================================
# 🆕 Singleton model + GPU semaphore (replaces ProcessPoolExecutor)
# ==========================================================================
_GLOBAL_YOLO = None
_GPU_SEMAPHORE = asyncio.Semaphore(1)  # serialize GPU across requests

# 🆕 Ghost pool for tracker ID re-stitching
_GHOST_POOL: dict = {}
_GHOST_TTL_S = 1.5
_GHOST_MAX_DIST_PX = 80


def get_yolo():
    global _GLOBAL_YOLO
    if _GLOBAL_YOLO is None:
        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'ai_models', 'best (cars).pt'
        )
        _GLOBAL_YOLO = YOLO(model_path)
        # Warm-up to amortize first-call latency
        _GLOBAL_YOLO.predict(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
        log.info("🟢 YOLO singleton initialized & warmed")
    return _GLOBAL_YOLO


# ==========================================================================
# Kalman tracker
# ==========================================================================
class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox, class_name):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = None
        self.class_history = [class_name]
        self.class_name = class_name
        self.hits = 0
        self.distance_est = -1

    def convert_bbox_to_z(self, bbox):
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = bbox[0] + w/2., bbox[1] + h/2.
        return np.array([x, y, w*h, w/float(h)]).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        w = np.sqrt(max(0, x[2] * x[3]))
        h = x[2] / w if w > 0 else 0
        return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.x)

    def update(self, bbox, new_class_name):
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
    return (np.array(matches) if len(matches) > 0
            else np.empty((0, 2), dtype=int)), np.array(unmatched)


# ==========================================================================
# 🆕 Synchronous YOLO chunk runner (singleton model)
# ==========================================================================
def _run_yolo_chunk(video_path, start_frame, end_frame, frame_skip=2):
    model = get_yolo()
    model_classes = model.names

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    FOCAL_LENGTH = 800
    REAL_SIGN_HEIGHT = 0.8
    MAX_DISTANCE = 45

    chunk_results = {}

    for current_frame in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if (current_frame - start_frame) % frame_skip != 0:
            continue

        results = model.predict(frame, conf=0.6, imgsz=640, verbose=False)

        raw_detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = model_classes[cls_id]

            box_w, box_h = x2 - x1, y2 - y1
            if (box_w * box_h) > (width * height) * 0.15:
                continue

            distance_est = -1
            if box_h > 0:
                distance_est = (FOCAL_LENGTH * REAL_SIGN_HEIGHT) / box_h
                if distance_est > MAX_DISTANCE:
                    continue

            raw_detections.append({
                'box': [x1, y1, x2, y2],
                'cls': class_name,
                'conf': conf,
                'distance_est': distance_est,
            })

        final_detections = []
        for raw_det in raw_detections:
            box = raw_det['box']
            is_dup = any(compute_iou(box, acc['box']) > 0.15 for acc in final_detections)
            if not is_dup:
                final_detections.append(raw_det)

        chunk_results[current_frame] = final_detections

    cap.release()
    return chunk_results


def _update_video_progress(test_id, percent, message):
    if test_id is None:
        return
    try:
        from app.routes.test_routes import update_progress
        update_progress(test_id, percent, message)
    except Exception:
        pass


# ==========================================================================
# 🎯 Main entry — async, GPU-serialized, ghost-pool stitching
# ==========================================================================
async def analyze_video_for_server(input_path: str,
                                   sensor_df: pd.DataFrame,
                                   test_id: str = None) -> list:
    KalmanBoxTracker.count = 0
    _GHOST_POOL.clear()

    log.info("🎥 Starting Video Analysis Pipeline...")

    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    vision_filter = VisionFilter(video_width=video_width)

    CHUNK_SIZE = 60
    chunks = []
    for start in range(0, total_frames, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total_frames)
        chunks.append((input_path, start, end))

    total_chunks = len(chunks)
    all_yolo_results = {}
    completed = 0
    start_time = time.time()

    # 🆕 GPU semaphore — serialize across concurrent /evaluate requests
    async with _GPU_SEMAPHORE:
        for vp, sf, ef in chunks:
            chunk_res = await asyncio.to_thread(_run_yolo_chunk, vp, sf, ef)
            all_yolo_results.update(chunk_res)
            completed += 1

            elapsed = time.time() - start_time
            avg = elapsed / completed
            eta = int((total_chunks - completed) * avg)
            pct = 10 + int(60 * completed / total_chunks)
            _update_video_progress(test_id, pct,
                                   f"YOLO Detection... ETA: ~{eta}s")

    log.info("✅ YOLO Done. Starting Tracking...")
    _update_video_progress(test_id, 70, "Tracking objects (Kalman)...")

    trackers = []
    final_video_events = []
    reported_event_ids = set()

    for frame_idx in range(total_frames):
        if frame_idx % max(1, total_frames // 20) == 0:
            pct = 70 + int(10 * frame_idx / total_frames)
            _update_video_progress(test_id, pct, "Filtering detections...")

        current_time_sec = frame_idx / fps

        detections, det_classes, det_distances = [], [], []
        if frame_idx in all_yolo_results:
            for det in all_yolo_results[frame_idx]:
                detections.append(det['box'])
                det_classes.append(det['cls'])
                det_distances.append(det['distance_est'])

        trks = np.zeros((len(trackers), 4))
        for t, trk in enumerate(trackers):
            pos = trk.predict()[0]
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]

        if len(detections) > 0:
            matches, unmatched_dets = associate_detections_to_trackers(detections, trks)
            for m in matches:
                trackers[m[1]].update(detections[m[0]], det_classes[m[0]])
                trackers[m[1]].distance_est = det_distances[m[0]]
            for i in unmatched_dets:
                trk = KalmanBoxTracker(detections[i], det_classes[i])
                trk.distance_est = det_distances[i]
                trackers.append(trk)

        active_trackers = []
        raw_frame_objects = []

        for trk in trackers:
            # Reporting tier
            if trk.time_since_update <= 15 and trk.hits >= 2:
                box = trk.kf.x.flatten()
                w = np.sqrt(max(0, box[2] * box[3]))
                h = box[2] / w if w > 0 else 0
                bbox = [int(box[0]-w/2), int(box[1]-h/2),
                        int(box[0]+w/2), int(box[1]+h/2)]

                # 🆕 Ghost-pool stitch on first confirmation
                if trk.hits == 2:
                    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                    best_gid, best_d = None, float('inf')
                    for gid, g in _GHOST_POOL.items():
                        if (current_time_sec - g["last_seen_t"]) > _GHOST_TTL_S:
                            continue
                        if g["class"] != trk.class_name:
                            continue
                        gx = (g["last_box"][0] + g["last_box"][2]) / 2
                        gy = (g["last_box"][1] + g["last_box"][3]) / 2
                        d = ((cx-gx)**2 + (cy-gy)**2) ** 0.5
                        if d < _GHOST_MAX_DIST_PX and d < best_d:
                            best_d, best_gid = d, gid
                    if best_gid is not None:
                        log.debug(f"🔗 Re-stitched ghost {best_gid} (Δ={best_d:.0f}px)")
                        trk.id = best_gid
                        _GHOST_POOL.pop(best_gid, None)

                raw_frame_objects.append({
                    'id': trk.id,
                    'class_name': trk.class_name,
                    'bbox': bbox,
                    'distance_est': getattr(trk, 'distance_est', -1),
                })

            # Drop or send to ghost pool
            if trk.time_since_update <= 30:
                active_trackers.append(trk)
            elif trk.id is not None:
                last = trk.kf.x.flatten()
                w = np.sqrt(max(0, last[2] * last[3]))
                h = last[2] / w if w > 0 else 0
                _GHOST_POOL[trk.id] = {
                    "last_box": [last[0]-w/2, last[1]-h/2, last[0]+w/2, last[1]+h/2],
                    "class": trk.class_name,
                    "last_seen_t": current_time_sec,
                }

        # Periodic ghost pool cleanup
        if frame_idx % 30 == 0:
            stale = [g for g, v in _GHOST_POOL.items()
                     if (current_time_sec - v["last_seen_t"]) > _GHOST_TTL_S]
            for g in stale:
                _GHOST_POOL.pop(g, None)

        trackers = active_trackers

        if raw_frame_objects:
            closest_idx = (sensor_df['time_seconds'] - current_time_sec).abs().idxmin()
            current_lat = sensor_df.loc[closest_idx, 'lat']
            current_lon = sensor_df.loc[closest_idx, 'lon']
            current_speed_kmh = sensor_df.loc[closest_idx, 'speed_kmh']

            filtered_objects = vision_filter.filter_detections(
                current_time=current_time_sec,
                current_lat=current_lat,
                current_lon=current_lon,
                frame_detections=raw_frame_objects,
                current_speed_kmh=current_speed_kmh,
            )

            for obj in filtered_objects:
                # 🆕 Wider time bucket + spatial bucket for dedup
                cx_bucket = (obj['id'] if obj.get('id') is not None else 0)
                event_key = f"{obj['type']}_{cx_bucket}_{int(current_time_sec) // 2}"
                if event_key not in reported_event_ids:
                    final_video_events.append(obj)
                    reported_event_ids.add(event_key)

    log.info(f"🏁 Pipeline Complete. Events: {len(final_video_events)}")
    return final_video_events