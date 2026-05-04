import cv2
import numpy as np
import concurrent.futures
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter
import multiprocessing
import os

class KalmanBoxTracker(object):
    count = 0
    def __init__(self, bbox, class_name):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],  
                              [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]])
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = None 
        self.class_history = [class_name]
        self.class_name = class_name
        self.hits = 0

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
        if len(self.class_history) > 10: self.class_history.pop(0)
        self.class_name = max(set(self.class_history), key=self.class_history.count)

def compute_iou(box1, box2):
    x_left, y_top = max(box1[0], box2[0]), max(box1[1], box2[1])
    x_right, y_bottom = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top: return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - intersection
    if union <= 0: return 0.0
    return intersection / float(union)

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.05): 
    if len(trackers) == 0: return np.empty((0, 2), dtype=int), np.arange(len(detections))
    iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(trackers):
            iou_matrix[d, t] = compute_iou(det, trk)
    np.nan_to_num(iou_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches, unmatched_detections = [], []
    for d in range(len(detections)):
        if d not in row_ind: unmatched_detections.append(d)
        else:
            t = col_ind[np.where(row_ind == d)[0][0]]
            if iou_matrix[d, t] < iou_threshold: unmatched_detections.append(d)
            else: matches.append(np.array([d, t]))
    return np.array(matches) if len(matches) > 0 else np.empty((0, 2), dtype=int), np.array(unmatched_detections)

def worker_yolo_extraction(args):
    video_path, start_frame, end_frame, chunk_id = args
    
    model = YOLO("best (cars).pt")
    model_classes = model.names 
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    chunk_results = {}
    print(f"Worker {chunk_id} starting: frames {start_frame} to {end_frame}")
    
    for current_frame in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=0.55, imgsz=1024, verbose=False)        
        
        raw_detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            class_name = model_classes[cls_id]
            
            box_w, box_h = x2 - x1, y2 - y1
            
            # סינון גודל אבסורדי
            if (box_w * box_h) > (width * height) * 0.15: continue
            aspect_ratio = box_w / float(box_h) if box_h > 0 else 0
            if aspect_ratio < 0.5 or aspect_ratio > 1.5: continue
            
            # --- התיקון: מסננים נתיב נגדי רק לתמרורים ולא לרכבים! ---
            center_x = x1 + (box_w / 2)
            is_car = "car" in class_name.lower()
            if not is_car:
                if center_x < width * 0.35: continue 

            FOCAL_LENGTH, REAL_SIGN_HEIGHT, MAX_DISTANCE = 800, 0.8, 45       
            if box_h > 0: 
                distance_est = (FOCAL_LENGTH * REAL_SIGN_HEIGHT) / box_h
                if distance_est > MAX_DISTANCE: continue 
            
            raw_detections.append({'box': [x1, y1, x2, y2], 'cls': class_name, 'conf': conf})
        
        final_detections = []
        for raw_det in raw_detections:
            box = raw_det['box']
            is_dup = False
            for acc in final_detections:
                if compute_iou(box, acc['box']) > 0.15: 
                    is_dup = True; break
            if not is_dup: final_detections.append(raw_det)
            
        chunk_results[current_frame] = final_detections

    cap.release()
    print(f"Worker {chunk_id} finished!")
    return chunk_results

def process_video(input_path, output_path):
    print("Extracting video info...")
    cap = cv2.VideoCapture(input_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    num_cores = multiprocessing.cpu_count()
    workers_to_use = max(1, num_cores - 1) 
    chunk_size = total_frames // workers_to_use
    
    chunks = []
    for i in range(workers_to_use):
        start = i * chunk_size
        end = total_frames if i == workers_to_use - 1 else (i + 1) * chunk_size
        chunks.append((input_path, start, end, i))

    print(f"Video has {total_frames} frames. Splitting into {workers_to_use} chunks.")

    all_yolo_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers_to_use) as executor:
        results = executor.map(worker_yolo_extraction, chunks)
        for res in results:
            all_yolo_results.update(res)

    print("YOLO extraction done. Starting tracking and drawing...")
    cap = cv2.VideoCapture(input_path)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    trackers = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        detections = []
        det_classes = []
        
        if frame_idx in all_yolo_results:
            for det in all_yolo_results[frame_idx]:
                detections.append(det['box'])
                det_classes.append(det['cls'])

        trks = np.zeros((len(trackers), 4))
        for t, trk in enumerate(trackers):
            pos = trk.predict()[0] 
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]

        if len(detections) > 0:
            matches, unmatched_dets = associate_detections_to_trackers(detections, trks)
            for m in matches:
                trackers[m[1]].update(detections[m[0]], det_classes[m[0]])
            for i in unmatched_dets:
                trackers.append(KalmanBoxTracker(detections[i], det_classes[i]))

        active_trackers = []
        for trk in trackers:
            if trk.time_since_update <= 15 and trk.hits >= 2:
                box = trk.kf.x.flatten()
                w = np.sqrt(max(0, box[2] * box[3]))
                h = box[2] / w if w > 0 else 0
                x1, y1, x2, y2 = int(box[0]-w/2), int(box[1]-h/2), int(box[0]+w/2), int(box[1]+h/2)
                
                # --- התיקון: צבעים מותאמים אישית לקלאסים שלך! ---
                cls_upper = trk.class_name.upper()
                if "STOP" in cls_upper: 
                    color = (0, 0, 255)       # אדום לתמרור עצור
                elif "YIELD" in cls_upper: 
                    color = (0, 255, 255)     # צהוב לזכות קדימה
                elif "ENTRY" in cls_upper: 
                    color = (0, 165, 255)     # כתום לאין כניסה
                elif "CAR" in cls_upper: 
                    color = (0, 255, 0)       # ירוק לרכבים!
                else: 
                    color = (255, 0, 0)       # כחול לכל השאר (גיבוי)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{trk.class_name} ID:{trk.id}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if trk.time_since_update <= 30:
                active_trackers.append(trk)
        
        trackers = active_trackers 
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Done! Saved to {output_path}")

if __name__ == '__main__':
    KalmanBoxTracker.count = 0
    # שים לב לשנות פה את השם של הסרטון שלך אם צריך
    process_video('input_video copy.mp4', 'output_processed.avi')