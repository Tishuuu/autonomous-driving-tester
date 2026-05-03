import math
import json
import os
from app.utils.logger import log

class VisionFilter:
    def __init__(self, video_width=1280):
        self.video_width = video_width
        self.vehicle_history = {}  
        
        self.known_traffic_lights = []
        lights_file = os.path.join(os.path.dirname(__file__), '..', 'ai_models', 'ness_ziona_lights.json')
        
        try:
            if os.path.exists(lights_file):
                with open(lights_file, 'r', encoding='utf-8') as f:
                    self.known_traffic_lights = json.load(f)
                log.info(f"🚦 Loaded {len(self.known_traffic_lights)} traffic lights from JSON.")
            else:
                log.warning("⚠️ Traffic lights JSON not found. Filter will not override stop signs.")
        except Exception as e:
            log.error(f"❌ Failed to load traffic lights: {e}")

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000  
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def is_near_traffic_light(self, current_lat, current_lon, threshold_meters=2):
        if not self.known_traffic_lights: return False
        for tl in self.known_traffic_lights:
            dist = self.haversine_distance(current_lat, current_lon, tl["lat"], tl["lon"])
            if dist < threshold_meters: return True
        return False

    def filter_detections(self, current_time, current_lat, current_lon, frame_detections):
        filtered_events = []
        near_traffic_light = self.is_near_traffic_light(current_lat, current_lon)

        # ✅ ניקוי תקופתי של vehicle_history - מונע דליפת זיכרון בנסיעות ארוכות.
        # מוחקים track_ids שלא ראינו אותם יותר מ-2 שניות (פי 4 מחלון ההיסטוריה).
        STALE_THRESHOLD = 2.0
        stale_ids = [
            tid for tid, history in self.vehicle_history.items()
            if not history or (current_time - history[-1]['time']) > STALE_THRESHOLD
        ]
        for tid in stale_ids:
            del self.vehicle_history[tid]

        for det in frame_detections:
            cls_name = det['class_name'].lower() 
            bbox = det['bbox'] 
            track_id = det.get('id', None)
            distance_est = det.get('distance_est', -1) 
            
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            is_stop_sign = "stop" in cls_name
            is_yield_sign = "yield" in cls_name
            is_no_entry = "entry" in cls_name
            is_car = "car" in cls_name or "vehicle" in cls_name

            # 1. סינון נתיב נגדי לתמרורים
            if is_stop_sign or is_yield_sign or is_no_entry:
                if center_x < (self.video_width * 0.35): 
                    continue 

            # 2. סינון רמזורים (מבטל עצור/זכות קדימה)
            if near_traffic_light and (is_stop_sign or is_yield_sign):
                log.debug(f"🚦 Skipped {cls_name} due to active traffic light override.")
                continue

            # ==========================================
            # 3. מסנן רכבים משולב (Cross-Traffic vs Parked)
            # ==========================================
            if is_car and track_id is not None:
                # מתעדים תנועה לכל רכב
                if track_id not in self.vehicle_history: 
                    self.vehicle_history[track_id] = []
                self.vehicle_history[track_id].append({'time': current_time, 'x': center_x, 'y': center_y})
                
                # שומרים רק את החצי שנייה האחרונה (היסטוריה קצרה ורלוונטית)
                self.vehicle_history[track_id] = [p for p in self.vehicle_history[track_id] if current_time - p['time'] <= 0.5]
                history = self.vehicle_history[track_id]

                # בדיקת סכנה מתפרצת מהשוליים!
                if center_x < (self.video_width * 0.20): # שוליים שמאליים
                    if len(history) < 3: 
                        continue # מחכים שייווצר כיוון תנועה (3 פריימים לפחות)
                    dx = history[-1]['x'] - history[0]['x']
                    # אם ה-X גדל (זז ימינה למרכז) משמע הוא מתפרץ לצומת!
                    if dx < 10: 
                        continue # לא מתפרץ למרכז -> רכב חונה, מחק אותו.

                elif center_x > (self.video_width * 0.80): # שוליים ימניים
                    if len(history) < 3: 
                        continue 
                    dx = history[-1]['x'] - history[0]['x']
                    # אם ה-X קטן (זז שמאלה למרכז) משמע הוא מתפרץ!
                    if dx > -10: 
                        continue # לא מתפרץ למרכז -> רכב חונה, מחק אותו.
            
            # מתקנים את השם לשם נקי
            clean_type = "car" if is_car else ("stop_sign" if is_stop_sign else ("yield_sign" if is_yield_sign else cls_name))

            filtered_events.append({
                "time_sec": current_time,
                "type": clean_type,
                "distance_meters": distance_est, 
                "id": track_id 
            })

        return filtered_events