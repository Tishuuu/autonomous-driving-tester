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
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def is_near_traffic_light(self, current_lat, current_lon, threshold_meters=12):
        if not self.known_traffic_lights:
            return False
        if current_lat is None or current_lon is None:
            return False
        try:
            current_lat = float(current_lat)
            current_lon = float(current_lon)
        except (TypeError, ValueError):
            return False

        for tl in self.known_traffic_lights:
            dist = self.haversine_distance(current_lat, current_lon, tl["lat"], tl["lon"])
            if dist < threshold_meters:
                return True
        return False

    def filter_detections(self, current_time, current_lat, current_lon, frame_detections, current_speed_kmh=None):
        filtered_events = []

        # PATCH: wider GPS radius, but only used when the vehicle is actually stationary.
        near_traffic_light = self.is_near_traffic_light(current_lat, current_lon, threshold_meters=12)
        try:
            speed = float(current_speed_kmh) if current_speed_kmh is not None else None
        except (TypeError, ValueError):
            speed = None
        is_stationary_near_light = near_traffic_light and speed is not None and speed <= 2.0

        # ✅ ניקוי תקופתי של vehicle_history - מונע דליפת זיכרון בנסיעות ארוכות.
        STALE_THRESHOLD = 2.0
        stale_ids = [
            tid for tid, history in self.vehicle_history.items()
            if not history or (current_time - history[-1]['time']) > STALE_THRESHOLD
        ]
        for tid in stale_ids:
            del self.vehicle_history[tid]

        for det in frame_detections:
            cls_name = str(det['class_name']).lower().replace('-', '_').replace(' ', '_')
            bbox = det['bbox']
            track_id = det.get('id', None)
            distance_est = det.get('distance_est', -1)

            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            is_stop_sign = "stop" in cls_name
            is_yield_sign = "yield" in cls_name
            is_no_entry = "no_entry" in cls_name or "noentry" in cls_name or "entry" in cls_name
            is_car = "car" in cls_name or "vehicle" in cls_name

            # 1. סינון נתיב נגדי לתמרורים
            if is_stop_sign or is_yield_sign or is_no_entry:
                if center_x < (self.video_width * 0.35):
                    continue

            # 2. סינון רמזורים (מבטל עצור/זכות קדימה רק אם הרכב עומד ליד רמזור)
            if is_stationary_near_light and (is_stop_sign or is_yield_sign):
                log.debug(f"🚦 Skipped {cls_name} due to stationary traffic light override.")
                continue

            # ==========================================
            # 3. מסנן רכבים משולב (Cross-Traffic vs Parked)
            # ==========================================
            if is_car and track_id is not None:
                if track_id not in self.vehicle_history:
                    self.vehicle_history[track_id] = []
                self.vehicle_history[track_id].append({
                    'time': current_time,
                    'x': center_x,
                    'y': center_y,
                    'distance': distance_est,
                })

                # שומרים חלון קצר, אבל מספיק כדי לראות אם המרחק יורד.
                self.vehicle_history[track_id] = [
                    p for p in self.vehicle_history[track_id]
                    if current_time - p['time'] <= 0.8
                ]
                history = self.vehicle_history[track_id]

                in_left_margin = center_x < (self.video_width * 0.20)
                in_right_margin = center_x > (self.video_width * 0.80)

                if in_left_margin or in_right_margin:
                    if len(history) >= 3:
                        dx = history[-1]['x'] - history[0]['x']
                        d0 = history[0].get('distance', -1)
                        d1 = history[-1].get('distance', -1)

                        valid_distance = d0 is not None and d1 is not None and float(d0) > 0 and float(d1) > 0
                        distance_decreasing = valid_distance and (float(d1) < float(d0) - 0.7)

                        moving_to_center = (
                            (in_left_margin and dx > 10) or
                            (in_right_margin and dx < -10)
                        )
                        laterally_static = abs(dx) < 10

                        # PATCH: delete margin vehicles only if they are static AND not approaching.
                        if laterally_static and not distance_decreasing and not moving_to_center:
                            continue
                    # len(history) < 3: keep it for now. Safer to keep possible cross-traffic.

            # מתקנים את השם לשם נקי ועקבי מול vector_builder.py
            if is_car:
                clean_type = "car"
            elif is_stop_sign:
                clean_type = "stop_sign"
            elif is_yield_sign:
                clean_type = "yield_sign"
            elif is_no_entry:
                clean_type = "no_entry"
            else:
                clean_type = cls_name

            filtered_events.append({
                "time_sec": current_time,
                "type": clean_type,
                "distance_meters": distance_est,
                "id": track_id,
            })

        return filtered_events
