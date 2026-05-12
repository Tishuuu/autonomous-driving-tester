import math
import json
import os
from app.utils.logger import log


class VisionFilter:
    """
    Post-YOLO logical filter.

    Goals:
    1. Keep real hazards: cross traffic, cars that move toward the ego lane,
       and cars whose distance is decreasing.
    2. Drop parked/irrelevant curb-side cars so they do not enter the feature
       vector and do not overload the AI decision stage.
    3. Keep STOP/YIELD/NO_ENTRY signs consistent with vector_builder.py.
    4. For M12, attach raw vehicle context fields to kept car events:
       relative_x, relative_y, motion_x, motion_y, static_score.

    Important: this filter runs AFTER YOLO/Kalman. It reduces events sent to
    vector_builder/model, not YOLO compute itself.
    """

    def __init__(self, video_width=1280, video_height=None):
        self.video_width = int(video_width or 1280)
        self.video_height = int(video_height or 720)
        self.vehicle_history = {}
        self.sign_history = {}
        self.confirmed_sign_keys = {}

        self.known_traffic_lights = []
        lights_file = os.path.join(
            os.path.dirname(__file__), '..', 'ai_models', 'ness_ziona_lights.json'
        )

        try:
            if os.path.exists(lights_file):
                with open(lights_file, 'r', encoding='utf-8') as f:
                    self.known_traffic_lights = json.load(f)
                log.info(f"🚦 Loaded {len(self.known_traffic_lights)} traffic lights from JSON.")
            else:
                log.warning("⚠️ Traffic lights JSON not found. Filter will not override stop signs.")
        except Exception as e:
            log.error(f"❌ Failed to load traffic lights: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = (
            math.sin(dphi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        )
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
            try:
                dist = self.haversine_distance(current_lat, current_lon, tl["lat"], tl["lon"])
                if dist < threshold_meters:
                    return True
            except Exception:
                continue
        return False

    def _safe_float(self, value, default=-1.0):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _clamp(self, value, lo, hi):
        return max(lo, min(hi, value))

    def _normalize_class(self, class_name):
        cls_name = str(class_name or "").lower().replace('-', '_').replace(' ', '_')

        is_stop_sign = "stop" in cls_name
        is_yield_sign = "yield" in cls_name
        no_entry_aliases = (
            "no_entry", "noentry", "no_entry_sign", "noentry_sign",
            "do_not_enter", "donotenter", "dont_enter", "do_not_enter_sign",
            "forbidden_entry", "wrong_way",
        )
        is_no_entry = any(alias in cls_name for alias in no_entry_aliases)
        is_car = "car" in cls_name or "vehicle" in cls_name or "truck" in cls_name or "bus" in cls_name
        is_person = "person" in cls_name or "pedestrian" in cls_name

        if is_car:
            return "car", True, False, False, False, is_person
        if is_stop_sign:
            return "stop_sign", False, True, False, False, is_person
        if is_yield_sign:
            return "yield_sign", False, False, True, False, is_person
        if is_no_entry:
            return "no_entry", False, False, False, True, is_person
        return cls_name, False, False, False, False, is_person

    def _cleanup_vehicle_history(self, current_time):
        STALE_THRESHOLD_S = 2.5
        stale_ids = [
            tid for tid, history in self.vehicle_history.items()
            if not history or (current_time - history[-1]['time']) > STALE_THRESHOLD_S
        ]
        for tid in stale_ids:
            del self.vehicle_history[tid]

    def _update_vehicle_history(self, track_id, current_time, center_x, center_y, distance_est):
        if track_id not in self.vehicle_history:
            self.vehicle_history[track_id] = []

        self.vehicle_history[track_id].append({
            'time': float(current_time),
            'x': float(center_x),
            'y': float(center_y),
            'distance': self._safe_float(distance_est, -1.0),
        })

        # A 1.4s window is long enough to tell static parked cars from cross traffic,
        # but short enough to react to a vehicle that starts moving.
        self.vehicle_history[track_id] = [
            p for p in self.vehicle_history[track_id]
            if (current_time - p['time']) <= 1.4
        ]
        return self.vehicle_history[track_id]

    def _vehicle_context_metrics(self, bbox, track_id, current_time, distance_est):
        """Return raw vehicle context fields for M12 and filtering.

        These are measurements/context, not final labels. The model can learn from
        them whether a vehicle looks parked/static/cross-traffic/lead-car.
        """
        x1, y1, x2, y2 = [self._safe_float(v, 0.0) for v in bbox]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        box_area_ratio = (box_w * box_h) / max(1.0, self.video_width * self.video_height)
        distance = self._safe_float(distance_est, -1.0)

        relative_x = self._clamp(
            (center_x - (self.video_width / 2.0)) / max(1.0, self.video_width / 2.0),
            -1.0,
            1.0,
        )
        relative_y = self._clamp(center_y / max(1.0, self.video_height), 0.0, 1.0)

        metrics = {
            'center_x': center_x,
            'center_y': center_y,
            'relative_x': relative_x,
            'relative_y': relative_y,
            'box_area_ratio': box_area_ratio,
            'motion_x': 0.0,
            'motion_y': 0.0,
            'px_speed': 0.0,
            'distance_delta': 0.0,
            'distance_rate_mps': 0.0,
            'history_len': 0,
            'static_score': 0.5 if track_id is not None else 0.0,
        }

        if track_id is None:
            return metrics

        history = self._update_vehicle_history(track_id, current_time, center_x, center_y, distance)
        metrics['history_len'] = len(history)

        if len(history) < 2:
            return metrics

        first = history[0]
        last = history[-1]
        dt = max(0.001, last['time'] - first['time'])
        dx = last['x'] - first['x']
        dy = last['y'] - first['y']
        px_speed = math.sqrt(dx * dx + dy * dy) / dt

        d0 = self._safe_float(first.get('distance'), -1.0)
        d1 = self._safe_float(last.get('distance'), -1.0)
        valid_distance = d0 > 0 and d1 > 0
        distance_delta = d1 - d0 if valid_distance else 0.0
        distance_rate_mps = distance_delta / dt if valid_distance else 0.0

        # Normalized image motion per second. Small enough to be stable for ML.
        motion_x = self._clamp((dx / max(1.0, self.video_width)) / dt, -2.0, 2.0)
        motion_y = self._clamp((dy / max(1.0, self.video_height)) / dt, -2.0, 2.0)

        # Static score is a raw heuristic for the model/debug, not a final relevance label.
        # 1.0 ~= visually/static-distance stable; 0.0 ~= clearly moving/changing.
        motion_activity = min(1.0, px_speed / 45.0)
        distance_activity = min(1.0, abs(distance_delta) / 1.5) if valid_distance else 0.0
        static_score = self._clamp(1.0 - max(motion_activity, distance_activity), 0.0, 1.0)

        metrics.update({
            'motion_x': motion_x,
            'motion_y': motion_y,
            'px_speed': px_speed,
            'distance_delta': distance_delta,
            'distance_rate_mps': distance_rate_mps,
            'static_score': static_score,
        })
        return metrics

    def _should_drop_parked_vehicle(self, bbox, track_id, current_time, distance_est, current_speed_kmh, metrics=None):
        """
        Returns True only for cars that are very likely parked/irrelevant.

        The filter is intentionally conservative:
        - New tracks are kept until we have enough history.
        - Vehicles in the driving corridor are kept.
        - Vehicles whose distance is decreasing are kept.
        - Vehicles moving from the side toward the center are kept.
        """
        if bbox is None or len(bbox) != 4:
            return False

        if metrics is None:
            metrics = self._vehicle_context_metrics(bbox, track_id, current_time, distance_est)

        center_x = metrics['center_x']
        box_area_ratio = metrics['box_area_ratio']
        distance = self._safe_float(distance_est, -1.0)
        ego_speed = self._safe_float(current_speed_kmh, 0.0)

        # Keep central vehicles. Even if one is stopped/parked, it is an object ahead.
        in_drive_corridor = (0.30 * self.video_width) <= center_x <= (0.70 * self.video_width)
        very_close = 0 < distance <= 7.0
        large_in_frame = box_area_ratio >= 0.055
        if in_drive_corridor or very_close or large_in_frame:
            return False

        # If no stable tracker id yet, keep it. Dropping new objects is unsafe.
        if track_id is None:
            return False

        if int(metrics.get('history_len', 0)) < 4:
            return False

        distance_delta = float(metrics.get('distance_delta', 0.0))
        distance_rate_mps = float(metrics.get('distance_rate_mps', 0.0))
        px_speed = float(metrics.get('px_speed', 0.0))
        motion_x = float(metrics.get('motion_x', 0.0))

        in_left_curb_zone = center_x < (self.video_width * 0.28)
        in_right_curb_zone = center_x > (self.video_width * 0.72)
        in_curb_zone = in_left_curb_zone or in_right_curb_zone

        # motion_x is normalized per second; convert the old pixel threshold to normalized-ish logic.
        moving_to_center = (
            (in_left_curb_zone and motion_x > 0.018) or
            (in_right_curb_zone and motion_x < -0.018)
        )
        approaching_fast = distance_delta < -1.0 or distance_rate_mps < -0.8
        visually_moving = px_speed > 35.0

        # When ego vehicle is moving, curb-side parked cars usually remain side/static.
        # Drop only if all danger signals are false.
        if in_curb_zone and not moving_to_center and not approaching_fast and not visually_moving:
            return True

        # Extra protection for long rows of parked cars at the far sides.
        far_side_object = in_curb_zone and distance > 12.0 and ego_speed > 3.0
        almost_static = px_speed < 18.0 and abs(distance_delta) < 0.7
        if far_side_object and almost_static:
            return True

        return False

    def _sign_passes_production_gate(self, clean_type, confidence, distance_m):
        """Class-specific sign gate after tracking."""
        conf = self._safe_float(confidence, 0.0)
        dist = self._safe_float(distance_m, 99.0)

        if clean_type == "yield_sign":
            return conf >= 0.08 and 0 < dist <= 35.0
        if clean_type == "stop_sign":
            return conf >= 0.14 and 0 < dist <= 35.0
        if clean_type == "no_entry":
            return conf >= 0.25 and 0 < dist <= 25.0
        return True

    def _cleanup_sign_history(self, current_time):
        """Remove stale sign evidence/confirmation state."""
        HISTORY_TTL_S = 2.2
        CONFIRMED_TTL_S = 1.2

        stale_hist = [
            key for key, hist in self.sign_history.items()
            if not hist or (current_time - hist[-1]["time"]) > HISTORY_TTL_S
        ]
        for key in stale_hist:
            self.sign_history.pop(key, None)

        stale_confirmed = [
            key for key, until_t in self.confirmed_sign_keys.items()
            if current_time > until_t + CONFIRMED_TTL_S
        ]
        for key in stale_confirmed:
            self.confirmed_sign_keys.pop(key, None)

    def _sign_key(self, clean_type, track_id, center_x, center_y):
        """Stable-ish key for sign evidence accumulation."""
        if track_id is not None:
            return f"{clean_type}:trk:{track_id}"
        # Fallback for detections before Kalman assigns a useful id.
        x_bucket = int(center_x // max(1.0, self.video_width * 0.12))
        y_bucket = int(center_y // max(1.0, self.video_height * 0.16))
        return f"{clean_type}:cell:{x_bucket}:{y_bucket}"

    def _sign_has_enough_evidence(
        self,
        clean_type,
        track_id,
        current_time,
        center_x,
        center_y,
        confidence,
        distance_m,
        time_since_update,
    ):
        """Reject sign ghosts from single-frame YOLO mistakes.

        Important detail: Kalman may keep a sign track alive for many frames after
        one YOLO hit. Counting those predicted frames as evidence creates false
        STOP/YIELD events in drives without signs. Therefore evidence is accumulated
        only from freshly observed tracks (`time_since_update <= 1`).
        """
        conf = self._safe_float(confidence, 0.0)
        dist = self._safe_float(distance_m, 99.0)
        tsu = int(self._safe_float(time_since_update, 99.0))
        key = self._sign_key(clean_type, track_id, center_x, center_y)

        # Allow a very short continuation only after a sign has already been
        # confirmed by real observations. This helps keep annotations stable but
        # prevents one-frame ghosts from becoming events.
        if tsu > 1:
            if clean_type == "yield_sign" and dist > 30.0:
                return False
            return key in self.confirmed_sign_keys and current_time <= self.confirmed_sign_keys[key]

        hist = self.sign_history.setdefault(key, [])
        hist.append({
            "time": float(current_time),
            "conf": conf,
            "dist": dist,
            "x": float(center_x),
            "y": float(center_y),
        })
        hist[:] = [p for p in hist if (current_time - p["time"]) <= 2.0]

        hits = len(hist)
        span = hist[-1]["time"] - hist[0]["time"] if hits >= 2 else 0.0
        max_conf = max(p["conf"] for p in hist)
        avg_conf = sum(p["conf"] for p in hist) / max(1, hits)
        min_dist = min(p["dist"] for p in hist if p["dist"] > 0) if any(p["dist"] > 0 for p in hist) else 99.0
        first_dist = hist[0]["dist"] if hist[0]["dist"] > 0 else 99.0
        last_dist = hist[-1]["dist"] if hist[-1]["dist"] > 0 else 99.0
        distance_decreased = (first_dist - last_dist) >= 1.2 or (first_dist - min_dist) >= 1.8

        confirmed = False
        if clean_type == "stop_sign":
            # STOP is emitted only after the sign gets very close.
            STOP_CLOSE_M = 11.5
            STOP_STRONG_SINGLE_M = 10.5
            STOP_VERY_CLOSE_M = 8.5
            confirmed = (
                (conf >= 0.65 and dist <= STOP_STRONG_SINGLE_M) or
                (hits >= 3 and span >= 0.35 and max_conf >= 0.45 and min_dist <= STOP_CLOSE_M) or
                (hits >= 2 and span >= 0.20 and max_conf >= 0.60 and min_dist <= STOP_CLOSE_M and distance_decreased) or
                (hits >= 2 and span >= 0.20 and min_dist <= STOP_VERY_CLOSE_M and max_conf >= 0.35)
            )
        elif clean_type == "yield_sign":
            confirmed = (
                (conf >= 0.65 and dist <= 27.0) or
                (hits >= 3 and span >= 0.45 and max_conf >= 0.20 and min_dist <= 31.0) or
                (hits >= 4 and span >= 0.75 and avg_conf >= 0.14 and min_dist <= 26.0 and distance_decreased)
            )
        elif clean_type == "no_entry":
            confirmed = (
                (conf >= 0.55 and dist <= 22.0) or
                (hits >= 2 and span >= 0.20 and max_conf >= 0.35 and min_dist <= 24.0)
            )
        else:
            confirmed = True

        if confirmed:
            if clean_type == "yield_sign" and dist > 30.0:
                log.debug(
                    f"🚧 Holding far confirmed-looking yield until closer "
                    f"key={key} dist={dist:.1f}m conf={conf:.2f} hits={hits} span={span:.2f}s"
                )
                return False

            self.confirmed_sign_keys[key] = current_time + 0.8
            return True

        log.debug(
            f"🚧 Pending/dropped unconfirmed sign {clean_type} "
            f"key={key} hits={hits} span={span:.2f}s max_conf={max_conf:.2f} "
            f"avg_conf={avg_conf:.2f} min_dist={min_dist:.1f}m tsu={tsu}"
        )
        return False

    def _vehicle_role_for_debug(self, metrics):
        rel_x = float(metrics.get('relative_x', 0.0))
        rel_y = float(metrics.get('relative_y', 0.0))
        motion_x = float(metrics.get('motion_x', 0.0))
        distance_delta = float(metrics.get('distance_delta', 0.0))

        if abs(rel_x) <= 0.35 and rel_y >= 0.30:
            return "lead_or_front_vehicle"
        if abs(motion_x) >= 0.018 or distance_delta < -1.0:
            return "dynamic_side_vehicle"
        return "kept_vehicle"

    # ------------------------------------------------------------------
    # Main filter
    # ------------------------------------------------------------------
    def filter_detections(self, current_time, current_lat, current_lon, frame_detections, current_speed_kmh=None):
        filtered_events = []
        self._cleanup_vehicle_history(current_time)
        self._cleanup_sign_history(current_time)

        near_traffic_light = self.is_near_traffic_light(current_lat, current_lon, threshold_meters=12)
        speed = self._safe_float(current_speed_kmh, None)
        is_stationary_near_light = (
            near_traffic_light and speed is not None and speed <= 2.0
        )

        for det in frame_detections:
            bbox = det.get('bbox')
            if bbox is None or len(bbox) != 4:
                continue

            clean_type, is_car, is_stop_sign, is_yield_sign, is_no_entry, _ = self._normalize_class(
                det.get('class_name')
            )

            track_id = det.get('id', None)
            distance_est = det.get('distance_est', -1)
            confidence = det.get("confidence", 0.0)

            if (is_stop_sign or is_yield_sign or is_no_entry) and not self._sign_passes_production_gate(
                clean_type, confidence, distance_est
            ):
                log.debug(
                    f"🚧 Dropped weak/far sign {clean_type} "
                    f"conf={self._safe_float(confidence, 0.0):.2f} "
                    f"dist={self._safe_float(distance_est, 99.0):.1f}m"
                )
                continue

            x1, y1, x2, y2 = [self._safe_float(v, 0.0) for v in bbox]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            time_since_update = det.get("time_since_update", 0)

            vehicle_metrics = None
            if is_car:
                vehicle_metrics = self._vehicle_context_metrics(
                    bbox=bbox,
                    track_id=track_id,
                    current_time=current_time,
                    distance_est=distance_est,
                )

            # 1. Sign ROI: ignore only extreme-left signs that are probably from
            # the opposite/side lane. The old 35% cutoff deleted valid yield signs.
            if is_stop_sign or is_yield_sign or is_no_entry:
                dist = self._safe_float(distance_est, 99.0)
                extreme_left = center_x < (self.video_width * 0.22)
                far_left = center_x < (self.video_width * 0.30)
                if extreme_left or (far_left and dist > 18.0):
                    continue

                if not self._sign_has_enough_evidence(
                    clean_type=clean_type,
                    track_id=track_id,
                    current_time=current_time,
                    center_x=center_x,
                    center_y=center_y,
                    confidence=confidence,
                    distance_m=distance_est,
                    time_since_update=time_since_update,
                ):
                    continue

            # 2. Traffic-light override: do not treat a nearby traffic light area as STOP/YIELD
            # only while stationary, so moving detections are not suppressed too aggressively.
            if is_stationary_near_light and (is_stop_sign or is_yield_sign):
                log.debug(f"🚦 Skipped {clean_type} due to stationary traffic light override.")
                continue

            # 3. Parked-car filter. This is the main anti-overload filter for the AI vector.
            if is_car:
                if self._should_drop_parked_vehicle(
                    bbox=bbox,
                    track_id=track_id,
                    current_time=current_time,
                    distance_est=distance_est,
                    current_speed_kmh=current_speed_kmh,
                    metrics=vehicle_metrics,
                ):
                    log.debug(
                        f"🅿️ Dropped likely parked car id={track_id} t={current_time:.2f}s "
                        f"rel_x={vehicle_metrics.get('relative_x', 0.0):.2f} "
                        f"static={vehicle_metrics.get('static_score', 0.0):.2f}"
                    )
                    continue

            event = {
                "time_sec": current_time,
                "type": clean_type,
                "distance_meters": distance_est,
                "id": track_id,
                "confidence": confidence,
                "time_since_update": time_since_update,
            }

            if is_car and vehicle_metrics is not None:
                event.update({
                    "relative_x": round(float(vehicle_metrics.get('relative_x', 0.0)), 5),
                    "relative_y": round(float(vehicle_metrics.get('relative_y', 0.0)), 5),
                    "motion_x": round(float(vehicle_metrics.get('motion_x', 0.0)), 5),
                    "motion_y": round(float(vehicle_metrics.get('motion_y', 0.0)), 5),
                    "static_score": round(float(vehicle_metrics.get('static_score', 0.0)), 5),
                    "box_area_ratio": round(float(vehicle_metrics.get('box_area_ratio', 0.0)), 6),
                    "role": self._vehicle_role_for_debug(vehicle_metrics),
                    "used_in_vector": True,
                })

            filtered_events.append(event)

        return filtered_events
