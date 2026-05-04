import pandas as pd
import numpy as np
from app.utils.logger import log


SIGN_MAPPING = {'none': 0, 'stop_sign': 1, 'yield_sign': 2, 'no_entry': 3}

# Balanced production gates. vision_service/vision_filter should already remove most
# weak detections, but replay files can still contain noisy signs. The vector builder
# is the last protection layer before M8.
SIGN_GATES = {
    'stop_sign': {
        'min_conf': 0.14,
        'max_dist': 35.0,
        'segment_min_conf': 0.18,
        'strong_conf': 0.50,
        'close_dist': 14.0,
        'weak_extra_s': 0.65,
        'confirmed_extra_s': 1.35,
        'cluster_gap_s': 0.90,
        'max_bridge_gap_s': 1.60,
    },
    'yield_sign': {
        'min_conf': 0.08,
        'max_dist': 35.0,
        'segment_min_conf': 0.12,
        'strong_conf': 0.28,
        'close_dist': 22.0,
        'weak_extra_s': 0.80,
        'confirmed_extra_s': 1.25,
        'cluster_gap_s': 1.00,
        'max_bridge_gap_s': 1.50,
    },
    'no_entry': {
        'min_conf': 0.25,
        'max_dist': 25.0,
        'segment_min_conf': 0.55,
        'strong_conf': 0.65,
        'close_dist': 12.0,
        'weak_extra_s': 0.45,
        'confirmed_extra_s': 0.85,
        'cluster_gap_s': 0.75,
        'max_bridge_gap_s': 1.10,
    },
}

# Conflict tie-breaker only. Confidence and distance are still more important.
SIGN_PRIORITY = {'yield_sign': 3, 'stop_sign': 2, 'no_entry': 1}

# Only allow dead-reckoning after the last YOLO observation for close/mid signs.
# Far signs must be refreshed by YOLO, otherwise they flood the feature vector.
DR_KEEP_MAX_DISTANCE_M = 28.0


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _normalize_sign_type(raw_type: str) -> str:
    """Normalize YOLO/vision-filter labels to the M8 sign vocabulary."""
    t = str(raw_type or "").lower().strip().replace("-", "_").replace(" ", "_")

    yield_aliases = (
        "yield", "yield_sign", "yieldsign",
        "give_way", "giveway", "giveway_sign", "give_way_sign",
        "give_priority", "priority", "right_of_way", "rightofway",
        "give_right_of_way", "give_rightofway",
        "triangle_sign", "triangular_sign",
    )
    no_entry_aliases = (
        "no_entry", "noentry", "no_entry_sign", "noentry_sign",
        "do_not_enter", "donotenter", "dont_enter", "do_not_enter_sign",
        "forbidden_entry", "wrong_way",
    )

    if "stop" in t:
        return "stop_sign"
    if any(a in t for a in yield_aliases):
        return "yield_sign"
    # Do not use generic "entry". It is too broad and causes false no_entry rows.
    if any(a in t for a in no_entry_aliases):
        return "no_entry"
    return "none"


def _is_vehicle_type(raw_type: str) -> bool:
    t = str(raw_type or "").lower().replace("-", "_").replace(" ", "_")
    return any(k in t for k in ("car", "vehicle", "bus", "truck"))


def _sign_gate_ok(sign_type: str, distance_m: float, confidence: float) -> bool:
    gate = SIGN_GATES.get(sign_type)
    if gate is None:
        return False
    if distance_m <= 0 or distance_m > gate['max_dist']:
        return False
    if confidence < gate['min_conf']:
        return False
    return True


def _prepare_events(video_events: list) -> pd.DataFrame:
    events_df = pd.DataFrame(video_events)
    if events_df.empty or 'type' not in events_df.columns or 'time_sec' not in events_df.columns:
        return pd.DataFrame()

    events_df = events_df.copy()
    events_df['time_sec'] = pd.to_numeric(events_df['time_sec'], errors='coerce')
    events_df['distance_meters'] = pd.to_numeric(
        events_df.get('distance_meters', 99.0), errors='coerce'
    ).fillna(99.0)
    if 'confidence' in events_df.columns:
        events_df['confidence'] = pd.to_numeric(events_df['confidence'], errors='coerce').fillna(1.0)
    else:
        events_df['confidence'] = 1.0
    if 'id' not in events_df.columns:
        events_df['id'] = None
    events_df = events_df.dropna(subset=['time_sec']).sort_values('time_sec').reset_index(drop=True)
    return events_df


def _build_sign_segments(events_df: pd.DataFrame) -> list[dict]:
    """
    Convert sparse sign detections into short, validated sign segments.

    Why this exists:
    - A single weak sign detection must not stay alive for 3-6 seconds.
    - Repeated/strong detections should persist briefly after YOLO blinks.
    - Conflicting signs at the same physical location are resolved later by score.
    """
    if events_df.empty:
        return []

    sign_rows = []
    for event_index, row in events_df.iterrows():
        sign_type = _normalize_sign_type(row.get('type'))
        if sign_type == 'none':
            continue
        dist = _safe_float(row.get('distance_meters'), 99.0)
        conf = _safe_float(row.get('confidence'), 1.0)
        if not _sign_gate_ok(sign_type, dist, conf):
            continue
        sign_rows.append({
            'event_index': int(event_index),
            'time_sec': _safe_float(row.get('time_sec'), 0.0),
            'type': sign_type,
            'distance': dist,
            'confidence': conf,
            'id': row.get('id'),
        })

    if not sign_rows:
        return []

    signs_df = pd.DataFrame(sign_rows).sort_values(['type', 'time_sec']).reset_index(drop=True)
    segments: list[dict] = []

    for sign_type, group in signs_df.groupby('type', sort=False):
        gate = SIGN_GATES[sign_type]
        current: list[dict] = []

        def flush_segment(items: list[dict]):
            if not items:
                return
            first_t = float(items[0]['time_sec'])
            last_t = float(items[-1]['time_sec'])
            max_conf = max(float(x['confidence']) for x in items)
            min_dist = min(float(x['distance']) for x in items)
            n_hits = len(items)
            duration = max(0.0, last_t - first_t)

            confirmed = (
                n_hits >= 2
                or duration >= 0.25
                or max_conf >= gate['strong_conf']
                or min_dist <= gate['close_dist']
            )

            # Very weak one-frame signs are usually false positives. Yield is allowed
            # slightly lower than stop/no_entry because it was empirically weaker.
            if max_conf < gate['segment_min_conf'] and not (confirmed and min_dist <= gate['close_dist']):
                return

            extra_s = gate['confirmed_extra_s'] if confirmed else gate['weak_extra_s']

            # Far signs are useful only while visible. Do not let far single-frame
            # detections live for seconds after YOLO loses them.
            if min_dist > DR_KEEP_MAX_DISTANCE_M:
                extra_s = min(extra_s, 0.45)

            # If a detection cluster itself is long, keep the whole visible interval,
            # then only a short post-YOLO tail.
            start_t = first_t
            end_t = last_t + extra_s

            obs = [
                {
                    'time': float(x['time_sec']),
                    'distance': float(x['distance']),
                    'confidence': float(x['confidence']),
                }
                for x in items
            ]
            segments.append({
                'type': sign_type,
                'start': start_t,
                'end': end_t,
                'first_seen': first_t,
                'last_seen': last_t,
                'n_hits': n_hits,
                'max_conf': max_conf,
                'min_dist': min_dist,
                'confirmed': confirmed,
                'obs': obs,
            })

        for _, row in group.iterrows():
            item = row.to_dict()
            if not current:
                current = [item]
                continue

            prev = current[-1]
            gap = float(item['time_sec']) - float(prev['time_sec'])
            dist_jump = abs(float(item['distance']) - float(prev['distance']))

            # Keep sparse but related detections together only when the gap is not
            # too large and distance is plausible. Otherwise start a new segment.
            if gap <= gate['cluster_gap_s'] and dist_jump <= 14.0:
                current.append(item)
            elif gap <= gate['max_bridge_gap_s'] and dist_jump <= 7.0:
                current.append(item)
            else:
                flush_segment(current)
                current = [item]

        flush_segment(current)

    return sorted(segments, key=lambda s: s['start'])


def _segment_distance_at(seg: dict, current_time: float, speed_ms: float) -> float:
    obs = seg.get('obs') or []
    if not obs:
        return 99.0

    last = obs[0]
    for item in obs:
        if float(item['time']) <= current_time:
            last = item
        else:
            break

    elapsed = max(0.0, current_time - float(last['time']))
    dist = float(last['distance']) - (speed_ms * elapsed)
    return max(0.0, min(99.0, dist))


def _select_active_segment(segments: list[dict], current_time: float, speed_ms: float) -> dict | None:
    candidates = []
    for seg in segments:
        if current_time < float(seg['start']) or current_time > float(seg['end']):
            continue
        dist = _segment_distance_at(seg, current_time, speed_ms)
        if dist > SIGN_GATES[seg['type']]['max_dist']:
            continue

        # Score: confidence dominates, then sign priority, then closer distance.
        # This prevents a generic weak stop/no_entry from hiding a yield sign, while
        # still allowing a strong nearby stop to win.
        priority = SIGN_PRIORITY.get(seg['type'], 0)
        score = (float(seg['max_conf']) * 3.0) + (priority * 0.06) - (dist * 0.012)
        if seg.get('confirmed'):
            score += 0.04
        candidates.append({
            'segment': seg,
            'distance': dist,
            'score': score,
        })

    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x['score'], reverse=True)[0]


def build_feature_vector(sensor_df: pd.DataFrame, video_events: list) -> pd.DataFrame:
    """
    Build M8 feature vector from sensors + filtered video events.

    v2 fixes:
    - sign detections are clustered into short validated segments;
    - weak/single detections get <1 second of influence;
    - repeated/strong detections get only a short post-YOLO tail;
    - far signs cannot remain alive after YOLO loses them;
    - confidence-first conflict resolution avoids false stop/no_entry overriding yield.
    """
    log.info("🧬 Building vector with segmented sign gates v2...")

    df = sensor_df.copy()
    df['car_distance'] = 99.0
    df['car_ttc'] = 99.0
    df['sign_type'] = "none"
    df['sign_distance'] = 99.0
    df['sign_ttc'] = 99.0

    FINAL_COLUMNS = ['time_seconds', 'speed_kmh', 'jerk', 'car_distance',
                     'car_ttc', 'sign_type', 'sign_distance', 'sign_ttc']

    if not video_events:
        log.warning("⚠️ No video events. Returning clean vector.")
        df['sign_type'] = 0
        return df[FINAL_COLUMNS]

    events_df = _prepare_events(video_events)
    if events_df.empty:
        log.warning("⚠️ Video events are missing required columns. Returning clean vector.")
        df['sign_type'] = 0
        return df[FINAL_COLUMNS]

    sign_segments = _build_sign_segments(events_df)
    log.info(f"🚦 Sign segments kept: {len(sign_segments)}")

    for idx, row in df.iterrows():
        current_time = _safe_float(row.get('time_seconds'), 0.0)
        speed_kmh = _safe_float(row.get('speed_kmh'), 0.0)
        speed_ms = speed_kmh / 3.6

        window = events_df[abs(events_df['time_sec'] - current_time) <= 0.2]

        if not window.empty:
            cars = window[window['type'].apply(_is_vehicle_type)]
            if not cars.empty:
                closest_car = cars.loc[cars['distance_meters'].idxmin()]
                car_dist = _safe_float(closest_car.get('distance_meters'), 99.0)
                df.at[idx, 'car_distance'] = round(car_dist, 2)
                if speed_ms > 0.5 and car_dist < 99.0:
                    df.at[idx, 'car_ttc'] = round(car_dist / speed_ms, 2)

        selected = _select_active_segment(sign_segments, current_time, speed_ms)
        if selected is not None:
            seg = selected['segment']
            dist = selected['distance']
            df.at[idx, 'sign_type'] = seg['type']
            df.at[idx, 'sign_distance'] = round(dist, 2)
            if speed_ms > 0.5:
                df.at[idx, 'sign_ttc'] = round(dist / speed_ms, 2)
            else:
                df.at[idx, 'sign_ttc'] = 99.0

    # Forward-fill car blinks only. Do not forward-fill signs.
    df.replace(99.0, np.nan, inplace=True)
    df['car_distance'] = df['car_distance'].ffill(limit=5)
    df['car_ttc'] = df['car_ttc'].ffill(limit=5)
    df.fillna(99.0, inplace=True)

    df['car_ttc'] = df['car_ttc'].replace(0, 99.0)
    df['sign_ttc'] = df['sign_ttc'].replace(0, 99.0)
    df['sign_type'] = df['sign_type'].map(SIGN_MAPPING).fillna(0).astype(int)

    counts = df['sign_type'].value_counts().to_dict()
    log.info(f"✅ Vector built. Shape: {df.shape} | sign_counts={counts}")
    return df[FINAL_COLUMNS]
