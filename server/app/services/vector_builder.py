import pandas as pd
import numpy as np
from app.utils.logger import log


SIGN_MAPPING = {'none': 0, 'stop_sign': 1, 'yield_sign': 2, 'no_entry': 3}

# Balanced production gates. vision_service/vision_filter should already remove most
# weak detections, but replay files can still contain noisy signs. The vector builder
# is the last protection layer before the AI model.
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

# M12 sign memory: context, not decision labels.
# These values let the model learn: "I saw STOP/YIELD earlier, I have driven X meters since then".
# They do NOT tell the model whether the driver stopped or yielded correctly.
SIGN_MEMORY_TTL_S = {
    'stop_sign': 8.0,
    'yield_sign': 8.0,
    'no_entry': 5.0,
}
SIGN_MEMORY_MAX_METERS = {
    'stop_sign': 38.0,
    'yield_sign': 38.0,
    'no_entry': 24.0,
}

M12_FINAL_COLUMNS = [
    'time_seconds',
    'speed_kmh',
    'jerk',

    # Vehicle context. car_distance/car_ttc are still the primary legacy fields.
    # The extra fields are raw-ish vision measurements for M12.
    'car_distance',
    'car_ttc',
    'car_relative_x',
    'car_relative_y',
    'car_motion_x',
    'car_motion_y',
    'car_static_score',

    # Current visible / short DR sign context.
    'sign_type',
    'sign_distance',
    'sign_ttc',

    # Sign memory context after the sign leaves the frame.
    'memory_sign_type',
    'memory_sign_distance',
    'time_since_sign_seen_sec',
    'meters_since_sign_seen',
    'last_seen_sign_distance',
    'memory_sign_ttc',
]


def _safe_float(value, default=0.0):
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _normalize_sign_type(raw_type: str) -> str:
    """Normalize YOLO/vision-filter labels to the AI sign vocabulary."""
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

    # Vehicle M12 raw/context fields. Missing fields mean older video_events were used.
    numeric_defaults = {
        'relative_x': 0.0,
        'relative_y': 0.0,
        'motion_x': 0.0,
        'motion_y': 0.0,
        'static_score': 0.0,
        'box_area_ratio': 0.0,
    }
    for col, default in numeric_defaults.items():
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors='coerce').fillna(default)
        else:
            events_df[col] = default

    if 'used_in_vector' not in events_df.columns:
        events_df['used_in_vector'] = True
    else:
        # Keep permissive parsing for json/bool/string sources.
        events_df['used_in_vector'] = events_df['used_in_vector'].apply(
            lambda v: False if str(v).lower() in ('false', '0', 'no', 'none') else bool(v)
        )

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
                'start': first_t,
                'end': last_t + extra_s,
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


def _init_m12_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['car_distance'] = 99.0
    df['car_ttc'] = 99.0
    df['car_relative_x'] = 0.0
    df['car_relative_y'] = 0.0
    df['car_motion_x'] = 0.0
    df['car_motion_y'] = 0.0
    df['car_static_score'] = 0.0
    df['_car_seen'] = 0

    df['sign_type'] = 'none'
    df['sign_distance'] = 99.0
    df['sign_ttc'] = 99.0

    df['memory_sign_type'] = 'none'
    df['memory_sign_distance'] = 99.0
    df['time_since_sign_seen_sec'] = 99.0
    df['meters_since_sign_seen'] = 99.0
    df['last_seen_sign_distance'] = 99.0
    df['memory_sign_ttc'] = 99.0

    return df


def _compute_odometer_m(df: pd.DataFrame) -> pd.Series:
    times = pd.to_numeric(df['time_seconds'], errors='coerce').fillna(0.0)
    speeds_ms = pd.to_numeric(df['speed_kmh'], errors='coerce').fillna(0.0) / 3.6
    dt = times.diff().fillna(0.0).clip(lower=0.0, upper=1.0)
    return (speeds_ms * dt).cumsum()


def _fill_vehicle_from_event(df: pd.DataFrame, idx, row, car_row, speed_ms: float) -> None:
    car_dist = _safe_float(car_row.get('distance_meters'), 99.0)
    df.at[idx, 'car_distance'] = round(car_dist, 2)
    if speed_ms > 0.5 and car_dist < 99.0:
        df.at[idx, 'car_ttc'] = round(car_dist / speed_ms, 2)

    df.at[idx, 'car_relative_x'] = round(_safe_float(car_row.get('relative_x'), 0.0), 4)
    df.at[idx, 'car_relative_y'] = round(_safe_float(car_row.get('relative_y'), 0.0), 4)
    df.at[idx, 'car_motion_x'] = round(_safe_float(car_row.get('motion_x'), 0.0), 4)
    df.at[idx, 'car_motion_y'] = round(_safe_float(car_row.get('motion_y'), 0.0), 4)
    df.at[idx, 'car_static_score'] = round(_safe_float(car_row.get('static_score'), 0.0), 4)
    df.at[idx, '_car_seen'] = 1


def _apply_car_blink_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill short car tracking blinks while keeping no-car rows clean."""
    car_cols = [
        'car_distance',
        'car_ttc',
        'car_relative_x',
        'car_relative_y',
        'car_motion_x',
        'car_motion_y',
        'car_static_score',
    ]

    no_car_mask = df['_car_seen'] == 0
    for col in car_cols:
        df.loc[no_car_mask, col] = np.nan
        df[col] = df[col].ffill(limit=5)

    df['car_distance'] = df['car_distance'].fillna(99.0)
    df['car_ttc'] = df['car_ttc'].fillna(99.0)
    for col in ['car_relative_x', 'car_relative_y', 'car_motion_x', 'car_motion_y', 'car_static_score']:
        df[col] = df[col].fillna(0.0)

    df.drop(columns=['_car_seen'], inplace=True, errors='ignore')
    return df


def build_feature_vector(sensor_df: pd.DataFrame, video_events: list) -> pd.DataFrame:
    """
    Build M12-ready feature vector from sensors + filtered video events.

    Principles:
    - The vector contains raw/context features, not final decisions.
    - STOP/YIELD memory persists briefly after the sign leaves the frame, so the
      model can infer behavior that occurs after seeing the sign.
    - Vehicle context uses only cars that passed VisionFilter; parked/irrelevant
      cars should not poison car_distance/car_ttc.
    """
    log.info("🧬 Building M12-ready vector with sign memory + vehicle context...")

    df = _init_m12_columns(sensor_df.copy())

    if 'time_seconds' not in df.columns:
        raise ValueError("sensor_df must contain time_seconds")
    if 'speed_kmh' not in df.columns:
        raise ValueError("sensor_df must contain speed_kmh")
    if 'jerk' not in df.columns:
        df['jerk'] = 0.0

    df['_odometer_m'] = _compute_odometer_m(df)

    if not video_events:
        log.warning("⚠️ No video events. Returning clean M12 vector.")
        df['sign_type'] = 0
        df['memory_sign_type'] = 0
        df.drop(columns=['_odometer_m', '_car_seen'], inplace=True, errors='ignore')
        return df[M12_FINAL_COLUMNS]

    events_df = _prepare_events(video_events)
    if events_df.empty:
        log.warning("⚠️ Video events are missing required columns. Returning clean M12 vector.")
        df['sign_type'] = 0
        df['memory_sign_type'] = 0
        df.drop(columns=['_odometer_m', '_car_seen'], inplace=True, errors='ignore')
        return df[M12_FINAL_COLUMNS]

    sign_segments = _build_sign_segments(events_df)
    log.info(f"🚦 Sign segments kept: {len(sign_segments)}")

    memory_type = 'none'
    memory_last_time = None
    memory_last_odometer = 0.0
    memory_last_distance = 99.0

    for idx, row in df.iterrows():
        current_time = _safe_float(row.get('time_seconds'), 0.0)
        speed_kmh = _safe_float(row.get('speed_kmh'), 0.0)
        speed_ms = speed_kmh / 3.6
        odometer_m = _safe_float(row.get('_odometer_m'), 0.0)

        # Vehicle selection: only events explicitly allowed by VisionFilter.
        # Use a slightly wider window because vision_service emits final video_events sparsely.
        car_window = events_df[
            (abs(events_df['time_sec'] - current_time) <= 0.35)
            & (events_df['type'].apply(_is_vehicle_type))
            & (events_df['used_in_vector'] == True)  # noqa: E712
        ]
        if not car_window.empty:
            closest_car = car_window.loc[car_window['distance_meters'].idxmin()]
            _fill_vehicle_from_event(df, idx, row, closest_car, speed_ms)

        # Current sign segment, if YOLO/tracking says the sign is still active.
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

            # Update sign memory from current sign observation/segment.
            memory_type = seg['type']
            memory_last_time = current_time
            memory_last_odometer = odometer_m
            memory_last_distance = float(dist)

        # M12 sign memory, valid even after sign_type returns to none.
        if memory_type != 'none' and memory_last_time is not None:
            time_since = max(0.0, current_time - float(memory_last_time))
            meters_since = max(0.0, odometer_m - float(memory_last_odometer))
            ttl = SIGN_MEMORY_TTL_S.get(memory_type, 6.0)
            max_m = SIGN_MEMORY_MAX_METERS.get(memory_type, 30.0)

            if time_since <= ttl and meters_since <= max_m:
                memory_dist = max(0.0, float(memory_last_distance) - meters_since)
                df.at[idx, 'memory_sign_type'] = memory_type
                df.at[idx, 'memory_sign_distance'] = round(memory_dist, 2)
                df.at[idx, 'time_since_sign_seen_sec'] = round(time_since, 2)
                df.at[idx, 'meters_since_sign_seen'] = round(meters_since, 2)
                df.at[idx, 'last_seen_sign_distance'] = round(float(memory_last_distance), 2)
                if speed_ms > 0.5:
                    df.at[idx, 'memory_sign_ttc'] = round(memory_dist / speed_ms, 2)
                else:
                    df.at[idx, 'memory_sign_ttc'] = 99.0
            else:
                memory_type = 'none'
                memory_last_time = None
                memory_last_odometer = 0.0
                memory_last_distance = 99.0

    # Forward-fill car blinks only. Do not forward-fill signs; sign memory handles temporal context.
    df = _apply_car_blink_ffill(df)

    # Sanitize TTC=0 and map sign strings to numeric codes.
    df['car_ttc'] = df['car_ttc'].replace(0, 99.0)
    df['sign_ttc'] = df['sign_ttc'].replace(0, 99.0)
    df['memory_sign_ttc'] = df['memory_sign_ttc'].replace(0, 99.0)

    df['sign_type'] = df['sign_type'].map(SIGN_MAPPING).fillna(0).astype(int)
    df['memory_sign_type'] = df['memory_sign_type'].map(SIGN_MAPPING).fillna(0).astype(int)

    # Ensure stable numeric output.
    for col in M12_FINAL_COLUMNS:
        if col != 'time_seconds':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(99.0 if 'distance' in col or 'ttc' in col else 0.0)

    sign_counts = df['sign_type'].value_counts().to_dict()
    memory_counts = df['memory_sign_type'].value_counts().to_dict()
    log.info(
        f"✅ M12 vector built. Shape: {df[M12_FINAL_COLUMNS].shape} | "
        f"sign_counts={sign_counts} | memory_sign_counts={memory_counts}"
    )

    df.drop(columns=['_odometer_m'], inplace=True, errors='ignore')
    return df[M12_FINAL_COLUMNS]
