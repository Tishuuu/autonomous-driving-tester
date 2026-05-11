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

# M13 keeps the exact M12 schema and adds only continuous, per-row physical
# measurements derived from raw sensors. No episode summaries / decisions.
M13_ADDED_COLUMNS = [
    # GPS quality / freshness
    'gps_accuracy',
    'gps_age_ms',
    'gps_update_count_delta',

    # Local GPS motion between consecutive rows, not absolute location.
    'gps_dx_m',
    'gps_dy_m',
    'gps_step_distance_m',
    'gps_speed_kmh',

    # Heading / turn dynamics from GPS movement.
    'heading_sin',
    'heading_cos',
    'heading_delta_deg',
    'turn_rate_deg_s',

    # Raw user-accelerometer channels from the client.
    'accel_x',
    'accel_y',
    'accel_z',
]

M13_FINAL_COLUMNS = M12_FINAL_COLUMNS + M13_ADDED_COLUMNS



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



def _wrap_degrees(delta: pd.Series) -> pd.Series:
    """Wrap angle differences to [-180, 180]."""
    return ((delta + 180.0) % 360.0) - 180.0



def _compute_m13_sensor_features(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build M13 added features from raw 10Hz sensor rows.

    This is intentionally a lightweight map-matching / gap-repair stage, not a
    road-rule engine:
    - uses only the driven GPS trace itself, not OSM road metadata;
    - rejects impossible GPS jumps;
    - smooths the remaining route points;
    - interpolates over GPS update gaps;
    - computes per-row local motion, heading and turn rate from the repaired path.

    The output remains pure timestamp-level measurements. It does not contain
    episode summaries such as turned_away_from_no_entry or unsafe_entry_score.
    """
    df = sensor_df.copy().reset_index(drop=True)
    n = len(df)
    out = pd.DataFrame(index=df.index)

    def num_col(col: str, default: float = 0.0) -> pd.Series:
        if col not in df.columns:
            return pd.Series([default] * n, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors='coerce').fillna(default).astype(float)

    if n == 0:
        for col in M13_ADDED_COLUMNS:
            out[col] = []
        return out[M13_ADDED_COLUMNS]

    time_s = num_col('time_seconds', 0.0)
    # Keep the original ordering but protect against duplicate / broken times.
    dt = time_s.diff().replace(0, np.nan).fillna(0.1).clip(0.05, 1.0)

    lat = num_col('lat', np.nan)
    lon = num_col('lon', np.nan)
    vehicle_speed_kmh = num_col('speed_kmh', 0.0).clip(0.0, 150.0)

    gps_accuracy = num_col('gps_accuracy', 999.0).clip(0.0, 999.0)
    gps_age_ms = num_col('gps_age_ms', 999999.0).clip(0.0, 999999.0)
    gps_update_count = num_col('gps_update_count', 0.0)
    gps_update_count_delta = gps_update_count.diff().fillna(0.0).clip(0.0, 10.0)

    valid_latlon = (
        np.isfinite(lat)
        & np.isfinite(lon)
        & lat.between(-90.0, 90.0)
        & lon.between(-180.0, 180.0)
        & ~((lat.abs() < 1e-9) & (lon.abs() < 1e-9))
    )

    # Defaults for clips with no usable GPS. This should be rare because
    # sensor_sync warns on frozen GPS, but replay should stay stable.
    matched_x = pd.Series([0.0] * n, index=df.index, dtype=float)
    matched_y = pd.Series([0.0] * n, index=df.index, dtype=float)

    if valid_latlon.sum() >= 2:
        earth_radius_m = 6371000.0
        lat0 = float(lat[valid_latlon].iloc[0])
        lon0 = float(lon[valid_latlon].iloc[0])
        lat0_rad = np.deg2rad(lat0)

        # Local tangent-plane approximation. The route clips are short, so this is
        # more than accurate enough and avoids pulling absolute lat/lon into the model.
        x_raw = (np.deg2rad(lon - lon0) * np.cos(lat0_rad) * earth_radius_m)
        y_raw = (np.deg2rad(lat - lat0) * earth_radius_m)

        raw_step = np.sqrt((x_raw.diff() ** 2) + (y_raw.diff() ** 2)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Real GPS update rows: prefer gps_update_count when present, fall back to
        # actual coordinate movement. Repeated resampled 10Hz rows are not treated
        # as fresh GPS observations.
        gps_quality_ok = (gps_accuracy <= 60.0) & (gps_age_ms <= 6000.0)
        fresh_update = (gps_update_count_delta > 0) | (raw_step >= 0.20)
        candidate_mask = valid_latlon & gps_quality_ok & fresh_update

        # Always seed with the first usable coordinate so interpolation has an anchor.
        first_valid_idx = int(np.where(valid_latlon.to_numpy())[0][0])
        candidate_mask.iloc[first_valid_idx] = True

        # If quality metadata is too strict/missing, fall back to real coordinate updates.
        if candidate_mask.sum() < 2:
            candidate_mask = valid_latlon & ((raw_step >= 0.20) | (df.index == first_valid_idx))

        pts = pd.DataFrame({
            'idx': np.arange(n),
            't': time_s,
            'x': x_raw,
            'y': y_raw,
            'vehicle_speed_kmh': vehicle_speed_kmh,
        })[candidate_mask].dropna(subset=['t', 'x', 'y']).copy()

        pts = pts.sort_values('t').drop_duplicates(subset=['t'], keep='last').reset_index(drop=True)

        # Reject impossible GPS jumps using the vehicle speed as a soft reference.
        # This is gap repair / measurement sanitation, not a driving decision.
        if len(pts) >= 2:
            kept = []
            for _, p in pts.iterrows():
                cur = {
                    't': float(p['t']),
                    'x': float(p['x']),
                    'y': float(p['y']),
                    'vehicle_speed_kmh': float(p['vehicle_speed_kmh']),
                }
                if not kept:
                    kept.append(cur)
                    continue

                prev = kept[-1]
                gap_s = max(0.001, cur['t'] - prev['t'])
                dist_m = float(np.hypot(cur['x'] - prev['x'], cur['y'] - prev['y']))
                gps_mps = dist_m / gap_s
                veh_mps = max(cur['vehicle_speed_kmh'], prev['vehicle_speed_kmh']) / 3.6

                # City-drive tolerant cap: accepts normal acceleration and GPS jitter,
                # rejects teleport jumps. The lower bound prevents over-pruning when
                # OBD/GPS speed is stale or zero during a turn.
                max_reasonable_mps = min(38.0, max(14.0, veh_mps * 3.0 + 8.0))

                if gps_mps <= max_reasonable_mps or dist_m <= 4.0:
                    kept.append(cur)

            if len(kept) >= 2:
                pts = pd.DataFrame(kept)

        if len(pts) >= 2:
            # Route smoothing. This is the practical replacement for full OSM HMM
            # at M13 stage: the path is matched to a smoothed local route line.
            pts['x_smooth'] = (
                pts['x']
                .rolling(window=3, center=True, min_periods=1).median()
                .ewm(alpha=0.45, adjust=False).mean()
            )
            pts['y_smooth'] = (
                pts['y']
                .rolling(window=3, center=True, min_periods=1).median()
                .ewm(alpha=0.45, adjust=False).mean()
            )

            # Interpolate across gaps to 10Hz rows. np.interp holds the first/last
            # value outside the observed range, which is safer than extrapolating.
            matched_x = pd.Series(np.interp(time_s.to_numpy(), pts['t'].to_numpy(), pts['x_smooth'].to_numpy()), index=df.index)
            matched_y = pd.Series(np.interp(time_s.to_numpy(), pts['t'].to_numpy(), pts['y_smooth'].to_numpy()), index=df.index)
        elif len(pts) == 1:
            matched_x = pd.Series([float(pts.iloc[0]['x'])] * n, index=df.index)
            matched_y = pd.Series([float(pts.iloc[0]['y'])] * n, index=df.index)

    # Per-row local motion from the repaired route. Since the route is already
    # interpolated, these no longer spike when GPS updates arrive at 1Hz.
    gps_dx_m = matched_x.diff().fillna(0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gps_dy_m = matched_y.diff().fillna(0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gps_step_distance_m = np.sqrt((gps_dx_m ** 2) + (gps_dy_m ** 2)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    gps_dx_m = gps_dx_m.clip(-8.0, 8.0)
    gps_dy_m = gps_dy_m.clip(-8.0, 8.0)
    gps_step_distance_m = gps_step_distance_m.clip(0.0, 12.0)

    gps_speed_kmh = ((gps_step_distance_m / dt) * 3.6).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gps_speed_kmh = (
        gps_speed_kmh
        .rolling(window=5, center=True, min_periods=1).median()
        .ewm(alpha=0.35, adjust=False).mean()
        .clip(0.0, 130.0)
    )

    # Heading from repaired route. Represent absolute heading as sin/cos to avoid
    # discontinuity at 359 -> 0 degrees. Delta/rate remain local measurements.
    step_ok = gps_step_distance_m >= 0.035
    raw_heading_deg = (np.degrees(np.arctan2(gps_dx_m, gps_dy_m)) + 360.0) % 360.0
    raw_heading_rad = np.deg2rad(raw_heading_deg)

    heading_sin_series = pd.Series(np.sin(raw_heading_rad), index=df.index).where(step_ok)
    heading_cos_series = pd.Series(np.cos(raw_heading_rad), index=df.index).where(step_ok)

    heading_sin_smooth = heading_sin_series.ffill().bfill().fillna(0.0).ewm(alpha=0.35, adjust=False).mean()
    heading_cos_smooth = heading_cos_series.ffill().bfill().fillna(1.0).ewm(alpha=0.35, adjust=False).mean()

    norm = np.sqrt((heading_sin_smooth ** 2) + (heading_cos_smooth ** 2)).replace(0, np.nan).fillna(1.0)
    heading_sin_smooth = heading_sin_smooth / norm
    heading_cos_smooth = heading_cos_smooth / norm

    heading_deg = (np.degrees(np.arctan2(heading_sin_smooth, heading_cos_smooth)) + 360.0) % 360.0
    heading_delta_deg = _wrap_degrees(heading_deg.diff().fillna(0.0))

    # Ignore deltas when the repaired route is effectively stationary; smooth/cap
    # turn rate to remove GPS jitter while preserving real turns.
    heading_delta_deg = heading_delta_deg.where(step_ok, 0.0)
    heading_delta_deg = heading_delta_deg.rolling(window=3, center=True, min_periods=1).median().clip(-45.0, 45.0)

    turn_rate_deg_s = (heading_delta_deg / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turn_rate_deg_s = (
        turn_rate_deg_s
        .rolling(window=5, center=True, min_periods=1).median()
        .clip(-120.0, 120.0)
    )

    out['gps_accuracy'] = gps_accuracy
    out['gps_age_ms'] = gps_age_ms
    out['gps_update_count_delta'] = gps_update_count_delta

    out['gps_dx_m'] = gps_dx_m.round(4)
    out['gps_dy_m'] = gps_dy_m.round(4)
    out['gps_step_distance_m'] = gps_step_distance_m.round(4)
    out['gps_speed_kmh'] = gps_speed_kmh.round(3)

    out['heading_sin'] = heading_sin_smooth.round(6)
    out['heading_cos'] = heading_cos_smooth.round(6)
    out['heading_delta_deg'] = heading_delta_deg.round(3)
    out['turn_rate_deg_s'] = turn_rate_deg_s.round(3)

    out['accel_x'] = num_col('accel.x', 0.0).clip(-5.0, 5.0)
    out['accel_y'] = num_col('accel.y', 0.0).clip(-5.0, 5.0)
    out['accel_z'] = num_col('accel.z', 0.0).clip(-5.0, 5.0)

    for col in M13_ADDED_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out[M13_ADDED_COLUMNS]

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


def _build_m12_feature_vector(sensor_df: pd.DataFrame, video_events: list) -> pd.DataFrame:
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



def build_feature_vector(sensor_df: pd.DataFrame, video_events: list) -> pd.DataFrame:
    """
    Build M13-ready feature vector from sensors + filtered video events.

    M13 = M12 features + continuous per-row GPS/heading/IMU measurements.
    The exported vector is 33 columns. M12 runtime can still select its known
    19 feature columns from this frame until the M13 model is trained.
    """
    log.info("🧬 Building M13-ready vector: M12 context + GPS/heading/IMU rows...")

    m12_df = _build_m12_feature_vector(sensor_df, video_events).reset_index(drop=True)
    m13_sensor_df = _compute_m13_sensor_features(sensor_df).reset_index(drop=True)

    if len(m13_sensor_df) != len(m12_df):
        # Should not happen when both derive from the same sensor_df, but keep the
        # export stable if a future caller passes pre-trimmed rows.
        m13_sensor_df = m13_sensor_df.reindex(m12_df.index).ffill().bfill().fillna(0.0)

    out = pd.concat([m12_df, m13_sensor_df[M13_ADDED_COLUMNS]], axis=1)

    for col in M13_FINAL_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)

    sign_counts = out['sign_type'].value_counts().to_dict()
    memory_counts = out['memory_sign_type'].value_counts().to_dict()
    log.info(
        f"✅ M13 vector built. Shape: {out[M13_FINAL_COLUMNS].shape} | "
        f"sign_counts={sign_counts} | memory_sign_counts={memory_counts}"
    )

    return out[M13_FINAL_COLUMNS]
