import pandas as pd
import numpy as np
from app.utils.logger import log


def build_feature_vector(sensor_df: pd.DataFrame, video_events: list) -> pd.DataFrame:
    """
    Multi-object vector with Dead Reckoning + lateral-acceleration
    geometric invalidation (turn detection).
    """
    log.info("🧬 Building vector with geometric DR invalidation...")

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

    events_df = pd.DataFrame(video_events)

    virtual_sign_type = "none"
    virtual_sign_dist = 99.0
    heading_drift = 0.0  # 🆕 lateral velocity integrated since last YOLO confirm
    HEADING_INVALIDATE = 0.4  # m/s lateral ≈ 25° heading change at urban speeds

    dt = 0.1

    for idx, row in df.iterrows():
        current_time = row['time_seconds']
        speed_ms = row['speed_kmh'] / 3.6
        a_lat = float(row.get('accel.x', 0.0) or 0.0)

        window = events_df[abs(events_df['time_sec'] - current_time) <= 0.2]
        yolo_saw_sign = False

        if not window.empty:
            cars = window[window['type'].str.contains('car|vehicle', case=False, na=False)]
            signs = window[window['type'].str.contains('stop|yield', case=False, na=False)]

            if not cars.empty:
                closest_car = cars.loc[cars['distance_meters'].idxmin()]
                df.at[idx, 'car_distance'] = round(closest_car['distance_meters'], 2)
                if speed_ms > 0.5:
                    df.at[idx, 'car_ttc'] = round(closest_car['distance_meters'] / speed_ms, 2)

            if not signs.empty:
                closest_sign = signs.loc[signs['distance_meters'].idxmin()]
                virtual_sign_type = "stop_sign" if "stop" in closest_sign['type'].lower() else "yield_sign"
                virtual_sign_dist = closest_sign['distance_meters']
                yolo_saw_sign = True
                heading_drift = 0.0  # 🆕 fresh confirm resets drift

        # ====================================================
        # Dead Reckoning + lateral drift check
        # ====================================================
        if not yolo_saw_sign and virtual_sign_dist < 15.0:
            virtual_sign_dist -= (speed_ms * dt)

            # 🆕 integrate lateral velocity during DR
            heading_drift += abs(a_lat) * 9.81 * dt

            if heading_drift > HEADING_INVALIDATE:
                log.debug(f"🔄 DR invalidated t={current_time:.2f}s "
                          f"drift={heading_drift:.2f} m/s")
                virtual_sign_type = "none"
                virtual_sign_dist = 99.0
                heading_drift = 0.0

            if virtual_sign_dist < -1.0:
                virtual_sign_type = "none"
                virtual_sign_dist = 99.0
                heading_drift = 0.0

        if virtual_sign_dist != 99.0:
            df.at[idx, 'sign_type'] = virtual_sign_type
            df.at[idx, 'sign_distance'] = round(max(0.0, virtual_sign_dist), 2)
            if speed_ms > 0.5:
                df.at[idx, 'sign_ttc'] = round(virtual_sign_dist / speed_ms, 2)
            else:
                df.at[idx, 'sign_ttc'] = 99.0

    # Forward-fill car blinks
    df.replace(99.0, np.nan, inplace=True)
    df['car_distance'] = df['car_distance'].ffill(limit=5)
    df['car_ttc'] = df['car_ttc'].ffill(limit=5)
    df.fillna(99.0, inplace=True)

    # Sanitize TTC=0
    df['car_ttc'] = df['car_ttc'].replace(0, 99.0)
    df['sign_ttc'] = df['sign_ttc'].replace(0, 99.0)

    sign_mapping = {'none': 0, 'stop_sign': 1, 'yield_sign': 2, 'no_entry': 3}
    df['sign_type'] = df['sign_type'].map(sign_mapping).fillna(0).astype(int)

    log.info(f"✅ Vector built. Shape: {df.shape}")
    return df[FINAL_COLUMNS]