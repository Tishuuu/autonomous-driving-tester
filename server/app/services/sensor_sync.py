import json
import pandas as pd
import numpy as np
from app.utils.logger import log


def process_sensor_json(json_file_path: str) -> pd.DataFrame:
    """
    Reads JSON, resamples to 10Hz, computes jerk, applies tier-aware
    smoothing for GPS plateaus (Tier 3 staircase fix).
    """
    log.info(f"🔄 Processing sensor data: {json_file_path}")

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        sensor_events = raw_data.get("data", [])
        if not sensor_events:
            raise ValueError("No 'data' array in JSON")

        video_offset_ms = raw_data.get("video_offset_ms", 0)
        if video_offset_ms != 0:
            log.info(f"🎥 Video offset: {video_offset_ms}ms")

        df = pd.json_normalize(sensor_events)
        df['time_ms'] = pd.to_timedelta(df['time_ms'], unit='ms')
        df.set_index('time_ms', inplace=True)

        log.info("⏱️ Resampling to 10Hz")
        resampled_df = df.resample('100ms').mean().interpolate(method='linear')

        # ==========================================================
        # 🆕 Tier-aware GPS plateau smoothing
        # ==========================================================
        if 'speed_source' in resampled_df.columns:
            tier = resampled_df['speed_source'].round()
            gps_mask = tier == 3
            speed = resampled_df['speed_kmh'].copy()

            same_as_prev = speed.diff().abs() < 1e-6
            blank_mask = gps_mask & same_as_prev

            n_blanked = int(blank_mask.sum())
            speed[blank_mask] = np.nan
            speed = speed.interpolate(method='linear', limit_direction='both')
            resampled_df['speed_kmh'] = speed

            log.info(f"🪛 Tier-3 plateau smoothing: {n_blanked} rows interpolated")

        # Jerk on cleaned speed signal
        dt = 0.1
        resampled_df['jerk'] = (resampled_df['g_force'].diff() / dt).fillna(0)

        # Time-seconds aligned to video PTS=0
        time_seconds_raw = resampled_df.index.total_seconds()
        resampled_df['time_seconds'] = time_seconds_raw - (video_offset_ms / 1000.0)

        before_count = len(resampled_df)
        resampled_df = resampled_df[resampled_df['time_seconds'] >= 0].reset_index(drop=False)
        dropped = before_count - len(resampled_df)
        if dropped > 0:
            log.info(f"🗑️ Dropped {dropped} pre-video rows")

        keep = ['time_seconds', 'speed_kmh', 'rpm', 'lat', 'lon',
                'g_force', 'jerk']
        accel_cols = [c for c in resampled_df.columns if c.startswith('accel.')]
        # Keep speed_source if present (vector_builder may use it)
        if 'speed_source' in resampled_df.columns:
            keep.append('speed_source')

        resampled_df = resampled_df[keep + accel_cols]

        log.info(f"✅ Sensor processing done. Rows: {len(resampled_df)}")
        return resampled_df

    except Exception as e:
        log.error(f"❌ Failed sensor JSON: {e}")
        raise