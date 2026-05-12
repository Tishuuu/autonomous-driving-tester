import json
from typing import Any

import numpy as np
import pandas as pd
from app.utils.logger import log


REQUIRED_COLUMNS = [
    "time_seconds",
    "speed_kmh",
    "rpm",
    "lat",
    "lon",
    "g_force",
    "jerk",
]

OPTIONAL_COLUMNS = [
    "speed_source",
    "obd_latency_ms",
    "gps_accuracy",
    "gps_age_ms",
    "gps_update_count",
]


def _to_number(df: pd.DataFrame, col: str, default: float = 0.0) -> None:
    if col not in df.columns:
        df[col] = default
    df[col] = pd.to_numeric(df[col], errors="coerce")


def _safe_unique_count(series: pd.Series) -> int:
    try:
        return int(series.dropna().round(7).nunique())
    except Exception:
        return 0


def _load_raw_json(json_file_path: str) -> tuple[list[dict[str, Any]], int]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    sensor_events = raw_data.get("data", [])
    if not isinstance(sensor_events, list) or not sensor_events:
        raise ValueError("No non-empty 'data' array in sensor JSON")

    video_offset_ms = int(raw_data.get("video_offset_ms", 0) or 0)
    return sensor_events, video_offset_ms


def process_sensor_json(json_file_path: str) -> pd.DataFrame:
    """
    Reads Flutter sensor JSON and converts it into the backend's 10Hz model input.

    Fixes included:
    - keeps fresh GPS metadata from the client: gps_accuracy/gps_age_ms/gps_update_count
    - protects resampling from non-numeric fields
    - warns when GPS is effectively frozen instead of silently accepting a fake route
    - keeps accel.x/y/z columns for vector_builder turn/drift invalidation
    - fails with a clear error if video offset removes every row
    """
    log.info(f"🔄 Processing sensor data: {json_file_path}")

    try:
        sensor_events, video_offset_ms = _load_raw_json(json_file_path)
        if video_offset_ms != 0:
            log.info(f"🎥 Video offset: {video_offset_ms}ms")

        df = pd.json_normalize(sensor_events)
        if "time_ms" not in df.columns:
            raise ValueError("Sensor JSON missing required 'time_ms' field")

        # Numeric sanitation before resampling.
        base_numeric = [
            "time_ms",
            "speed_kmh",
            "rpm",
            "lat",
            "lon",
            "g_force",
            "speed_source",
            "obd_latency_ms",
            "gps_accuracy",
            "gps_age_ms",
            "gps_update_count",
            "accel.x",
            "accel.y",
            "accel.z",
        ]
        for col in base_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col, default in {
            "speed_kmh": 0.0,
            "rpm": 0.0,
            "lat": np.nan,
            "lon": np.nan,
            "g_force": 0.0,
            "speed_source": 3,
            "obd_latency_ms": 0.0,
            "gps_accuracy": 999.0,
            "gps_age_ms": 999999.0,
            "gps_update_count": 0,
            "accel.x": 0.0,
            "accel.y": 0.0,
            "accel.z": 0.0,
        }.items():
            _to_number(df, col, default)

        df = df.dropna(subset=["time_ms"]).sort_values("time_ms")
        df = df.drop_duplicates(subset=["time_ms"], keep="last")
        if df.empty:
            raise ValueError("Sensor JSON has no valid numeric timestamps")

        # Keep only numeric columns for safe pandas resampling.
        df["time_ms"] = pd.to_timedelta(df["time_ms"], unit="ms")
        df.set_index("time_ms", inplace=True)
        numeric_df = df.select_dtypes(include=[np.number]).copy()

        log.info("⏱️ Resampling to 10Hz")
        resampled_df = numeric_df.resample("100ms").mean().interpolate(method="linear", limit_direction="both")

        # Tier-aware GPS plateau smoothing for GPS speed only.
        if "speed_source" in resampled_df.columns:
            tier = resampled_df["speed_source"].round()
            gps_mask = tier == 3
            speed = resampled_df["speed_kmh"].copy()

            same_as_prev = speed.diff().abs() < 1e-6
            blank_mask = gps_mask & same_as_prev

            n_blanked = int(blank_mask.sum())
            if n_blanked > 0:
                speed[blank_mask] = np.nan
                speed = speed.interpolate(method="linear", limit_direction="both")
                resampled_df["speed_kmh"] = speed
            log.info(f"🪛 Tier-3 plateau smoothing: {n_blanked} rows interpolated")

        # Jerk on cleaned g-force signal.
        dt = 0.1
        resampled_df["jerk"] = (resampled_df["g_force"].diff() / dt).fillna(0)

        time_seconds_raw = resampled_df.index.total_seconds()
        resampled_df["time_seconds"] = time_seconds_raw - (video_offset_ms / 1000.0)

        before_count = len(resampled_df)
        resampled_df = resampled_df[resampled_df["time_seconds"] >= 0].reset_index(drop=True)
        dropped = before_count - len(resampled_df)
        if dropped > 0:
            log.info(f"🗑️ Dropped {dropped} pre-video rows")
        if resampled_df.empty:
            raise ValueError("No sensor rows remain after applying video_offset_ms")

        # GPS quality diagnostics. If this triggers, the client GPS stream did not update.
        unique_lat = _safe_unique_count(resampled_df["lat"])
        unique_lon = _safe_unique_count(resampled_df["lon"])
        duration_s = float(resampled_df["time_seconds"].iloc[-1] - resampled_df["time_seconds"].iloc[0])
        if duration_s >= 10 and unique_lat <= 1 and unique_lon <= 1:
            log.warning(
                "⚠️ GPS appears frozen for the whole drive: "
                f"duration={duration_s:.1f}s unique_lat={unique_lat} unique_lon={unique_lon}"
            )

        keep = REQUIRED_COLUMNS.copy()
        for col in OPTIONAL_COLUMNS:
            if col in resampled_df.columns:
                keep.append(col)

        accel_cols = [c for c in ["accel.x", "accel.y", "accel.z"] if c in resampled_df.columns]
        out = resampled_df[keep + accel_cols].copy()

        # Final sanitation for ML pipeline.
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out.ffill(inplace=True)
        out.bfill(inplace=True)
        out.fillna(0, inplace=True)

        log.info(f"✅ Sensor processing done. Rows: {len(out)}")
        return out

    except Exception as e:
        log.error(f"❌ Failed sensor JSON: {e}")
        raise
