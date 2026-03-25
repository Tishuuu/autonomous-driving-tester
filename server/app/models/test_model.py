from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# --- 1. מודל לאירוע בודד (ה"קופסה השחורה") ---
class DrivingEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    type: str = Field(..., description="סוג האירוע: LaneDeparture, HardBrake, Tailgating, SignMissed")
    severity: str = Field(..., pattern="^(low|medium|critical)$")
    location: dict = Field(default={}, description="קוורדינטות GPS של האירוע {lat, lng}")
    details: str = Field(default="", description="פירוט נוסף, למשל: 'תמרור עצור זוהה אך הרכב לא עצר'")

# --- 2. נתוני סביבה (חשוב לניתוח ה-AI) ---
class EnvironmentData(BaseModel):
    weather: str = Field(default="unknown", description="sunny, rainy, cloudy, night")
    road_type: str = Field(default="urban", description="urban, highway, mixed")
    traffic_density: str = Field(default="medium", description="low, medium, high")

# --- 3. סיכומי חיישנים ---
class SensorMetrics(BaseModel):
    # OBD - האמת בקרקע
    max_speed_kmh: float = 0
    avg_speed_kmh: float = 0
    fuel_consumption: float = 0
    
    # IMU - יציבות
    smoothness_score: float = Field(default=100, description="ציון נהיגה חלקה 0-100")
    brake_events_count: int = 0
    
    # GPS Reliability
    gps_signal_quality: float = Field(default=100, description="אחוז הזמן שהיה GPS תקין")

# --- המודל הראשי המעודכן ---
class DrivingTest(BaseModel):
    # זיהוי
    student_id: str = Field(..., min_length=5)
    tester_id: str = "AUTO-SYSTEM"
    test_id: str = Field(..., description="מזהה ייחודי של הטסט מהאפליקציה")
    
    # זמנים
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: int = 0
    distance_meters: float = 0 # הוספנו מרחק כולל
    
    # ציון וסטטוס
    final_score: float = Field(..., ge=0, le=100)
    status: str = Field(default="pending", pattern="^(pending|passed|failed|aborted)$")
    
    # הנתונים החדשים שהוספנו
    environment: EnvironmentData = EnvironmentData()
    metrics: SensorMetrics = SensorMetrics()
    
    events_log: List[DrivingEvent] = []
    
    route_path: List[dict] = [] # List of {lat, lng}

class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "student_id": "216312355",
                "test_id": "TEST-123456",  # <--- הוספנו את זה
                "start_time": "2026-02-11T12:00:00", # <--- הוספנו את זה
                "final_score": 88.0,
                "status": "passed",
                "duration_seconds": 1200,
                "environment": {
                    "weather": "sunny",
                    "road_type": "urban",
                    "traffic_density": "medium"
                },
                "metrics": {
                    "obd": {"max_speed": 110.5, "average_speed": 45.2, "engine_load_peak": 80.0},
                    "imu": {"hard_brakes": 1, "rapid_accelerations": 0, "jerk_score": 95.0},
                    "camera": {"lane_departures": 0, "signs_missed": 0, "tailgating_events": 0},
                    "gps_signal_quality": 100.0
                },
                "events_log": [
                    {
                        "timestamp": "2026-02-11T12:15:00",
                        "type": "HardBrake",
                        "severity": "medium",
                        "location": {"lat": 32.0853, "lng": 34.7818},
                        "details": "בלימה פתאומית לפני מעבר חציה"
                    }
                ]
            }
        }