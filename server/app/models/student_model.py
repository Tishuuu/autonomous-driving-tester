from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class StudentCreate(BaseModel):
    """מודל ליצירת תלמיד חדש."""
    student_id: str = Field(..., min_length=5, max_length=15,
                             description="תעודת זהות של התלמיד")
    name: str = Field(..., min_length=2, max_length=50)
    tester_email: str = Field(..., description="אימייל המורה/בודק שהוסיף את התלמיד")


class Student(StudentCreate):
    """מודל תלמיד מלא, עם תאריך הוספה."""
    created_at: datetime = Field(default_factory=datetime.now)


class TestSaveRequest(BaseModel):
    """מבנה הנתונים לשמירת טסט מנותח ב-DB."""
    student_id: str = Field(..., description="תז התלמיד שעבר את הטסט")
    tester_email: str = Field(..., description="אימייל המורה שביצע את הטסט")
    # grade is kept for backwards compatibility only: 100=PASS, 0=FAIL.
    grade: int = Field(..., ge=0, le=100)
    result: str = "PASS"
    passed: bool = True
    mistakes_count: int = 0
    mistake_codes: list = Field(default_factory=list)
    violations_codes: list = Field(default_factory=list)
    violation_events_count: int = 0
    ignored_warning_codes: list = Field(default_factory=list)
    ignored_warning_events_count: int = 0
    ignored_warning_events: list = Field(default_factory=list)
    xai_explanations: dict = Field(default_factory=dict)
    windows_analyzed: int = 0
    test_id: Optional[str] = None
    test_date: Optional[datetime] = None
    # 🆕 לוג חשיבת המודל - נשמר ב-MongoDB לבקרה ולניתוח
    decision_log: list = Field(default_factory=list)
    action_sequences: list = Field(default_factory=list)