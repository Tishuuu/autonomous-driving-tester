from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StudentCreate(BaseModel):
    """Payload for creating a new student. tester_email is ignored by the server and kept optional for old clients."""
    student_id: str = Field(..., min_length=5, max_length=15, description="Student ID")
    name: str = Field(..., min_length=2, max_length=50)
    tester_email: Optional[str] = Field(default=None, description="Deprecated. Server uses JWT token owner.")


class Student(StudentCreate):
    """Full student DB model."""
    created_at: datetime = Field(default_factory=datetime.now)


class TestSaveRequest(BaseModel):
    """Payload for saving an analyzed driving test in MongoDB."""
    student_id: str = Field(..., description="Student ID")
    tester_email: Optional[str] = Field(default=None, description="Deprecated. Server uses JWT token owner.")
    grade: int = Field(..., ge=0, le=100)
    violations_codes: list = Field(default_factory=list)
    violation_events_count: int = 0
    xai_explanations: dict = Field(default_factory=dict)
    windows_analyzed: int = 0
    test_id: Optional[str] = None
    test_date: Optional[datetime] = None
    decision_log: list = Field(default_factory=list)
    action_sequences: list = Field(default_factory=list)
    positive_actions: list = Field(default_factory=list)
