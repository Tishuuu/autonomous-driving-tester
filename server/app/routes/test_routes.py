from fastapi import APIRouter, HTTPException, Body, UploadFile, File, Form
from typing import List
from datetime import datetime
import shutil
import json
import os
import uuid

from app.models.test_model import DrivingTest # ודא שהנתיב הזה תואם לפרויקט שלך
from app.core.database import db
from app.utils.logger import log

router = APIRouter()

# ==========================================
# 1. הנתיב החדש: מקבל קבצים, מנתח ושומר
# ==========================================
@router.post("/process", response_model=dict)
async def process_and_save_test(
    student_id: str = Form(...),
    video: UploadFile = File(...),
    sensor_data: UploadFile = File(...)
):
    """
    נתיב זה מקבל וידאו ונתוני חיישנים מהאפליקציה,
    מריץ עליהם את מודל ה-AI, ושומר את התוצאה למסד הנתונים.
    """
    test_id = f"TEST-{str(uuid.uuid4())[:8].upper()}"
    temp_vid = f"temp_{test_id}.mp4"
    temp_json = f"temp_{test_id}.json"

    try:
        log.info(f" Receiving test files for student: {student_id}")

        # 1. שמירת הקבצים זמנית על הלפטופ
        with open(temp_vid, "wb") as f: shutil.copyfileobj(video.file, f)
        with open(temp_json, "wb") as f: shutil.copyfileobj(sensor_data.file, f)
            
        with open(temp_json, "r") as f: sensors_dict = json.load(f)

        log.info(f" Analyzing video for student {student_id}...")

        # 2. --- כאן ירוץ ה-YOLO וה-LSTM בעתיד ---
        # כרגע אנחנו שמים תוצאות מדומוֹת (Mock)
        ai_score = 88.0 
        
        # 3. בניית האובייקט בעזרת המודל שלך!
        test_record = DrivingTest(
            student_id=student_id,
            test_id=test_id,
            start_time=datetime.now(),
            final_score=ai_score,
            status="passed",
            # תוכל להוסיף פה את שאר השדות מהסכמה שלך כמו environment ו-metrics
        )

        # 4. שמירה ישירה ל-DB (בדיוק כמו שעשית בנתיב /save)
        test_dict = test_record.dict(by_alias=True)
        new_test = await db.db["tests"].insert_one(test_dict)
        log.info(f" AI Test Processed and Saved! DB ID: {new_test.inserted_id}")

        # 5. מחיקת הוידאו מהלפטופ (חשוב מאוד!)
        os.remove(temp_vid)
        os.remove(temp_json)

        return {
            "message": "Test processed and saved successfully", 
            "test_id": test_id, 
            "score": ai_score
        }

    except Exception as e:
        # במקרה של קריסה, ננקה את הקבצים הזמניים
        if os.path.exists(temp_vid): os.remove(temp_vid)
        if os.path.exists(temp_json): os.remove(temp_json)
        log.error(f" Error processing test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 2. הנתיבים המקוריים שלך (ללא שינוי!)
# ==========================================
@router.post("/save", response_model=dict)
async def save_test_result(test: DrivingTest = Body(...)):
    """
    שמירת טסט רגיל (טקסט בלבד)
    """
    try:
        log.info(f" Receiving new test for student: {test.student_id}")
        
        test_dict = test.dict(by_alias=True)
        new_test = await db.db["tests"].insert_one(test_dict)
        
        log.info(f" Test saved successfully! ID: {new_test.inserted_id}")
        return {"message": "Test saved successfully", "id": str(new_test.inserted_id)}

    except Exception as e:
        log.error(f" Error saving test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user/{student_id}", response_model=List[DrivingTest])
async def get_student_history(student_id: str):
    """
    שליפת היסטוריית טסטים לתלמיד
    """
    try:
        log.info(f" Fetching history for: {student_id}")
        
        tests = []
        cursor = db.db["tests"].find({"student_id": student_id}).sort("start_time", -1)
        
        async for document in cursor:
            tests.append(DrivingTest(**document))
            
        log.info(f"Found {len(tests)} tests")
        return tests

    except Exception as e:
        log.error(f" Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")