from fastapi import APIRouter, HTTPException, Body
from typing import List
from app.models.test_model import DrivingTest
from app.core.database import db
from app.utils.logger import log

router = APIRouter()

# --- 1. שמירת טסט חדש (POST) ---
@router.post("/save", response_model=dict)
async def save_test_result(test: DrivingTest = Body(...)):
    """
    מקבל את תוצאות הטסט מהאפליקציה ושומר במונגו.
    מבצע ולידציה אוטומטית לפי המודל שיצרנו.
    """
    try:
        log.info(f" Receiving new test for student: {test.student_id}")
        
        # המרה למילון כדי שמונגו יבין את זה
        test_dict = test.dict(by_alias=True)
        
        # שמירה במסד הנתונים (Collection בשם 'tests')
        new_test = await db.db["tests"].insert_one(test_dict)
        
        log.info(f" Test saved successfully! ID: {new_test.inserted_id}")
        return {"message": "Test saved successfully", "id": str(new_test.inserted_id)}

    except Exception as e:
        log.error(f" Error saving test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. שליפת היסטוריה לסטודנט (GET) ---
@router.get("/user/{student_id}", response_model=List[DrivingTest])
async def get_student_history(student_id: str):
    """
    מחזיר את רשימת הטסטים של סטודנט ספציפי, מהחדש לישן.
    """
    try:
        log.info(f"🔍 Fetching history for: {student_id}")
        
        tests = []
        # שליפה מהדאטה בייס, מיון לפי זמן (descending)
        cursor = db.db["tests"].find({"student_id": student_id}).sort("start_time", -1)
        
        async for document in cursor:
            tests.append(DrivingTest(**document))
            
        log.info(f"Found {len(tests)} tests")
        return tests

    except Exception as e:
        log.error(f" Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")