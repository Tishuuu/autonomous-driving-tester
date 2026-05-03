from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from app.core.database import db
from app.models.student_model import StudentCreate
from app.utils.logger import log
from app.routes.auth_routes import get_current_tester

router = APIRouter()


@router.post("/")
async def create_student(
    student: StudentCreate,
    tester: dict = Depends(get_current_tester),
):
    """Creates student under authenticated tester. tester_email from token only."""
    # 🛡️ Override any client-supplied tester_email with token identity
    tester_email = tester["email"]

    existing = await db.db["students"].find_one({
        "student_id": student.student_id,
        "tester_email": tester_email,
    })
    if existing:
        raise HTTPException(
            status_code=409,
            detail="Student with this ID already exists for this tester",
        )

    student_dict = student.model_dump()
    student_dict["tester_email"] = tester_email  # force token value
    student_dict["created_at"] = datetime.now()

    await db.db["students"].insert_one(student_dict)
    log.info(f"➕ Student {student.name} ({student.student_id}) by {tester_email}")
    return {
        "status": "success",
        "student_id": student.student_id,
        "name": student.name,
    }


@router.get("/")
async def list_students(tester: dict = Depends(get_current_tester)):
    """Returns students belonging to authenticated tester."""
    cursor = db.db["students"].find({"tester_email": tester["email"]})
    students = []
    async for doc in cursor:
        doc.pop("_id", None)
        if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
            doc["created_at"] = doc["created_at"].isoformat()
        students.append(doc)
    log.info(f"📋 Listed {len(students)} students for {tester['email']}")
    return students


@router.delete("/{student_id}")
async def delete_student(
    student_id: str,
    tester: dict = Depends(get_current_tester),
):
    """Deletes student only if owned by authenticated tester."""
    result = await db.db["students"].delete_one({
        "student_id": student_id,
        "tester_email": tester["email"],
    })
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Student not found")
    log.info(f"🗑️ Student {student_id} deleted by {tester['email']}")
    return {"status": "success", "deleted": student_id}