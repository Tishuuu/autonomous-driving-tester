import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.database import connect_db, close_db
from app.utils.logger import log
from app.routes import test_routes, auth_routes, student_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """מחליף את on_event שהוסר ב-FastAPI החדש."""
    # Startup
    log.info(" Server Starting Up")
    await connect_db()
    yield
    # Shutdown
    log.info(" Server Shutting Down")
    await close_db()


app = FastAPI(
    title="Auto Tester API",
    description="Backend for Autonomous Driving Test System",
    version="1.1.0",
    lifespan=lifespan,
)

# --- Routes ---
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(test_routes.router, prefix="/api/tests", tags=["Driving Tests"])
app.include_router(student_routes.router, prefix="/api/students", tags=["Students"])


@app.get("/")
async def read_root():
    return {
        "status": "online",
        "system": "Auto Tester API",
        "device": "Ready for connection",
    }


if __name__ == "__main__":
    # ✅ reload=False - מונע double-loading של 7+ workers, ומונע WinError 10055
    # (אם תרצה reload בזמן פיתוח, הפעל מחוץ ל-main.py: uvicorn main:app --reload)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)