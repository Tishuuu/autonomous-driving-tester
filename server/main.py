import uvicorn
from fastapi import FastAPI
from app.core.database import connect_db, close_db
from app.utils.logger import log
from app.routes import test_routes, auth_routes 

app = FastAPI(
    title="Auto Tester API",
    description="Backend for Autonomous Driving Test System",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    log.info(" Server Starting Up")
    await connect_db()

@app.on_event("shutdown")
async def shutdown_event():
    log.info(" Server Shutting Down")
    await close_db()

# --- Routes ---
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(test_routes.router, prefix="/api/tests", tags=["Driving Tests"])

@app.get("/")
async def read_root():
    return {"status": "online", "system": "Auto Tester API", "device": "Ready for connection"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)