from motor.motor_asyncio import AsyncIOMotorClient
from app.utils.logger import log
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "auto_tester_db"

class Database:
    client: AsyncIOMotorClient = None
    db = None

db = Database()

async def connect_db():
    """פונקציה שמופעלת כשהשרת עולה"""
    try:
        db.client = AsyncIOMotorClient(MONGO_URI)
        db.db = db.client[DB_NAME]
        
        await db.db.command("ping")
        
        log.info(f" MongoDB Connected Successfully! Database: {DB_NAME}")
    except Exception as e:
        log.error(f" MongoDB Connection Failed: {e}")
        raise e

async def close_db():
    """פונקציה שמופעלת כשהשרת נכבה"""
    if db.client:
        db.client.close()
        log.info(" MongoDB Connection Closed")