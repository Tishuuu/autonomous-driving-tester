import sys
from loguru import logger
import os

# הגדרת נתיב לתיקיית הלוגים (רמה אחת מעל app)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

# יצירת התיקייה אם היא לא קיימת
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ניקוי לוגר ברירת המחדל
logger.remove()

# 1. לוגר לקונסול (צבעוני ויפה)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

# 2. קובץ לשגיאות בלבד (Error) - בתוך תיקיית logs
logger.add(
    os.path.join(LOG_DIR, "error.log"),
    rotation="10 MB", 
    retention="1 month", 
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# 3. קובץ לכל הפעילות (Combined)
logger.add(
    os.path.join(LOG_DIR, "combined.log"),
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)

# משתנה לייצוא
log = logger