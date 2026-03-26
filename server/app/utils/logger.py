import sys
from loguru import logger
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

logger.remove()

logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

logger.add(
    os.path.join(LOG_DIR, "error.log"),
    rotation="10 MB", 
    retention="1 month", 
    level="ERROR",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

logger.add(
    os.path.join(LOG_DIR, "combined.log"),
    rotation="10 MB",
    retention="10 days",
    level="INFO"
)


log = logger