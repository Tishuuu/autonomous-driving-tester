from fastapi import APIRouter, HTTPException, Body
from app.core.database import db
from app.models.user_model import UserRegister, UserLogin
from app.utils.logger import log
from passlib.context import CryptContext

router = APIRouter()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post("/register")
async def register(user: UserRegister):
    existing_user = await db.db["users"].find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=409, detail="Email already registered")
    
    hashed_password = pwd_context.hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed_password
    
    await db.db["users"].insert_one(user_dict)
    log.info(f"New user registered: {user.email}")
    return {"message": "User created successfully", "name": user.name}

@router.post("/login")
async def login(credentials: UserLogin):
    log.info(f" Login attempt: {credentials.email}")
    
    user = await db.db["users"].find_one({"email": credentials.email})
    
    if not user or not pwd_context.verify(credentials.password, user["password"]):
        log.warning(f" Failed login for: {credentials.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    log.info(f" Successful login: {user['name']}")
    return {
        "message": "Login successful",
        "name": user["name"],
        "email": user["email"]
    }