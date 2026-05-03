import os
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.database import db
from app.models.user_model import UserRegister, UserLogin
from app.utils.logger import log

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 🆕 JWT config
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE-ME-IN-PROD-SET-IN-DOTENV")
JWT_ALGORITHM = "HS256"
JWT_TTL = timedelta(days=7)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=True)


def make_token(email: str, name: str) -> str:
    payload = {
        "sub": email,
        "name": name,
        "exp": datetime.utcnow() + JWT_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


async def get_current_tester(token: str = Depends(oauth2_scheme)) -> dict:
    """Dependency — extracts authenticated tester from JWT."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": email, "name": payload.get("name", "")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.post("/register")
async def register(user: UserRegister):
    existing = await db.db["users"].find_one({"email": user.email})
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = pwd_context.hash(user.password)
    user_dict = user.dict()
    user_dict["password"] = hashed
    await db.db["users"].insert_one(user_dict)

    log.info(f"New user registered: {user.email}")
    token = make_token(user.email, user.name)
    return {
        "message": "User created successfully",
        "name": user.name,
        "email": user.email,
        "access_token": token,
        "token_type": "bearer",
    }


@router.post("/login")
async def login(credentials: UserLogin):
    log.info(f"Login attempt: {credentials.email}")
    user = await db.db["users"].find_one({"email": credentials.email})
    if not user or not pwd_context.verify(credentials.password, user["password"]):
        log.warning(f"Failed login: {credentials.email}")
        raise HTTPException(status_code=401, detail="Invalid email or password")

    log.info(f"Successful login: {user['name']}")
    token = make_token(user["email"], user["name"])
    return {
        "message": "Login successful",
        "name": user["name"],
        "email": user["email"],
        "access_token": token,
        "token_type": "bearer",
    }


@router.get("/me")
async def whoami(tester: dict = Depends(get_current_tester)):
    """Verify current token validity."""
    return tester