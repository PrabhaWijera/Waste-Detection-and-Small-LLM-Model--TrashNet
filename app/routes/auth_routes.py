from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from passlib.context import CryptContext
from app.database.connector_mysql import db
from app.utils.jwt_utils import create_access_token

router = APIRouter()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ------------------- Pydantic Models -------------------
class UrbanUserRegister(BaseModel):
    username: str
    email: str
    password: str


class UrbanUserLogin(BaseModel):
    username: str
    password: str


# ------------------- Auth Routes -------------------

@router.post("/register")
def register(user: UrbanUserRegister):
    cursor = db.cursor()
    hashed = pwd_context.hash(user.password)

    try:
        cursor.execute(
            "INSERT INTO users (username, email, password, role) VALUES (%s, %s, %s, %s)",
            (user.username, user.email, hashed, "urban")
        )
        db.commit()
        return {"message": "Urban user registered successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=f"Error: {e}")
    finally:
        cursor.close()


@router.post("/login")
def login(user: UrbanUserLogin):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s AND role='urban'", (user.username,))
    db_user = cursor.fetchone()
    cursor.close()

    if not db_user or not pwd_context.verify(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"id": db_user["id"], "role": db_user["role"]})
    return {"access_token": token, "token_type": "bearer"}
