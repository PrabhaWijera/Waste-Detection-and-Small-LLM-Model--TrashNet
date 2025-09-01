# app/routes/public_user_routes.py
import os
import uuid
from datetime import datetime, timezone
import aiosmtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from fastapi import APIRouter, UploadFile, Form, File, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from app.database.connector_mysql import db
from app.services.training_manager import start_retrain_async, NEW_DATA_DIR, count_new_samples, MIN_SAMPLES
from app.utils.jwt_utils import decode_access_token, create_access_token

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

os.makedirs(NEW_DATA_DIR, exist_ok=True)

# ----------------------------
# Email helper (async with aiosmtplib)
# ----------------------------
async def send_welcome_email_async(to_email: str):
    from_email = "prabhashwlive2001@gmail.com"
    password = "shaaumbtribzkecz"  # Gmail App Password

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = "Welcome to Waste Management System"

    body = """
    <html>
    <body>
        <p>✅ Welcome! You are registered in the Waste Management System.</p>
        <p>You can login using your email or phone number (no password required).</p>
        <p>Thank you for using our system!</p>
    </body>
    </html>
    """
    msg.attach(MIMEText(body, "html", "utf-8"))

    try:
        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=465,
            username=from_email,
            password=password,
            use_tls=True,
        )
        print(f"✅ Welcome email sent to {to_email}")
    except Exception as e:
        import traceback
        print(f"❌ Failed to send email to {to_email}: {e}")
        print(traceback.format_exc())

# ----------------------------
# Submit Waste Endpoint
# ----------------------------
@router.post("/submit_waste")
async def submit_waste(
    image: UploadFile = File(...),
    waste_type: str = Form(...),
    name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    location: str = Form(...),
    urban_center: str = Form(...),
):
    """Submit waste image and auto-register user if new"""
    
    # Save uploaded image
    contents = await image.read()
    ext = os.path.splitext(image.filename)[1] or ".jpg"
    class_dir = os.path.join(NEW_DATA_DIR, waste_type)
    os.makedirs(class_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(class_dir, filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    created_at = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

    # -----------------------------
    # Insert submission
    # -----------------------------
    with db.cursor(dictionary=True, buffered=True) as cursor:
        cursor.execute("""
            INSERT INTO submissions 
            (name, email, phone, location, urban_center, waste_type, source, processed, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 'public', FALSE, %s)
        """, (name, email, phone, location, urban_center, waste_type, created_at))
        db.commit()

    # -----------------------------
    # Auto-register new user if not exists
    # -----------------------------
    new_user_registered = False
    with db.cursor(dictionary=True, buffered=True) as cursor:
        cursor.execute("SELECT id FROM users WHERE email=%s OR username=%s", (email, phone))
        user = cursor.fetchone()

    if not user:
        with db.cursor(dictionary=True) as cursor:
            cursor.execute("""
                INSERT INTO users (username, email, password, role, created_at)
                VALUES (%s, %s, %s, 'normal', %s)
            """, (phone, email, "nopass", created_at))
            db.commit()
        new_user_registered = True

    # -----------------------------
    # Send welcome email if new user
    # -----------------------------
    if new_user_registered:
        print(f"✅ New user registered. Sending email to: {email}")
        try:
            await send_welcome_email_async(email)
        except Exception as e:
            print(f"❌ Failed to send welcome email: {e}")

    # -----------------------------
    # Retraining status
    # -----------------------------
    total_new = count_new_samples()
    retrain_status = (
        start_retrain_async()
        if total_new >= MIN_SAMPLES
        else f"Waiting for more samples ({total_new}/{MIN_SAMPLES})."
    )

    return {
        "message": "Submission received. User registered (if new).",
        "saved_file": f"{waste_type}/{filename}",
        "retrain_status": retrain_status,
    }

# ----------------------------
# Login Endpoint
# ----------------------------
class LoginRequest(BaseModel):
    identifier: str  # email or phone

@router.post("/login")
async def login(req: LoginRequest):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email=%s OR username=%s", (req.identifier, req.identifier))
    user = cursor.fetchone()
    cursor.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Please submit waste first.")

    token = create_access_token({"sub": user["email"], "phone": user["username"]})
    return {"access_token": token, "email": user["email"], "phone": user["username"]}

# ----------------------------
# Auth Helpers
# ----------------------------
async def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return payload

# ----------------------------
# Get My Submissions
# ----------------------------
@router.get("/my_submissions")
async def my_submissions(user=Depends(get_current_user)):
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, waste_type, location, urban_center, created_at
        FROM submissions
        WHERE email=%s OR phone=%s
        ORDER BY created_at DESC
    """, (user["sub"], user["phone"]))
    rows = cursor.fetchall()
    cursor.close()
    return rows

# Expose router
user_routes = router




















# # app/routes/public_user_routes.py
# import os
# import uuid
# from datetime import datetime, timezone
# from fastapi import APIRouter, UploadFile, Form, File
# from app.database.connector_mysql import db
# from app.services.training_manager import start_retrain_async, NEW_DATA_DIR, count_new_samples, MIN_SAMPLES

# router = APIRouter()

# # Ensure the new data directory exists
# os.makedirs(NEW_DATA_DIR, exist_ok=True)

# @router.post("/submit_waste")
# async def submit_waste(
#     image: UploadFile = File(...),
#     waste_type: str = Form(...),
#     region: str = Form("LK-11"),
#     city: str = Form("Colombo"),
# ):
#     """Endpoint for public users to submit new waste images."""

#     # Read image contents
#     contents = await image.read()
#     ext = os.path.splitext(image.filename)[1] or ".jpg"

#     # 1️⃣ Save metadata in mysql
#     doc = {
#         "waste_type": waste_type,
#         "region": region,
#         "city": city,
#         "source": "public",
#         "processed": False,
#         "created_at": datetime.now(timezone.utc),
#     }
#     db.submissions.insert_one(doc)

#     # 2️⃣ Save image locally in class-specific folder
#     class_dir = os.path.join(NEW_DATA_DIR, waste_type)
#     os.makedirs(class_dir, exist_ok=True)
#     filename = f"{uuid.uuid4().hex}{ext}"
#     save_path = os.path.join(class_dir, filename)
#     with open(save_path, "wb") as f:
#         f.write(contents)

#     # 3️⃣ Check if retraining threshold is met
#     total_new = count_new_samples()
#     if total_new >= MIN_SAMPLES:
#         retrain_status = start_retrain_async()
#     else:
#         retrain_status = f"Waiting for more samples ({total_new}/{MIN_SAMPLES})."

#     return {
#         "message": "Submission received.",
#         "saved_file": f"{waste_type}/{filename}",
#         "retrain_status": retrain_status,
#     }

# # Expose router
# user_routes = router
# # 