# app/routes/public_user_routes.py
import os
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, Form, File
from app.database.mongo_client import db
from app.services.training_manager import start_retrain_async, NEW_DATA_DIR, count_new_samples, MIN_SAMPLES

router = APIRouter()

# Ensure the new data directory exists
os.makedirs(NEW_DATA_DIR, exist_ok=True)

@router.post("/submit_waste")
async def submit_waste(
    image: UploadFile = File(...),
    waste_type: str = Form(...),
    region: str = Form("LK-11"),
    city: str = Form("Colombo"),
):
    """Endpoint for public users to submit new waste images."""

    # Read image contents
    contents = await image.read()
    ext = os.path.splitext(image.filename)[1] or ".jpg"

    # 1️⃣ Save metadata in MongoDB
    doc = {
        "waste_type": waste_type,
        "region": region,
        "city": city,
        "source": "public",
        "processed": False,
        "created_at": datetime.now(timezone.utc),
    }
    db.submissions.insert_one(doc)

    # 2️⃣ Save image locally in class-specific folder
    class_dir = os.path.join(NEW_DATA_DIR, waste_type)
    os.makedirs(class_dir, exist_ok=True)
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(class_dir, filename)
    with open(save_path, "wb") as f:
        f.write(contents)

    # 3️⃣ Check if retraining threshold is met
    total_new = count_new_samples()
    if total_new >= MIN_SAMPLES:
        retrain_status = start_retrain_async()
    else:
        retrain_status = f"Waiting for more samples ({total_new}/{MIN_SAMPLES})."

    return {
        "message": "Submission received.",
        "saved_file": f"{waste_type}/{filename}",
        "retrain_status": retrain_status,
    }

# Expose router
user_routes = router
