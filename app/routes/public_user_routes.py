# app/routes/public_user_routes.py
import os
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, Form, File
from fastapi.responses import JSONResponse

from app.database.mongo_client import db
from app.services.training_manager import check_and_trigger_retrain

router = APIRouter()

# Folder for saving new submissions
NEW_DATA_DIR = "data/new_waste"
os.makedirs(NEW_DATA_DIR, exist_ok=True)


@router.post("/submit_waste")
async def submit_waste(
    image: UploadFile = File(...),
    waste_type: str = Form(...),
    region: str = Form("LK-11"),
    city: str = Form("Colombo"),
):
    """
    Public user submits image + waste type.
    System saves:
      1. Image + metadata to MongoDB
      2. Physical image file into data/new_waste/{waste_type}/
    Retraining is triggered automatically.
    """
    try:
        contents = await image.read()
    except Exception as e:
        return JSONResponse({"error": f"Failed to read image: {e}"}, status_code=400)

    # ---- 1) Save to MongoDB ----
    doc = {
        "image_bytes": contents,
        "waste_type": waste_type,
        "region": region,
        "city": city,
        "source": "public",
        "processed": False,
        "created_at": datetime.now(timezone.utc)
    }
    db.submissions.insert_one(doc)

    # ---- 2) Save image to local folder by class ----
    class_dir = os.path.join(NEW_DATA_DIR, waste_type)
    os.makedirs(class_dir, exist_ok=True)

    ext = os.path.splitext(image.filename)[1] or ".jpg"
    unique_filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(class_dir, unique_filename)

    try:
        with open(save_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        return JSONResponse({"error": f"Failed to save image to disk: {e}"}, status_code=500)

    # ---- 3) Trigger retrain if needed ----
    retrain_msg = check_and_trigger_retrain()

    return {
        "message": "Submission received.",
        "saved_file": f"{waste_type}/{unique_filename}",
        "retrain_status": retrain_msg
    }

# Expose router so run.py can import it
user_routes = router
