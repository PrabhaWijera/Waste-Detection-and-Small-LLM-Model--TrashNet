from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from datetime import datetime, timezone

from app.services.model_registry import get_classifier
from app.models.recommendation_engine import RecommendationEngine
from app.models.waste_llm_yolo import WasteLLM
from app.database.mongo_client import db

router = APIRouter()
recommender = RecommendationEngine()
llm = WasteLLM()

@router.post("/classify")
async def classify_image(
    image: UploadFile = File(...),
    region: str = Form("LK-11"),
    city: str = Form("Colombo"),
    user_id: str = Form(None),
):
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    classifier = get_classifier()
    label, confidence, _ = classifier.predict(img)

    # Safe default if nothing detected
    if not label:
        label, confidence = "trash", 0.0

    recommendation = recommender.recommend(label, region=region, city=city)

    # Save to DB
    doc = {
        "user_id": user_id,
        "region": region,
        "city": city,
        "prediction": label,
        "confidence": confidence,
        "recommendation": recommendation,
        "source": "image",
        "created_at": datetime.now(timezone.utc),
    }
    db.submissions.insert_one(doc)

    return JSONResponse({
        "prediction": label,
        "confidence": confidence,
        "recommendation": recommendation
    })

@router.post("/llm_recommend")
async def llm_recommend(description: str = Form(...), region: str = Form("LK-11"), city: str = Form("Colombo")):
    result = llm.recommend(description, region=region, city=city)
    return {"input": description, "recommendation": result}
