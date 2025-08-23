import io
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from datetime import datetime, timezone

from app.models.waste_classifier import WasteClassifier
from app.models.recommendation_engine import RecommendationEngine
from models.waste_llm import WasteLLM
from app.database.mongo_client import db

router = APIRouter()

# Load models once
classifier = WasteClassifier()  # lazy loads model on first predict
recommender = RecommendationEngine()
llm = WasteLLM()  # LLM for text recommendations

class TextOnlyRequest(BaseModel):
    description: str
    region: Optional[str] = "LK-11"
    city: Optional[str] = "Colombo"

# ---------------- IMAGE CLASSIFICATION + RULE-BASED ----------------
@router.post("/classify")
async def classify_image(
    image: UploadFile = File(...),
    region: str = Form("LK-11"),
    city: str = Form("Colombo"),
    user_id: Optional[str] = Form(None)
):
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image. {e}")

    # Step 1: Classify
    pred, confidence = classifier.predict(pil_img)
    # Step 2: Rule-based recommendation
    recommendation = recommender.recommend(pred, region=region, city=city)

    # Save to DB
    doc = {
        "user_id": user_id,
        "region": region,
        "city": city,
        "prediction": pred,
        "confidence": float(confidence),
        "recommendation": recommendation,
        "source": "image",
        "created_at": datetime.now(timezone.utc)
    }
    try:
        db.submissions.insert_one(doc)
    except Exception as e:
        doc["_db_error"] = str(e)

    return JSONResponse({
        "prediction": pred,
        "confidence": confidence,
        "recommendation": recommendation,
        "metadata": {"region": region, "city": city}
    })

# ---------------- TEXT-ONLY RULE-BASED ----------------
@router.post("/recommend")
async def recommend_from_text(payload: TextOnlyRequest):
    text = payload.description.lower()
    label = "trash"
    keywords = {
        "plastic": ["plastic", "poly", "pet", "bottle", "polythene"],
        "paper": ["paper", "magazine", "newspaper", "carton paper"],
        "glass": ["glass", "jar", "bottle glass"],
        "metal": ["metal", "tin", "aluminum", "steel", "can"],
        "cardboard": ["cardboard", "carton", "box"]
    }
    for k, arr in keywords.items():
        if any(w in text for w in arr):
            label = k
            break

    recommendation = recommender.recommend(label, region=payload.region, city=payload.city)

    doc = {
        "user_id": None,
        "region": payload.region,
        "city": payload.city,
        "prediction": label,
        "confidence": None,
        "recommendation": recommendation,
        "source": "text",
        "created_at": datetime.now(timezone.utc),
        "raw_text": payload.description
    }
    try:
        db.submissions.insert_one(doc)
    except Exception as e:
        doc["_db_error"] = str(e)

    return {"prediction": label, "recommendation": recommendation}

# ---------------- HEALTH CHECK ----------------
@router.get("/health")
async def health():
    db_ok = True
    try:
        db.command("ping")
    except Exception:
        db_ok = False
    return {"status": "ok", "db": db_ok, "model_loaded": classifier.model_loaded}

# ---------------- LLM RECOMMENDATION ----------------
# ---------------- LLM RECOMMENDATION ----------------
@router.post("/llm_recommend")
async def llm_recommend(
    description: str = Form(...),   # <-- use Form instead of plain str
    region: str = Form("LK-11"),
    city: str = Form("Colombo")
):
    """
    Generates a recommendation from LLM based on user input text.
    """
    result = llm.recommend(description, region=region, city=city)
    return {"input": description, "recommendation": result}
# ---------------- IMAGE -> CLASSIFIER -> LLM (Optional future) ----------------
@router.post("/classify_llm")
async def classify_with_llm(
    image: UploadFile = File(...),
    region: str = Form("LK-11"),
    city: str = Form("Colombo")
):
    """
    Combines classifier prediction with LLM for richer recommendations.
    """
    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image. {e}")

    label, confidence = classifier.predict(pil_img)
    prompt_text = f"Predicted waste: {label}, region: {region}, city: {city}"
    llm_result = llm.recommend(prompt_text, region=region, city=city)

    return {
        "prediction": label,
        "confidence": confidence,
        "llm_recommendation": llm_result
    }
