from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from PIL import Image
import io, json

from app.database.connector_mysql import db
from app.services.model_registry import get_classifier
from app.models.recommendation_engine import RecommendationEngine
from app.models.waste_llm_yolo import WasteLLM
from app.utils.dependencies import get_current_user

router = APIRouter()
recommender = RecommendationEngine()
llm = WasteLLM()

@router.post("/classify")
async def classify_image(image: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    try:
        img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    label, confidence, _ = get_classifier().predict(img)
    if not label:
        label, confidence = "trash", 0.0
    recommendation = recommender.recommend(label)

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO urban_submissions 
        (urban_user_id, urban_center, prediction, confidence, recommendation, source, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        current_user["id"],
        "UNKNOWN",
        label,
        confidence,
        json.dumps(recommendation),
        "image",
        datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    ))
    db.commit()
    cursor.close()

    return JSONResponse({"prediction": label, "confidence": confidence, "recommendation": recommendation})

@router.post("/llm_recommend")
async def llm_recommend(description: str = Form(...), current_user: dict = Depends(get_current_user)):
    result = llm.recommend(description)

    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO urban_llm_recommendations 
        (urban_user_id, urban_center, description, recommendation, created_at)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        current_user["id"],
        "UNKNOWN",
        description,
        result,
        datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    ))
    db.commit()
    cursor.close()

    return {"input": description, "recommendation": result}


class SubmissionClassification(BaseModel):
    submission_id: int
    prediction: str
    confidence: float
    recommendation: str

@router.put("/classify-update")
async def classify_submission(data: SubmissionClassification):
    conn = db  # Use your existing DB connection
    cursor = conn.cursor()

    try:
        # Update the processed field in submissions table
        cursor.execute("""
            UPDATE submissions
            SET processed = TRUE
            WHERE id = %s
        """, (data.submission_id,))

        # Insert classification record into urban_submissions
        cursor.execute("""
            INSERT INTO urban_submissions
            (urban_user_id, urban_center, prediction, confidence, recommendation, source, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            data.submission_id,
            "UNKNOWN",
            data.prediction,
            data.confidence,
            data.recommendation,
            "manual",
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        return JSONResponse({"message": "Submission classified successfully"})

    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()





















# from fastapi import APIRouter, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# from datetime import datetime, timezone
# import json

# from app.services.model_registry import get_classifier
# from app.models.recommendation_engine import RecommendationEngine
# from app.models.waste_llm_yolo import WasteLLM
# from app.database.connector_mysql import db

# router = APIRouter()
# recommender = RecommendationEngine()
# llm = WasteLLM()

# @router.post("/classify")
# async def classify_image(
#     image: UploadFile = File(...),
#     urban_user_id: str = Form(None),
#     urban_center: str = Form(None),  # optional field
# ):
#     try:
#         img = Image.open(io.BytesIO(await image.read())).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

#     classifier = get_classifier()
#     label, confidence, _ = classifier.predict(img)

#     # Safe default if nothing detected
#     if not label:
#         label, confidence = "trash", 0.0

#     recommendation = recommender.recommend(label)

#     # Save to MySQL (urban table)
#     cursor = db.cursor()
#     query = """
#         INSERT INTO urban_submissions 
#         (urban_user_id, urban_center, prediction, confidence, recommendation, source, created_at)
#         VALUES (%s, %s, %s, %s, %s, %s, %s)
#     """
#     cursor.execute(query, (
#         urban_user_id,
#         urban_center,
#         label,
#         confidence,
#         json.dumps(recommendation),  # save as JSON string
#         "image",
#         datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#     ))
#     db.commit()
#     cursor.close()

#     return JSONResponse({
#         "prediction": label,
#         "confidence": confidence,
#         "recommendation": recommendation
#     })


# @router.post("/llm_recommend")
# async def llm_recommend(
#     description: str = Form(...),
#     urban_user_id: str = Form(None),
#     urban_center: str = Form(None)  # optional field
# ):
#     result = llm.recommend(description)

#     # Save to urban LLM table
#     cursor = db.cursor()
#     query = """
#         INSERT INTO urban_llm_recommendations 
#         (urban_user_id, urban_center, description, recommendation, created_at)
#         VALUES (%s, %s, %s, %s, %s)
#     """
#     cursor.execute(query, (
#         urban_user_id,
#         urban_center,
#         description,
#         result,
#         datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
#     ))
#     db.commit()
#     cursor.close()

#     return {"input": description, "recommendation": result}

























# from fastapi import APIRouter, UploadFile, File, Form, HTTPException
# from fastapi.responses import JSONResponse
# from PIL import Image
# import io
# from datetime import datetime, timezone

# from app.services.model_registry import get_classifier
# from app.models.recommendation_engine import RecommendationEngine
# from app.models.waste_llm_yolo import WasteLLM
# from app.database.connector_mysql import db

# router = APIRouter()
# recommender = RecommendationEngine()
# llm = WasteLLM()

# @router.post("/classify")
# async def classify_image(
#     image: UploadFile = File(...),
#     user_id: str = Form(None),
# ):
#     try:
#         img = Image.open(io.BytesIO(await image.read())).convert("RGB")
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

#     classifier = get_classifier()
#     label, confidence, _ = classifier.predict(img)

#     # Safe default if nothing detected
#     if not label:
#         label, confidence = "trash", 0.0

#     recommendation = recommender.recommend(label)

#     # Save to DB
#     doc = {
#         "user_id": user_id,
#         "prediction": label,
#         "confidence": confidence,
#         "recommendation": recommendation,
#         "source": "image",
#         "created_at": datetime.now(timezone.utc),
#     }
#     db.submissions.insert_one(doc)

#     return JSONResponse({
#         "prediction": label,
#         "confidence": confidence,
#         "recommendation": recommendation
#     })

# @router.post("/llm_recommend")
# async def llm_recommend(description: str = Form(...)):
#     result = llm.recommend(description)
#     return {"input": description, "recommendation": result}
