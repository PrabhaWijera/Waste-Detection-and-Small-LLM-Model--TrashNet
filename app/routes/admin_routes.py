from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Query
from typing import Optional

from app.database.mongo_client import db

router = APIRouter()

@router.get("/metrics")
def metrics(days: int = Query(30, ge=1, le=365)):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {
            "_id": {"prediction": "$prediction", "city": "$city"},
            "count": {"$sum": 1}
        }}
    ]
    by_class_city = list(db.submissions.aggregate(pipeline))
    total = db.submissions.count_documents({"created_at": {"$gte": since}})
    return {"since": since.isoformat(), "total": total, "by_class_city": by_class_city}

@router.get("/submissions")
def submissions(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)):
    skip = (page - 1) * page_size
    docs = list(db.submissions.find().sort("created_at", -1).skip(skip).limit(page_size))
    # Convert ObjectIds to strings
    for d in docs:
        d["_id"] = str(d["_id"])
        if "created_at" in d and hasattr(d["created_at"], "isoformat"):
            d["created_at"] = d["created_at"].isoformat()
    return {"page": page, "page_size": page_size, "items": docs}


