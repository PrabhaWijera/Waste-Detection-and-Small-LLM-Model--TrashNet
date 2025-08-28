from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Query
from typing import Optional
from app.database.connector_mysql import db

router = APIRouter()

@router.get("/metrics")
def metrics(days: int = Query(30, ge=1, le=365)):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_str = since.strftime('%Y-%m-%d %H:%M:%S')
    
    cursor = db.cursor(dictionary=True)
    
    # Total count
    cursor.execute("SELECT COUNT(*) AS total FROM submissions WHERE created_at >= %s", (since_str,))
    total = cursor.fetchone()["total"]
    
    # Group by waste_type and city
    cursor.execute("""
        SELECT waste_type AS prediction, city, COUNT(*) AS count
        FROM submissions
        WHERE created_at >= %s
        GROUP BY waste_type, city
    """, (since_str,))
    
    by_class_city = cursor.fetchall()
    cursor.close()
    
    return {"since": since.isoformat(), "total": total, "by_class_city": by_class_city}


@router.get("/submissions")
def submissions(page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=200)):
    offset = (page - 1) * page_size
    cursor = db.cursor(dictionary=True)
    
    cursor.execute(f"""
        SELECT * FROM submissions
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
    """, (page_size, offset))
    
    items = cursor.fetchall()
    
    # Convert datetime to ISO format
    for d in items:
        if "created_at" in d and isinstance(d["created_at"], datetime):
            d["created_at"] = d["created_at"].isoformat()
    
    cursor.close()
    return {"page": page, "page_size": page_size, "items": items}
