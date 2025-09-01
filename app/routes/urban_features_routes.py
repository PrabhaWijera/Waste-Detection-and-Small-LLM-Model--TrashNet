from fastapi import APIRouter, Depends, HTTPException, status, Query
from app.database.connector_mysql import db
from app.utils.dependencies import get_current_user
import csv
from fastapi.responses import StreamingResponse
from io import StringIO
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import aiosmtplib
from fastapi.responses import JSONResponse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
router = APIRouter()

# Admin-only dependency
def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "urban":  # assuming urban users are admin
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# ------------------- Urban Users -------------------
@router.get("/urban_users")
def list_urban_users(admin: dict = Depends(get_current_admin)):
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE role='urban'")
    users = cursor.fetchall()
    cursor.close()
    return {"urban_users": users}

# ------------------- Normal Submissions -------------------
@router.get("/normal_submissions")
def list_normal_submissions(
    admin: dict = Depends(get_current_admin), 
    start_date: str = None, 
    end_date: str = None
):
    cursor = db.cursor(dictionary=True)
    query = """
        SELECT id, name, email, phone, location, urban_center, waste_type, source, processed, created_at
        FROM submissions
        WHERE 1=1
    """
    params = []

    if start_date:
        query += " AND created_at >= %s "
        params.append(start_date + " 00:00:00")
    if end_date:
        query += " AND created_at <= %s "
        params.append(end_date + " 23:59:59")

    query += " ORDER BY created_at DESC"
    cursor.execute(query, tuple(params))
    submissions = cursor.fetchall()
    cursor.close()
    return {"submissions": submissions}


# ------------------- Monthly Waste Impact -------------------
@router.get("/waste_report/monthly")
def monthly_report(admin: dict = Depends(get_current_admin), 
                   start_date: str = None, end_date: str = None):
    cursor = db.cursor(dictionary=True)
    query = """
        SELECT DATE_FORMAT(created_at, '%Y-%m') as month, waste_type, COUNT(*) as total
        FROM submissions
        WHERE 1=1
    """
    params = []

    if start_date:
        query += " AND created_at >= %s "
        params.append(start_date)
    if end_date:
        query += " AND created_at <= %s "
        params.append(end_date)

    query += " GROUP BY month, waste_type ORDER BY month DESC"
    cursor.execute(query, tuple(params))
    report = cursor.fetchall()
    cursor.close()
    return {"monthly_report": report}

# ------------------- Yearly Waste Impact -------------------


# ------------------- Download CSV -------------------
@router.get("/download_report")
def download_report(admin: dict = Depends(get_current_admin),
                    period: str = Query("monthly", enum=["monthly", "yearly"])):
    si = StringIO()
    writer = csv.writer(si)

    if period == "monthly":
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT DATE_FORMAT(created_at, '%Y-%m') as month, waste_type, COUNT(*) as total
            FROM submissions
            GROUP BY month, waste_type ORDER BY month DESC
        """)
        report = cursor.fetchall()
        writer.writerow(["Month", "Waste Type", "Total"])
        for row in report:
            writer.writerow([row['month'], row['waste_type'], row['total']])
        cursor.close()
    else:
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT YEAR(created_at) as year, waste_type, COUNT(*) as total
            FROM submissions
            GROUP BY year, waste_type ORDER BY year DESC
        """)
        report = cursor.fetchall()
        writer.writerow(["Year", "Waste Type", "Total"])
        for row in report:
            writer.writerow([row['year'], row['waste_type'], row['total']])
        cursor.close()

    si.seek(0)
    return StreamingResponse(si, media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={period}_waste_report.csv"})

# In your FastAPI router

# Sri Lanka city coordinates (you can expand this list)
LOCATION_COORDS = {
    "Colombo": {"lat": 6.9271, "lng": 79.8612},
    "Kandy": {"lat": 7.2906, "lng": 80.6337},
    "Galle": {"lat": 6.0535, "lng": 80.2210},
    "Jaffna": {"lat": 9.6615, "lng": 80.0255},
    "Negombo": {"lat": 7.2008, "lng": 79.8737},
    # fallback default
    "DEFAULT": {"lat": 7.8731, "lng": 80.7718}
}


@router.get("/risk_zones")
def risk_zones(admin: dict = Depends(get_current_admin)):
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT location, COUNT(*) as count
        FROM submissions
        WHERE location IS NOT NULL AND location != ''
        GROUP BY location
    """)
    rows = cursor.fetchall()
    cursor.close()

    # compute threshold (top 20% = high risk)
    counts = sorted([r["count"] for r in rows], reverse=True)
    threshold = counts[int(len(counts) * 0.2)] if counts else 0

    zones = []
    for r in rows:
        coords = LOCATION_COORDS.get(r["location"], LOCATION_COORDS["DEFAULT"])
        zones.append({
            "location": r["location"],
            "count": r["count"],
            "risk": "high" if r["count"] >= threshold else "low",
            "lat": coords["lat"],
            "lng": coords["lng"]
        })

    return {"zones": zones}

class EmailRequest(BaseModel):
    email: str
    name: str
    waste_type: str

@router.post("/send-user-email")
async def send_user_email(req: EmailRequest):
    from_email = "prabhashwlive2001@gmail.com"
    password = "shaaumbtribzkecz"

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = req.email
    msg["Subject"] = "Your Waste Submission Has Been Processed"

    body = f"""
    <html>
    <body>
        <p>✅ Hello {req.name},</p>
        <p>Thank you for your public service! Your submission regarding <b>{req.waste_type}</b> waste has been successfully processed and is now ready for recycling.</p>
        <p>🌿 We appreciate your contribution to a cleaner environment!</p>
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
        return JSONResponse(content={"status": "success", "message": f"Email sent to {req.email}"})
    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})