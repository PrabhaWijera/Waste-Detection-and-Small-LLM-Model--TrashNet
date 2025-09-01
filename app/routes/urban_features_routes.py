from fastapi import APIRouter, Depends, HTTPException, status, Query
from app.database.connector_mysql import db
from app.utils.dependencies import get_current_user
import csv
from fastapi.responses import StreamingResponse
from io import StringIO
from datetime import datetime

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
@router.get("/waste_report/yearly")
def yearly_report(admin: dict = Depends(get_current_admin)):
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT YEAR(created_at) as year, waste_type, COUNT(*) as total
        FROM submissions
        GROUP BY year, waste_type
        ORDER BY year DESC
    """)
    report = cursor.fetchall()
    cursor.close()
    return {"yearly_report": report}

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
