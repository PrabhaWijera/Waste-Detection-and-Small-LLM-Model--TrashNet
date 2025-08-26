# run.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.user_routes import router as user_router
from app.routes.admin_routes import router as admin_router
from app.routes.public_user_routes import user_routes

app = FastAPI(title="YOLOv8 Waste Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/api", tags=["User"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
app.include_router(user_routes, prefix="/api/public", tags=["Public"])

@app.get("/")
def root():
    return {"status": "ok", "service": "YOLOv8 Waste API"}

if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
