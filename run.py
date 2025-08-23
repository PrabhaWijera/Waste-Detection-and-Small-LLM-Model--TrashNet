import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.routes.user_routes import router as user_router
from app.routes.admin_routes import router as admin_router

load_dotenv()

app = FastAPI(
    title="AI-Powered Waste Classification & Recommendation API",
    version="1.0.0",
    description="Backend for waste classification using CNN (ResNet) + rule-based/LLM-ready recommendations."
)

# CORS (adjust for your frontend origins)
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(user_router, prefix="/api", tags=["User"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])

@app.get("/")
def root():
    return {"status": "ok", "service": "waste-management-backend", "time": os.getenv("TZ", "UTC")}

# Run: uvicorn run:app --reload
