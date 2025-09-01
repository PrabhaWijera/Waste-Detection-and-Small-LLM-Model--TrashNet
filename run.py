# run.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.urban_routes import router as user_router
from app.routes.admin_routes import router as admin_router
from app.routes.public_user_routes import user_routes
from app.routes.auth_routes import router as auth_router   # NEW
from app.routes.urban_features_routes import router as features_router
app = FastAPI(title="YOLOv8 Waste Management API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router, prefix="/api", tags=["Urban"])  # <-- this was 'User', rename to 'Urban'
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])
app.include_router(user_routes, prefix="/api/public", tags=["Public"])

# New routers
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])     # Register/Login
app.include_router(features_router, prefix="/api/admin", tags=["Urban Features"])

@app.get("/")
def root():
    return {"status": "ok", "service": "YOLOv8 Waste API"}

if __name__ == "__main__":
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True)
