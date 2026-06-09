import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.frameworks.routes import audio_routes

app = FastAPI(title="VietASR Web API Gateway")

# Enable strict CORS constraints as required by the Constitution
allowed_origins_env = os.getenv(
    "ALLOWED_CORS_ORIGINS", 
    "http://localhost:5173,http://localhost:3000"
)
origins = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register endpoints
app.include_router(audio_routes.router, prefix="/api")

@app.get("/health")
def health_check():
    return {"status": "ok"}
