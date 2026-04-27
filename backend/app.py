from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from service import service
except ImportError:  # pragma: no cover - supports `uvicorn backend.app:app`
    from .service import service


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"


class RecommendationRequest(BaseModel):
    student_id: Optional[str] = None
    branch: str = "Computer Science"
    cgpa: float = Field(default=7.5, ge=0.0, le=10.0)
    skills: str = "Python, SQL"
    interests: str = "Data Science"
    location: str = "Remote"
    college_tier: str = "Tier-2"
    region: str = "Urban"
    preferred_work_mode: str = "No Preference"
    objective_learning: float = Field(default=0.4, ge=0.0, le=1.0)
    objective_career_fit: float = Field(default=0.4, ge=0.0, le=1.0)
    objective_compensation: float = Field(default=0.2, ge=0.0, le=1.0)
    top_k: int = Field(default=8, ge=1, le=20)

app = FastAPI(title="Internship Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")


@app.on_event("startup")
def startup_event() -> None:
    try:
        service.initialize()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize recommendation service: {exc}") from exc


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return service.health()


@app.get("/api/options")
def options() -> Dict[str, Any]:
    return service.options()


@app.get("/api/model-info")
def model_info() -> Dict[str, Any]:
    return service.model_info()


@app.post("/api/recommendations")
def recommendations(payload: RecommendationRequest) -> Dict[str, Any]:
    try:
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        return service.recommend(data, top_k=payload.top_k, student_id=payload.student_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/")
def index() -> FileResponse:
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_file)
