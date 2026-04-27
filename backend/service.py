from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommendation_engine import (  # noqa: E402
    BRANCH_OPTIONS,
    LOCATION_OPTIONS,
    REGION_OPTIONS,
    TIER_OPTIONS,
    WORK_MODE_OPTIONS,
    InternshipRecommendationEngine,
    prepare_profile_for_inference,
    validate_profile,
)


class RecommendationService:
    def __init__(self) -> None:
        self.engine = InternshipRecommendationEngine()
        self.ready = False

    def _find_file(self, candidates: List[str]) -> Optional[str]:
        for candidate in candidates:
            for root in (BASE_DIR, SRC_DIR, BASE_DIR / "data"):
                path = root / candidate
                if path.exists():
                    return str(path)
        return None

    def initialize(self) -> None:
        if self.ready:
            return

        students_path = self._find_file(["students_cleaned.csv", "students_uncleaned_new_v2.csv"])
        internships_path = self._find_file(["internships_cleaned.csv", "internships_uncleaned_new_v2.csv"])

        if not students_path or not internships_path:
            raise FileNotFoundError("Could not locate the student and internship datasets.")

        self.engine.load_data(students_path, internships_path)
        self.engine.fit()
        self.ready = True

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "students": 0 if self.engine.students_df is None else int(len(self.engine.students_df)),
            "internships": 0 if self.engine.internships_df is None else int(len(self.engine.internships_df)),
            "model_type": self.engine.ranker.model_type,
        }

    def model_info(self) -> Dict[str, Any]:
        if not self.ready:
            self.initialize()

        return {
            "name": "AI Internship Recommendation Engine",
            "purpose": "Personalized, fair, and explainable recommendations for PM Internship style student opportunity discovery.",
            "pipeline": [
                {
                    "stage": "Candidate retrieval",
                    "description": "Compares student skills, interests, branch, and work-mode preference with internship text.",
                },
                {
                    "stage": "ML ranking",
                    "description": "Ranks candidates using skill overlap, domain fit, CGPA fit, location fit, stipend, and popularity.",
                },
                {
                    "stage": "Fairness and diversity",
                    "description": "Promotes qualified rural or Tier-3 profiles and avoids repetitive companies or domains in top results.",
                },
                {
                    "stage": "Explainability",
                    "description": "Shows matched skills, missing skills, score drivers, and practical improvement actions.",
                },
            ],
            "model_type": self.engine.ranker.model_type,
            "feature_importance": self.engine.ranker.feature_importance,
            "datasets": {
                "students": 0 if self.engine.students_df is None else int(len(self.engine.students_df)),
                "internships": 0 if self.engine.internships_df is None else int(len(self.engine.internships_df)),
                "interactions": 0 if self.engine.interactions_df is None else int(len(self.engine.interactions_df)),
            },
        }

    def options(self) -> Dict[str, List[str]]:
        return {
            "branches": BRANCH_OPTIONS,
            "tiers": TIER_OPTIONS,
            "regions": REGION_OPTIONS,
            "locations": LOCATION_OPTIONS,
            "work_modes": WORK_MODE_OPTIONS,
        }

    def recommend(self, profile: Dict[str, Any], top_k: int = 8, student_id: Optional[str] = None) -> Dict[str, Any]:
        if not self.ready:
            self.initialize()

        validated = validate_profile(profile)
        prepared = prepare_profile_for_inference(self.engine, validated)
        recommendations = self.engine.recommend(
            prepared,
            top_k=top_k,
            student_id=student_id or prepared.get("student_id"),
        )

        fairness_input = pd.DataFrame(
            [
                {
                    "Domain": item.get("domain", "Unknown"),
                    "Company": item.get("company", "Unknown"),
                    "Stipend (INR)": item.get("stipend", 0),
                }
                for item in recommendations
            ]
        )

        fairness_report = self.engine.policy.fairness_report(fairness_input)

        return {
            "profile": prepared,
            "recommendations": recommendations,
            "fairness_report": fairness_report,
            "model": {
                "model_type": self.engine.ranker.model_type,
                "pipeline": "Retrieval -> ML Ranking -> Fairness Reranking -> Explainability",
            },
        }


service = RecommendationService()
