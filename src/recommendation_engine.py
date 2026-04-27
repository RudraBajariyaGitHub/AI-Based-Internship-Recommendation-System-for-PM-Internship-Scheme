import json
import logging
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.ensemble import RandomForestRegressor

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Falling back to rule-based scoring.")

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from preprocessing import MLPreprocessor

    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    logger.warning("preprocessing.py not available. Using raw CSV loading.")


# ---------------------------- Constants ------------------------------------

FEATURE_NAMES = [
    "retrieval_score",
    "skill_overlap",
    "weighted_skill_overlap",
    "domain_weight_score",
    "domain_interest",
    "cgpa_fit",
    "location_fit",
    "stipend_norm",
    "work_mode_fit",
    "company_popularity",
    "collaborative_score",
]

WEIGHT_FALLBACK = {
    "retrieval_score": 0.23,
    "skill_overlap": 0.16,
    "weighted_skill_overlap": 0.12,
    "domain_weight_score": 0.07,
    "domain_interest": 0.10,
    "cgpa_fit": 0.06,
    "location_fit": 0.06,
    "stipend_norm": 0.06,
    "work_mode_fit": 0.05,
    "company_popularity": 0.04,
    "collaborative_score": 0.05,
}

FAIRNESS_TIER_BONUS = {"Tier-3": 0.05, "Tier-2": 0.02, "Tier-1": 0.0}
FAIRNESS_RURAL_BONUS = 0.03

CGPA_EXCELLENT = 8.5
CGPA_GOOD = 7.0


# ---------------------------- Utility --------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _norm_text(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _split_csv_like(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def _row_value(row: Any, key: str, default: Any = "") -> Any:
    """Safely get a scalar value from dict-like or Series-like rows.

    Handles duplicate column labels where pandas may return a Series instead of a scalar.
    """
    try:
        if isinstance(row, dict):
            return row.get(key, default)

        if hasattr(row, "index"):
            if key not in row.index:
                return default
            value = row[key]
            if isinstance(value, pd.Series):
                return value.iloc[0] if len(value) > 0 else default
            return value

        getter = getattr(row, "get", None)
        if callable(getter):
            return getter(key, default)
    except Exception:
        return default
    return default


def _normalize_skill_text(text: str) -> str:
    mapping = {
        "py": "Python",
        "python3": "Python",
        "ml": "Machine Learning",
        "machinelearning": "Machine Learning",
        "sql": "SQL",
        "js": "JavaScript",
        "javascript": "JavaScript",
        "ai": "Artificial Intelligence",
        "ds": "Data Science",
        "cse": "Computer Science",
        "cs": "Computer Science",
    }
    parts = []
    for item in _split_csv_like(text):
        key = item.strip().lower().replace(" ", "")
        parts.append(mapping.get(key, item.strip().title()))
    return ", ".join(dict.fromkeys(parts))


def _normalize_branch_text(text: str) -> str:
    value = _norm_text(text).lower().replace("&", "and")
    mapping = {
        "cse": "Computer Science",
        "computer science": "Computer Science",
        "it": "Information Technology",
        "information technology": "Information Technology",
        "data science": "Data Science",
        "ai": "Artificial Intelligence",
        "artificial intelligence": "Artificial Intelligence",
        "ece": "Electronics & Communication",
        "electronics and communication": "Electronics & Communication",
        "electronics & communication": "Electronics & Communication",
        "ee": "Electrical Engineering",
        "electrical engineering": "Electrical Engineering",
        "me": "Mechanical Engineering",
        "mechanical engineering": "Mechanical Engineering",
        "ce": "Civil Engineering",
        "civil engineering": "Civil Engineering",
        "che": "Chemical Engineering",
        "chemical engineering": "Chemical Engineering",
        "biotech": "Biotechnology",
        "biotechnology": "Biotechnology",
    }
    return mapping.get(value, _norm_text(text).title() or "Other")


def _normalize_location_text(text: str) -> str:
    value = _norm_text(text).lower()
    mapping = {
        "bangalore": "Bangalore",
        "bengaluru": "Bangalore",
        "mumbai": "Mumbai",
        "delhi": "Delhi",
        "hyderabad": "Hyderabad",
        "chennai": "Chennai",
        "pune": "Pune",
        "noida": "Noida",
        "gurgaon": "Gurgaon",
        "gurugram": "Gurgaon",
        "kolkata": "Kolkata",
        "remote": "Remote",
        "on site": "On-site",
        "onsite": "On-site",
        "hybrid": "Hybrid",
    }
    return mapping.get(value, _norm_text(text).title() or "Other")


def _normalize_work_mode_text(text: str) -> str:
    value = _norm_text(text).lower()
    mapping = {
        "remote": "Remote",
        "hybrid": "Hybrid",
        "on-site": "On-site",
        "onsite": "On-site",
        "no preference": "No Preference",
    }
    return mapping.get(value, _norm_text(text).title() or "No Preference")


def validate_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    validated = {
        "student_id": _norm_text(profile.get("student_id", "")) or None,
        "branch": _normalize_branch_text(profile.get("branch", "Other")),
        "cgpa": _safe_float(profile.get("cgpa", 7.5), 7.5),
        "skills": _normalize_skill_text(profile.get("skills", "Python, SQL")),
        "interests": _normalize_skill_text(profile.get("interests", "Data Science")),
        "location": _normalize_location_text(profile.get("location", "Remote")),
        "college_tier": _norm_text(profile.get("college_tier", "Tier-2")) or "Tier-2",
        "region": _norm_text(profile.get("region", "Urban")) or "Urban",
        "preferred_work_mode": _normalize_work_mode_text(profile.get("preferred_work_mode", "No Preference")),
        "objective_learning": _safe_float(profile.get("objective_learning", 0.4), 0.4),
        "objective_career_fit": _safe_float(profile.get("objective_career_fit", 0.4), 0.4),
        "objective_compensation": _safe_float(profile.get("objective_compensation", 0.2), 0.2),
    }

    validated["cgpa"] = max(0.0, min(10.0, float(validated["cgpa"])))
    if validated["college_tier"] not in TIER_OPTIONS:
        validated["college_tier"] = "Tier-2"
    if validated["region"] not in REGION_OPTIONS:
        validated["region"] = "Urban"
    if validated["preferred_work_mode"] not in WORK_MODE_OPTIONS:
        validated["preferred_work_mode"] = "No Preference"
    if validated["branch"] not in BRANCH_OPTIONS:
        validated["branch"] = "Other"
    if validated["location"] not in LOCATION_OPTIONS:
        validated["location"] = "Other"

    if not _split_csv_like(validated["skills"]):
        validated["skills"] = "Python"
    if not _split_csv_like(validated["interests"]):
        validated["interests"] = "Technology"

    weights = {
        "objective_learning": max(0.0, validated["objective_learning"]),
        "objective_career_fit": max(0.0, validated["objective_career_fit"]),
        "objective_compensation": max(0.0, validated["objective_compensation"]),
    }
    total = sum(weights.values())
    if total <= 0:
        weights = {"objective_learning": 0.4, "objective_career_fit": 0.4, "objective_compensation": 0.2}
    else:
        weights = {k: v / total for k, v in weights.items()}
    validated.update(weights)
    return validated


def load_student_from_json(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Profile JSON not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Profile JSON must contain a single JSON object.")
    return validate_profile(payload)


def prepare_profile_for_inference(engine: Any, profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = validate_profile(profile)
    preprocessor = getattr(engine, "preprocessor", None)
    if preprocessor is None:
        return profile

    clean_fn = getattr(preprocessor, "clean_student_profile", None)
    if callable(clean_fn):
        try:
            cleaned = clean_fn(profile)
            if isinstance(cleaned, dict):
                profile.update(cleaned)
                profile = validate_profile(profile)
        except Exception as ex:
            logger.warning("Preprocessor profile cleaning failed (%s). Using validated profile.", ex)

    return profile


class CollaborativeSignalEngine:
    """Collaborative signal provider using item popularity from labelled interactions."""

    def __init__(self):
        self._fitted = False
        self._model = None
        self._dataset = None
        self._item_popularity: Dict[str, float] = {}

    def fit(self, interactions_df: Optional[pd.DataFrame]) -> None:
        self._fitted = True
        self._item_popularity = {}

        if interactions_df is None or interactions_df.empty:
            return

        if "internship_id" in interactions_df.columns:
            pop = interactions_df["internship_id"].astype(str).value_counts()
            mx = float(pop.max()) if len(pop) else 1.0
            self._item_popularity = {k: float(v / mx) for k, v in pop.items()}

    def score_items(self, student_id: Optional[str], internship_ids: List[str]) -> Dict[str, float]:
        if not internship_ids:
            return {}

        return {iid: float(self._item_popularity.get(str(iid), 0.0)) for iid in internship_ids}


# ------------------------ Stage 1: Retrieval -------------------------------

class CandidateRetrievalEngine:
    """
    Semantic retrieval engine.

    Priority order:
      1) sentence-transformers + FAISS
      2) sentence-transformers + brute-force cosine
      3) TF-IDF cosine
      4) lightweight token overlap fallback
    """

    def __init__(self, retrieval_k: int = 300):
        self.retrieval_k = retrieval_k
        self._ids: List[str] = []
        self._fitted = False

        self._mode = "token"
        self._embedder = None
        self._faiss_index = None
        self._emb_matrix = None

        self._tfidf = None
        self._tfidf_matrix = None

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
                self._mode = "embedding"
            except Exception as ex:
                logger.info("Local sentence-transformer unavailable (%s). Using TF-IDF retrieval.", ex)
                self._embedder = None

        if self._embedder is None and SKLEARN_AVAILABLE:
            self._tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
            self._mode = "tfidf"

    @staticmethod
    def _internship_text(row: pd.Series) -> str:
        parts = [
            _norm_text(row.get("Title", "")),
            _norm_text(row.get("Domain", "")),
            _norm_text(row.get("Required Skills", "")),
            _norm_text(row.get("Required Skills", "")),
            _norm_text(row.get("Description", "")),
            _norm_text(row.get("Company", "")),
        ]
        return " ".join([p for p in parts if p]).lower()

    @staticmethod
    def _student_text(profile: Dict[str, Any]) -> str:
        parts = [
            _norm_text(profile.get("skills", "")),
            _norm_text(profile.get("skills", "")),
            _norm_text(profile.get("interests", "")),
            _norm_text(profile.get("branch", "")),
            _norm_text(profile.get("preferred_work_mode", "")),
        ]
        return " ".join([p for p in parts if p]).lower()

    def fit(self, internships_df: pd.DataFrame) -> None:
        self._ids = internships_df["Internship ID"].astype(str).tolist()
        corpus = [self._internship_text(row) for _, row in internships_df.iterrows()]

        if self._mode == "embedding" and self._embedder is not None:
            emb = self._embedder.encode(corpus, show_progress_bar=False, normalize_embeddings=True)
            self._emb_matrix = np.asarray(emb, dtype=np.float32)
            if FAISS_AVAILABLE:
                index = faiss.IndexFlatIP(self._emb_matrix.shape[1])
                index.add(self._emb_matrix)
                self._faiss_index = index
                self._mode = "embedding_faiss"
            else:
                self._mode = "embedding_bruteforce"

        elif self._mode == "tfidf" and self._tfidf is not None:
            self._tfidf_matrix = self._tfidf.fit_transform(corpus)

        self._token_corpus = [set(text.split()) for text in corpus]
        self._fitted = True
        logger.info("CandidateRetrievalEngine fitted with mode=%s on %d internships", self._mode, len(corpus))

    def retrieve(self, profile: Dict[str, Any]) -> List[Tuple[str, float]]:
        if not self._fitted:
            raise RuntimeError("Call fit() before retrieve().")

        query = self._student_text(profile)
        query_k = min(self.retrieval_k, len(self._ids))

        if self._mode == "embedding_faiss":
            q = self._embedder.encode([query], show_progress_bar=False, normalize_embeddings=True)
            q = np.asarray(q, dtype=np.float32)
            sims, idxs = self._faiss_index.search(q, query_k)
            return [(self._ids[int(i)], float(s)) for s, i in zip(sims[0], idxs[0])]

        if self._mode == "embedding_bruteforce":
            q = self._embedder.encode([query], show_progress_bar=False, normalize_embeddings=True)
            q = np.asarray(q, dtype=np.float32)[0]
            sims = np.dot(self._emb_matrix, q)
            order = np.argsort(-sims)[:query_k]
            return [(self._ids[int(i)], float(sims[int(i)])) for i in order]

        if self._mode == "tfidf" and self._tfidf is not None and self._tfidf_matrix is not None:
            qv = self._tfidf.transform([query])
            sims = cosine_similarity(qv, self._tfidf_matrix)[0]
            order = np.argsort(-sims)[:query_k]
            return [(self._ids[int(i)], float(sims[int(i)])) for i in order]

        qset = set(query.split())
        scores = []
        for i, cset in enumerate(self._token_corpus):
            scores.append((self._ids[i], _jaccard(qset, cset)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:query_k]


# ------------------------ Stage 2: ML Ranking ------------------------------

@dataclass
class RankTrainingArtifacts:
    model_type: str
    feature_importance: Dict[str, float]
    trained: bool


class MLRankingEngine:
    """Trained ranking model with fallbacks."""

    def __init__(self):
        self.model = None
        self.model_type = "weighted_fallback"
        self.feature_importance: Dict[str, float] = dict(WEIGHT_FALLBACK)
        self._trained = False

    def _fit_lightgbm(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                verbose=-1,
            )
            model.fit(X, y)
            self.model = model
            self.model_type = "lightgbm"
            importances = model.feature_importances_.astype(float)
            denom = float(importances.sum()) if float(importances.sum()) > 0 else 1.0
            self.feature_importance = {
                f: float(v / denom) for f, v in zip(FEATURE_NAMES, importances)
            }
            return True
        except Exception:
            return False

    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray) -> bool:
        try:
            model = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
            )
            model.fit(X, y)
            self.model = model
            self.model_type = "xgboost"
            importances = model.feature_importances_.astype(float)
            denom = float(importances.sum()) if float(importances.sum()) > 0 else 1.0
            self.feature_importance = {
                f: float(v / denom) for f, v in zip(FEATURE_NAMES, importances)
            }
            return True
        except Exception:
            return False

    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        try:
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)
            self.model = model
            self.model_type = "random_forest"
            importances = model.feature_importances_.astype(float)
            denom = float(importances.sum()) if float(importances.sum()) > 0 else 1.0
            self.feature_importance = {
                f: float(v / denom) for f, v in zip(FEATURE_NAMES, importances)
            }
            return True
        except Exception:
            return False

    def fit(self, feature_rows: List[Dict[str, Any]]) -> RankTrainingArtifacts:
        if not feature_rows:
            self._trained = False
            return RankTrainingArtifacts(self.model_type, self.feature_importance, False)

        df = pd.DataFrame(feature_rows)
        if "label" not in df.columns:
            self._trained = False
            return RankTrainingArtifacts(self.model_type, self.feature_importance, False)

        X = df[FEATURE_NAMES].fillna(0.0).values.astype(float)
        y = df["label"].fillna(0.0).values.astype(float)

        trained = False
        if LIGHTGBM_AVAILABLE:
            trained = self._fit_lightgbm(X, y)
        if not trained and XGBOOST_AVAILABLE:
            trained = self._fit_xgboost(X, y)
        if not trained:
            trained = self._fit_sklearn(X, y)

        self._trained = trained
        logger.info("MLRankingEngine fit complete. model_type=%s trained=%s", self.model_type, trained)
        return RankTrainingArtifacts(self.model_type, self.feature_importance, trained)

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        X = feature_df[FEATURE_NAMES].fillna(0.0).values.astype(float)
        if self._trained and self.model is not None:
            try:
                pred = self.model.predict(X)
                return np.asarray(pred, dtype=float)
            except Exception:
                pass

        # Weighted fallback if model unavailable
        w = np.array([self.feature_importance.get(f, WEIGHT_FALLBACK[f]) for f in FEATURE_NAMES], dtype=float)
        wsum = float(w.sum()) if float(w.sum()) > 0 else 1.0
        w = w / wsum
        return (X * w).sum(axis=1)


# ---------------------- Stage 3: Policy Reranking --------------------------

class PolicyReranker:
    """
    Constrained reranking for fairness and diversity.

    Constraints used:
      - Quality floor: do not promote very poor matches
      - Max items per domain/company in top-K
      - Fair visibility bonus for Tier-3/Rural students
      - Optional stipend diversity encouragement
    """

    def __init__(
        self,
        quality_floor: float = 0.12,
        max_per_domain: int = 2,
        max_per_company: int = 1,
    ):
        self.quality_floor = quality_floor
        self.max_per_domain = max_per_domain
        self.max_per_company = max_per_company

    @staticmethod
    def _stipend_bucket(stipend: float) -> str:
        if stipend < 8000:
            return "low"
        if stipend < 20000:
            return "mid"
        return "high"

    def rerank(
        self,
        scored_df: pd.DataFrame,
        student_profile: Dict[str, Any],
        top_k: int,
    ) -> pd.DataFrame:
        if scored_df.empty:
            return scored_df

        tier = _norm_text(student_profile.get("college_tier", "Tier-2")) or "Tier-2"
        region = _norm_text(student_profile.get("region", "Urban")) or "Urban"

        fairness_bonus = FAIRNESS_TIER_BONUS.get(tier, 0.0)
        if region == "Rural":
            fairness_bonus += FAIRNESS_RURAL_BONUS

        df = scored_df.copy()
        df["policy_score"] = df["rank_score"].astype(float)
        # Apply fairness bonus only above quality floor
        mask = df["policy_score"] >= self.quality_floor
        df.loc[mask, "policy_score"] = np.minimum(1.0, df.loc[mask, "policy_score"] + fairness_bonus)

        # Diversity-aware constrained selection
        df = df.sort_values("policy_score", ascending=False).reset_index(drop=True)

        selected_idx = []
        domain_count = Counter()
        company_count = Counter()
        stipend_count = Counter()

        for idx, row in df.iterrows():
            if len(selected_idx) >= top_k:
                break

            domain = _norm_text(row.get("Domain", "Unknown")) or "Unknown"
            company = _norm_text(row.get("Company", "Unknown")) or "Unknown"
            stipend_bucket = self._stipend_bucket(_safe_float(row.get("Stipend (INR)", 0), 0.0))

            if domain_count[domain] >= self.max_per_domain:
                continue
            if company_count[company] >= self.max_per_company:
                continue

            # Encourage stipend diversity softly
            if len(selected_idx) >= 3 and stipend_count[stipend_bucket] >= max(1, top_k // 2):
                continue

            selected_idx.append(idx)
            domain_count[domain] += 1
            company_count[company] += 1
            stipend_count[stipend_bucket] += 1

        # Fill remaining slots by policy_score if constraints were strict
        if len(selected_idx) < top_k:
            for idx in range(len(df)):
                if idx not in selected_idx:
                    selected_idx.append(idx)
                    if len(selected_idx) >= top_k:
                        break

        out = df.iloc[selected_idx].copy().reset_index(drop=True)
        out["fairness_bonus"] = fairness_bonus
        return out

    @staticmethod
    def fairness_report(final_df: pd.DataFrame) -> Dict[str, Any]:
        if final_df.empty:
            return {}
        domains = final_df["Domain"].fillna("Unknown").astype(str).tolist()
        companies = final_df["Company"].fillna("Unknown").astype(str).tolist()
        stipend = final_df["Stipend (INR)"].fillna(0).astype(float).tolist()
        buckets = [PolicyReranker._stipend_bucket(s) for s in stipend]
        return {
            "unique_domains": len(set(domains)),
            "unique_companies": len(set(companies)),
            "domain_distribution": dict(Counter(domains)),
            "stipend_bucket_distribution": dict(Counter(buckets)),
        }


# -------------------- Stage 4: Explainability Engine -----------------------

class ExplainabilityEngine:
    """Feature attribution + counterfactual recommendations."""

    def __init__(self, ranker: MLRankingEngine):
        self.ranker = ranker

    def feature_attribution(self, feature_row: Dict[str, float]) -> Dict[str, float]:
        # Default attribution uses learned global importance * local value.
        imp = self.ranker.feature_importance
        raw = {k: imp.get(k, 0.0) * float(feature_row.get(k, 0.0)) for k in FEATURE_NAMES}
        total = sum(max(v, 0.0) for v in raw.values())
        if total <= 0:
            return {k: 0.0 for k in FEATURE_NAMES}
        return {k: round(max(v, 0.0) / total, 4) for k, v in raw.items()}

    def shap_values_if_available(self, feature_df: pd.DataFrame) -> Optional[np.ndarray]:
        if not SHAP_AVAILABLE or self.ranker.model is None or not self.ranker._trained:
            return None
        try:
            explainer = shap.Explainer(self.ranker.model)
            values = explainer(feature_df[FEATURE_NAMES].fillna(0.0)).values
            return values
        except Exception:
            return None

    def build_explanation(
        self,
        student_profile: Dict[str, Any],
        internship_row: pd.Series,
        feature_row: Dict[str, Any],
        rank_score: float,
    ) -> Dict[str, Any]:
        student_skills = [s.lower() for s in _split_csv_like(_norm_text(student_profile.get("skills", "")))]
        intern_skills = _split_csv_like(_norm_text(internship_row.get("Required Skills", "")))

        matched = [s for s in intern_skills if s.lower() in set(student_skills)]
        missing = [s for s in intern_skills if s.lower() not in set(student_skills)]
        match_pct = int((len(matched) / len(intern_skills)) * 100) if intern_skills else 0

        reasons: List[str] = []
        if match_pct >= 80:
            reasons.append(f"Strong skill match ({match_pct}%).")
        elif match_pct >= 50:
            reasons.append(f"Good skill alignment ({match_pct}%).")
        elif match_pct > 0:
            reasons.append(f"Partial skill overlap ({match_pct}%).")
        else:
            reasons.append("Low direct skill overlap; selected via broader profile fit.")

        if float(feature_row.get("domain_interest", 0.0)) >= 0.7:
            reasons.append(f"Domain aligns with your interests ({_norm_text(internship_row.get('Domain', 'N/A'))}).")

        if float(feature_row.get("location_fit", 0.0)) >= 0.9:
            reasons.append("Location/work mode is strongly compatible with your preference.")

        stipend = int(_safe_float(internship_row.get("Stipend (INR)", 0), 0.0))
        if stipend >= 20000:
            reasons.append(f"High stipend opportunity (INR {stipend:,}/month).")

        attribution = self.feature_attribution({k: float(feature_row.get(k, 0.0)) for k in FEATURE_NAMES})
        top_contrib = sorted(attribution.items(), key=lambda x: x[1], reverse=True)[:3]

        actions: List[str] = []
        if missing:
            actions.append(f"Learn one priority missing skill: {missing[0]}.")
        if float(feature_row.get("domain_interest", 0.0)) < 0.5:
            actions.append("Add domain-specific projects in your resume/profile.")
        if float(feature_row.get("location_fit", 0.0)) < 0.5:
            actions.append("Enable remote preference or add more location flexibility.")
        if float(student_profile.get("cgpa", 0.0)) < 7.0:
            actions.append("Improve CGPA or add stronger project proof to offset academics.")

        counterfactual = {
            "if_add_top_missing_skill": round(min(1.0, rank_score + 0.05), 4),
            "if_domain_project_added": round(min(1.0, rank_score + 0.04), 4),
            "if_location_flexible": round(min(1.0, rank_score + 0.03), 4),
        }

        return {
            "match_percentage": match_pct,
            "matched_skills": matched,
            "missing_skills": missing,
            "reasons": reasons,
            "top_feature_contributions": [{"feature": k, "contribution": v} for k, v in top_contrib],
            "improvement_actions": actions,
            "counterfactual_scores": counterfactual,
        }


# -------------------------- Evaluation -------------------------------------

class EvaluationEngine:
    @staticmethod
    def precision_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if k <= 0:
            return 0.0
        rec = recommended_ids[:k]
        return len(set(rec) & set(relevant_ids)) / k

    @staticmethod
    def recall_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        if not relevant_ids:
            return 0.0
        rec = recommended_ids[:k]
        return len(set(rec) & set(relevant_ids)) / len(set(relevant_ids))

    @staticmethod
    def ndcg_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        rel = set(relevant_ids)
        dcg = sum((1 / math.log2(i + 2)) for i, rid in enumerate(recommended_ids[:k]) if rid in rel)
        ideal_hits = min(k, len(rel))
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg > 0 else 0.0


# ------------------- Master Heavy-ML Recommendation Engine -----------------

class InternshipRecommendationEngine:
    def __init__(self):
        self.preprocessor = MLPreprocessor() if PREPROCESSOR_AVAILABLE else None

        self.retrieval = CandidateRetrievalEngine(retrieval_k=350)
        self.ranker = MLRankingEngine()
        self.collaborative = CollaborativeSignalEngine()
        self.policy = PolicyReranker(quality_floor=0.12, max_per_domain=2, max_per_company=1)
        self.explainer = ExplainabilityEngine(self.ranker)
        self.evaluator = EvaluationEngine()

        self.students_df: Optional[pd.DataFrame] = None
        self.internships_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None

        self.company_popularity: Dict[str, float] = {}
        self._intern_lookup: Dict[str, pd.Series] = {}
        self._fitted = False
        self._default_objective_weights = {
            "learning": 0.40,
            "career_fit": 0.40,
            "compensation": 0.20,
        }

    # ---------------- Data ----------------
    @staticmethod
    def _validate_required_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{table_name} missing required columns: {missing}")

    def _raw_load(self, students_path: str, internships_path: str) -> None:
        self.students_df = pd.read_csv(students_path)
        self.internships_df = pd.read_csv(internships_path)
        for col in ["Required Skills", "required_skills", "skills"]:
            if col in self.internships_df.columns:
                self.internships_df.rename(columns={col: "Required Skills"}, inplace=True)
                break

    def load_data(
        self,
        students_path: str,
        internships_path: str,
        interactions_path: Optional[str] = None,
    ) -> None:
        logger.info("Loading data...")

        if self.preprocessor is not None:
            try:
                s_df, i_df = self.preprocessor.load_and_clean(students_path, internships_path)
                self.students_df, self.internships_df = s_df, i_df
                self.interactions_df = self.preprocessor.generate_labelled_dataset(s_df, i_df)
            except Exception as ex:
                logger.warning("Preprocessor failed (%s). Falling back to raw load.", ex)
                self._raw_load(students_path, internships_path)
        else:
            self._raw_load(students_path, internships_path)

        if interactions_path and os.path.exists(interactions_path):
            self.interactions_df = pd.read_csv(interactions_path)

        if self.internships_df is None or self.students_df is None:
            raise RuntimeError("Failed to load datasets.")

        self._validate_required_columns(
            self.students_df,
            ["Student ID", "Skills", "Interests", "Branch", "CGPA", "Location", "College Tier", "Region"],
            "students_df",
        )
        self._validate_required_columns(
            self.internships_df,
            ["Internship ID", "Title", "Company", "Domain", "Location", "Work Type", "Stipend (INR)", "Required Skills"],
            "internships_df",
        )

        # Ensure string IDs
        self.students_df["Student ID"] = self.students_df["Student ID"].astype(str)
        self.internships_df["Internship ID"] = self.internships_df["Internship ID"].astype(str)

        # Build lookups
        self._intern_lookup = {
            row["Internship ID"]: row for _, row in self.internships_df.iterrows()
        }

        # Company popularity (from interactions if present, else frequency fallback)
        if self.interactions_df is not None and not self.interactions_df.empty:
            if "internship_id" in self.interactions_df.columns:
                pop = self.interactions_df["internship_id"].astype(str).value_counts()
                maxv = float(pop.max()) if len(pop) else 1.0
                per_intern = {k: float(v / maxv) for k, v in pop.items()}
                comp_vals = defaultdict(list)
                for iid, pv in per_intern.items():
                    row = self._intern_lookup.get(iid)
                    if row is not None:
                        comp_vals[_norm_text(row.get("Company", "Unknown"))].append(pv)
                self.company_popularity = {
                    c: float(np.mean(vals)) for c, vals in comp_vals.items()
                }
            else:
                self.company_popularity = {}
        else:
            comp_count = self.internships_df["Company"].fillna("Unknown").astype(str).value_counts()
            m = float(comp_count.max()) if len(comp_count) else 1.0
            self.company_popularity = {c: float(v / m) for c, v in comp_count.items()}

        logger.info(
            "Data loaded. Students=%d Internships=%d Interactions=%s",
            len(self.students_df), len(self.internships_df),
            0 if self.interactions_df is None else len(self.interactions_df),
        )

    # -------------- Feature Engineering --------------
    def _weighted_skill_overlap(self, student_skills: set, row: pd.Series) -> float:
        req = _split_csv_like(_norm_text(row.get("Required Skills", "")))
        if not req:
            return 0.0

        if self.preprocessor is None:
            return _jaccard({s.lower() for s in req}, {s.lower() for s in student_skills})

        domain = _norm_text(row.get("Domain", ""))
        total = 0.0
        hit = 0.0
        for sk in req:
            w = 1.0
            try:
                cluster = self.preprocessor._get_primary_skill_cluster(req)
                w = self.preprocessor._get_skill_weight_with_fallback(sk, domain, cluster)
            except Exception:
                w = 1.0
            total += w
            if sk.lower() in {x.lower() for x in student_skills}:
                hit += w
        return (hit / total) if total > 0 else 0.0

    def _domain_weight_score(self, student_skills: set, row: pd.Series) -> float:
        """Score based on domain_skill_weights coverage when preprocessor exposes them."""
        req = _split_csv_like(_norm_text(row.get("Required Skills", "")))
        if not req:
            return 0.0

        if self.preprocessor is None or not hasattr(self.preprocessor, "domain_skill_weights"):
            return self._weighted_skill_overlap(student_skills, row)

        domain = _norm_text(row.get("Domain", ""))
        dsw = getattr(self.preprocessor, "domain_skill_weights", {}) or {}
        domain_weights = dsw.get(domain, {}) if isinstance(dsw, dict) else {}
        if not isinstance(domain_weights, dict) or not domain_weights:
            return self._weighted_skill_overlap(student_skills, row)

        req_lower = [s.lower() for s in req]
        stu_lower = {s.lower() for s in student_skills}
        vals = [float(domain_weights.get(s, domain_weights.get(s.title(), 1.0))) for s in req]
        max_w = max(vals) if vals else 1.0
        if max_w <= 0:
            max_w = 1.0

        total = 0.0
        hit = 0.0
        for orig, s in zip(req, req_lower):
            w = float(domain_weights.get(orig, domain_weights.get(orig.title(), domain_weights.get(s, 1.0))))
            total += w
            if s in stu_lower:
                hit += w
        return (hit / total) if total > 0 else 0.0

    def _domain_interest_score(self, interests: str, domain: str) -> float:
        if not interests or not domain:
            return 0.0
        if self.preprocessor is not None:
            try:
                return float(self.preprocessor._get_domain_similarity_score(interests, domain))
            except Exception:
                pass
        i = interests.lower()
        d = domain.lower()
        return 0.8 if any(w in i for w in d.split() if len(w) > 3) else 0.2

    def _location_fit(self, student_location: str, intern_location: str, work_type: str) -> float:
        sl = student_location.lower()
        il = intern_location.lower()
        wt = work_type.lower()
        if "remote" in wt or il == "remote":
            return 1.0
        if sl and il and sl == il:
            return 1.0
        if self.preprocessor is not None:
            try:
                return float(self.preprocessor._get_location_match_level(student_location, intern_location))
            except Exception:
                pass
        return 0.35

    def _cgpa_fit(self, cgpa: float) -> float:
        if cgpa >= CGPA_EXCELLENT:
            return 1.0
        if cgpa >= CGPA_GOOD:
            return 0.75
        if cgpa >= 6.0:
            return 0.5
        return 0.3

    def _work_mode_fit(self, preferred_mode: str, work_type: str) -> float:
        if not preferred_mode:
            return 0.6
        pm = preferred_mode.lower()
        wt = work_type.lower()
        if pm in wt:
            return 1.0
        if "hybrid" in wt:
            return 0.8
        return 0.4

    def _build_feature_row(
        self,
        student_profile: Dict[str, Any],
        intern_row: pd.Series,
        retrieval_score: float,
        collaborative_score: float,
    ) -> Dict[str, Any]:
        student_skills = set(_split_csv_like(_norm_text(student_profile.get("skills", ""))))
        intern_skills = set(_split_csv_like(_norm_text(_row_value(intern_row, "Required Skills", ""))))

        overlap = _jaccard({s.lower() for s in student_skills}, {s.lower() for s in intern_skills})
        weighted_overlap = self._weighted_skill_overlap(student_skills, intern_row)
        domain_weight_score = self._domain_weight_score(student_skills, intern_row)
        domain_interest = self._domain_interest_score(
            _norm_text(student_profile.get("interests", "")),
            _norm_text(_row_value(intern_row, "Domain", "")),
        )
        cgpa_fit = self._cgpa_fit(_safe_float(student_profile.get("cgpa", 7.0), 7.0))
        location_fit = self._location_fit(
            _norm_text(student_profile.get("location", "")),
            _norm_text(_row_value(intern_row, "Location", "")),
            _norm_text(_row_value(intern_row, "Work Type", "")),
        )

        max_stipend = float(self.internships_df["Stipend (INR)"].fillna(0).max())
        if max_stipend <= 0:
            max_stipend = 25000.0
        stipend_norm = _safe_float(_row_value(intern_row, "Stipend (INR)", 0), 0.0) / max_stipend

        work_mode_fit = self._work_mode_fit(
            _norm_text(student_profile.get("preferred_work_mode", "")),
            _norm_text(_row_value(intern_row, "Work Type", "")),
        )

        company = _norm_text(_row_value(intern_row, "Company", "Unknown"))
        company_pop = float(self.company_popularity.get(company, 0.5))

        row = {
            "retrieval_score": float(max(0.0, retrieval_score)),
            "skill_overlap": float(max(0.0, min(1.0, overlap))),
            "weighted_skill_overlap": float(max(0.0, min(1.0, weighted_overlap))),
            "domain_weight_score": float(max(0.0, min(1.0, domain_weight_score))),
            "domain_interest": float(max(0.0, min(1.0, domain_interest))),
            "cgpa_fit": float(max(0.0, min(1.0, cgpa_fit))),
            "location_fit": float(max(0.0, min(1.0, location_fit))),
            "stipend_norm": float(max(0.0, min(1.0, stipend_norm))),
            "work_mode_fit": float(max(0.0, min(1.0, work_mode_fit))),
            "company_popularity": float(max(0.0, min(1.0, company_pop))),
            "collaborative_score": float(max(0.0, min(1.0, collaborative_score))),
        }
        return row

    def _objective_weights(self, profile: Dict[str, Any]) -> Dict[str, float]:
        # User can pass objective priorities explicitly; else use defaults.
        w = {
            "learning": _safe_float(profile.get("objective_learning", self._default_objective_weights["learning"]), self._default_objective_weights["learning"]),
            "career_fit": _safe_float(profile.get("objective_career_fit", self._default_objective_weights["career_fit"]), self._default_objective_weights["career_fit"]),
            "compensation": _safe_float(profile.get("objective_compensation", self._default_objective_weights["compensation"]), self._default_objective_weights["compensation"]),
        }
        s = sum(max(v, 0.0) for v in w.values())
        if s <= 0:
            return dict(self._default_objective_weights)
        return {k: max(v, 0.0) / s for k, v in w.items()}

    def _multi_objective_score(self, row: pd.Series, profile: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        learning = float(0.50 * row["domain_interest"] + 0.35 * row["domain_weight_score"] + 0.15 * row["weighted_skill_overlap"])
        career_fit = float(0.40 * row["skill_overlap"] + 0.30 * row["location_fit"] + 0.20 * row["work_mode_fit"] + 0.10 * row["collaborative_score"])
        compensation = float(0.75 * row["stipend_norm"] + 0.25 * row["company_popularity"])
        w = self._objective_weights(profile)
        total = w["learning"] * learning + w["career_fit"] * career_fit + w["compensation"] * compensation
        details = {
            "learning_score": round(learning, 4),
            "career_fit_score": round(career_fit, 4),
            "compensation_score": round(compensation, 4),
            "objective_composite": round(float(total), 4),
        }
        return float(total), details

    @staticmethod
    def _confidence_score(row: pd.Series) -> float:
        # Confidence is higher with stronger retrieval, stronger agreement across signals, and less sparsity.
        vals = [
            float(row["retrieval_score"]),
            float(row["skill_overlap"]),
            float(row["weighted_skill_overlap"]),
            float(row["domain_interest"]),
            float(row["location_fit"]),
            float(row["collaborative_score"]),
        ]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        agreement = max(0.0, 1.0 - std_v)
        conf = 0.65 * mean_v + 0.35 * agreement
        return float(max(0.0, min(1.0, conf)))

    # -------------- Train ranker --------------
    def _build_training_features(self) -> List[Dict[str, Any]]:
        if self.interactions_df is None or self.interactions_df.empty:
            return []

        cols = set(self.interactions_df.columns)
        required = {"student_id", "internship_id"}
        if not required.issubset(cols):
            return []

        out = []
        student_map = {row["Student ID"]: row for _, row in self.students_df.iterrows()}

        # Label definition prioritizes richer outcomes when available.
        def row_label(r: pd.Series) -> float:
            y = 0.0
            if "applied" in r.index:
                y += 0.35 * _safe_float(r.get("applied", 0), 0.0)
            if "interview" in r.index:
                y += 0.20 * _safe_float(r.get("interview", 0), 0.0)
            if "got_offer" in r.index:
                y += 0.30 * _safe_float(r.get("got_offer", 0), 0.0)
            if "accepted" in r.index:
                y += 0.15 * _safe_float(r.get("accepted", 0), 0.0)
            # Fallback if only applied exists
            if y == 0.0 and "applied" in r.index:
                y = _safe_float(r.get("applied", 0), 0.0)
            return float(max(0.0, min(1.0, y)))

        # Build features from historical pairs
        interactions_view = self.interactions_df
        max_training_rows = 200000
        if len(interactions_view) > max_training_rows:
            logger.warning(
                "Interactions are large (%d). Sampling %d rows for training speed.",
                len(interactions_view),
                max_training_rows,
            )
            interactions_view = interactions_view.sample(n=max_training_rows, random_state=42)

        for _, ir in interactions_view.iterrows():
            sid = str(ir.get("student_id"))
            iid = str(ir.get("internship_id"))
            srow = student_map.get(sid)
            irow = self._intern_lookup.get(iid)
            if srow is None or irow is None:
                continue

            profile = {
                "skills": _norm_text(srow.get("Skills", "")),
                "interests": _norm_text(srow.get("Interests", "")),
                "branch": _norm_text(srow.get("Branch", "")),
                "cgpa": _safe_float(srow.get("CGPA", 7.0), 7.0),
                "location": _norm_text(srow.get("Location", "")),
                "college_tier": _norm_text(srow.get("College Tier", "Tier-2")),
                "region": _norm_text(srow.get("Region", "Urban")),
                "preferred_work_mode": "",
            }

            # Retrieval score surrogate in training
            retrieval_score = self._domain_interest_score(profile["interests"], _norm_text(_row_value(irow, "Domain", "")))
            collab = self.collaborative.score_items(sid, [iid]).get(iid, 0.0)
            feats = self._build_feature_row(profile, irow, retrieval_score, collab)
            feats["label"] = row_label(ir)
            out.append(feats)

        return out

    def fit(self) -> None:
        if self.students_df is None or self.internships_df is None:
            raise RuntimeError("Call load_data() first")

        logger.info("Stage 1/3: fitting candidate retrieval...")
        self.retrieval.fit(self.internships_df)

        logger.info("Stage 2/3: fitting collaborative signal...")
        self.collaborative.fit(self.interactions_df)

        logger.info("Stage 3/3: training ranker...")
        rows = self._build_training_features()
        artifacts = self.ranker.fit(rows)
        logger.info("Ranker training: model=%s trained=%s", artifacts.model_type, artifacts.trained)

        self._fitted = True
        logger.info("Engine ready.")

    # -------------- Recommend --------------
    def recommend(
        self,
        student_profile: Dict[str, Any],
        top_k: int = 10,
        student_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self._fitted:
            raise RuntimeError("Call fit() before recommend().")

        # Stage 1: retrieval
        retrieved = self.retrieval.retrieve(student_profile)
        if not retrieved:
            return []

        # Build candidate feature table
        cand_ids = [iid for iid, _ in retrieved]
        collab_scores = self.collaborative.score_items(student_id, cand_ids)
        cand_rows = []
        for iid, r_score in retrieved:
            irow = self._intern_lookup.get(str(iid))
            if irow is None:
                continue
            feats = self._build_feature_row(
                student_profile,
                irow,
                retrieval_score=r_score,
                collaborative_score=collab_scores.get(str(iid), 0.0),
            )
            row = {
                "Internship ID": str(iid),
                "Title": _norm_text(_row_value(irow, "Title", "N/A")),
                "Company": _norm_text(_row_value(irow, "Company", "N/A")),
                "Domain": _norm_text(_row_value(irow, "Domain", "N/A")),
                "Location": _norm_text(_row_value(irow, "Location", "N/A")),
                "Work Type": _norm_text(_row_value(irow, "Work Type", "N/A")),
                "Stipend (INR)": _safe_float(_row_value(irow, "Stipend (INR)", 0), 0.0),
                "Duration": _safe_float(_row_value(irow, "Duration", 0), 0.0),
                "Required Skills": _norm_text(_row_value(irow, "Required Skills", "")),
                "Description": _norm_text(_row_value(irow, "Description", "")),
            }
            row.update(feats)
            cand_rows.append(row)

        if not cand_rows:
            return []

        cand_df = pd.DataFrame(cand_rows)

        # Stage 2: ranking score
        cand_df["rank_score"] = self.ranker.predict(cand_df)
        # Normalize for stability
        mn, mx = float(cand_df["rank_score"].min()), float(cand_df["rank_score"].max())
        if mx > mn:
            cand_df["rank_score"] = (cand_df["rank_score"] - mn) / (mx - mn)

        # Multi-objective optimization layer (salary vs learning vs career fit)
        objective_scores = cand_df.apply(lambda r: self._multi_objective_score(r, student_profile), axis=1)
        cand_df["objective_score"] = [x[0] for x in objective_scores]
        cand_df["objective_details"] = [x[1] for x in objective_scores]
        # Blend learned rank with personalized objective utility.
        cand_df["rank_score"] = 0.75 * cand_df["rank_score"] + 0.25 * cand_df["objective_score"]

        # Stage 3: policy reranking
        final_df = self.policy.rerank(cand_df, student_profile, top_k=top_k)

        # Stage 4: explanation
        results = []
        for idx, row in final_df.iterrows():
            f_row = {k: float(row[k]) for k in FEATURE_NAMES}
            irow = self._intern_lookup[str(row["Internship ID"])]
            explanation = self.explainer.build_explanation(
                student_profile=student_profile,
                internship_row=irow,
                feature_row=f_row,
                rank_score=float(row["policy_score"]),
            )

            result = {
                "rank": int(idx + 1),
                "internship_id": str(row["Internship ID"]),
                "title": str(row["Title"]),
                "company": str(row["Company"]),
                "domain": str(row["Domain"]),
                "location": str(row["Location"]),
                "work_type": str(row["Work Type"]),
                "stipend": int(round(float(row["Stipend (INR)"]))),
                "duration_weeks": int(round(float(row["Duration"]))),
                "required_skills": str(row["Required Skills"]),
                "description": str(row["Description"]),
                "scores": {
                    "retrieval_score": round(float(row["retrieval_score"]), 4),
                    "rank_score": round(float(row["rank_score"]), 4),
                    "objective_score": round(float(row.get("objective_score", 0.0)), 4),
                    "policy_score": round(float(row["policy_score"]), 4),
                    "fairness_bonus": round(float(row.get("fairness_bonus", 0.0)), 4),
                    "confidence": round(self._confidence_score(row), 4),
                },
                "feature_values": {k: round(float(row[k]), 4) for k in FEATURE_NAMES},
                "objective_breakdown": row.get("objective_details", {}),
                "explanation": explanation,
            }
            results.append(result)

        return results

    # -------------- Evaluation --------------
    def evaluate_offline(self, k: int = 5, sample_size: int = 50) -> Dict[str, Any]:
        if self.interactions_df is None or self.interactions_df.empty:
            return {"error": "No interactions available for evaluation."}

        if "applied" not in self.interactions_df.columns:
            return {"error": "Interactions missing 'applied' label."}

        positive = self.interactions_df[self.interactions_df["applied"] == 1]
        relevant_map = positive.groupby("student_id")["internship_id"].apply(list).to_dict()

        student_ids = list(relevant_map.keys())[:sample_size]
        p_vals, r_vals, n_vals = [], [], []

        for sid in student_ids:
            rows = self.students_df[self.students_df["Student ID"] == str(sid)]
            if rows.empty:
                continue
            s = rows.iloc[0]
            profile = {
                "skills": _norm_text(s.get("Skills", "")),
                "interests": _norm_text(s.get("Interests", "")),
                "branch": _norm_text(s.get("Branch", "")),
                "cgpa": _safe_float(s.get("CGPA", 7.0), 7.0),
                "location": _norm_text(s.get("Location", "")),
                "college_tier": _norm_text(s.get("College Tier", "Tier-2")),
                "region": _norm_text(s.get("Region", "Urban")),
                "preferred_work_mode": "",
            }
            recs = self.recommend(profile, top_k=k, student_id=str(sid))
            rec_ids = [r["internship_id"] for r in recs]
            rel = [str(x) for x in relevant_map.get(sid, [])]

            p_vals.append(self.evaluator.precision_at_k(rec_ids, rel, k))
            r_vals.append(self.evaluator.recall_at_k(rec_ids, rel, k))
            n_vals.append(self.evaluator.ndcg_at_k(rec_ids, rel, k))

        return {
            f"Precision@{k}": float(np.mean(p_vals)) if p_vals else 0.0,
            f"Recall@{k}": float(np.mean(r_vals)) if r_vals else 0.0,
            f"NDCG@{k}": float(np.mean(n_vals)) if n_vals else 0.0,
            "students_evaluated": len(p_vals),
        }

# ---------------------------- Input Utilities -------------------------------

TIER_OPTIONS = ["Tier-1", "Tier-2", "Tier-3"]
REGION_OPTIONS = ["Urban", "Rural"]
BRANCH_OPTIONS = [
    "Computer Science",
    "Information Technology",
    "Data Science",
    "Artificial Intelligence",
    "Electronics & Communication",
    "Electrical Engineering",
    "Mechanical Engineering",
    "Civil Engineering",
    "Chemical Engineering",
    "Biotechnology",
    "Other",
]
LOCATION_OPTIONS = [
    "Bangalore",
    "Mumbai",
    "Delhi",
    "Hyderabad",
    "Chennai",
    "Pune",
    "Noida",
    "Gurgaon",
    "Kolkata",
    "Remote",
    "Other",
]
WORK_MODE_OPTIONS = ["Remote", "Hybrid", "On-site", "No Preference"]


def _prompt_choice(prompt: str, options: List[str], default_idx: int = 0) -> str:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")

    while True:
        raw = input(f"Enter number (default {default_idx + 1}): ").strip()
        if raw == "":
            return options[default_idx]
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("Invalid choice. Please pick a valid number.")


def _prompt_float(prompt: str, min_val: float, max_val: float, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{min_val}-{max_val}] (default {default}): ").strip()
        if raw == "":
            return default
        try:
            v = float(raw)
            if min_val <= v <= max_val:
                return v
        except ValueError:
            pass
        print("Invalid number. Try again.")


def _prompt_int(prompt: str, min_val: int, max_val: int, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{min_val}-{max_val}] (default {default}): ").strip()
        if raw == "":
            return default
        try:
            v = int(raw)
            if min_val <= v <= max_val:
                return v
        except ValueError:
            pass
        print("Invalid integer. Try again.")


def _prompt_skills(prompt: str, default: str = "") -> str:
    print(f"\n{prompt}")
    print("Comma-separated (example: Python, SQL, Machine Learning)")
    raw = input("-> ").strip()
    if not raw:
        return default
    parts = [x.strip().title() for x in raw.split(",") if x.strip()]
    return ", ".join(parts)


def collect_student_profile() -> Dict[str, Any]:
    print("\n" + "=" * 84)
    print("STUDENT PROFILE INPUT")
    print("Press Enter to accept defaults")
    print("=" * 84)

    student_id = input("Student ID (optional, for tracking): ").strip()
    branch = _prompt_choice("Select Branch:", BRANCH_OPTIONS, default_idx=0)
    cgpa = _prompt_float("Enter CGPA", 0.0, 10.0, 7.5)
    skills = _prompt_skills("Enter technical skills", default="Python, SQL")
    interests = _prompt_skills("Enter interests/domains", default="Data Science")
    location = _prompt_choice("Preferred location:", LOCATION_OPTIONS, default_idx=0)
    tier = _prompt_choice("College tier:", TIER_OPTIONS, default_idx=1)
    region = _prompt_choice("Region:", REGION_OPTIONS, default_idx=0)
    pref_mode = _prompt_choice("Preferred work mode:", WORK_MODE_OPTIONS, default_idx=3)
    print("\nObjective priorities (0 to 1). Higher means more important.")
    w_learning = _prompt_float("Learning priority", 0.0, 1.0, 0.4)
    w_career = _prompt_float("Career-fit priority", 0.0, 1.0, 0.4)
    w_comp = _prompt_float("Compensation priority", 0.0, 1.0, 0.2)

    return {
        "student_id": student_id or None,
        "branch": branch,
        "cgpa": cgpa,
        "skills": skills,
        "interests": interests,
        "location": location,
        "college_tier": tier,
        "region": region,
        "preferred_work_mode": pref_mode,
        "objective_learning": w_learning,
        "objective_career_fit": w_career,
        "objective_compensation": w_comp,
    }


# ---------------------------- CLI Display -----------------------------------

def display_results(results: List[Dict[str, Any]], student_profile: Dict[str, Any], fairness_report: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("HEAVY-ML RECOMMENDATION RESULTS")
    print("=" * 100)

    print("\nStudent Summary")
    print(f"  Branch        : {student_profile.get('branch', 'N/A')}")
    print(f"  CGPA          : {student_profile.get('cgpa', 'N/A')}")
    print(f"  Skills        : {student_profile.get('skills', 'N/A')}")
    print(f"  Interests     : {student_profile.get('interests', 'N/A')}")
    print(f"  Location      : {student_profile.get('location', 'N/A')}")
    print(f"  College Tier  : {student_profile.get('college_tier', 'N/A')}")
    print(f"  Region        : {student_profile.get('region', 'N/A')}")
    print(f"  Work Mode     : {student_profile.get('preferred_work_mode', 'N/A')}")

    print("\nTop Recommendations")
    print("-" * 100)

    for rec in results:
        match_pct = rec["explanation"].get("match_percentage", 0)
        filled = int(match_pct / 5)
        bar = "#" * filled + "." * (20 - filled)

        print(f"\n#{rec['rank']}  {rec['title']} @ {rec['company']}")
        print(f"  Domain/Type   : {rec['domain']} | {rec['work_type']}")
        print(f"  Location      : {rec['location']}")
        print(f"  Stipend       : INR {rec['stipend']:,}/month")
        print(f"  Duration      : {rec['duration_weeks']} weeks")
        print(f"  Score         : {rec['scores']['policy_score']:.4f}  [{bar}] {match_pct}%")
        print(f"  Confidence    : {rec['scores']['confidence']:.4f}")
        obj = rec.get("objective_breakdown", {})
        if obj:
            print(
                "  Objectives    : "
                f"learning={obj.get('learning_score', 0):.3f}, "
                f"career_fit={obj.get('career_fit_score', 0):.3f}, "
                f"compensation={obj.get('compensation_score', 0):.3f}"
            )

        print("  Why this match:")
        for reason in rec["explanation"].get("reasons", [])[:3]:
            print(f"    - {reason}")

        print("  Top contributions:")
        for c in rec["explanation"].get("top_feature_contributions", [])[:3]:
            print(f"    - {c['feature']}: {c['contribution']:.3f}")

        print("  Action plan:")
        acts = rec["explanation"].get("improvement_actions", [])
        if not acts:
            print("    - Profile already strong for this role")
        else:
            for a in acts[:2]:
                print(f"    - {a}")

    print("\nFairness/Diversity Report")
    print(f"  Unique domains            : {fairness_report.get('unique_domains', 0)}")
    print(f"  Unique companies          : {fairness_report.get('unique_companies', 0)}")
    print(f"  Domain distribution       : {fairness_report.get('domain_distribution', {})}")
    print(f"  Stipend bucket distribution: {fairness_report.get('stipend_bucket_distribution', {})}")
    print("=" * 100 + "\n")


# ---------------------------- Main ------------------------------------------

def main() -> None:
    print("\n" + "=" * 100)
    print("AI Internship Recommendation Engine - Heavy ML Architecture")
    print("Stages: Retrieval -> ML Ranking -> Policy Reranking -> Explainability -> Continuous Learning")
    print("=" * 100)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    def _find(filename: str) -> Optional[str]:
        candidates = [
            os.path.join(base_dir, filename),
            os.path.join(base_dir, "src", filename),
            os.path.join(base_dir, "data", filename),
            filename,
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    students_path = _find("students_uncleaned_new_v2.csv") or _find("students_cleaned.csv")
    internships_path = _find("internships_uncleaned_new_v2.csv") or _find("internships_cleaned.csv")

    if not students_path or not internships_path:
        print("Could not locate student/internship CSV files.")
        sys.exit(1)

    engine = InternshipRecommendationEngine()
    print("\nLoading datasets...")
    engine.load_data(students_path, internships_path)

    print("Training retrieval + ranking modules...")
    engine.fit()
    print("System ready.\n")

    print("Enter the student details below to get recommendations.\n")
    profile = prepare_profile_for_inference(engine, collect_student_profile())
    top_k = _prompt_int("How many recommendations", 1, 20, 8)
    recs = engine.recommend(
        profile,
        top_k=top_k,
        student_id=profile.get("student_id"),
    )
    fair = engine.policy.fairness_report(pd.DataFrame([
        {
            "Domain": r["domain"],
            "Company": r["company"],
            "Stipend (INR)": r["stipend"],
        }
        for r in recs
    ]))
    display_results(recs, profile, fair)


if __name__ == "__main__":
    main()
