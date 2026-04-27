# AI Internship Recommendation Engine

A FastAPI and static frontend project for recommending internships to students using retrieval, ML ranking, fairness-aware reranking, and explainable scoring.

## Project Structure

```text
backend/      FastAPI app and service layer
frontend/     Browser UI served by FastAPI or opened separately
src/          Preprocessing and recommendation engine code
data/         Raw input datasets
docs/         Project documentation PDFs
```

## Requirements

- Python 3.10 or newer
- pip
- Git

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run The App

Start the backend and frontend together:

```powershell
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/api/health
```

## Preprocess Data

To regenerate the cleaned datasets and labelled interaction file:

```powershell
python src/preprocessing.py
```

## Command Line Recommendation Engine

You can also run the recommendation engine from the terminal:

```powershell
python src/recommendation_engine.py
```

## GitHub Upload

Initialize Git and commit:

```powershell
git init
git add .
git commit -m "Initial project setup"
```

Connect a GitHub repository and push:

```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

