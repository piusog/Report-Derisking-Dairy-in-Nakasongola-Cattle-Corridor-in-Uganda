from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np

from schemas import InvestmentRequest, InvestmentResponse

app = FastAPI(title="Nakasongola Dairy Derisking API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "artifacts/best_model.pkl"
FEATURES_PATH = "artifacts/feature_names.pkl"

model = None
feature_names = None
model_name = None

@app.on_event("startup")
def load_artifacts():
    global model, feature_names, model_name
    print("[API] Loading model and feature names...")
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    model_name = model.__class__.__name__
    print(f"[API] Loaded model: {model_name}")
    print(f"[API] Feature order: {feature_names}")

@app.post("/predict", response_model=InvestmentResponse)
def predict_investment(request: InvestmentRequest):
    print("[API] Received request:", request.dict())

    x_dict = {
        "temp_c_mean": request.temp_c_mean,
        "rain_mm_total": request.rain_mm_total,
        "ndvi": request.ndvi,
        "milk_liters": request.milk_liters,
        "feed_cost_ugx": request.feed_cost_ugx,
        "vet_visits": request.vet_visits,
        "loan_access": request.loan_access,
        "outages_days": request.outages_days,
        "market_price_ugx_per_liter": request.market_price_ugx_per_liter,
    }
    X = pd.DataFrame([x_dict])[feature_names]

    proba = float(model.predict_proba(X)[:, 1][0])
    label = int(proba >= 0.5)

    # Simple feature importance ranking
    top_features = []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:3]]
    elif hasattr(model, "coef_"):
        coef = model.coef_[0]
        sorted_idx = np.argsort(np.abs(coef))[::-1]
        top_features = [feature_names[i] for i in sorted_idx[:3]]

    msg = "Investment appears feasible." if label == 1 else "Investment appears risky."

    print(f"[API] Prediction → label={label}, proba={proba:.3f}, top_features={top_features}")

    return InvestmentResponse(
        investment_feasible=label,
        probability=proba,
        model_used=model_name,
        top_features=top_features,
        message=msg,
    )
