from pydantic import BaseModel, Field, confloat, conint
from typing import List

class InvestmentRequest(BaseModel):
    temp_c_mean: float
    rain_mm_total: confloat(ge=0)
    ndvi: confloat(ge=0, le=1)
    milk_liters: confloat(ge=0)
    feed_cost_ugx: confloat(ge=0)
    vet_visits: conint(ge=0)
    loan_access: conint(ge=0, le=1)
    outages_days: conint(ge=0)
    market_price_ugx_per_liter: confloat(ge=0)

class InvestmentResponse(BaseModel):
    investment_feasible: int
    probability: float
    model_used: str
    top_features: List[str]
    message: str
