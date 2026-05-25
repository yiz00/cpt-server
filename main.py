from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.stats import norm

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model  = joblib.load("cpt_kmeans_model.pkl")
scaler = joblib.load("cpt_robust_scaler.pkl")

class CPTInput(BaseModel):
    omissions: float
    commissions: float
    hit_rt: float
    varse: float

@app.post("/predict")
def predict(data: CPTInput):
    TOTAL_TARGETS    = 324
    TOTAL_NONTARGETS = 36

    hr  = np.clip((TOTAL_TARGETS - data.omissions) / TOTAL_TARGETS, 0.005, 0.995)
    far = np.clip(data.commissions / TOTAL_NONTARGETS, 0.005, 0.995)
    z_hr  = norm.ppf(hr)
    z_far = norm.ppf(far)
    d_prime = z_hr - z_far
    beta    = np.exp(0.5 * (z_far**2 - z_hr**2))

    # 학습 피처 순서: Omissions, Commissions, HitRT, RT_SD, d_prime, beta
    features = np.array([[
        data.omissions,
        data.commissions,
        data.hit_rt,
        data.varse,
        d_prime,
        beta
    ]])
    features_scaled = scaler.transform(features)

    prob = model.predict_proba(features_scaled)[0][1]

    T_RISK    = 0.54
    T_CAUTION = 0.39

    if prob >= T_RISK:
        core = {
            "Omissions": data.omissions,
            "Commissions": data.commissions,
            "RT_SD": data.varse,
        }
        worst = max(core, key=core.get)
        if worst == "Omissions":
            detail = "목표 자극을 자주 놓치는 '부주의(Inattention)' 우세 유형입니다."
        elif worst == "Commissions":
            detail = "억제 통제력이 저하된 '충동성(Impulsivity)' 우세 유형입니다."
        else:
            detail = "반응 속도 편차가 심한 '주의력 유지 실패' 유형입니다."
        result = f"[위험군] ADHD 위험 확률 {prob*100:.1f}점 | {detail}"

    elif prob >= T_CAUTION:
        result = f"[주의군] ADHD 위험 확률 {prob*100:.1f}점 | 경계선 수준입니다. 전문가 상담을 권장합니다."
    else:
        result = f"[정상군] ADHD 위험 확률 {prob*100:.1f}점 | 안정적인 인지 기능을 보입니다."

    return {"result": result}
