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

# 2. 모델 및 스케일러 로드
model  = joblib.load("cpt_kmeans_model.pkl")
scaler = joblib.load("cpt_robust_scaler.pkl")

# 루트 경로 정의 추가
@app.get("/")
def read_root():
    return {"message": "CPT Prediction Server is running"}

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

    # [수정된 부분]: 확률이 '주의군' 이상(T_CAUTION 이상)이면 무조건 유형 분석을 실행합니다.
    if prob >= T_CAUTION:
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

        # 세부 분석 결과를 바탕으로 점수대에 따라 앞부분 라벨(위험군/주의군)만 다르게 붙여줍니다.
        if prob >= T_RISK:
            result = f"[위험군] ADHD 위험 확률 {prob*100:.1f}점 | {detail} 즉각적인 임상 평가가 필요합니다."
        else:
            result = f"[주의군] ADHD 위험 확률 {prob*100:.1f}점 | 경계선 수준입니다. {detail}"

    else:
        # 정상군은 세부 분석 없이 안정적이라는 메시지만 출력합니다.
        result = f"[정상군] ADHD 위험 확률 {prob*100:.1f}점 | 안정적인 인지 기능을 보입니다."

    return {"result": result}
