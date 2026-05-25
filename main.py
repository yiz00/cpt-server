from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from scipy.stats import norm  # d_prime과 beta 계산을 위한 라이브러리 추가

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,  
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    kmeans_model = joblib.load('cpt_kmeans_model.pkl')
    scaler = joblib.load('cpt_robust_scaler.pkl')
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")

# [수정됨] 웹에서 나이 대신 반응속도(hit_rt)를 정확히 받아오도록 수정
class CptData(BaseModel):
    age: int
    omissions: float
    commissions: float
    hit_rt: float  
    varse: float

@app.post("/predict")
def predict_cpt(data: CptData):
    try:
        # 1. d_prime 및 beta 자동 계산 (모델이 요구하는 6개 변수 완성)
        TOTAL_TARGETS = 324
        TOTAL_NONTARGETS = 36
        
        hr = np.clip((TOTAL_TARGETS - data.omissions) / TOTAL_TARGETS, 0.005, 0.995)
        far = np.clip(data.commissions / TOTAL_NONTARGETS, 0.005, 0.995)
        z_hr = norm.ppf(hr)
        z_far = norm.ppf(far)
        d_prime = z_hr - z_far
        beta = np.exp(0.5 * (z_far**2 - z_hr**2))
        
        # 2. 스케일러가 요구하는 정확한 6개 변수 순서: [누락, 오경보, 반응속도, 변동성, d_prime, beta]
        new_user_data = np.array([[data.omissions, data.commissions, data.hit_rt, data.varse, d_prime, beta]])
        
        # 3. 스케일링 및 군집 예측
        scaled_data = scaler.transform(new_user_data)
        cluster_label = kmeans_model.predict(scaled_data)[0]
        
        # 4. 취약 유형 분석
        features = {
            "누락 오류(Omissions)": data.omissions,
            "오경보 오류(Commissions)": data.commissions,
            "반응 속도 표준편차(RT_SD)": data.varse
        }
        worst_feature = max(features, key=features.get)
        
        # 5. 결과 그룹 및 스펙트럼 점수 산출
        total_errors = data.omissions + data.commissions
        
        if cluster_label == 0:
            group = "정상군"
            patient_prob = min(0.39, max(0.05, total_errors / 100.0))
        elif cluster_label == 1:
            group = "주의군"
            patient_prob = min(0.69, max(0.40, 0.40 + (total_errors / 80.0)))
        else:
            group = "위험군"
            patient_prob = min(0.99, max(0.70, 0.70 + (total_errors / 60.0)))

        report = f"▶ [스펙트럼 점수]: {patient_prob*100:.1f}점 -> {group}\n"
        
        # 정상군이 아닐 때만 취약 유형 덧붙임
        if group != "정상군":
            report += f"▶ [취약 유형 분석]: 귀하의 가장 두드러진 결함 지표는 **{worst_feature}** 입니다.\n"
            if "Omissions" in worst_feature:
                report += "   -> 분석: 목표 자극을 자주 놓치는 '부주의' 유형입니다.\n"
            elif "Commissions" in worst_feature:
                report += "   -> 분석: 억제 통제력이 저하되어 성급하게 반응하는 '충동성' 유형입니다.\n"
            elif "RT_SD" in worst_feature:
                report += "   -> 분석: 반응 속도의 편차가 심한 '주의력 유지 실패' 유형입니다.\n"
        else:
            report += "▶ [분석 결과]: 전반적인 주의력 및 억제 통제 지표가 양호하며, 두드러진 결함이 관찰되지 않습니다.\n"
            
        return {"result": report}

    except Exception as e:
        return {"result": f"서버 내부 연산 오류가 발생했습니다: {str(e)}"}
