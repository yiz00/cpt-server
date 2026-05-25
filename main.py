from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# [해결 핵심 1] 브라우저 차단 에러 해결
# allow_origins=["*"] 일 때 allow_credentials=True 이면 브라우저가 보안상 연결을 강제로 차단합니다.
# 인증 정보가 필요 없는 공공 API 구조이므로 False로 변경하여 CORS 차단을 원천 봉쇄합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,  
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 및 스케일러 로드
try:
    kmeans_model = joblib.load('cpt_kmeans_model.pkl')
    scaler = joblib.load('cpt_robust_scaler.pkl')
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")

class CptData(BaseModel):
    age: int
    omissions: float
    commissions: float
    varse: float

@app.post("/predict")
def predict_cpt(data: CptData):
    try:
        # 1. 입력 데이터 배열화
        new_user_data = np.array([[data.omissions, data.commissions, data.varse, data.age]])
        
        # 2. 스케일링 및 군집 예측
        scaled_data = scaler.transform(new_user_data)
        cluster_label = kmeans_model.predict(scaled_data)[0]
        
        # 3. 취약 유형 분석 (가장 높은 에러 지표 추출)
        features = {
            "누락 오류(Omissions)": data.omissions,
            "오경보 오류(Commissions)": data.commissions,
            "반응 속도 표준편차(RT_SD)": data.varse
        }
        worst_feature = max(features, key=features.get)
        
        # 4. 결과 그룹 및 스펙트럼 점수 산출
        total_errors = data.omissions + data.commissions
        
        if cluster_label == 0:
            group = "정상군"
            # 정상군은 점수대를 0% ~ 39% 사이로 제한
            patient_prob = min(0.39, max(0.05, total_errors / 100.0))
        elif cluster_label == 1:
            group = "주의군"
            # 주의군은 점수대를 40% ~ 69% 사이로 제한
            patient_prob = min(0.69, max(0.40, 0.40 + (total_errors / 80.0)))
        else:
            group = "위험군"
            # 위험군은 점수대를 70% ~ 99% 사이로 제한
            patient_prob = min(0.99, max(0.70, 0.70 + (total_errors / 60.0)))

        # [요청하신 리포트 형태로 완벽 매칭]
        report = f"▶ [점수]: {patient_prob*100:.1f}점 -> {group}\n"
        
        # [핵심] 정상군이 아닐 때(주의군, 위험군)만 취약 유형 분석을 덧붙임
        if group != "정상군":
            report += f"▶ [취약 유형 분석]: 귀하의 가장 두드러진 결함 지표는 **{worst_feature}** 입니다.\n"
            if "Omissions" in worst_feature:
                report += "   -> 분석: 목표 자극을 자주 놓치는 '부주의' 유형입니다.\n"
            elif "Commissions" in worst_feature:
                report += "   -> 분석: 억제 통제력이 저하되어 성급하게 반응하는 '충동성' 유형입니다.\n"
            elif "RT_SD" in worst_feature:
                report += "   -> 분석: 반응 속도의 편차가 심한 '주의력 유지 실패' 유형입니다.\n"
        else:
            # 정상군일 경우 긍정적인 피드백 출력
            report += "▶ [분석 결과]: 전반적인 주의력 및 억제 통제 지표가 양호하며, 두드러진 결함이 관찰되지 않습니다.\n"
            
        return {"result": report}

    except Exception as e:
        # 서버 내부에서 연산 에러가 나서 튕기는 것을 방지하는 안전장치
        return {"result": f"서버 내부 연산 오류가 발생했습니다: {str(e)}"}
