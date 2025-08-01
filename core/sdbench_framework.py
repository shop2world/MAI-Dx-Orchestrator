from typing import List, Dict, Any, Optional
from models.medical_models import (
    SDBenchEvaluation, Diagnosis, MedicalTest, 
    CostAnalysis, PatientInfo, ActionType, DecisionResult
)
import openai
import os
from dotenv import load_dotenv
import json
import time

load_dotenv()

class SDBenchFramework:
    """SDBench Framework - 의료 진단 평가 시스템"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 평가 기준 데이터 (실제로는 더 정교한 데이터베이스가 필요)
        self.evaluation_criteria = {
            "accuracy_weights": {
                "symptom_match": 0.3,
                "diagnosis_confidence": 0.25,
                "evidence_support": 0.25,
                "differential_diagnosis": 0.2
            },
            "cost_efficiency_weights": {
                "total_cost": 0.4,
                "cost_effectiveness": 0.3,
                "insurance_coverage": 0.2,
                "patient_burden": 0.1
            },
            "safety_weights": {
                "risk_assessment": 0.4,
                "safety_protocols": 0.3,
                "follow_up_plan": 0.2,
                "emergency_readiness": 0.1
            }
        }
    
    def evaluate_diagnosis(self, 
                          diagnosis: Diagnosis,
                          patient_info: PatientInfo,
                          cost_analysis: Optional[CostAnalysis] = None,
                          decision_result: Optional[DecisionResult] = None,
                          context: Dict[str, Any] = None) -> SDBenchEvaluation:
        """진단 결과를 종합적으로 평가"""
        
        print("🔍 SDBench Framework 평가 시작...")
        
        # 정확도 평가
        accuracy_score = self._evaluate_accuracy(diagnosis, patient_info, context)
        
        # 비용 효율성 평가
        cost_efficiency = self._evaluate_cost_efficiency(
            diagnosis, cost_analysis, patient_info
        )
        
        # 안전성 평가
        safety_score = self._evaluate_safety(diagnosis, patient_info, decision_result)
        
        # 종합 점수 계산
        overall_score = self._calculate_overall_score(
            accuracy_score, cost_efficiency, safety_score
        )
        
        # 피드백 생성
        feedback = self._generate_feedback(
            accuracy_score, cost_efficiency, safety_score, diagnosis
        )
        
        # 개선 제안 생성
        improvement_suggestions = self._generate_improvement_suggestions(
            accuracy_score, cost_efficiency, safety_score, diagnosis, patient_info
        )
        
        return SDBenchEvaluation(
            accuracy_score=accuracy_score,
            cost_efficiency=cost_efficiency,
            safety_score=safety_score,
            overall_score=overall_score,
            feedback=feedback,
            improvement_suggestions=improvement_suggestions
        )
    
    def _evaluate_accuracy(self, diagnosis: Diagnosis, 
                          patient_info: PatientInfo,
                          context: Dict[str, Any] = None) -> float:
        """진단 정확도 평가"""
        
        try:
            # AI 기반 정확도 평가
            accuracy_prompt = f"""
다음 의료 진단의 정확도를 0.0-1.0 사이로 평가해주세요:

진단: {diagnosis.condition}
신뢰도: {diagnosis.confidence}
증상: {', '.join(patient_info.symptoms) if patient_info.symptoms else '없음'}
증상 설명: {diagnosis.reasoning}
감별진단: {', '.join(diagnosis.differential_diagnoses) if diagnosis.differential_diagnoses else '없음'}

평가 기준:
1. 증상과 진단의 일치도 (30%)
2. 진단 신뢰도 (25%)
3. 증거의 충분성 (25%)
4. 감별진단의 적절성 (20%)

정확도 점수만 숫자로 응답해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 진단 정확도 평가 전문가입니다."},
                    {"role": "user", "content": accuracy_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            try:
                accuracy = float(response.choices[0].message.content.strip())
                return min(1.0, max(0.0, accuracy))
            except:
                # 기본 정확도 계산
                return self._calculate_basic_accuracy(diagnosis, patient_info)
                
        except Exception as e:
            print(f"정확도 평가 중 오류: {str(e)}")
            return self._calculate_basic_accuracy(diagnosis, patient_info)
    
    def _calculate_basic_accuracy(self, diagnosis: Diagnosis, 
                                patient_info: PatientInfo) -> float:
        """기본 정확도 계산"""
        
        accuracy = 0.0
        
        # 신뢰도 기반 점수
        accuracy += diagnosis.confidence * 0.4
        
        # 증상 일치도 (간단한 키워드 매칭)
        if patient_info.symptoms:
            symptom_keywords = [
                "발열", "기침", "통증", "피로", "메스꺼움", "구토", "설사", "변비",
                "두통", "어지러움", "호흡곤란", "가슴통증", "복통", "관절통"
            ]
            
            matched_symptoms = sum(
                1 for symptom in patient_info.symptoms 
                if any(keyword in symptom for keyword in symptom_keywords)
            )
            
            if matched_symptoms > 0:
                accuracy += (matched_symptoms / len(patient_info.symptoms)) * 0.3
        
        # 감별진단 존재 여부
        if diagnosis.differential_diagnoses:
            accuracy += 0.2
        
        # 중증도 고려
        if diagnosis.severity in ["mild", "moderate"]:
            accuracy += 0.1
        
        return min(1.0, accuracy)
    
    def _evaluate_cost_efficiency(self, diagnosis: Diagnosis,
                                cost_analysis: Optional[CostAnalysis],
                                patient_info: PatientInfo) -> float:
        """비용 효율성 평가"""
        
        if not cost_analysis:
            return 0.5  # 기본값
        
        try:
            # AI 기반 비용 효율성 평가
            cost_prompt = f"""
다음 의료 진단의 비용 효율성을 0.0-1.0 사이로 평가해주세요:

진단: {diagnosis.condition}
총 비용: {cost_analysis.total_cost:,}원
보험 적용률: {cost_analysis.insurance_coverage:.1%}
환자 부담: {cost_analysis.patient_responsibility:,}원
비용 효율성: {cost_analysis.cost_effectiveness}

평가 기준:
1. 총 비용의 적절성 (40%)
2. 비용 대비 효과 (30%)
3. 보험 적용률 (20%)
4. 환자 부담 (10%)

비용 효율성 점수만 숫자로 응답해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 비용 효율성 평가 전문가입니다."},
                    {"role": "user", "content": cost_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            try:
                efficiency = float(response.choices[0].message.content.strip())
                return min(1.0, max(0.0, efficiency))
            except:
                return self._calculate_basic_cost_efficiency(cost_analysis)
                
        except Exception as e:
            print(f"비용 효율성 평가 중 오류: {str(e)}")
            return self._calculate_basic_cost_efficiency(cost_analysis)
    
    def _calculate_basic_cost_efficiency(self, cost_analysis: CostAnalysis) -> float:
        """기본 비용 효율성 계산"""
        
        efficiency = 0.0
        
        # 총 비용 평가
        if cost_analysis.total_cost < 100000:
            efficiency += 0.4
        elif cost_analysis.total_cost < 300000:
            efficiency += 0.3
        elif cost_analysis.total_cost < 500000:
            efficiency += 0.2
        else:
            efficiency += 0.1
        
        # 보험 적용률
        efficiency += cost_analysis.insurance_coverage * 0.3
        
        # 비용 효율성 등급
        if cost_analysis.cost_effectiveness == "high":
            efficiency += 0.3
        elif cost_analysis.cost_effectiveness == "medium":
            efficiency += 0.2
        else:
            efficiency += 0.1
        
        return min(1.0, efficiency)
    
    def _evaluate_safety(self, diagnosis: Diagnosis,
                        patient_info: PatientInfo,
                        decision_result: Optional[DecisionResult]) -> float:
        """안전성 평가"""
        
        try:
            # AI 기반 안전성 평가
            safety_prompt = f"""
다음 의료 진단의 안전성을 0.0-1.0 사이로 평가해주세요:

진단: {diagnosis.condition}
중증도: {diagnosis.severity}
나이: {patient_info.age}
성별: {patient_info.gender}
기존 병력: {', '.join(patient_info.medical_history) if patient_info.medical_history else '없음'}
복용 약물: {', '.join(patient_info.current_medications) if patient_info.current_medications else '없음'}

평가 기준:
1. 위험도 평가 (40%)
2. 안전 프로토콜 준수 (30%)
3. 후속 조치 계획 (20%)
4. 응급 상황 대비 (10%)

안전성 점수만 숫자로 응답해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 안전성 평가 전문가입니다."},
                    {"role": "user", "content": safety_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            try:
                safety = float(response.choices[0].message.content.strip())
                return min(1.0, max(0.0, safety))
            except:
                return self._calculate_basic_safety(diagnosis, patient_info)
                
        except Exception as e:
            print(f"안전성 평가 중 오류: {str(e)}")
            return self._calculate_basic_safety(diagnosis, patient_info)
    
    def _calculate_basic_safety(self, diagnosis: Diagnosis, 
                              patient_info: PatientInfo) -> float:
        """기본 안전성 계산"""
        
        safety = 0.5  # 기본값
        
        # 중증도에 따른 조정
        if diagnosis.severity == "mild":
            safety += 0.2
        elif diagnosis.severity == "moderate":
            safety += 0.1
        elif diagnosis.severity == "severe":
            safety -= 0.1
        elif diagnosis.severity == "critical":
            safety -= 0.2
        
        # 나이 관련 안전성
        if patient_info.age:
            if patient_info.age > 65:
                safety -= 0.1
            elif patient_info.age < 18:
                safety -= 0.05
        
        # 기존 병력
        if patient_info.medical_history:
            safety -= 0.1
        
        # 복용 약물
        if patient_info.current_medications:
            safety -= 0.05
        
        return min(1.0, max(0.0, safety))
    
    def _calculate_overall_score(self, accuracy: float, 
                               cost_efficiency: float, 
                               safety: float) -> float:
        """종합 점수 계산"""
        
        # 가중 평균 계산
        overall = (
            accuracy * 0.4 +
            cost_efficiency * 0.3 +
            safety * 0.3
        )
        
        return min(1.0, max(0.0, overall))
    
    def _generate_feedback(self, accuracy: float, 
                          cost_efficiency: float, 
                          safety: float,
                          diagnosis: Diagnosis) -> List[str]:
        """평가 결과에 대한 피드백 생성"""
        
        feedback = []
        
        # 정확도 피드백
        if accuracy >= 0.8:
            feedback.append("진단 정확도가 매우 높습니다")
        elif accuracy >= 0.6:
            feedback.append("진단 정확도가 양호합니다")
        else:
            feedback.append("진단 정확도를 개선할 필요가 있습니다")
        
        # 비용 효율성 피드백
        if cost_efficiency >= 0.8:
            feedback.append("비용 효율성이 매우 우수합니다")
        elif cost_efficiency >= 0.6:
            feedback.append("비용 효율성이 적절합니다")
        else:
            feedback.append("비용 효율성을 개선할 필요가 있습니다")
        
        # 안전성 피드백
        if safety >= 0.8:
            feedback.append("안전성 수준이 매우 높습니다")
        elif safety >= 0.6:
            feedback.append("안전성 수준이 적절합니다")
        else:
            feedback.append("안전성을 개선할 필요가 있습니다")
        
        return feedback
    
    def _generate_improvement_suggestions(self, accuracy: float,
                                        cost_efficiency: float,
                                        safety: float,
                                        diagnosis: Diagnosis,
                                        patient_info: PatientInfo) -> List[str]:
        """개선 제안 생성"""
        
        suggestions = []
        
        try:
            # AI 기반 개선 제안
            improvement_prompt = f"""
다음 의료 진단의 개선 방안을 제시해주세요:

진단: {diagnosis.condition}
정확도: {accuracy:.2f}
비용 효율성: {cost_efficiency:.2f}
안전성: {safety:.2f}
증상: {', '.join(patient_info.symptoms) if patient_info.symptoms else '없음'}

각 영역별로 1-2개의 구체적인 개선 방안을 제시해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 진단 개선 전문가입니다."},
                    {"role": "user", "content": improvement_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            ai_suggestions = response.choices[0].message.content.split('\n')
            suggestions.extend([s.strip() for s in ai_suggestions if s.strip()])
            
        except Exception as e:
            print(f"개선 제안 생성 중 오류: {str(e)}")
        
        # 기본 개선 제안
        if accuracy < 0.7:
            suggestions.append("추가 검사를 통한 진단 정확도 향상")
        
        if cost_efficiency < 0.6:
            suggestions.append("비용 효율적인 대안 검사 고려")
        
        if safety < 0.7:
            suggestions.append("안전 프로토콜 강화 및 모니터링 강화")
        
        return suggestions[:5]  # 최대 5개 제안으로 제한 