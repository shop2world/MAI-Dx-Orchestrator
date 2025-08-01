from typing import List, Dict, Any
from models.medical_models import (
    CostAnalysis, DiagnosisConfirmation, MedicalTest, 
    Diagnosis, PatientInfo, ActionType
)
import openai
import os
from dotenv import load_dotenv
import json

load_dotenv()

class CostAnalysisModule:
    """의료 비용 분석 모듈"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 기본 의료 검사 비용 데이터 (실제로는 DB에서 가져와야 함)
        self.test_costs = {
            "혈액검사": {"cost": 50000, "insurance_coverage": 0.8},
            "X-ray": {"cost": 80000, "insurance_coverage": 0.7},
            "CT": {"cost": 200000, "insurance_coverage": 0.6},
            "MRI": {"cost": 400000, "insurance_coverage": 0.5},
            "초음파": {"cost": 60000, "insurance_coverage": 0.8},
            "심전도": {"cost": 30000, "insurance_coverage": 0.9},
            "소변검사": {"cost": 20000, "insurance_coverage": 0.9},
            "대변검사": {"cost": 25000, "insurance_coverage": 0.9},
            "알레르기검사": {"cost": 100000, "insurance_coverage": 0.6},
            "내시경": {"cost": 150000, "insurance_coverage": 0.7}
        }
    
    def analyze_costs(self, tests: List[MedicalTest], 
                     patient_info: PatientInfo = None) -> CostAnalysis:
        """검사 비용을 분석"""
        
        total_cost = 0
        cost_breakdown = {}
        
        for test in tests:
            # 검사 비용 계산
            base_cost = self._get_test_cost(test.test_name)
            cost_breakdown[test.test_name] = base_cost
            total_cost += base_cost
        
        # 보험 적용 계산
        insurance_coverage = self._calculate_insurance_coverage(tests, patient_info)
        patient_responsibility = total_cost * (1 - insurance_coverage)
        
        # 비용 효율성 평가
        cost_effectiveness = self._evaluate_cost_effectiveness(tests, total_cost)
        
        # 추천사항 생성
        recommendations = self._generate_cost_recommendations(
            tests, total_cost, patient_responsibility
        )
        
        return CostAnalysis(
            total_cost=total_cost,
            insurance_coverage=insurance_coverage,
            patient_responsibility=patient_responsibility,
            cost_breakdown=cost_breakdown,
            cost_effectiveness=cost_effectiveness,
            recommendations=recommendations
        )
    
    def _get_test_cost(self, test_name: str) -> float:
        """검사별 기본 비용 반환"""
        for key, value in self.test_costs.items():
            if key in test_name or test_name in key:
                return value["cost"]
        return 50000  # 기본 비용
    
    def _calculate_insurance_coverage(self, tests: List[MedicalTest], 
                                    patient_info: PatientInfo) -> float:
        """보험 적용률 계산"""
        
        if not tests:
            return 0.0
        
        total_coverage = 0
        for test in tests:
            for key, value in self.test_costs.items():
                if key in test.test_name or test.test_name in key:
                    total_coverage += value["insurance_coverage"]
                    break
            else:
                total_coverage += 0.7  # 기본 보험 적용률
        
        return total_coverage / len(tests)
    
    def _evaluate_cost_effectiveness(self, tests: List[MedicalTest], 
                                   total_cost: float) -> str:
        """비용 효율성 평가"""
        
        if total_cost < 100000:
            return "high"
        elif total_cost < 300000:
            return "medium"
        else:
            return "low"
    
    def _generate_cost_recommendations(self, tests: List[MedicalTest], 
                                     total_cost: float, 
                                     patient_responsibility: float) -> List[str]:
        """비용 관련 추천사항 생성"""
        
        recommendations = []
        
        if total_cost > 500000:
            recommendations.append("고비용 검사이므로 단계적 접근을 고려하세요")
        
        if patient_responsibility > 200000:
            recommendations.append("환자 부담이 큽니다. 보험 혜택을 확인하세요")
        
        if len(tests) > 5:
            recommendations.append("검사 수가 많습니다. 우선순위를 정해 진행하세요")
        
        # AI 기반 비용 최적화 추천
        try:
            cost_prompt = f"""
다음 의료 검사들의 비용을 분석하고 최적화 방안을 제시해주세요:

검사 목록: {[test.test_name for test in tests]}
총 비용: {total_cost:,}원
환자 부담: {patient_responsibility:,}원

비용 효율적인 대안이나 최적화 방안을 2-3개 제시해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 비용 최적화 전문가입니다."},
                    {"role": "user", "content": cost_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            ai_recommendations = response.choices[0].message.content.split('\n')
            recommendations.extend([rec.strip() for rec in ai_recommendations if rec.strip()])
            
        except Exception as e:
            recommendations.append("AI 비용 분석 중 오류가 발생했습니다")
        
        return recommendations


class DiagnosisConfirmationModule:
    """진단 확정 모듈"""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def confirm_diagnosis(self, diagnosis: Diagnosis, 
                         patient_info: PatientInfo,
                         supporting_evidence: List[str] = None) -> DiagnosisConfirmation:
        """진단을 확정하고 검증"""
        
        # 진단 확정 방법들
        confirmation_methods = self._get_confirmation_methods(diagnosis, patient_info)
        
        # 신뢰도 계산
        confidence_level = self._calculate_confidence_level(
            diagnosis, patient_info, supporting_evidence
        )
        
        # 위험 요인 식별
        risk_factors = self._identify_risk_factors(diagnosis, patient_info)
        
        # 후속 조치 필요성 판단
        follow_up_required = self._assess_follow_up_need(diagnosis, confidence_level)
        follow_up_plan = self._generate_follow_up_plan(diagnosis) if follow_up_required else None
        
        return DiagnosisConfirmation(
            confirmed_diagnosis=diagnosis,
            confirmation_methods=confirmation_methods,
            confidence_level=confidence_level,
            risk_factors=risk_factors,
            follow_up_required=follow_up_required,
            follow_up_plan=follow_up_plan
        )
    
    def _get_confirmation_methods(self, diagnosis: Diagnosis, 
                                patient_info: PatientInfo) -> List[str]:
        """진단 확정 방법들 결정"""
        
        methods = []
        
        # 기본 확정 방법들
        if diagnosis.confidence > 0.8:
            methods.append("높은 신뢰도 기반 확정")
        
        if diagnosis.icd_code:
            methods.append("ICD 코드 매칭")
        
        if patient_info.symptoms:
            methods.append("증상 패턴 분석")
        
        if patient_info.medical_history:
            methods.append("과거 병력 검토")
        
        # AI 기반 추가 확정 방법
        try:
            confirmation_prompt = f"""
다음 진단에 대한 추가 확정 방법을 제시해주세요:

진단: {diagnosis.condition}
증상: {', '.join(patient_info.symptoms)}
신뢰도: {diagnosis.confidence}

추가로 고려할 수 있는 확정 방법 2-3개를 제시해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 진단 확정 전문가입니다."},
                    {"role": "user", "content": confirmation_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            ai_methods = response.choices[0].message.content.split('\n')
            methods.extend([method.strip() for method in ai_methods if method.strip()])
            
        except Exception as e:
            methods.append("AI 확정 분석 중 오류 발생")
        
        return methods
    
    def _calculate_confidence_level(self, diagnosis: Diagnosis, 
                                  patient_info: PatientInfo,
                                  supporting_evidence: List[str] = None) -> float:
        """진단 확정 신뢰도 계산"""
        
        base_confidence = diagnosis.confidence
        
        # 증상 일치도 평가
        symptom_match = self._evaluate_symptom_match(diagnosis, patient_info)
        
        # 증거 강도 평가
        evidence_strength = self._evaluate_evidence_strength(supporting_evidence)
        
        # 위험 요인 고려
        risk_adjustment = self._calculate_risk_adjustment(patient_info)
        
        # 최종 신뢰도 계산
        final_confidence = (
            base_confidence * 0.4 +
            symptom_match * 0.3 +
            evidence_strength * 0.2 +
            risk_adjustment * 0.1
        )
        
        return min(1.0, max(0.0, final_confidence))
    
    def _evaluate_symptom_match(self, diagnosis: Diagnosis, 
                              patient_info: PatientInfo) -> float:
        """증상 일치도 평가"""
        
        if not patient_info.symptoms:
            return 0.5
        
        # AI 기반 증상 일치도 평가
        try:
            symptom_prompt = f"""
진단 "{diagnosis.condition}"과 다음 증상들의 일치도를 0.0-1.0 사이로 평가해주세요:

증상: {', '.join(patient_info.symptoms)}
진단 설명: {diagnosis.reasoning}

일치도 점수만 숫자로 응답해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 증상 분석 전문가입니다."},
                    {"role": "user", "content": symptom_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            try:
                return float(response.choices[0].message.content.strip())
            except:
                return 0.7
                
        except Exception as e:
            return 0.7
    
    def _evaluate_evidence_strength(self, supporting_evidence: List[str] = None) -> float:
        """증거 강도 평가"""
        
        if not supporting_evidence:
            return 0.5
        
        # 증거 수와 품질에 따른 평가
        evidence_count = len(supporting_evidence)
        if evidence_count >= 5:
            return 0.9
        elif evidence_count >= 3:
            return 0.7
        elif evidence_count >= 1:
            return 0.6
        else:
            return 0.5
    
    def _calculate_risk_adjustment(self, patient_info: PatientInfo) -> float:
        """위험 요인에 따른 조정"""
        
        risk_factors = 0
        
        # 나이 관련 위험
        if patient_info.age and patient_info.age > 65:
            risk_factors += 0.1
        
        # 기존 병력
        if patient_info.medical_history:
            risk_factors += 0.1
        
        # 복용 약물
        if patient_info.current_medications:
            risk_factors += 0.05
        
        return 1.0 - risk_factors  # 위험 요인이 많을수록 신뢰도 감소
    
    def _identify_risk_factors(self, diagnosis: Diagnosis, 
                             patient_info: PatientInfo) -> List[str]:
        """위험 요인 식별"""
        
        risk_factors = []
        
        # 기본 위험 요인들
        if diagnosis.severity in ["severe", "critical"]:
            risk_factors.append("중증도가 높은 진단")
        
        if patient_info.age and patient_info.age > 65:
            risk_factors.append("고령 환자")
        
        if patient_info.medical_history:
            risk_factors.append("기존 병력 존재")
        
        # AI 기반 위험 요인 분석
        try:
            risk_prompt = f"""
다음 진단과 환자 정보에서 추가 위험 요인을 식별해주세요:

진단: {diagnosis.condition}
나이: {patient_info.age}
성별: {patient_info.gender}
기존 병력: {', '.join(patient_info.medical_history)}
복용 약물: {', '.join(patient_info.current_medications)}

추가 위험 요인 2-3개를 제시해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 위험 요인 분석 전문가입니다."},
                    {"role": "user", "content": risk_prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            ai_risks = response.choices[0].message.content.split('\n')
            risk_factors.extend([risk.strip() for risk in ai_risks if risk.strip()])
            
        except Exception as e:
            risk_factors.append("AI 위험 분석 중 오류 발생")
        
        return risk_factors
    
    def _assess_follow_up_need(self, diagnosis: Diagnosis, 
                             confidence_level: float) -> bool:
        """후속 조치 필요성 판단"""
        
        # 신뢰도가 낮거나 중증도가 높은 경우 후속 조치 필요
        if confidence_level < 0.7:
            return True
        
        if diagnosis.severity in ["severe", "critical"]:
            return True
        
        return False
    
    def _generate_follow_up_plan(self, diagnosis: Diagnosis) -> str:
        """후속 조치 계획 생성"""
        
        try:
            follow_up_prompt = f"""
다음 진단에 대한 후속 조치 계획을 수립해주세요:

진단: {diagnosis.condition}
중증도: {diagnosis.severity}

구체적인 후속 조치 계획을 제시해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 후속 조치 계획 전문가입니다."},
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return "후속 조치 계획 생성 중 오류가 발생했습니다." 