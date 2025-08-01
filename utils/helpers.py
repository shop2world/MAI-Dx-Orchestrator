"""
MAI-Dx Orchestrator 유틸리티 함수들
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
import uuid

def setup_logging(level: str = "INFO") -> logging.Logger:
    """로깅 설정"""
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('mai_dx.log')
        ]
    )
    
    return logging.getLogger(__name__)

def generate_session_id() -> str:
    """세션 ID 생성"""
    return str(uuid.uuid4())

def calculate_confidence_score(scores: List[float]) -> float:
    """신뢰도 점수 계산"""
    if not scores:
        return 0.0
    
    # 가중 평균 계산 (최신 점수에 더 높은 가중치)
    weights = [i + 1 for i in range(len(scores))]
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    
    return weighted_sum / total_weight

def validate_patient_info(patient_data: Dict[str, Any]) -> List[str]:
    """환자 정보 유효성 검사"""
    errors = []
    
    # 필수 필드 검사
    if not patient_data.get('symptoms'):
        errors.append("증상 정보가 필요합니다")
    
    # 나이 검사
    age = patient_data.get('age')
    if age is not None and (age < 0 or age > 150):
        errors.append("나이는 0-150 사이여야 합니다")
    
    # 성별 검사
    gender = patient_data.get('gender')
    if gender and gender not in ['남성', '여성', '기타']:
        errors.append("성별은 '남성', '여성', '기타' 중 하나여야 합니다")
    
    return errors

def format_cost(cost: float) -> str:
    """비용 포맷팅"""
    if cost >= 1000000:
        return f"{cost/1000000:.1f}M원"
    elif cost >= 1000:
        return f"{cost/1000:.0f}K원"
    else:
        return f"{cost:.0f}원"

def calculate_risk_score(patient_info: Dict[str, Any]) -> float:
    """위험도 점수 계산"""
    risk_score = 0.0
    
    # 나이 위험도
    age = patient_info.get('age', 0)
    if age > 65:
        risk_score += 0.3
    elif age > 50:
        risk_score += 0.2
    elif age < 18:
        risk_score += 0.1
    
    # 기존 병력 위험도
    medical_history = patient_info.get('medical_history', [])
    if medical_history:
        risk_score += 0.2
    
    # 복용 약물 위험도
    medications = patient_info.get('current_medications', [])
    if medications:
        risk_score += 0.1
    
    # 활력징후 위험도
    vital_signs = patient_info.get('vital_signs', {})
    if vital_signs:
        # 혈압 위험도
        bp = vital_signs.get('blood_pressure')
        if bp:
            try:
                systolic, diastolic = map(int, bp.split('/'))
                if systolic > 140 or diastolic > 90:
                    risk_score += 0.2
            except:
                pass
        
        # 체온 위험도
        temp = vital_signs.get('temperature')
        if temp and temp > 38.0:
            risk_score += 0.3
    
    return min(1.0, risk_score)

def parse_symptoms(symptoms_text: str) -> List[str]:
    """증상 텍스트 파싱"""
    if not symptoms_text:
        return []
    
    # 쉼표, 세미콜론, 줄바꿈으로 구분
    symptoms = []
    for symptom in symptoms_text.replace('\n', ',').replace(';', ',').split(','):
        symptom = symptom.strip()
        if symptom:
            symptoms.append(symptom)
    
    return symptoms

def create_summary_report(state_data: Dict[str, Any]) -> str:
    """진단 결과 요약 리포트 생성"""
    
    report = []
    report.append("=" * 50)
    report.append("Shop2world MAI-Dx Orchestrator 진단 리포트")
    report.append("=" * 50)
    report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 환자 정보
    patient_info = state_data.get('patient_info', {})
    report.append("👤 환자 정보:")
    report.append(f"  - 나이: {patient_info.get('age', 'N/A')}세")
    report.append(f"  - 성별: {patient_info.get('gender', 'N/A')}")
    report.append(f"  - 증상: {', '.join(patient_info.get('symptoms', []))}")
    report.append("")
    
    # 진단 결과
    diagnosis = state_data.get('proposed_diagnosis')
    if diagnosis:
        report.append("🏥 진단 결과:")
        report.append(f"  - 진단명: {diagnosis.get('condition', 'N/A')}")
        report.append(f"  - 신뢰도: {diagnosis.get('confidence', 0):.1%}")
        report.append(f"  - 중증도: {diagnosis.get('severity', 'N/A')}")
        report.append(f"  - 근거: {diagnosis.get('reasoning', 'N/A')}")
        report.append("")
    
    # 비용 분석
    cost_analysis = state_data.get('cost_analysis')
    if cost_analysis:
        report.append("💰 비용 분석:")
        report.append(f"  - 총 비용: {format_cost(cost_analysis.get('total_cost', 0))}")
        report.append(f"  - 환자 부담: {format_cost(cost_analysis.get('patient_responsibility', 0))}")
        report.append(f"  - 보험 적용률: {cost_analysis.get('insurance_coverage', 0):.1%}")
        report.append("")
    
    # SDBench 평가
    sdbench = state_data.get('sdbench_evaluation')
    if sdbench:
        report.append("📊 SDBench 평가:")
        report.append(f"  - 정확도: {sdbench.get('accuracy_score', 0):.1%}")
        report.append(f"  - 비용 효율성: {sdbench.get('cost_efficiency', 0):.1%}")
        report.append(f"  - 안전성: {sdbench.get('safety_score', 0):.1%}")
        report.append(f"  - 종합 점수: {sdbench.get('overall_score', 0):.1%}")
        report.append("")
    
    # 최종 결정
    final_decision = state_data.get('final_decision')
    if final_decision:
        report.append("🎯 최종 결정:")
        report.append(f"  - 결정: {final_decision.get('decision', 'N/A')}")
        report.append(f"  - 근거: {final_decision.get('reasoning', 'N/A')}")
        report.append(f"  - 다음 단계: {', '.join(final_decision.get('next_steps', []))}")
        report.append("")
    
    report.append("=" * 50)
    
    return "\n".join(report)

def export_to_json(data: Dict[str, Any], filename: str = None) -> str:
    """데이터를 JSON으로 내보내기"""
    
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mai_dx_export_{timestamp}.json"
    
    # datetime 객체를 문자열로 변환
    def datetime_converter(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=datetime_converter)
    
    return filename

def import_from_json(filename: str) -> Dict[str, Any]:
    """JSON 파일에서 데이터 가져오기"""
    
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def calculate_hash(data: str) -> str:
    """데이터 해시 계산"""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def sanitize_filename(filename: str) -> str:
    """파일명 정리"""
    import re
    # 특수문자 제거 및 공백을 언더스코어로 변경
    sanitized = re.sub(r'[^\w\s-]', '', filename)
    sanitized = re.sub(r'[-\s]+', '_', sanitized)
    return sanitized.strip('_')

def format_duration(seconds: float) -> str:
    """시간 지속 시간 포맷팅"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}분"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}시간" 