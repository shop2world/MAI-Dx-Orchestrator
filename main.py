#!/usr/bin/env python3
"""
MAI-Dx Orchestrator 메인 실행 파일

이 파일은 MAI-Dx Orchestrator 시스템을 실행하는 메인 엔트리 포인트입니다.
"""

import asyncio
import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.medical_models import PatientInfo
from core.orchestrator import MAIDxOrchestrator

def print_banner():
    """시스템 배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🏥 MAI-Dx Orchestrator                    ║
    ║              Shop2world AI-powered Medical Diagnosis         ║
    ║                        Version 1.0.0                         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_usage():
    """사용법 출력"""
    usage = """
    사용법:
    
    1. 웹 인터페이스 실행:
       streamlit run app.py
    
    2. API 서버 실행:
       uvicorn api:app --reload
    
    3. 명령줄 데모 실행:
       python main.py --demo
    
    4. 환경 설정:
       .env 파일에 OPENAI_API_KEY를 설정하세요.
    """
    print(usage)

async def run_demo():
    """데모 실행"""
    print("🚀 MAI-Dx Orchestrator 데모를 시작합니다...")
    
    # 오케스트레이터 초기화
    orchestrator = MAIDxOrchestrator()
    
    # 샘플 환자 정보
    patient_info = PatientInfo(
        age=35,
        gender="남성",
        symptoms=["발열", "기침", "피로감", "두통"],
        medical_history=["고혈압"],
        current_medications=["혈압약"],
        vital_signs={
            "blood_pressure": "140/90",
            "heart_rate": 85,
            "temperature": 38.2
        }
    )
    
    print(f"👤 환자 정보:")
    print(f"   - 나이: {patient_info.age}세")
    print(f"   - 성별: {patient_info.gender}")
    print(f"   - 증상: {', '.join(patient_info.symptoms)}")
    print(f"   - 과거 병력: {', '.join(patient_info.medical_history)}")
    print(f"   - 복용 약물: {', '.join(patient_info.current_medications)}")
    print()
    
    # 세션 시작
    print("🏥 진단 세션을 시작합니다...")
    session_id = await orchestrator.start_diagnosis_session(patient_info)
    print(f"   세션 ID: {session_id}")
    print()
    
    # 진단 프로세스 실행
    print("🔍 AI 의사들이 토론을 시작합니다...")
    result = await orchestrator.process_diagnosis(session_id)
    
    if result.success:
        print("✅ 진단이 완료되었습니다!")
        print()
        
        # 결과 요약
        state = orchestrator.get_session_state(session_id)
        if state:
            print("📋 진단 결과 요약:")
            
            if state.current_action:
                print(f"   - 결정된 액션: {state.current_action.value}")
            
            if state.proposed_diagnosis:
                print(f"   - 진단명: {state.proposed_diagnosis.condition}")
                print(f"   - 신뢰도: {state.proposed_diagnosis.confidence:.1%}")
                print(f"   - 중증도: {state.proposed_diagnosis.severity}")
                print(f"   - 근거: {state.proposed_diagnosis.reasoning}")
            
            if state.cost_analysis:
                print(f"   - 총 비용: {state.cost_analysis.total_cost:,}원")
                print(f"   - 환자 부담: {state.cost_analysis.patient_responsibility:,}원")
            
            if state.sdbench_evaluation:
                print(f"   - SDBench 종합 점수: {state.sdbench_evaluation.overall_score:.1%}")
            
            if state.final_decision:
                print(f"   - 최종 결정: {state.final_decision.decision}")
                print(f"   - 다음 단계: {', '.join(state.final_decision.next_steps)}")
        
        print()
        print("🎉 데모가 성공적으로 완료되었습니다!")
        
    else:
        print(f"❌ 진단 중 오류가 발생했습니다: {result.message}")

def check_environment():
    """환경 설정 확인"""
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  경고: OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일을 생성하고 OpenAI API 키를 설정하세요.")
        print("   예시: OPENAI_API_KEY=your_api_key_here")
        print()
        return False
    return True

def main():
    """메인 함수"""
    print_banner()
    
    # 명령줄 인수 처리
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            if check_environment():
                asyncio.run(run_demo())
            else:
                print("환경 설정을 확인한 후 다시 시도하세요.")
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print_usage()
        else:
            print(f"알 수 없는 옵션: {sys.argv[1]}")
            print_usage()
    else:
        print_usage()

if __name__ == "__main__":
    main() 