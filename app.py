import streamlit as st
import asyncio
import json
from datetime import datetime
from models.medical_models import PatientInfo
from core.orchestrator import MAIDxOrchestrator
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="MAI-Dx Orchestrator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = MAIDxOrchestrator()
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'diagnosis_result' not in st.session_state:
    st.session_state.diagnosis_result = None

def main():
    # 메인 헤더
    st.markdown('<h1 class="main-header">🏥 MAI-Dx Orchestrator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Shop2world AI-powered Medical Diagnosis System</p>', unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("🔧 시스템 제어")
        
        # 새 세션 시작
        if st.button("🆕 새 진단 세션 시작", use_container_width=True):
            st.session_state.current_session_id = None
            st.session_state.diagnosis_result = None
            st.rerun()
        
        # 기존 세션 목록
        if st.session_state.orchestrator.sessions:
            st.subheader("📋 기존 세션")
            for session_id in list(st.session_state.orchestrator.sessions.keys())[:5]:
                if st.button(f"세션: {session_id[:8]}...", key=f"session_{session_id}"):
                    st.session_state.current_session_id = session_id
                    st.rerun()
        
        st.divider()
        
        # 시스템 정보
        st.subheader("ℹ️ 시스템 정보")
        st.info(f"활성 세션: {len(st.session_state.orchestrator.sessions)}")
        
        if st.button("🗑️ 모든 세션 삭제", use_container_width=True):
            st.session_state.orchestrator.sessions.clear()
            st.session_state.current_session_id = None
            st.session_state.diagnosis_result = None
            st.rerun()
    
    # 메인 컨텐츠
    if st.session_state.current_session_id is None:
        show_new_session_form()
    else:
        show_session_details()

def show_new_session_form():
    """새 세션 폼 표시"""
    
    st.markdown('<h2 class="sub-header">📝 환자 정보 입력</h2>', unsafe_allow_html=True)
    
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("나이", min_value=0, max_value=120, value=30)
            gender = st.selectbox("성별", ["남성", "여성", "기타"])
            
            # 증상 입력
            st.subheader("증상")
            symptoms_input = st.text_area(
                "증상을 쉼표로 구분하여 입력하세요",
                placeholder="예: 발열, 기침, 피로감, 두통"
            )
            symptoms = [s.strip() for s in symptoms_input.split(',') if s.strip()] if symptoms_input else []
        
        with col2:
            # 과거 병력
            st.subheader("과거 병력")
            medical_history_input = st.text_area(
                "과거 병력을 쉼표로 구분하여 입력하세요",
                placeholder="예: 고혈압, 당뇨병"
            )
            medical_history = [h.strip() for h in medical_history_input.split(',') if h.strip()] if medical_history_input else []
            
            # 현재 복용 중인 약물
            st.subheader("현재 복용 중인 약물")
            medications_input = st.text_area(
                "복용 중인 약물을 쉼표로 구분하여 입력하세요",
                placeholder="예: 아스피린, 혈압약"
            )
            current_medications = [m.strip() for m in medications_input.split(',') if m.strip()] if medications_input else []
        
        # 활력징후
        st.subheader("활력징후 (선택사항)")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            blood_pressure = st.text_input("혈압 (mmHg)", placeholder="120/80")
        with col4:
            heart_rate = st.number_input("심박수 (회/분)", min_value=40, max_value=200, value=80)
        with col5:
            temperature = st.number_input("체온 (°C)", min_value=35.0, max_value=42.0, value=36.5, step=0.1)
        
        vital_signs = {}
        if blood_pressure:
            vital_signs["blood_pressure"] = blood_pressure
        if heart_rate:
            vital_signs["heart_rate"] = heart_rate
        if temperature:
            vital_signs["temperature"] = temperature
        
        submitted = st.form_submit_button("🚀 진단 시작", use_container_width=True)
        
        if submitted:
            if not symptoms:
                st.error("최소 하나의 증상을 입력해주세요.")
                return
            
            # PatientInfo 객체 생성
            patient_info = PatientInfo(
                age=age,
                gender=gender,
                symptoms=symptoms,
                medical_history=medical_history,
                current_medications=current_medications,
                vital_signs=vital_signs if vital_signs else None
            )
            
            # 세션 시작
            with st.spinner("진단 세션을 시작하는 중..."):
                session_id = asyncio.run(
                    st.session_state.orchestrator.start_diagnosis_session(patient_info)
                )
                st.session_state.current_session_id = session_id
            
            st.success(f"진단 세션이 시작되었습니다! 세션 ID: {session_id[:8]}...")
            st.rerun()

def show_session_details():
    """세션 상세 정보 표시"""
    
    session_id = st.session_state.current_session_id
    state = st.session_state.orchestrator.get_session_state(session_id)
    
    if not state:
        st.error("세션을 찾을 수 없습니다.")
        return
    
    # 헤더
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f'<h2 class="sub-header">🏥 진단 세션: {session_id[:8]}...</h2>', unsafe_allow_html=True)
    with col2:
        st.info(f"생성: {state.created_at.strftime('%Y-%m-%d %H:%M')}")
    with col3:
        if st.button("🔄 진단 실행", use_container_width=True):
            run_diagnosis(session_id)
    
    # 환자 정보 표시
    with st.expander("👤 환자 정보", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**나이:** {state.patient_info.age}세")
            st.write(f"**성별:** {state.patient_info.gender}")
            st.write(f"**증상:** {', '.join(state.patient_info.symptoms)}")
        
        with col2:
            st.write(f"**과거 병력:** {', '.join(state.patient_info.medical_history) if state.patient_info.medical_history else '없음'}")
            st.write(f"**복용 약물:** {', '.join(state.patient_info.current_medications) if state.patient_info.current_medications else '없음'}")
            if state.patient_info.vital_signs:
                st.write(f"**활력징후:** {state.patient_info.vital_signs}")
    
    # 진단 결과가 있는 경우 표시
    if st.session_state.diagnosis_result:
        show_diagnosis_results(state)
    else:
        st.info("진단을 실행하려면 '진단 실행' 버튼을 클릭하세요.")

def run_diagnosis(session_id):
    """진단 실행"""
    
    with st.spinner("AI 의사들이 토론하고 있습니다..."):
        result = asyncio.run(st.session_state.orchestrator.process_diagnosis(session_id))
        st.session_state.diagnosis_result = result
    
    if result.success:
        st.success("진단이 완료되었습니다!")
        st.rerun()
    else:
        st.error(f"진단 중 오류가 발생했습니다: {result.message}")

def show_diagnosis_results(state):
    """진단 결과 표시"""
    
    # 탭으로 결과 구분
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 진단 결과", "💬 토론 과정", "💰 비용 분석", "📊 SDBench 평가", "📋 전체 과정"])
    
    with tab1:
        show_diagnosis_summary(state)
    
    with tab2:
        show_debate_process(state)
    
    with tab3:
        show_cost_analysis(state)
    
    with tab4:
        show_sdbench_evaluation(state)
    
    with tab5:
        show_full_process(state)

def show_diagnosis_summary(state):
    """진단 요약 표시"""
    
    if state.current_action:
        st.markdown(f'<div class="info-box"><h3>결정된 액션: {state.current_action.value}</h3></div>', unsafe_allow_html=True)
    
    if state.proposed_diagnosis:
        st.markdown('<h3>🏥 진단 결과</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("진단명", state.proposed_diagnosis.condition)
            st.metric("신뢰도", f"{state.proposed_diagnosis.confidence:.1%}")
            st.metric("중증도", state.proposed_diagnosis.severity)
        
        with col2:
            if state.diagnosis_confirmation:
                st.metric("확정 신뢰도", f"{state.diagnosis_confirmation.confidence_level:.1%}")
                st.metric("후속 조치 필요", "예" if state.diagnosis_confirmation.follow_up_required else "아니오")
        
        # 진단 상세 정보
        st.subheader("진단 상세")
        st.write(f"**근거:** {state.proposed_diagnosis.reasoning}")
        
        if state.proposed_diagnosis.differential_diagnoses:
            st.write(f"**감별진단:** {', '.join(state.proposed_diagnosis.differential_diagnoses)}")
        
        if state.diagnosis_confirmation and state.diagnosis_confirmation.follow_up_plan:
            st.write(f"**후속 조치 계획:** {state.diagnosis_confirmation.follow_up_plan}")
    
    if state.final_decision:
        st.markdown('<h3>🎯 최종 결정</h3>', unsafe_allow_html=True)
        
        decision_color = "success" if state.final_decision.decision == "proceed" else "warning"
        st.markdown(f'<div class="{decision_color}-box"><strong>결정:</strong> {state.final_decision.decision}</div>', unsafe_allow_html=True)
        
        st.write(f"**근거:** {state.final_decision.reasoning}")
        st.write(f"**다음 단계:** {', '.join(state.final_decision.next_steps)}")

def show_debate_process(state):
    """토론 과정 표시"""
    
    if not state.debate_rounds:
        st.info("토론 과정이 없습니다.")
        return
    
    st.markdown('<h3>💬 Virtual Doctor Panel 토론 과정</h3>', unsafe_allow_html=True)
    
    for i, round_data in enumerate(state.debate_rounds):
        with st.expander(f"라운드 {round_data.round_number}", expanded=i==len(state.debate_rounds)-1):
            
            # 합의 결과
            if round_data.consensus:
                st.markdown(f'<div class="info-box"><strong>합의:</strong> {round_data.consensus}</div>', unsafe_allow_html=True)
            
            # 각 에이전트 응답
            for response in round_data.agent_responses:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{response.agent_role.value}:** {response.response}")
                        if response.reasoning:
                            st.caption(f"근거: {response.reasoning}")
                    
                    with col2:
                        st.metric("신뢰도", f"{response.confidence:.1%}")
                    
                    if response.recommendations:
                        st.write(f"**추천사항:** {', '.join(response.recommendations)}")
                    
                    if response.concerns:
                        st.write(f"**우려사항:** {', '.join(response.concerns)}")
                    
                    st.divider()
            
            # 불일치 사항
            if round_data.disagreements:
                st.markdown(f'<div class="warning-box"><strong>불일치 사항:</strong> {", ".join(round_data.disagreements)}</div>', unsafe_allow_html=True)

def show_cost_analysis(state):
    """비용 분석 표시"""
    
    if not state.cost_analysis:
        st.info("비용 분석이 없습니다.")
        return
    
    st.markdown('<h3>💰 비용 분석</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 비용", f"{state.cost_analysis.total_cost:,}원")
        st.metric("보험 적용률", f"{state.cost_analysis.insurance_coverage:.1%}")
    
    with col2:
        st.metric("환자 부담", f"{state.cost_analysis.patient_responsibility:,}원")
        st.metric("비용 효율성", state.cost_analysis.cost_effectiveness)
    
    with col3:
        if state.proposed_tests:
            st.metric("제안된 검사 수", len(state.proposed_tests))
    
    # 비용 세부 내역
    if state.cost_analysis.cost_breakdown:
        st.subheader("비용 세부 내역")
        
        cost_data = []
        for test_name, cost in state.cost_analysis.cost_breakdown.items():
            cost_data.append({
                "검사명": test_name,
                "비용": cost
            })
        
        df = pd.DataFrame(cost_data)
        fig = px.bar(df, x="검사명", y="비용", title="검사별 비용")
        st.plotly_chart(fig, use_container_width=True)
    
    # 추천사항
    if state.cost_analysis.recommendations:
        st.subheader("비용 관련 추천사항")
        for rec in state.cost_analysis.recommendations:
            st.write(f"• {rec}")

def show_sdbench_evaluation(state):
    """SDBench 평가 표시"""
    
    if not state.sdbench_evaluation:
        st.info("SDBench 평가가 없습니다.")
        return
    
    st.markdown('<h3>📊 SDBench Framework 평가</h3>', unsafe_allow_html=True)
    
    # 점수 표시
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("정확도", f"{state.sdbench_evaluation.accuracy_score:.1%}")
    with col2:
        st.metric("비용 효율성", f"{state.sdbench_evaluation.cost_efficiency:.1%}")
    with col3:
        st.metric("안전성", f"{state.sdbench_evaluation.safety_score:.1%}")
    with col4:
        st.metric("종합 점수", f"{state.sdbench_evaluation.overall_score:.1%}")
    
    # 점수 차트
    scores = {
        "정확도": state.sdbench_evaluation.accuracy_score,
        "비용 효율성": state.sdbench_evaluation.cost_efficiency,
        "안전성": state.sdbench_evaluation.safety_score,
        "종합 점수": state.sdbench_evaluation.overall_score
    }
    
    fig = go.Figure(data=[
        go.Bar(x=list(scores.keys()), y=list(scores.values()))
    ])
    fig.update_layout(title="SDBench 평가 점수", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    
    # 피드백
    if state.sdbench_evaluation.feedback:
        st.subheader("평가 피드백")
        for feedback in state.sdbench_evaluation.feedback:
            st.write(f"• {feedback}")
    
    # 개선 제안
    if state.sdbench_evaluation.improvement_suggestions:
        st.subheader("개선 제안")
        for suggestion in state.sdbench_evaluation.improvement_suggestions:
            st.write(f"• {suggestion}")

def show_full_process(state):
    """전체 과정 표시"""
    
    st.markdown('<h3>📋 전체 진단 과정</h3>', unsafe_allow_html=True)
    
    # JSON 형태로 전체 데이터 표시
    st.json(state.dict())

if __name__ == "__main__":
    main() 