import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from models.medical_models import (
    OrchestratorState, PatientInfo, ActionType, MedicalTest, 
    Diagnosis, DecisionResult, SystemResponse
)
from core.debate_system import ChainOfDebate
from core.analysis_modules import CostAnalysisModule, DiagnosisConfirmationModule
from core.sdbench_framework import SDBenchFramework
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class MAIDxOrchestrator:
    """MAI-Dx Orchestrator 메인 시스템"""
    
    def __init__(self):
        self.debate_system = ChainOfDebate()
        self.cost_analyzer = CostAnalysisModule()
        self.diagnosis_confirmer = DiagnosisConfirmationModule()
        self.sdbench = SDBenchFramework()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 세션 저장소 (실제로는 데이터베이스 사용)
        self.sessions: Dict[str, OrchestratorState] = {}
    
    async def start_diagnosis_session(self, patient_info: PatientInfo) -> str:
        """새로운 진단 세션 시작"""
        
        session_id = str(uuid.uuid4())
        
        # 초기 상태 생성
        initial_state = OrchestratorState(
            patient_info=patient_info,
            session_id=session_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.sessions[session_id] = initial_state
        print(f"🏥 새로운 진단 세션 시작: {session_id}")
        
        return session_id
    
    async def process_diagnosis(self, session_id: str) -> SystemResponse:
        """진단 프로세스 실행"""
        
        if session_id not in self.sessions:
            return SystemResponse(
                success=False,
                message="세션을 찾을 수 없습니다",
                session_id=session_id
            )
        
        state = self.sessions[session_id]
        
        try:
            print(f"🔍 진단 프로세스 시작 - 세션: {session_id}")
            
            # 1단계: Chain of Debate 실행
            print("📋 1단계: Virtual Doctor Panel 토론 시작...")
            debate_rounds = await self.debate_system.conduct_debate(
                state.patient_info, max_rounds=3
            )
            
            state.debate_rounds = debate_rounds
            
            # 2단계: 합의 결과에서 액션 결정
            if debate_rounds:
                last_round = debate_rounds[-1]
                action_type = self.debate_system.extract_action_from_consensus(
                    last_round.consensus
                )
                state.current_action = action_type
                
                print(f"🎯 결정된 액션: {action_type.value}")
            
            # 3단계: 액션별 처리
            if state.current_action == ActionType.ASK_QUESTION:
                result = await self._handle_ask_question(state)
            elif state.current_action == ActionType.REQUEST_TEST:
                result = await self._handle_request_test(state)
            elif state.current_action == ActionType.PROVIDE_DIAGNOSIS:
                result = await self._handle_provide_diagnosis(state)
            else:
                result = await self._handle_ask_question(state)  # 기본값
            
            # 4단계: 최종 결정
            final_decision = await self._make_final_decision(state)
            state.final_decision = final_decision
            
            # 5단계: SDBench 평가
            if state.proposed_diagnosis:
                sdbench_evaluation = self.sdbench.evaluate_diagnosis(
                    diagnosis=state.proposed_diagnosis,
                    patient_info=state.patient_info,
                    cost_analysis=state.cost_analysis,
                    decision_result=state.final_decision
                )
                state.sdbench_evaluation = sdbench_evaluation
            
            # 상태 업데이트
            state.updated_at = datetime.now()
            self.sessions[session_id] = state
            
            return SystemResponse(
                success=True,
                message="진단 프로세스가 성공적으로 완료되었습니다",
                data=state.dict(),
                session_id=session_id
            )
            
        except Exception as e:
            print(f"❌ 진단 프로세스 오류: {str(e)}")
            return SystemResponse(
                success=False,
                message=f"진단 프로세스 중 오류가 발생했습니다: {str(e)}",
                session_id=session_id
            )
    
    async def _handle_ask_question(self, state: OrchestratorState) -> Dict[str, Any]:
        """질문 요청 처리"""
        
        print("❓ 질문 요청 처리 중...")
        
        # AI 기반 추가 질문 생성
        try:
            question_prompt = f"""
환자의 증상과 병력을 바탕으로 추가로 필요한 정보를 파악하기 위한 질문을 생성해주세요.

환자 정보:
- 나이: {state.patient_info.age}
- 성별: {state.patient_info.gender}
- 증상: {', '.join(state.patient_info.symptoms)}
- 과거 병력: {', '.join(state.patient_info.medical_history)}
- 복용 약물: {', '.join(state.patient_info.current_medications)}

추가로 필요한 정보를 파악하기 위한 3-5개의 구체적인 질문을 제시해주세요.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 진단을 위한 질문 생성 전문가입니다."},
                    {"role": "user", "content": question_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            questions = response.choices[0].message.content.split('\n')
            questions = [q.strip() for q in questions if q.strip()]
            
            return {
                "action": "ask_question",
                "questions": questions,
                "reasoning": "추가 정보가 필요하여 구체적인 질문을 생성했습니다."
            }
            
        except Exception as e:
            return {
                "action": "ask_question",
                "questions": ["증상의 지속 기간은 얼마나 되나요?", "통증의 강도는 어느 정도인가요?"],
                "reasoning": f"질문 생성 중 오류 발생: {str(e)}"
            }
    
    async def _handle_request_test(self, state: OrchestratorState) -> Dict[str, Any]:
        """검사 요청 처리"""
        
        print("🔬 검사 요청 처리 중...")
        
        # AI 기반 검사 추천
        try:
            test_prompt = f"""
환자의 증상과 병력을 바탕으로 필요한 의료 검사를 추천해주세요.

환자 정보:
- 나이: {state.patient_info.age}
- 성별: {state.patient_info.gender}
- 증상: {', '.join(state.patient_info.symptoms)}
- 과거 병력: {', '.join(state.patient_info.medical_history)}

토론 결과: {state.debate_rounds[-1].consensus if state.debate_rounds else '없음'}

필요한 검사들을 다음 형식으로 제시해주세요:
검사명: 설명 (비용, 긴급도)
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 검사 추천 전문가입니다."},
                    {"role": "user", "content": test_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            test_recommendations = response.choices[0].message.content.split('\n')
            
            # MedicalTest 객체로 변환
            proposed_tests = []
            for rec in test_recommendations:
                if ':' in rec:
                    parts = rec.split(':')
                    if len(parts) >= 2:
                        test_name = parts[0].strip()
                        description = parts[1].strip()
                        
                        # 기본값 설정
                        cost = 50000
                        urgency = "medium"
                        category = "other"
                        
                        # 비용 추출 (간단한 파싱)
                        if "원" in description:
                            try:
                                cost_str = description.split("원")[0].split()[-1]
                                cost = int(cost_str.replace(",", ""))
                            except:
                                pass
                        
                        test = MedicalTest(
                            test_name=test_name,
                            test_code=f"TEST_{len(proposed_tests)+1}",
                            description=description,
                            cost=cost,
                            urgency=urgency,
                            category=category
                        )
                        proposed_tests.append(test)
            
            state.proposed_tests = proposed_tests
            
            # 비용 분석
            cost_analysis = self.cost_analyzer.analyze_costs(
                proposed_tests, state.patient_info
            )
            state.cost_analysis = cost_analysis
            
            return {
                "action": "request_test",
                "proposed_tests": [test.dict() for test in proposed_tests],
                "cost_analysis": cost_analysis.dict(),
                "reasoning": "진단을 위해 추가 검사가 필요합니다."
            }
            
        except Exception as e:
            return {
                "action": "request_test",
                "proposed_tests": [],
                "reasoning": f"검사 추천 중 오류 발생: {str(e)}"
            }
    
    async def _handle_provide_diagnosis(self, state: OrchestratorState) -> Dict[str, Any]:
        """진단 제공 처리"""
        
        print("🏥 진단 제공 처리 중...")
        
        # AI 기반 진단 생성
        try:
            diagnosis_prompt = f"""
환자의 증상과 병력을 바탕으로 의료 진단을 제공해주세요.

환자 정보:
- 나이: {state.patient_info.age}
- 성별: {state.patient_info.gender}
- 증상: {', '.join(state.patient_info.symptoms)}
- 과거 병력: {', '.join(state.patient_info.medical_history)}
- 복용 약물: {', '.join(state.patient_info.current_medications)}

토론 결과: {state.debate_rounds[-1].consensus if state.debate_rounds else '없음'}

다음 형식으로 진단을 제공해주세요:
진단명: [진단명]
신뢰도: [0.0-1.0]
중증도: [mild/moderate/severe/critical]
근거: [진단 근거]
감별진단: [감별진단 목록]
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 진단 전문가입니다."},
                    {"role": "user", "content": diagnosis_prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            diagnosis_text = response.choices[0].message.content
            
            # 진단 파싱
            diagnosis = self._parse_diagnosis(diagnosis_text)
            state.proposed_diagnosis = diagnosis
            
            # 진단 확정
            diagnosis_confirmation = self.diagnosis_confirmer.confirm_diagnosis(
                diagnosis=diagnosis,
                patient_info=state.patient_info
            )
            state.diagnosis_confirmation = diagnosis_confirmation
            
            return {
                "action": "provide_diagnosis",
                "diagnosis": diagnosis.dict(),
                "confirmation": diagnosis_confirmation.dict(),
                "reasoning": "충분한 정보를 바탕으로 진단을 제공합니다."
            }
            
        except Exception as e:
            return {
                "action": "provide_diagnosis",
                "diagnosis": None,
                "reasoning": f"진단 생성 중 오류 발생: {str(e)}"
            }
    
    def _parse_diagnosis(self, diagnosis_text: str) -> Diagnosis:
        """진단 텍스트를 파싱하여 Diagnosis 객체로 변환"""
        
        try:
            lines = diagnosis_text.split('\n')
            condition = "미확인 진단"
            confidence = 0.5
            severity = "moderate"
            reasoning = "증상 분석 결과"
            differential_diagnoses = []
            
            for line in lines:
                line = line.strip()
                if line.startswith("진단명:"):
                    condition = line.replace("진단명:", "").strip()
                elif line.startswith("신뢰도:"):
                    try:
                        confidence = float(line.replace("신뢰도:", "").strip())
                    except:
                        confidence = 0.5
                elif line.startswith("중증도:"):
                    severity = line.replace("중증도:", "").strip()
                elif line.startswith("근거:"):
                    reasoning = line.replace("근거:", "").strip()
                elif line.startswith("감별진단:"):
                    diff_text = line.replace("감별진단:", "").strip()
                    differential_diagnoses = [d.strip() for d in diff_text.split(',') if d.strip()]
            
            return Diagnosis(
                condition=condition,
                confidence=confidence,
                severity=severity,
                reasoning=reasoning,
                differential_diagnoses=differential_diagnoses
            )
            
        except Exception as e:
            return Diagnosis(
                condition="진단 파싱 오류",
                confidence=0.0,
                severity="moderate",
                reasoning=f"진단 파싱 중 오류: {str(e)}",
                differential_diagnoses=[]
            )
    
    async def _make_final_decision(self, state: OrchestratorState) -> DecisionResult:
        """최종 결정 생성"""
        
        print("🎯 최종 결정 생성 중...")
        
        try:
            decision_prompt = f"""
다음 진단 프로세스의 결과를 바탕으로 최종 결정을 내려주세요.

현재 액션: {state.current_action.value if state.current_action else '없음'}
진단: {state.proposed_diagnosis.condition if state.proposed_diagnosis else '없음'}
비용: {state.cost_analysis.total_cost if state.cost_analysis else 0}원
진단 확정 신뢰도: {state.diagnosis_confirmation.confidence_level if state.diagnosis_confirmation else 0}

다음 중 하나로 결정해주세요:
1. "proceed" - 현재 결정을 진행
2. "reconsider" - 재검토 필요

결정: [proceed/reconsider]
근거: [결정 근거]
다음 단계: [구체적인 다음 단계]
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 의사결정 전문가입니다."},
                    {"role": "user", "content": decision_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            decision_text = response.choices[0].message.content
            
            # 결정 파싱
            decision = "proceed"  # 기본값
            reasoning = "진단 프로세스 완료"
            next_steps = ["환자에게 결과 전달"]
            
            if "reconsider" in decision_text.lower():
                decision = "reconsider"
                reasoning = "추가 검토가 필요합니다"
                next_steps = ["추가 정보 수집", "재평가"]
            
            return DecisionResult(
                action_taken=state.current_action or ActionType.ASK_QUESTION,
                decision=decision,
                reasoning=reasoning,
                next_steps=next_steps
            )
            
        except Exception as e:
            return DecisionResult(
                action_taken=state.current_action or ActionType.ASK_QUESTION,
                decision="proceed",
                reasoning=f"결정 생성 중 오류: {str(e)}",
                next_steps=["오류 복구"]
            )
    
    def get_session_state(self, session_id: str) -> Optional[OrchestratorState]:
        """세션 상태 조회"""
        return self.sessions.get(session_id)
    
    def list_sessions(self) -> List[str]:
        """모든 세션 ID 목록 반환"""
        return list(self.sessions.keys())
    
    def clear_session(self, session_id: str) -> bool:
        """세션 삭제"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False 