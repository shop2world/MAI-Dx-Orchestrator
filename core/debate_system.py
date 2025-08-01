from typing import List, Dict, Any
from models.medical_models import (
    AgentResponse, DebateRound, PatientInfo, 
    AgentRole, ActionType, MedicalTest, Diagnosis
)
from agents.medical_agents import (
    DrHypothesisAgent, DrTestChooserAgent, DrChallengerAgent,
    DrStewardshipAgent, DrChecklistAgent
)
import asyncio
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class ChainOfDebate:
    """Chain of Debate 시스템 - 가상 의사들이 협의하는 시스템"""
    
    def __init__(self):
        self.agents = {
            AgentRole.HYPOTHESIS: DrHypothesisAgent(),
            AgentRole.TEST_CHOOSER: DrTestChooserAgent(),
            AgentRole.CHALLENGER: DrChallengerAgent(),
            AgentRole.STEWARDSHIP: DrStewardshipAgent(),
            AgentRole.CHECKLIST: DrChecklistAgent()
        }
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    async def conduct_debate(self, patient_info: PatientInfo, 
                           max_rounds: int = 3, 
                           context: Dict[str, Any] = None) -> List[DebateRound]:
        """토론을 진행하고 라운드별 결과를 반환"""
        
        debate_rounds = []
        current_context = context or {}
        
        for round_num in range(1, max_rounds + 1):
            print(f"🔍 토론 라운드 {round_num} 시작...")
            
            # 각 에이전트의 응답 수집
            agent_responses = await self._collect_agent_responses(
                patient_info, current_context, round_num
            )
            
            # 합의 도출
            consensus = await self._reach_consensus(agent_responses, round_num)
            
            # 불일치 사항 식별
            disagreements = self._identify_disagreements(agent_responses)
            
            # 라운드 결과 생성
            debate_round = DebateRound(
                round_number=round_num,
                agent_responses=agent_responses,
                consensus=consensus,
                disagreements=disagreements
            )
            
            debate_rounds.append(debate_round)
            
            # 다음 라운드를 위한 컨텍스트 업데이트
            current_context = self._update_context(current_context, debate_round)
            
            # 합의가 도출되면 조기 종료
            if consensus and "합의" in consensus:
                print(f"✅ 라운드 {round_num}에서 합의 도출")
                break
        
        return debate_rounds
    
    async def _collect_agent_responses(self, patient_info: PatientInfo, 
                                     context: Dict[str, Any], 
                                     round_num: int) -> List[AgentResponse]:
        """모든 에이전트의 응답을 수집"""
        
        responses = []
        
        # 각 에이전트가 순차적으로 응답
        for role, agent in self.agents.items():
            print(f"  🤖 {agent.get_role_description()} 분석 중...")
            
            # 이전 라운드의 정보를 컨텍스트에 추가
            round_context = {
                **context,
                "round_number": round_num,
                "previous_responses": [r.dict() for r in responses] if responses else []
            }
            
            response = agent.analyze(patient_info, round_context)
            responses.append(response)
            
            # API 호출 간격 조절
            await asyncio.sleep(0.5)
        
        return responses
    
    async def _reach_consensus(self, agent_responses: List[AgentResponse], 
                             round_num: int) -> str:
        """에이전트들의 응답을 바탕으로 합의 도출"""
        
        try:
            # 모든 에이전트의 응답을 요약
            responses_summary = "\n\n".join([
                f"{resp.agent_role.value}: {resp.response}" 
                for resp in agent_responses
            ])
            
            consensus_prompt = f"""
다음은 의료 진단을 위한 가상 의사들의 토론 라운드 {round_num} 결과입니다:

{responses_summary}

위 응답들을 종합하여 다음 중 하나로 합의를 도출해주세요:

1. "질문 필요" - 추가 정보가 필요함
2. "검사 요청" - 특정 검사가 필요함  
3. "진단 제공" - 충분한 정보로 진단 가능
4. "재검토 필요" - 더 많은 토론이 필요함

합의 결과: [결과]
합의 근거: [간단한 설명]
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 의료 토론의 합의 도출 전문가입니다."},
                    {"role": "user", "content": consensus_prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"합의 도출 중 오류 발생: {str(e)}"
    
    def _identify_disagreements(self, agent_responses: List[AgentResponse]) -> List[str]:
        """에이전트들 간의 불일치 사항을 식별"""
        
        disagreements = []
        
        # 신뢰도 차이가 큰 경우
        confidences = [resp.confidence for resp in agent_responses]
        if max(confidences) - min(confidences) > 0.3:
            disagreements.append("에이전트들 간 신뢰도 차이가 큽니다")
        
        # 우려사항이 있는 경우
        all_concerns = []
        for resp in agent_responses:
            all_concerns.extend(resp.concerns)
        
        if all_concerns:
            disagreements.append(f"우려사항 발견: {', '.join(all_concerns[:3])}")
        
        # 추천사항이 서로 다른 경우
        recommendations = [resp.recommendations for resp in agent_responses]
        if len(set(tuple(rec) for rec in recommendations)) > 1:
            disagreements.append("에이전트들의 추천사항이 서로 다릅니다")
        
        return disagreements
    
    def _update_context(self, current_context: Dict[str, Any], 
                       debate_round: DebateRound) -> Dict[str, Any]:
        """다음 라운드를 위한 컨텍스트 업데이트"""
        
        return {
            **current_context,
            "previous_round": debate_round.dict(),
            "consensus": debate_round.consensus,
            "disagreements": debate_round.disagreements
        }
    
    def extract_action_from_consensus(self, consensus: str) -> ActionType:
        """합의 결과에서 액션 타입을 추출"""
        
        consensus_lower = consensus.lower()
        
        if "질문" in consensus_lower or "추가 정보" in consensus_lower:
            return ActionType.ASK_QUESTION
        elif "검사" in consensus_lower or "테스트" in consensus_lower:
            return ActionType.REQUEST_TEST
        elif "진단" in consensus_lower:
            return ActionType.PROVIDE_DIAGNOSIS
        else:
            return ActionType.ASK_QUESTION  # 기본값 