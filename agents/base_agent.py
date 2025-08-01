from abc import ABC, abstractmethod
from typing import Dict, Any, List
from models.medical_models import AgentResponse, AgentRole, PatientInfo
import openai
import os
from dotenv import load_dotenv

load_dotenv()

class BaseAgent(ABC):
    """모든 의료 에이전트의 기본 클래스"""
    
    def __init__(self, role: AgentRole, model_name: str = "gpt-4"):
        self.role = role
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    @abstractmethod
    def get_system_prompt(self) -> str:
        """에이전트별 시스템 프롬프트를 반환"""
        pass
    
    @abstractmethod
    def get_role_description(self) -> str:
        """에이전트의 역할 설명을 반환"""
        pass
    
    def analyze(self, patient_info: PatientInfo, context: Dict[str, Any] = None) -> AgentResponse:
        """환자 정보를 분석하고 응답을 생성"""
        
        # 시스템 프롬프트 구성
        system_prompt = self.get_system_prompt()
        
        # 사용자 프롬프트 구성
        user_prompt = self._build_user_prompt(patient_info, context)
        
        try:
            # OpenAI API 호출
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # 응답 파싱
            content = response.choices[0].message.content
            
            # AgentResponse 객체 생성
            return self._parse_response(content, patient_info)
            
        except Exception as e:
            # 에러 발생 시 기본 응답 생성
            return AgentResponse(
                agent_role=self.role,
                response=f"분석 중 오류가 발생했습니다: {str(e)}",
                confidence=0.0,
                reasoning="시스템 오류로 인해 분석을 완료할 수 없습니다.",
                recommendations=[],
                concerns=[f"시스템 오류: {str(e)}"]
            )
    
    def _build_user_prompt(self, patient_info: PatientInfo, context: Dict[str, Any] = None) -> str:
        """사용자 프롬프트를 구성"""
        
        prompt = f"""
환자 정보:
- 나이: {patient_info.age or '알 수 없음'}
- 성별: {patient_info.gender or '알 수 없음'}
- 증상: {', '.join(patient_info.symptoms) if patient_info.symptoms else '없음'}
- 과거 병력: {', '.join(patient_info.medical_history) if patient_info.medical_history else '없음'}
- 현재 복용 중인 약물: {', '.join(patient_info.current_medications) if patient_info.current_medications else '없음'}
- 활력징후: {patient_info.vital_signs or '알 수 없음'}

추가 컨텍스트: {context or '없음'}

위 정보를 바탕으로 {self.get_role_description()} 역할을 수행해주세요.
응답은 다음 형식으로 제공해주세요:

RESPONSE: [에이전트의 주요 응답]
CONFIDENCE: [0.0-1.0 사이의 신뢰도]
REASONING: [판단 근거]
RECOMMENDATIONS: [추천사항들, 쉼표로 구분]
CONCERNS: [우려사항들, 쉼표로 구분]
"""
        return prompt
    
    def _parse_response(self, content: str, patient_info: PatientInfo) -> AgentResponse:
        """API 응답을 파싱하여 AgentResponse 객체로 변환"""
        
        try:
            # 응답에서 각 섹션 추출
            lines = content.split('\n')
            response = ""
            confidence = 0.5
            reasoning = ""
            recommendations = []
            concerns = []
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith("RESPONSE:"):
                    current_section = "response"
                    response = line.replace("RESPONSE:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    current_section = "confidence"
                    try:
                        confidence = float(line.replace("CONFIDENCE:", "").strip())
                    except:
                        confidence = 0.5
                elif line.startswith("REASONING:"):
                    current_section = "reasoning"
                    reasoning = line.replace("REASONING:", "").strip()
                elif line.startswith("RECOMMENDATIONS:"):
                    current_section = "recommendations"
                    recs = line.replace("RECOMMENDATIONS:", "").strip()
                    recommendations = [r.strip() for r in recs.split(',') if r.strip()]
                elif line.startswith("CONCERNS:"):
                    current_section = "concerns"
                    cons = line.replace("CONCERNS:", "").strip()
                    concerns = [c.strip() for c in cons.split(',') if c.strip()]
                elif current_section:
                    # 멀티라인 섹션 처리
                    if current_section == "response":
                        response += " " + line
                    elif current_section == "reasoning":
                        reasoning += " " + line
            
            return AgentResponse(
                agent_role=self.role,
                response=response or "분석 완료",
                confidence=confidence,
                reasoning=reasoning or "제공된 정보를 바탕으로 분석했습니다.",
                recommendations=recommendations,
                concerns=concerns
            )
            
        except Exception as e:
            # 파싱 실패 시 기본 응답
            return AgentResponse(
                agent_role=self.role,
                response=content[:200] + "..." if len(content) > 200 else content,
                confidence=0.5,
                reasoning="응답 파싱 중 오류가 발생했습니다.",
                recommendations=[],
                concerns=[f"파싱 오류: {str(e)}"]
            ) 