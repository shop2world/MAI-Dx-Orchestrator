from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime


class ActionType(str, Enum):
    """가능한 액션 타입들"""
    ASK_QUESTION = "ask_question"
    REQUEST_TEST = "request_test"
    PROVIDE_DIAGNOSIS = "provide_diagnosis"


class AgentRole(str, Enum):
    """에이전트 역할들"""
    HYPOTHESIS = "hypothesis"
    TEST_CHOOSER = "test_chooser"
    CHALLENGER = "challenger"
    STEWARDSHIP = "stewardship"
    CHECKLIST = "checklist"


class PatientInfo(BaseModel):
    """환자 정보 모델"""
    age: Optional[int] = None
    gender: Optional[str] = None
    symptoms: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    vital_signs: Optional[Dict[str, Any]] = None


class MedicalTest(BaseModel):
    """의료 검사 모델"""
    test_name: str
    test_code: str
    description: str
    cost: float
    urgency: str  # "low", "medium", "high", "emergency"
    category: str  # "blood", "imaging", "physical", "other"


class Diagnosis(BaseModel):
    """진단 결과 모델"""
    condition: str
    icd_code: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    differential_diagnoses: List[str] = Field(default_factory=list)
    reasoning: str
    severity: str  # "mild", "moderate", "severe", "critical"


class AgentResponse(BaseModel):
    """에이전트 응답 모델"""
    agent_role: AgentRole
    response: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommendations: List[str] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)


class DebateRound(BaseModel):
    """토론 라운드 모델"""
    round_number: int
    agent_responses: List[AgentResponse]
    consensus: Optional[str] = None
    disagreements: List[str] = Field(default_factory=list)


class CostAnalysis(BaseModel):
    """비용 분석 모델"""
    total_cost: float
    insurance_coverage: float
    patient_responsibility: float
    cost_breakdown: Dict[str, float]
    cost_effectiveness: str  # "low", "medium", "high"
    recommendations: List[str] = Field(default_factory=list)


class DiagnosisConfirmation(BaseModel):
    """진단 확정 모델"""
    confirmed_diagnosis: Diagnosis
    confirmation_methods: List[str]
    confidence_level: float = Field(ge=0.0, le=1.0)
    risk_factors: List[str] = Field(default_factory=list)
    follow_up_required: bool = False
    follow_up_plan: Optional[str] = None


class DecisionResult(BaseModel):
    """최종 결정 결과 모델"""
    action_taken: ActionType
    decision: str  # "proceed" or "reconsider"
    reasoning: str
    next_steps: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class SDBenchEvaluation(BaseModel):
    """SDBench 평가 결과 모델"""
    accuracy_score: float = Field(ge=0.0, le=1.0)
    cost_efficiency: float = Field(ge=0.0, le=1.0)
    safety_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    feedback: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    """오케스트레이터 상태 모델"""
    patient_info: PatientInfo
    current_action: Optional[ActionType] = None
    debate_rounds: List[DebateRound] = Field(default_factory=list)
    proposed_tests: List[MedicalTest] = Field(default_factory=list)
    proposed_diagnosis: Optional[Diagnosis] = None
    cost_analysis: Optional[CostAnalysis] = None
    diagnosis_confirmation: Optional[DiagnosisConfirmation] = None
    final_decision: Optional[DecisionResult] = None
    sdbench_evaluation: Optional[SDBenchEvaluation] = None
    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class SystemResponse(BaseModel):
    """시스템 응답 모델"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now) 