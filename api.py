from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import uuid
from datetime import datetime

from models.medical_models import PatientInfo, SystemResponse
from core.orchestrator import MAIDxOrchestrator

# FastAPI 앱 생성
app = FastAPI(
    title="MAI-Dx Orchestrator API",
    description="Shop2world AI-powered Medical Diagnosis System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 오케스트레이터 인스턴스
orchestrator = MAIDxOrchestrator()

# API 모델들
class PatientInfoRequest(BaseModel):
    age: Optional[int] = None
    gender: Optional[str] = None
    symptoms: List[str]
    medical_history: List[str] = []
    current_medications: List[str] = []
    vital_signs: Optional[Dict[str, Any]] = None

class DiagnosisRequest(BaseModel):
    session_id: str

class SessionResponse(BaseModel):
    session_id: str
    message: str
    timestamp: datetime

class DiagnosisResponse(BaseModel):
    success: bool
    message: str
    session_id: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime

# 헬스체크 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "MAI-Dx Orchestrator API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_sessions": len(orchestrator.sessions),
        "timestamp": datetime.now()
    }

# 세션 관리 엔드포인트
@app.post("/sessions", response_model=SessionResponse)
async def create_session(patient_info: PatientInfoRequest):
    """새로운 진단 세션 생성"""
    
    try:
        # PatientInfo 객체 생성
        patient = PatientInfo(
            age=patient_info.age,
            gender=patient_info.gender,
            symptoms=patient_info.symptoms,
            medical_history=patient_info.medical_history,
            current_medications=patient_info.current_medications,
            vital_signs=patient_info.vital_signs
        )
        
        # 세션 시작
        session_id = await orchestrator.start_diagnosis_session(patient)
        
        return SessionResponse(
            session_id=session_id,
            message="진단 세션이 성공적으로 생성되었습니다",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"세션 생성 중 오류: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """모든 세션 목록 조회"""
    
    sessions = []
    for session_id in orchestrator.list_sessions():
        state = orchestrator.get_session_state(session_id)
        if state:
            sessions.append({
                "session_id": session_id,
                "created_at": state.created_at,
                "updated_at": state.updated_at,
                "patient_age": state.patient_info.age,
                "patient_gender": state.patient_info.gender,
                "symptoms": state.patient_info.symptoms,
                "has_diagnosis": state.proposed_diagnosis is not None
            })
    
    return {
        "sessions": sessions,
        "total_count": len(sessions),
        "timestamp": datetime.now()
    }

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """특정 세션 정보 조회"""
    
    state = orchestrator.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return {
        "session_id": session_id,
        "state": state.dict(),
        "timestamp": datetime.now()
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """세션 삭제"""
    
    success = orchestrator.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return {
        "message": "세션이 성공적으로 삭제되었습니다",
        "session_id": session_id,
        "timestamp": datetime.now()
    }

# 진단 프로세스 엔드포인트
@app.post("/diagnose", response_model=DiagnosisResponse)
async def start_diagnosis(request: DiagnosisRequest):
    """진단 프로세스 시작"""
    
    try:
        # 진단 실행
        result = await orchestrator.process_diagnosis(request.session_id)
        
        return DiagnosisResponse(
            success=result.success,
            message=result.message,
            session_id=result.session_id,
            data=result.data,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"진단 프로세스 중 오류: {str(e)}")

@app.post("/diagnose/async")
async def start_diagnosis_async(request: DiagnosisRequest, background_tasks: BackgroundTasks):
    """비동기 진단 프로세스 시작"""
    
    # 세션 존재 확인
    state = orchestrator.get_session_state(request.session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    # 백그라운드에서 진단 실행
    background_tasks.add_task(run_diagnosis_background, request.session_id)
    
    return {
        "message": "진단 프로세스가 백그라운드에서 시작되었습니다",
        "session_id": request.session_id,
        "status": "processing",
        "timestamp": datetime.now()
    }

async def run_diagnosis_background(session_id: str):
    """백그라운드에서 진단 실행"""
    
    try:
        await orchestrator.process_diagnosis(session_id)
        print(f"백그라운드 진단 완료: {session_id}")
    except Exception as e:
        print(f"백그라운드 진단 오류: {session_id} - {str(e)}")

# 특정 결과 조회 엔드포인트
@app.get("/sessions/{session_id}/diagnosis")
async def get_diagnosis_result(session_id: str):
    """진단 결과 조회"""
    
    state = orchestrator.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if not state.proposed_diagnosis:
        return {
            "message": "아직 진단이 완료되지 않았습니다",
            "session_id": session_id,
            "has_diagnosis": False,
            "timestamp": datetime.now()
        }
    
    return {
        "session_id": session_id,
        "has_diagnosis": True,
        "diagnosis": state.proposed_diagnosis.dict() if state.proposed_diagnosis else None,
        "confirmation": state.diagnosis_confirmation.dict() if state.diagnosis_confirmation else None,
        "cost_analysis": state.cost_analysis.dict() if state.cost_analysis else None,
        "final_decision": state.final_decision.dict() if state.final_decision else None,
        "sdbench_evaluation": state.sdbench_evaluation.dict() if state.sdbench_evaluation else None,
        "timestamp": datetime.now()
    }

@app.get("/sessions/{session_id}/debate")
async def get_debate_process(session_id: str):
    """토론 과정 조회"""
    
    state = orchestrator.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    return {
        "session_id": session_id,
        "debate_rounds": [round_data.dict() for round_data in state.debate_rounds],
        "current_action": state.current_action.value if state.current_action else None,
        "timestamp": datetime.now()
    }

@app.get("/sessions/{session_id}/cost-analysis")
async def get_cost_analysis(session_id: str):
    """비용 분석 조회"""
    
    state = orchestrator.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if not state.cost_analysis:
        return {
            "message": "비용 분석이 없습니다",
            "session_id": session_id,
            "has_cost_analysis": False,
            "timestamp": datetime.now()
        }
    
    return {
        "session_id": session_id,
        "has_cost_analysis": True,
        "cost_analysis": state.cost_analysis.dict(),
        "proposed_tests": [test.dict() for test in state.proposed_tests],
        "timestamp": datetime.now()
    }

@app.get("/sessions/{session_id}/sdbench")
async def get_sdbench_evaluation(session_id: str):
    """SDBench 평가 결과 조회"""
    
    state = orchestrator.get_session_state(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다")
    
    if not state.sdbench_evaluation:
        return {
            "message": "SDBench 평가가 없습니다",
            "session_id": session_id,
            "has_evaluation": False,
            "timestamp": datetime.now()
        }
    
    return {
        "session_id": session_id,
        "has_evaluation": True,
        "sdbench_evaluation": state.sdbench_evaluation.dict(),
        "timestamp": datetime.now()
    }

# 시스템 관리 엔드포인트
@app.get("/system/status")
async def get_system_status():
    """시스템 상태 조회"""
    
    return {
        "active_sessions": len(orchestrator.sessions),
        "system_health": "healthy",
        "orchestrator_initialized": orchestrator is not None,
        "timestamp": datetime.now()
    }

@app.post("/system/clear-all")
async def clear_all_sessions():
    """모든 세션 삭제"""
    
    session_count = len(orchestrator.sessions)
    orchestrator.sessions.clear()
    
    return {
        "message": f"{session_count}개의 세션이 모두 삭제되었습니다",
        "cleared_sessions": session_count,
        "timestamp": datetime.now()
    }

# 에러 핸들러
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "요청한 리소스를 찾을 수 없습니다",
            "timestamp": datetime.now()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버 내부 오류가 발생했습니다",
            "timestamp": datetime.now()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 