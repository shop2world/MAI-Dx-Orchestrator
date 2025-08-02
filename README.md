# MAI-Dx Orchestrator

Shop2world의 MAI-Dx Orchestrator 시스템을 구현한 의료 진단 AI 시스템입니다.

## 📺 Demo Video

[![Watch the video](https://img.youtube.com/vi/영상ID/0.jpg)](https://www.youtube.com/watch?v=영상ID)


## 🏥 시스템 개요

이 시스템은 다음과 같은 구성 요소로 이루어져 있습니다:

### Virtual Doctor Panel (가상 의사 패널)
- **Dr. Hypothesis Agent**: 진단 가설 생성
- **Dr. Test Chooser Agent**: 적절한 검사 선택
- **Dr. Challenger Agent**: 가설과 결정에 대한 도전적 검토
- **Dr. Stewardship Agent**: 자원 관리 및 윤리적 고려사항
- **Dr. Checklist Agent**: 필수 절차 및 기준 확인

### 의사결정 파이프라인
1. **Chain of Debate**: 에이전트들 간의 협의 과정
2. **Action Selection**: 질문, 검사 요청, 진단 제공 중 선택
3. **Cost Analysis**: 비용 분석
4. **Diagnosis Confirmation**: 진단 확정
5. **Decision to Proceed**: 최종 실행 결정

### SDBench Framework
- 의료 진단의 정확도 및 비용효율성 평가
- 피드백 루프를 통한 시스템 개선

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:
```
OPENAI_API_KEY=your_api_key_here
```

### 3. 실행
```bash
# Streamlit 웹 인터페이스 실행
streamlit run app.py

# 또는 FastAPI 서버 실행
uvicorn api:app --reload
```

## 📁 프로젝트 구조

```
MAI-Dx-Orchestrator/
├── agents/           # AI 에이전트들
├── core/            # 핵심 시스템 로직
├── models/          # 데이터 모델
├── utils/           # 유틸리티 함수
├── api.py           # FastAPI 서버
├── app.py           # Streamlit 웹 앱
├── main.py          # 메인 실행 파일
└── requirements.txt # 의존성 패키지
```

## 🎯 주요 기능

- **멀티 에이전트 협업**: 5명의 가상 의사가 Chain of Debate 방식으로 협의
- **의료 진단**: 증상 기반 진단 및 검사 추천
- **비용 분석**: 의료 비용 및 보험 적용 분석
- **진단 확정**: 다중 검증을 통한 진단 정확도 향상
- **평가 시스템**: SDBench 기반 성능 평가

## 🔧 기술 스택

- **AI/ML**: OpenAI GPT-4, LangChain, LangGraph
- **멀티 에이전트**: CrewAI, AutoGen
- **웹 프레임워크**: FastAPI, Streamlit
- **데이터 처리**: Pandas, NumPy
- **시각화**: Matplotlib, Seaborn

## 📝 사용 예시

```python
from core.orchestrator import MAIDxOrchestrator

# 시스템 초기화
orchestrator = MAIDxOrchestrator()

# 환자 증상 입력
symptoms = "발열, 기침, 피로감이 3일간 지속됨"

# 진단 프로세스 실행
result = orchestrator.diagnose(symptoms)
print(result)
```

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 
