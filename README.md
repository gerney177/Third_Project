# 🤖 LLM 기반 투자 조언 서비스

**RAG (Retrieval Augmented Generation)** 시스템을 통해 개인화된 투자 조언을 제공하는 서비스입니다.

## 📋 프로젝트 개요

### 🎯 주요 기능

- **🌐 외부 전략 분석**: 투자 전략 글 URL을 분석하여 지식베이스에 자동 추가
- **📝 지능형 텍스트 청킹**: 의미 단위로 텍스트를 분할하여 검색 효율성 향상
- **🔍 앙상블 검색**: BM25 (희소 검색) + Dense Vector (밀집 검색) 결합 (4:6 비율)
- **🧠 GPT-OSS 기반 LLM**: 고품질 투자 조언 생성
- **🌐 REST API**: 백엔드 시스템과의 원활한 연동
- **🔧 종합 테스트**: 전체 시스템 검증 및 성능 측정
- **📊 백엔드 데이터 연동**: 백엔드 개발자가 전달하는 금융 데이터 활용
- **🎯 통합 투자 조언**: 외부 전략 분석과 백엔드 데이터를 결합한 투자 조언 생성

### 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 투자 전략 글 URL   │ -> │   웹 인제스터     │ -> │     텍스트 청킹    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐                                    │
│   백엔드 데이터   │ ──────────────────────────────────┼─┐
│ (금융 데이터)    │                                    │ │
└─────────────────┘                                    │ │
        │                                               │ │
        ▼                                               ▼ │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   투자 조언       │ <- │   GPT-OSS LLM   │ <- │  벡터 데이터베이스   │
│ (백엔드 데이터   │    │ (전략 + 데이터    │    │ (ChromaDB+BM25)    │
│  + 전략 분석)    │    │  통합 분석)       │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                                               │
        ▼                                               │
┌─────────────────┐    ┌─────────────────┐              │
│   Flask API     │ -> │   사용자 쿼리      │ -----------> │
│    서버          │    │ + 전략 URL        │   검색        │
└─────────────────┘    └─────────────────┘              │
```

## 🛠️ 기술 스택

### 핵심 라이브러리
- **LangChain**: RAG 시스템 구현
- **ChromaDB**: 벡터 데이터베이스
- **GPT-OSS**: 대규모 언어 모델
- **Sentence Transformers**: 다국어 임베딩
- **BM25**: 희소 검색 알고리즘

### 서버 & API
- **Flask**: REST API 서버
- **Flask-CORS**: CORS 지원
- **Requests**: HTTP 클라이언트

### 데이터 처리
- **NLTK**: 자연어 처리
- **tiktoken**: 토큰 계산

### 투자 조언 시스템
- **전략 분석**: 외부 투자 전략 글 자동 분석
- **백엔드 연동**: 백엔드 개발자가 전달하는 금융 데이터 활용
- **RAG 시스템**: 벡터 검색과 LLM을 통한 지능형 조언 생성

## 📦 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. NLTK 데이터 다운로드

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 3. Hugging Face 토큰 설정 (GPT-OSS 사용시)

```bash
export HF_TOKEN="your_huggingface_token"
```

### 4. 백엔드 연동 설정

현재 시스템은 백엔드 개발자가 전달하는 금융 데이터를 활용하며, 투자 전략 분석에 집중합니다.

## 🚀 사용 방법

### 1. API 서버 시작

```bash
# 기본 실행 (CPU 모드)
python api_server.py

# GPU 모드 (권장)
python api_server.py --enable-gpu

# 커스텀 설정
python api_server.py --host 0.0.0.0 --port 8000 --debug
```

### 2. 서비스 테스트

```bash
# 전체 테스트 실행
python test_service.py

# 서버 상태 확인만
python test_service.py --check-server

# 테스트 결과 저장
python test_service.py --save-results
```

### 3. 개별 컴포넌트 테스트

```bash
# 텍스트 청킹 테스트
python text_chunker.py

# 벡터 데이터베이스 테스트
python vector_database.py

# LLM 서비스 테스트
python llm_service.py

# 종합 서비스 테스트
python investment_advisor_service.py
```

## 🌐 API 엔드포인트

### 📊 투자 조언 요청

```bash
curl -X POST http://localhost:5000/api/get-advice \
  -H "Content-Type: application/json" \
  -d '{
    "stock_symbol": "005930",
    "quantity": 10,
    "price": 70000,
    "strategy": "https://example.com/investment-strategy-article"
    # 외부 투자 전략 글 URL을 제공하면 자동으로 분석하여 조언에 활용됩니다.
  }'
```

**응답 예시:**
```json
{
  "advice": "상세한 투자 분석 내용...",
  "investment_summary": {
    "stock_symbol": "005930",
    "quantity": 10,
    "price_per_share": 70000,
    "total_investment": 700000,
    "currency": "KRW"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 📊 투자 전략 글 분석

```bash
curl -X POST http://localhost:5000/api/analyze-strategy \
  -H "Content-Type: application/json" \
  -d '{
    "article_url": "https://gall.dcinside.com/mgallery/board/view/?id=chartanalysis&no=2023207"
  }'
```

**응답 예시:**
```json
{
  "success": true,
  "article_info": {
    "title": "차트 분석 게시글",
    "url": "https://gall.dcinside.com/mgallery/board/view/?id=chartanalysis&no=2023207",
    "content_length": 2500,
    "chunks_created": 3
  },
  "processing_stats": {
    "total_chunks": 3,
    "avg_chunk_size": 833
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 🔍 지식베이스 검색

```bash
curl -X POST http://localhost:5000/api/search-knowledge \
  -H "Content-Type: application/json" \
  -d '{
    "query": "삼성전자 투자 전략",
    "top_k": 5,
    "search_type": "ensemble"
  }'
```

### 📈 시스템 정보 조회

```bash
# 헬스 체크
curl http://localhost:5000/api/health

# 서비스 정보
curl http://localhost:5000/api/service-info

# 지식베이스 통계
curl http://localhost:5000/api/knowledge-stats
```

### 📊 투자 조언 시스템

현재 시스템은 외부 투자 전략 글 분석에 집중하며, 백엔드 개발자가 전달하는 금융 데이터를 활용합니다.

## 📁 프로젝트 구조

```
investment-advisor-service/
├── 📄 README.md                        # 프로젝트 문서 (한글)
├── 📄 requirements.txt                 # 의존성 목록
├── 📝 text_chunker.py                 # 텍스트 청킹 서비스
├── 🔍 vector_database.py              # 벡터 DB (앙상블 검색)
├── 🧠 llm_service.py                  # GPT-OSS LLM 서비스
├── 🎯 investment_advisor_service.py   # 메인 통합 서비스
├── 🌐 api_server.py                   # Flask API 서버
├── 🧪 test_service.py                 # 종합 테스트 스크립트
└── 📂 chroma_db/                      # ChromaDB 데이터 저장소
    └── ...
```

## 📚 투자 전략 분석 파이프라인

- **웹 인제스터**: `web_ingestor.py`가 HTML을 파싱하여 본문 텍스트 추출
- **전략 분석**: `investment_advisor_service.py`의 `analyze_strategy_article`이 전략 글을 분석
- **지식베이스 통합**: 추출된 내용을 청킹하여 벡터 DB에 저장
- **사용 API**: `POST /api/analyze-strategy` (입력: `{"article_url": "..."}`)

### 사용 예시
```bash
# 1. 투자 전략 글 분석
curl -X POST http://localhost:5000/api/analyze-strategy \
  -H "Content-Type: application/json" \
  -d '{"article_url": "https://example.com/strategy"}'

# 2. 분석된 전략으로 투자 조언 요청
curl -X POST http://localhost:5000/api/get-advice \
  -H "Content-Type: application/json" \
  -d '{
    "stock_symbol": "005930",
    "quantity": 10,
    "price": 70000,
    "strategy": "https://example.com/strategy"
  }'
```

## ⚙️ 주요 설정

### 앙상블 검색 가중치
- **BM25 (희소 검색)**: 40%
- **Dense Vector (밀집 검색)**: 60%

### 텍스트 청킹 설정
- **청크 크기**: 1000 토큰
- **오버랩**: 200 토큰
- **언어 지원**: 한국어, 영어

### LLM 설정
- **모델**: `openai/gpt-oss-20b`
- **최대 토큰**: 1000
- **Temperature**: 0.7
- **추론 강도**: high/medium/low

## 🔧 백엔드 연동 가이드

### 팀 프로젝트 통합 예시

```python
import requests

# 투자 조언 요청
def get_investment_advice(stock_symbol, quantity, price, strategy):
    response = requests.post(
        "http://localhost:5000/api/get-advice",
        json={
            "stock_symbol": stock_symbol,
            "quantity": quantity,
            "price": price,
            "strategy": strategy
        }
    )
    return response.json()

# 사용 예시
advice = get_investment_advice(
    stock_symbol="005930",
    quantity=10,
    price=70000,
    strategy="https://youtube.com/watch?v=investment_analysis"
)
print(advice["advice"])
```

### 환경별 설정

```bash
# 개발 환경
python api_server.py --debug

# 운영 환경
python api_server.py --host 0.0.0.0 --port 8080

# Docker 환경
python api_server.py --host 0.0.0.0 --port 5000
```

## 🧪 테스트 및 검증

### 컴포넌트 테스트
- ✅ 텍스트 청킹 (의미 단위 분할)
- ✅ 벡터 데이터베이스 (앙상블 검색)
- ✅ LLM 서비스 (투자 조언 생성)
- ✅ API 서버 (REST 엔드포인트)

### 통합 테스트
- ✅ 전체 투자 조언 플로우
- ✅ 동시 요청 처리
- ✅ 성능 측정
- ✅ 오류 처리

### 성능 지표
- **응답 시간**: < 1초 (일반 요청)
- **조언 생성**: < 30초 (GPU 환경)
- **동시 처리**: 최대 10개 요청

## 🚨 주의사항

### GPU 요구사항
- **GPT-OSS 모델**: NVIDIA GPU 권장 (최소 8GB VRAM)
- **CPU 모드**: 제한된 기능으로 작동 (Mock 응답)

### API 사용 제한
- **ChromaDB**: 로컬 저장소 용량 고려
- **동시 요청**: 서버 리소스에 따른 제한

### 보안 고려사항
- **API 인증**: 운영 환경에서 토큰 기반 인증 추가 권장
- **CORS 설정**: 특정 도메인으로 제한 권장
- **입력 검증**: SQL 인젝션 등 공격 방어

## 🔮 향후 개선 계획

### 기능 확장
- [x] 백엔드 연동
- [x] 외부 전략 분석
- [ ] 뉴스 기사 자동 수집 및 분석
- [ ] 포트폴리오 관리 기능
- [ ] 리스크 평가 모델 고도화

### 성능 최적화
- [ ] 모델 양자화를 통한 추론 속도 향상
- [ ] 캐싱 시스템 도입
- [ ] 비동기 처리 개선
- [ ] 분산 처리 아키텍처

### 사용성 개선
- [ ] 웹 대시보드 구축
- [ ] 모바일 앱 지원
- [ ] 다국어 인터페이스
- [ ] 사용자 피드백 시스템

## 📞 문의 및 지원

프로젝트 관련 문의나 이슈는 다음을 통해 연락주세요:

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **팀 연락처**: 팀 프로젝트 관련 협업 문의

## 📄 라이선스

이 프로젝트는 팀 프로젝트용으로 개발되었습니다.

---

**⚡ 빠른 시작 가이드**

1. `pip install -r requirements.txt` - 의존성 설치
2. `python api_server.py` - 서버 시작
3. `python test_service.py --check-server` - 서버 확인
4. API 엔드포인트로 투자 조언 요청

**🎯 핵심 특징: RAG 시스템 + 앙상블 검색 + GPT-OSS로 구현된 차세대 투자 조언 서비스**
