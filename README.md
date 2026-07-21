# skala-data-analysis

NYC Yellow Taxi 데이터를 활용한 End2End 데이터 분석 프로젝트 (Day2 종합 실습).

## 분석 주제
신용카드 결제 승객은 어떤 운행 조건에서 팁을 더 많이 주는가?
(시간대, 거리, 요금, 승객 수에 따른 팁 지급률·평균 팁 비율 분석)

- 타겟 변수: `tip_pct = tip_amount / fare_amount`
- 필터 조건: `payment_type == 1`, `fare_amount > 0`, `trip_distance > 0`

## 실습 범위
1. **데이터 준비**: Pandas / Polars 두 방식으로 각각 로딩·비교, 결측치·중복 처리, EDA
2. **시각화**: Seaborn 정적 차트 1개 이상 + Plotly 인터랙티브 차트 1개 이상 (제목·축 레이블 필수)
3. **통계 분석**: 기술통계(평균·표준편차·분위수), 상관계수, `scipy.stats.ttest_ind` 기반 t-test 및 p-value 해석
4. **ML Pipeline**: `sklearn.pipeline.Pipeline`으로 전처리+모델 학습, 평가 지표 출력, `joblib` 모델 저장
5. **자동화**: 분석 결과를 `reports/report.md`로 자동 생성

## 데이터 출처
- NYC Yellow Taxi 2026-05 Parquet: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet
- `data/raw/`는 git에 커밋하지 않는다. `src/download.py`를 실행해 각자 로컬에 받는다.

```bash
python src/download.py
```

## 폴더 구조
```
skala-data-analysis/
├── data/
│   ├── raw/            # 원본 parquet (git 미추적)
│   ├── processed/      # 전처리 완료 데이터 (git 미추적)
│   └── external/       # 외부 참조 데이터
├── notebooks/
│   ├── 01_eda.ipynb              # Pandas/Polars 로딩 비교 + EDA
│   ├── 02_visualization.ipynb    # Seaborn + Plotly 시각화
│   └── 03_stats_and_ml.ipynb     # 기술통계·t-test + ML Pipeline
├── src/                 # 노트북에서 검증된 로직을 옮겨 담는 재사용 모듈
│   ├── download.py       # 데이터 다운로드
│   ├── load.py            # Pandas/Polars 양쪽 로딩 및 비교
│   ├── clean.py           # 결측치·중복 처리, 필터링
│   ├── schema.py          # Pydantic v2 스키마 검증
│   ├── viz.py              # 시각화 함수
│   ├── stats.py            # 기술통계·상관계수·t-test
│   ├── ml.py                # sklearn Pipeline·모델 저장
│   └── report.py            # report.md 자동 생성
├── models/               # joblib으로 저장된 모델 (git 미추적)
├── reports/
│   └── report.md          # 자동 생성 리포트
└── tests/                 # pytest 테스트
```

## 환경 설정
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements-dev.txt   # 런타임 + 개발 도구(pytest, ruff, pre-commit) 포함
# 런타임 패키지만 필요하면: pip install -r requirements.txt
```

### pre-commit 훅 설치
```bash
pre-commit install
```
커밋 시 `ruff`(lint + format)가 자동으로 실행된다.

### 테스트 / 린트
```bash
pytest
ruff check .
```

## 협업 규칙
- 탐색은 `notebooks/`, 검증된 재사용 로직은 `src/`로 이동 — 두 역할을 섞지 않는다.
- `data/raw/`, `data/processed/`, `models/`는 git에 올리지 않는다 (`.gitignore` 참고).
- 처음부터 완벽한 구조를 만들 필요 없음 — 분석이 반복되면 함수화하고, 함수가 쌓이면 모듈화한다.
