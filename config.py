"""프로젝트 공통 설정.

모든 팀원이 같은 전처리 기준과 구간 정의를 사용하도록 한곳에서 관리한다.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_credit_card_taxi.parquet"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
INTERACTIVE_DIR = OUTPUT_DIR / "interactive"
MODEL_DIR = PROJECT_ROOT / "models"

START_DATE = "2026-05-01"
END_DATE = "2026-06-01"

# 신용카드 결제 코드
CREDIT_CARD_PAYMENT_TYPE = 1

# 이상치 기준
MIN_TRIP_DISTANCE = 0.1
MAX_TRIP_DISTANCE = 30.0
MIN_FARE_AMOUNT = 3.0
MAX_FARE_AMOUNT = 150.0
MIN_TRIP_DURATION_MIN = 1.0
MAX_TRIP_DURATION_MIN = 180.0
MIN_PASSENGER_COUNT = 1
MAX_PASSENGER_COUNT = 6
MIN_TIP_RATE = 0.0
MAX_TIP_RATE = 1.0

# 분석 구간
TIME_ORDER = ["아침", "낮", "저녁", "심야"]

DISTANCE_BINS = [0, 1, 3, 5, 10, 30]
DISTANCE_LABELS = [
    "1마일 이하",
    "1~3마일",
    "3~5마일",
    "5~10마일",
    "10마일 초과",
]

FARE_BINS = [0, 10, 20, 30, 50, 150]
FARE_LABELS = [
    "$10 이하",
    "$10~20",
    "$20~30",
    "$30~50",
    "$50 초과",
]

PASSENGER_BINS = [0, 1, 2, 3, 6]
PASSENGER_LABELS = ["1명", "2명", "3명", "4명 이상"]

# 전체 데이터는 EDA에 사용하고, ML은 실행시간과 메모리 안정을 위해 표본을 사용할 수 있다.
MODEL_SAMPLE_SIZE = 300_000
RANDOM_STATE = 42
TEST_SIZE = 0.2
