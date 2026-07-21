"""
==============================================================================
 프로그램명 : schema
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   정제된 trip 레코드의 필드 타입·범위를 Pydantic v2 모델로 검증한다.
     1) TripRecord : tip_amount, fare_amount, trip_distance, payment_type,
                      passenger_count 등의 타입·범위 계약 정의
     2) validate_sample() : DataFrame에서 일부를 샘플링해 TripRecord로 검증

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (필드 뼈대만, 검증 로직은 팀원 작업 예정)
   v1.1  2026-07-21  장상민  validate_sample() 구현
==============================================================================
"""

import logging

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) TripRecord — 정제된 trip 레코드 스키마
# ---------------------------------------------------------------------------
class TripRecord(BaseModel):
    payment_type: int = Field(..., description="결제 수단 코드 (1 = 신용카드)")
    fare_amount: float = Field(..., gt=0, description="운행 요금")
    trip_distance: float = Field(..., gt=0, description="운행 거리")
    tip_amount: float = Field(..., ge=0, description="팁 금액")
    passenger_count: int = Field(..., ge=0, description="승객 수")


# ---------------------------------------------------------------------------
# 2) validate_sample
# ---------------------------------------------------------------------------
# WHY: 268만행 전체를 행 단위로 Pydantic 검증하면 느리다. 정제 로직(clean.py)이
# 맞게 동작했는지 확인하는 목적이므로, 무작위 샘플만 검증해도 충분하다.
def validate_sample(df: pd.DataFrame, n: int = 1000, seed: int = 42) -> dict:
    """df에서 n행을 무작위 샘플링해 TripRecord로 검증한다.

    반환값: {"n": 샘플 수, "valid": 통과 수, "invalid": 실패 수, "errors": 실패 상세 목록}
    """
    tag = "[validate_sample]"
    sample = df.sample(n=min(n, len(df)), random_state=seed)

    valid_count = 0
    errors: list[dict] = []
    for idx, row in sample.iterrows():
        try:
            TripRecord(
                payment_type=int(row["payment_type"]),
                fare_amount=float(row["fare_amount"]),
                trip_distance=float(row["trip_distance"]),
                tip_amount=float(row["tip_amount"]),
                passenger_count=int(row["passenger_count"]),
            )
            valid_count += 1
        except ValidationError as e:
            errors.append({"index": idx, "error": str(e)})

    result = {"n": len(sample), "valid": valid_count, "invalid": len(errors), "errors": errors}
    logger.info("%s n=%d valid=%d invalid=%d", tag, result["n"], result["valid"], result["invalid"])
    return result


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    from src.clean import add_time_features, add_tip_pct, filter_valid_trips
    from src.load import load_with_pandas

    print("\n=== 1단계 : 정제된 데이터로 스키마 검증 ===\n")
    df, _ = load_with_pandas()
    df = filter_valid_trips(df)
    df = add_tip_pct(df)
    df = add_time_features(df)

    result = validate_sample(df, n=2000)
    assert result["invalid"] == 0, f"Checkpoint : 스키마 검증 실패 {result['invalid']}건"

    print(f"\n\nCheckpoint : 샘플 {result['n']}건 전부 스키마 통과")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
