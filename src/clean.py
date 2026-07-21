"""
==============================================================================
 프로그램명 : clean
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   load.py로 로딩한 데이터의 결측치·중복 처리, 분석 대상 필터링, 파생변수
   생성을 담당한다.
     1) drop_missing_and_duplicates() : 결측치·완전중복행 제거
     2) filter_valid_trips() : payment_type==1, fare_amount>0, trip_distance>0
        + 이동시간/팁비율 이상치 제거
     3) add_tip_pct() : tip_pct = tip_amount / fare_amount, has_tip, high_tip 컬럼 추가
     4) add_time_features() : pickup_hour, is_weekend, is_congestion 등 시간·혼잡구역 파생변수
        (분석 주제가 "시간대·혼잡구역별 팁 패턴"이라 EDA/통계/ML 전 단계에서 필요해 추가)

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  4개 함수 로직 구현 (add_time_features는 분석 중 필요해 추가)
==============================================================================
"""

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) drop_missing_and_duplicates
# ---------------------------------------------------------------------------
# WHY: 결측치는 passenger_count/RatecodeID/congestion_surcharge/Airport_fee/
# store_and_fwd_flag 5개 컬럼에서만 나타나는데, payment_type==0(미분류 결제)에
# 100% 쏠려있고 신용카드(payment_type==1)에는 0%다. 그래서 filter_valid_trips()로
# 신용카드만 남기면 이 결측치는 자연히 해소된다 — 이 함수는 그 사실을 검증하는
# 용도(및 완전중복행 제거)로 쓴다.
def drop_missing_and_duplicates(
    df: pd.DataFrame, subset_cols: list[str] | None = None
) -> pd.DataFrame:
    """완전 중복행과 subset_cols(기본: 전체 컬럼)의 결측치가 있는 행을 제거한다."""
    tag = "[drop_missing_and_duplicates]"
    before = len(df)

    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates()

    df = df.dropna(subset=subset_cols)

    logger.info(
        "%s 중복 %d행 제거, 결측치 포함 행 제거 후 %s -> %s",
        tag, n_dupes, f"{before:,}", f"{len(df):,}",
    )
    return df


# ---------------------------------------------------------------------------
# 2) filter_valid_trips
# ---------------------------------------------------------------------------
# WHY: README에 명시된 필터(payment_type==1, fare_amount>0, trip_distance>0)에
# 더해, 이동시간이 0 이하이거나 3시간(180분) 이상인 트립도 GPS/미터 오류로 보고
# 제거한다. 안 그러면 tip_pct(=tip/fare) 계산 시 분모가 비정상적으로 작은 트립이
# 표준편차를 왜곡한다 (add_tip_pct에서 별도로 한 번 더 다룸).
def filter_valid_trips(df: pd.DataFrame) -> pd.DataFrame:
    """신용카드 결제, 정상 요금·거리·이동시간 범위의 트립만 남긴다."""
    tag = "[filter_valid_trips]"
    before = len(df)

    df = df[df["payment_type"] == 1].copy()
    logger.info("%s 신용카드 결제 필터링 : %s -> %s행", tag, f"{before:,}", f"{len(df):,}")

    pickup_to_dropoff = df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    trip_duration_min = pickup_to_dropoff.dt.total_seconds() / 60
    before_outlier = len(df)
    df = df[
        (df["fare_amount"] > 0)
        & (df["trip_distance"] > 0)
        & (trip_duration_min > 0)
        & (trip_duration_min < 180)
    ]
    logger.info(
        "%s 요금/거리/이동시간 이상치 제거 : %s -> %s행",
        tag, f"{before_outlier:,}", f"{len(df):,}",
    )
    return df


# ---------------------------------------------------------------------------
# 3) add_tip_pct
# ---------------------------------------------------------------------------
# WHY: tip_pct(=tip_amount/fare_amount)가 100%를 넘는 행(초저요금 트립 등)이
# 0.17% 있는데, 이게 표준편차를 심하게 왜곡해 통계 검정 효과크기를 실제보다
# 작게 보이게 만든다는 걸 확인했다. 그래서 100% 초과는 이상치로 제거한다.
def add_tip_pct(df: pd.DataFrame) -> pd.DataFrame:
    """tip_pct, has_tip(팁 지급 여부), high_tip(팁비율 20% 이상) 컬럼을 추가한다."""
    tag = "[add_tip_pct]"
    df = df.copy()
    df["tip_pct"] = df["tip_amount"] / df["fare_amount"]

    before = len(df)
    df = df[df["tip_pct"] <= 1.0]
    logger.info("%s 팁비율>100%% 이상치 제거 : %s -> %s행", tag, f"{before:,}", f"{len(df):,}")

    df["has_tip"] = df["tip_amount"] > 0
    df["high_tip"] = df["tip_pct"] >= 0.2
    return df


# ---------------------------------------------------------------------------
# 4) add_time_features
# ---------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """pickup_hour, is_weekend, is_night, trip_duration_min, is_congestion 파생변수를 추가한다."""
    df = df.copy()
    df["trip_duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["is_weekend"] = df["tpep_pickup_datetime"].dt.dayofweek >= 5
    df["is_night"] = df["pickup_hour"].between(0, 5)
    df["is_congestion"] = df["congestion_surcharge"] > 0
    logger.info("[add_time_features] pickup_hour/is_weekend/is_night/is_congestion 추가 완료")
    return df


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    from src.load import load_with_pandas

    print("\n=== 1단계 : 원본 로딩 ===\n")
    df, _ = load_with_pandas()

    print("\n=== 2단계 : 필터링 + 파생변수 ===\n")
    df = filter_valid_trips(df)
    df = drop_missing_and_duplicates(df, subset_cols=["passenger_count", "RatecodeID"])
    df = add_tip_pct(df)
    df = add_time_features(df)

    assert df["payment_type"].nunique() == 1, "Checkpoint : 신용카드 결제만 남아있어야 한다"
    assert df.isnull().sum().sum() == 0, "Checkpoint : 결측치가 남아있으면 안 된다"

    print(f"\n\nCheckpoint : 정제 완료, 최종 {len(df):,}행")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
