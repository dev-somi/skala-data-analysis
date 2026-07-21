"""
==============================================================================
 프로그램명 : clean
 작성자     : 광주 4반 박소미, 이다영
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   원본 택시 데이터의 결측치·중복·이상치를 처리하고,
   신용카드 결제 운행만 필터링하여 분석용 파생변수를 생성한다.

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 구조 작성
   v1.1  2026-07-21  이다영  데이터 정제 및 파생변수 생성 로직 구현
==============================================================================
"""

from __future__ import annotations

import pandas as pd

from config import (
    CREDIT_CARD_PAYMENT_TYPE,
    DATA_PATH,
    DISTANCE_BINS,
    DISTANCE_LABELS,
    END_DATE,
    FARE_BINS,
    FARE_LABELS,
    MAX_FARE_AMOUNT,
    MAX_PASSENGER_COUNT,
    MAX_TIP_RATE,
    MAX_TRIP_DISTANCE,
    MAX_TRIP_DURATION_MIN,
    MIN_FARE_AMOUNT,
    MIN_PASSENGER_COUNT,
    MIN_TIP_RATE,
    MIN_TRIP_DISTANCE,
    MIN_TRIP_DURATION_MIN,
    PASSENGER_BINS,
    PASSENGER_LABELS,
    PROCESSED_DATA_PATH,
    START_DATE,
    TABLE_DIR,
    TIME_ORDER,
)
from src.common import classify_time, ensure_directories, save_json


REQUIRED_COLUMNS = {
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
}


def drop_missing_and_duplicates(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """완전히 동일한 중복 행과 분석 필수 컬럼의 결측치를 제거한다."""
    quality_log = {
        "raw_rows": len(dataframe),
        "raw_columns": dataframe.shape[1],
        "duplicate_rows_before": int(dataframe.duplicated().sum()),
    }

    missing_summary = (
        dataframe.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(
            missing_rate=lambda frame: frame["missing_count"] / len(dataframe)
        )
        .reset_index(names="column")
    )
    missing_summary.to_csv(
        TABLE_DIR / "missing_values_before.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cleaned = dataframe.drop_duplicates().copy()
    quality_log["rows_after_duplicate_removal"] = len(cleaned)

    before = len(cleaned)
    cleaned = cleaned.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    quality_log["removed_missing_required"] = before - len(cleaned)

    return cleaned, quality_log


def filter_valid_trips(
    dataframe: pd.DataFrame,
    quality_log: dict[str, int | float | str],
) -> pd.DataFrame:
    """분석 월, 결제 방식 및 운행 범위 조건을 만족하는 행만 선택한다."""
    cleaned = dataframe.copy()

    before = len(cleaned)
    target_month_mask = cleaned["tpep_pickup_datetime"].between(
        START_DATE,
        END_DATE,
        inclusive="left",
    )
    cleaned = cleaned.loc[target_month_mask].copy()
    quality_log["removed_outside_target_month"] = before - len(cleaned)

    before = len(cleaned)
    cleaned = cleaned.loc[
        cleaned["payment_type"] == CREDIT_CARD_PAYMENT_TYPE
    ].copy()
    quality_log["removed_non_credit_card"] = before - len(cleaned)

    cleaned["trip_duration_min"] = (
        cleaned["tpep_dropoff_datetime"] - cleaned["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    before = len(cleaned)
    valid_mask = (
        cleaned["trip_distance"].between(
            MIN_TRIP_DISTANCE,
            MAX_TRIP_DISTANCE,
        )
        & cleaned["fare_amount"].between(
            MIN_FARE_AMOUNT,
            MAX_FARE_AMOUNT,
        )
        & (cleaned["tip_amount"] >= 0)
        & cleaned["passenger_count"].between(
            MIN_PASSENGER_COUNT,
            MAX_PASSENGER_COUNT,
        )
        & cleaned["trip_duration_min"].between(
            MIN_TRIP_DURATION_MIN,
            MAX_TRIP_DURATION_MIN,
        )
    )

    cleaned = cleaned.loc[valid_mask].copy()
    quality_log["removed_basic_outliers"] = before - len(cleaned)

    return cleaned


def add_tip_pct(
    dataframe: pd.DataFrame,
    quality_log: dict[str, int | float | str],
) -> pd.DataFrame:
    """팁 지급 여부, 팁 비율 및 분석용 범주형 파생변수를 생성한다."""
    cleaned = dataframe.copy()

    cleaned["tip_paid"] = (cleaned["tip_amount"] > 0).astype("int8")
    cleaned["tip_rate"] = cleaned["tip_amount"] / cleaned["fare_amount"]

    before = len(cleaned)
    cleaned = cleaned.loc[
        cleaned["tip_rate"].between(MIN_TIP_RATE, MAX_TIP_RATE)
    ].copy()
    quality_log["removed_tip_rate_outliers"] = before - len(cleaned)

    cleaned["tip_rate_pct"] = cleaned["tip_rate"] * 100

    cleaned["pre_tip_total"] = (
        cleaned["total_amount"] - cleaned["tip_amount"]
    )
    cleaned["tip_rate_total"] = (
        cleaned["tip_amount"]
        / cleaned["pre_tip_total"].where(cleaned["pre_tip_total"] > 0)
    )

    cleaned["pickup_hour"] = (
        cleaned["tpep_pickup_datetime"].dt.hour.astype("int8")
    )
    cleaned["time_group"] = cleaned["pickup_hour"].map(classify_time)
    cleaned["time_group"] = pd.Categorical(
        cleaned["time_group"],
        categories=TIME_ORDER,
        ordered=True,
    )

    cleaned["distance_group"] = pd.cut(
        cleaned["trip_distance"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        include_lowest=True,
    )
    cleaned["fare_group"] = pd.cut(
        cleaned["fare_amount"],
        bins=FARE_BINS,
        labels=FARE_LABELS,
        include_lowest=True,
    )
    cleaned["passenger_group"] = pd.cut(
        cleaned["passenger_count"],
        bins=PASSENGER_BINS,
        labels=PASSENGER_LABELS,
        include_lowest=True,
    )

    return cleaned


def clean_trip_data() -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """전체 전처리 파이프라인을 실행하고 분석용 데이터를 저장한다."""
    ensure_directories()

    dataframe = pd.read_parquet(DATA_PATH)

    missing_required = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_required:
        raise ValueError(f"필수 컬럼 누락: {sorted(missing_required)}")

    dataframe, quality_log = drop_missing_and_duplicates(dataframe)
    dataframe = filter_valid_trips(dataframe, quality_log)
    dataframe = add_tip_pct(dataframe, quality_log)

    quality_log["final_rows"] = len(dataframe)
    quality_log["final_columns"] = dataframe.shape[1]
    quality_log["final_tip_payment_rate"] = round(
        dataframe["tip_paid"].mean(),
        6,
    )

    dataframe.to_parquet(PROCESSED_DATA_PATH, index=False)
    save_json(quality_log, TABLE_DIR / "data_quality_log.json")

    final_missing = (
        dataframe.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(
            missing_rate=lambda frame: frame["missing_count"] / len(dataframe)
        )
        .reset_index(names="column")
    )
    final_missing.to_csv(
        TABLE_DIR / "missing_values_after.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return dataframe, quality_log


def main() -> None:
    """전체 데이터 정제 작업을 실행한다."""
    _, quality_log = clean_trip_data()

    print("\n[데이터 품질 처리 결과]")
    print(pd.Series(quality_log).to_string())
    print(f"\n전처리 데이터 저장: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()
