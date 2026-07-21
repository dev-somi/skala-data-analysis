"""3. 기술통계, 상관계수, Welch t-test 및 p-value 해석."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from config import PROCESSED_DATA_PATH, TABLE_DIR
from src.common import ensure_directories, save_json


NUMERIC_COLUMNS = [
    "trip_distance",
    "fare_amount",
    "trip_duration_min",
    "passenger_count",
    "tip_amount",
    "tip_rate",
]


def load_processed() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "전처리 데이터가 없습니다. 먼저 python3 -m src.clean 명령을 실행하세요."
        )
    return pd.read_parquet(PROCESSED_DATA_PATH)


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[NUMERIC_COLUMNS].describe(
        percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]
    ).T
    stats.to_csv(
        TABLE_DIR / "descriptive_statistics.csv",
        encoding="utf-8-sig",
    )
    return stats


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    corr = df[NUMERIC_COLUMNS + ["tip_paid"]].corr(method="pearson")
    corr.to_csv(
        TABLE_DIR / "correlation_matrix.csv",
        encoding="utf-8-sig",
    )
    return corr


def run_ttest(df: pd.DataFrame) -> dict:
    """장거리(5마일 초과)와 단거리(3마일 이하)의 평균 팁 비율 비교.

    H0: 두 집단의 평균 팁 비율이 같다.
    H1: 두 집단의 평균 팁 비율이 다르다.
    Welch t-test를 사용해 두 집단의 분산이 같다고 가정하지 않는다.
    """
    short_trip = df.loc[
        df["trip_distance"] <= 3, "tip_rate"
    ].dropna()
    long_trip = df.loc[
        df["trip_distance"] > 5, "tip_rate"
    ].dropna()

    t_stat, p_value = ttest_ind(
        short_trip,
        long_trip,
        equal_var=False,
        nan_policy="omit",
    )

    alpha = 0.05
    significant = bool(p_value < alpha)
    interpretation = (
        "p-value가 0.05보다 작으므로 귀무가설을 기각한다. "
        "단거리와 장거리 운행의 평균 팁 비율 차이는 통계적으로 유의하다."
        if significant
        else
        "p-value가 0.05 이상이므로 귀무가설을 기각하지 못한다. "
        "단거리와 장거리 운행의 평균 팁 비율 차이가 있다고 볼 근거가 충분하지 않다."
    )

    # 표본이 매우 크면 작은 차이도 유의해질 수 있으므로 효과크기(Cohen's d)도 계산한다.
    n1, n2 = len(short_trip), len(long_trip)
    s1, s2 = short_trip.std(ddof=1), long_trip.std(ddof=1)
    pooled_sd = np.sqrt(
        ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    )
    cohens_d = (
        (long_trip.mean() - short_trip.mean()) / pooled_sd
        if pooled_sd > 0
        else np.nan
    )

    result = {
        "test": "Welch independent two-sample t-test",
        "null_hypothesis": "단거리와 장거리의 평균 팁 비율은 같다.",
        "alternative_hypothesis": "단거리와 장거리의 평균 팁 비율은 다르다.",
        "short_trip_definition": "trip_distance <= 3 miles",
        "long_trip_definition": "trip_distance > 5 miles",
        "short_trip_n": n1,
        "long_trip_n": n2,
        "short_trip_mean_tip_rate": float(short_trip.mean()),
        "long_trip_mean_tip_rate": float(long_trip.mean()),
        "mean_difference_long_minus_short": float(
            long_trip.mean() - short_trip.mean()
        ),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "statistically_significant": significant,
        "cohens_d": float(cohens_d),
        "interpretation": interpretation,
        "caution": (
            "통계적 유의성과 실질적 크기는 다르므로 평균 차이와 Cohen's d를 함께 본다."
        ),
    }

    save_json(result, TABLE_DIR / "ttest_result.json")
    (TABLE_DIR / "ttest_interpretation.txt").write_text(
        interpretation + "\n" + result["caution"],
        encoding="utf-8",
    )
    return result


def main() -> None:
    ensure_directories()
    df = load_processed()
    stats = descriptive_statistics(df)
    corr = correlation_analysis(df)
    ttest_result = run_ttest(df)

    print("\n[기술통계]")
    print(stats.to_string())
    print("\n[상관계수]")
    print(corr.round(4).to_string())
    print("\n[t-test]")
    print(json.dumps(ttest_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
