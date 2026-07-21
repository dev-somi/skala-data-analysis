"""
==============================================================================
 프로그램명 : stats
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   기술통계, 상관계수, t-test 산출을 담당한다.
     1) describe_stats() : 평균·표준편차·분위수 등 기술통계 산출
     2) correlation_matrix() : 변수 간 상관계수 계산
     3) run_ttest() : scipy.stats.ttest_ind(Welch) 기반 t-test, p-value·효과크기 해석
     4) run_significance_suite() : 주제 변수(시간대/거리/혼잡구역/승객수/요금) x
        종속변수(팁 지급률/팁 비율) 전체 조합에 3)을 일괄 적용
        (표본이 268만행이라 p-value만으론 판단이 안 돼 3)에서 만든 효과크기까지
        같이 봐야 해서 추가)
     5) groupby_tip_summary() : 조건별(시간대/거리구간/혼잡구역/승객수/주중주말)
        팁 지급률·평균 팁 비율 집계 (report.py의 EDA 절에서 필요해 추가)

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  5개 함수 로직 구현 (4, 5는 분석 중 필요해 추가)
==============================================================================
"""

import logging

import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1) describe_stats
# ---------------------------------------------------------------------------
def describe_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """columns에 대한 평균·표준편차·분위수 등 기술통계를 반환한다."""
    result = df[columns].describe().T
    logger.info("[describe_stats] \n%s", result.to_string())
    return result


# ---------------------------------------------------------------------------
# 2) correlation_matrix
# ---------------------------------------------------------------------------
def correlation_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """columns 간 피어슨 상관계수 행렬을 반환한다."""
    result = df[columns].corr()
    logger.info("[correlation_matrix] \n%s", result.to_string())
    return result


# ---------------------------------------------------------------------------
# 3) run_ttest
# ---------------------------------------------------------------------------
# WHY: 표본이 수백만 건이면 아주 작은 차이도 p-value가 항상 유의하게 나온다.
# 그래서 p<0.05 여부만이 아니라 Cohen's d(효과크기)로 실질적 중요도를 같이 판단한다.
# 관례: |d| < 0.1 무시가능, < 0.3 작음, < 0.5 중간, 그 이상 큼.
def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    pooled_std = ((a.std() ** 2 + b.std() ** 2) / 2) ** 0.5
    return (a.mean() - b.mean()) / pooled_std


def _classify_effect_size(d: float) -> str:
    v = abs(d)
    if v < 0.1:
        return "무시가능(통계적 유의 ≠ 실질적 의미)"
    if v < 0.3:
        return "작음"
    if v < 0.5:
        return "중간"
    return "큼"


def run_ttest(group_a: pd.Series, group_b: pd.Series, label: str = "") -> dict:
    """Welch's t-test를 수행하고 t-통계량·p-value·Cohen's d·해석을 담은 dict를 반환한다."""
    t_stat, p_value = scipy_stats.ttest_ind(group_a, group_b, equal_var=False)
    d = _cohens_d(group_a, group_b)
    result = {
        "label": label,
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": d,
        "effect_size": _classify_effect_size(d),
    }
    logger.info(
        "[run_ttest] %s t=%.3f p=%.2e d=%.4f(%s)",
        label, t_stat, p_value, d, result["effect_size"],
    )
    return result


# ---------------------------------------------------------------------------
# 4) run_significance_suite
# ---------------------------------------------------------------------------
def run_significance_suite(card: pd.DataFrame) -> pd.DataFrame:
    """시간대(심야)/혼잡구역/주중주말/합승 여부에 대해 has_tip·tip_pct 각각 t-test를
    수행하고, 거리·요금은 상관계수를 산출해 하나의 표로 합쳐 반환한다."""
    rows: list[dict] = []
    outcomes = {"has_tip": card["has_tip"].astype(int), "tip_pct": card["tip_pct"]}

    def add_ttest_pair(name: str, mask_a: pd.Series, mask_b: pd.Series):
        for outcome_name, series in outcomes.items():
            res = run_ttest(series[mask_a], series[mask_b], label=f"{name} / {outcome_name}")
            rows.append({"변수": name, "종속변수": outcome_name, **res})

    add_ttest_pair("혼잡구역 여부", card["is_congestion"], ~card["is_congestion"])
    add_ttest_pair("심야(0-5시) 여부", card["is_night"], ~card["is_night"])
    add_ttest_pair("주중/주말", card["is_weekend"], ~card["is_weekend"])
    add_ttest_pair(
        "합승 여부(1명 vs 2명+)", card["passenger_count"] > 1, card["passenger_count"] == 1
    )

    corr_vars = [("거리(trip_distance)", "trip_distance"), ("요금(fare_amount)", "fare_amount")]
    for var_name, col in corr_vars:
        for outcome_name, series in outcomes.items():
            r, p_value = scipy_stats.pearsonr(card[col], series)
            rows.append(
                {
                    "변수": var_name,
                    "종속변수": outcome_name,
                    "label": f"{var_name} / {outcome_name}",
                    "t_stat": None,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "cohens_d": r,
                    "effect_size": _classify_effect_size(r),
                }
            )

    result = pd.DataFrame(rows)
    logger.info("[run_significance_suite] 결과:\n%s", result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 5) groupby_tip_summary
# ---------------------------------------------------------------------------
def groupby_tip_summary(card: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """group_col 기준으로 팁 지급률(tip_rate)·평균 팁 비율(avg_tip_pct)·표본수(n)를 집계한다."""
    result = card.groupby(group_col, observed=True).agg(
        tip_rate=("has_tip", "mean"),
        avg_tip_pct=("tip_pct", "mean"),
        n=("tip_pct", "size"),
    )
    logger.info("[groupby_tip_summary] %s 기준:\n%s", group_col, result.to_string())
    return result


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    from src.clean import add_time_features, add_tip_pct, filter_valid_trips
    from src.load import load_with_pandas

    print("\n=== 1단계 : 데이터 준비 ===\n")
    df, _ = load_with_pandas()
    df = filter_valid_trips(df)
    df = add_tip_pct(df)
    df = add_time_features(df)

    print("\n=== 2단계 : 기술통계·상관계수 ===\n")
    describe_stats(df, ["trip_distance", "fare_amount", "tip_amount", "tip_pct"])
    correlation_matrix(df, ["trip_distance", "fare_amount", "trip_duration_min", "tip_pct"])

    print("\n=== 3단계 : t-test 일괄 수행 ===\n")
    sig_df = run_significance_suite(df)

    assert sig_df["p_value"].notna().all(), "Checkpoint : 모든 검정에 p-value가 있어야 한다"

    print(f"\n\nCheckpoint : 통계 검정 {len(sig_df)}건 완료")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
