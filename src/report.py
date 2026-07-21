"""
==============================================================================
 프로그램명 : report
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   분석 결과를 종합해 reports/report.md 를 자동 생성한다.
     1) generate_report() : 로딩비교·EDA·통계·ML 결과를 취합해 report.md 작성

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  generate_report() 로직 구현
==============================================================================
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _rank_significant_vars(sig_df: pd.DataFrame, outcome: str) -> list[str]:
    subset = sig_df[(sig_df["종속변수"] == outcome) & (sig_df["significant"])].copy()
    subset["abs_effect"] = subset["cohens_d"].abs()
    return subset.sort_values("abs_effect", ascending=False)["변수"].tolist()


# ---------------------------------------------------------------------------
# 1) generate_report
# ---------------------------------------------------------------------------
def generate_report(
    load_comparison: pd.DataFrame,
    eda_tables: dict[str, pd.DataFrame],
    sig_df: pd.DataFrame,
    ml_results_by_target: dict[str, dict],
    report_path: Path,
) -> Path:
    """분석 파이프라인 전체 결과를 report_path(reports/report.md)에 markdown으로 작성한다."""
    lines: list[str] = []
    lines.append("# 신용카드 결제 승객의 팁 지급 패턴 분석\n\n")
    lines.append(
        "운행 조건(시간대·거리·혼잡구역·승객수·요금)이 (1) 팁 지급률과 (2) 팁 비율에 "
        "미치는 영향을 분석한다.\n\n"
    )

    lines.append("## 1. 데이터 준비\n\n")
    lines.append("Pandas vs Polars(Lazy) 로딩 비교:\n\n")
    lines.append(load_comparison.to_markdown(index=False) + "\n\n")
    lines.append(
        "결측치는 passenger_count/RatecodeID/congestion_surcharge/Airport_fee/"
        "store_and_fwd_flag 5개 컬럼에서 나타나는데, 전부 미분류 결제수단"
        "(payment_type=0)에 100% 집중되어 있다. 신용카드(payment_type=1)만 "
        "필터링하면 결측치가 자연히 해소된다 (임의 대체 불필요).\n\n"
    )

    lines.append("## 2. EDA 요약\n\n")
    for name, tbl in eda_tables.items():
        lines.append(f"### {name}\n\n")
        lines.append(tbl.to_markdown() + "\n\n")

    lines.append("## 3. 통계 검정 결과\n\n")
    lines.append(
        "표본이 265만 건 이상이라 아주 작은 차이도 p-value가 거의 항상 유의하게 나온다. "
        "그래서 p<0.05 여부와 별개로 **효과크기(Cohen's d / r)** 로 실질적 중요도를 판단한다.\n\n"
    )
    lines.append(sig_df.to_markdown(index=False) + "\n\n")

    for outcome in ["has_tip", "tip_pct"]:
        ranked = _rank_significant_vars(sig_df, outcome)
        ranked_str = ", ".join(ranked) if ranked else "없음"
        lines.append(f"- **{outcome} 유의 변수(효과크기 순)**: {ranked_str}\n")
    lines.append(
        "\n> 주의: 요금(fare_amount)은 `tip_pct = tip_amount / fare_amount`의 분모이므로 "
        "상관관계가 유의해도 인과관계로 해석하지 않는다.\n\n"
    )

    lines.append("## 4. ML Pipeline 결과\n\n")
    lines.append(
        "두 개의 타겟을 각각 예측한다 — (1) `has_tip`(지급률에 대응), "
        "(2) `high_tip`(팁 비율 20% 이상, 팁 비율에 대응). accuracy는 클래스 불균형 때문에 "
        "실제 성능을 과장해 보일 수 있어 **macro F1과 클래스별 recall**을 함께 표시한다.\n\n"
    )
    for target_col, res in ml_results_by_target.items():
        lines.append(f"### {res.get('label', target_col)}\n\n")
        lines.append(f"베이스라인(다수클래스) accuracy: {res['baseline_acc']:.4f}\n\n")
        lines.append(res["metrics_df"].to_markdown(index=False) + "\n\n")
        mcfadden = res["mcfadden"]
        lines.append(
            f"McFadden 의사결정계수(R²) = {mcfadden['r2']:.4f}, "
            f"수정된 R² = {mcfadden['adj_r2']:.4f} "
            f"(train n={mcfadden['n']:,}, k={mcfadden['k']})\n\n"
        )
        lines.append(res["coef_df"].head(5).to_markdown(index=False) + "\n\n")

    if "has_tip" in ml_results_by_target and "high_tip" in ml_results_by_target:
        r2_has = ml_results_by_target["has_tip"]["mcfadden"]["r2"]
        r2_high = ml_results_by_target["high_tip"]["mcfadden"]["r2"]
        f1_has = ml_results_by_target["has_tip"]["metrics_df"]["macro_f1"].max()
        f1_high = ml_results_by_target["high_tip"]["metrics_df"]["macro_f1"].max()
        lines.append(
            f"**두 타겟 비교**: `has_tip`(McFadden R²={r2_has:.3f}, macro F1={f1_has:.3f})이 "
            f"`high_tip`(McFadden R²={r2_high:.3f}, macro F1={f1_high:.3f})보다 "
            "훨씬 설명력이 높다. 즉 **누가 팁을 주는지는 잘 설명되지만, "
            "얼마나 후하게 주는지는 잘 설명되지 않는다.**\n\n"
        )

    lines.append(
        "**추가 발견**: 애초 가설에는 없었지만 `RatecodeID`(요금 체계)가 두 모델 모두에서 "
        "가장 강한 신호로 확인됐다. 협상요금·공항 요금 등 특수 요금제 트립일수록 고팁 확률이 "
        "높은 반면, 미분류 요금(RatecodeID=99)은 오즈비가 극단적으로 낮았다.\n\n"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")
    logger.info("[generate_report] 저장 완료 : %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    from src.clean import add_time_features, add_tip_pct, filter_valid_trips
    from src.load import compare_loaders, load_with_pandas
    from src.ml import TARGET_CONFIGS, run_target_pipeline
    from src.stats import groupby_tip_summary, run_significance_suite

    print("\n=== 1단계 : 데이터 준비 ===\n")
    load_comparison = compare_loaders()
    df, _ = load_with_pandas()
    df = filter_valid_trips(df)
    df = add_tip_pct(df)
    df = add_time_features(df)

    print("\n=== 2단계 : EDA ===\n")
    df["distance_bin"] = pd.cut(df["trip_distance"], bins=[0, 1, 3, 5, 10, 20, 1000])
    eda_tables = {
        "시간대별": groupby_tip_summary(df, "pickup_hour"),
        "거리구간별": groupby_tip_summary(df, "distance_bin"),
        "혼잡구역여부별": groupby_tip_summary(df, "is_congestion"),
        "승객수별": groupby_tip_summary(df, "passenger_count"),
        "주중_주말별": groupby_tip_summary(df, "is_weekend"),
    }

    print("\n=== 3단계 : 통계 검정 ===\n")
    sig_df = run_significance_suite(df)

    print("\n=== 4단계 : ML Pipeline ===\n")
    models_dir = Path(__file__).resolve().parent.parent / "models"
    ml_results = {}
    for target_col, cfg in TARGET_CONFIGS.items():
        res = run_target_pipeline(df, target_col, cfg["num_features"], models_dir)
        res["label"] = cfg["label"]
        ml_results[target_col] = res

    print("\n=== 5단계 : report.md 생성 ===\n")
    report_path = Path(__file__).resolve().parent.parent / "reports" / "report.md"
    generate_report(load_comparison, eda_tables, sig_df, ml_results, report_path)

    report_ok = report_path.exists() and report_path.stat().st_size > 0
    assert report_ok, "Checkpoint : report.md가 생성되어야 한다"

    print(f"\n\nCheckpoint : report.md 생성 완료 ({report_path})")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
