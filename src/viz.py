"""
==============================================================================
 프로그램명 : viz
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   Seaborn 정적 차트 및 Plotly 인터랙티브 차트 생성 함수를 제공한다.
     1) plot_tip_pct_distribution() : Seaborn 정적 차트 — 시간대별 팁 지급률/팁비율
        그룹비교 (제목·축 레이블 포함)
     2) plot_tip_pct_interactive() : Plotly 인터랙티브 차트 — 거리구간 x 혼잡구역
        그룹비교

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  차트 생성 로직 구현
==============================================================================
"""

import logging
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# WHY: 한글이 깨지는(tofu box) 문제를 피하려고 시스템에 설치된 한글 폰트를 찾아서
# 쓴다. macOS는 AppleGothic, Windows는 Malgun Gothic이 보통 기본 내장이다.
_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]


def _set_korean_font() -> None:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _KOREAN_FONT_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    logger.warning("[_set_korean_font] 한글 폰트를 찾지 못했습니다 (한글이 깨질 수 있음)")


_set_korean_font()


# ---------------------------------------------------------------------------
# 1) plot_tip_pct_distribution — Seaborn 정적 차트 (그룹비교)
# ---------------------------------------------------------------------------
def plot_tip_pct_distribution(card: pd.DataFrame, output_path: Path) -> Path:
    """시간대별 팁 지급률 / 평균 팁 비율을 나란히 비교하는 Seaborn 막대그래프를 저장한다."""
    by_hour = card.groupby("pickup_hour", observed=True).agg(
        tip_rate=("has_tip", "mean"),
        avg_tip_pct=("tip_pct", "mean"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    sns.barplot(x=by_hour.index, y=by_hour["tip_rate"], ax=axes[0], color="#4C72B0")
    axes[0].set_title("시간대별 팁 지급률")
    axes[0].set_xlabel("시간대(pickup_hour)")
    axes[0].set_ylabel("팁 지급률")

    sns.barplot(x=by_hour.index, y=by_hour["avg_tip_pct"], ax=axes[1], color="#DD8452")
    axes[1].set_title("시간대별 평균 팁 비율")
    axes[1].set_xlabel("시간대(pickup_hour)")
    axes[1].set_ylabel("평균 tip_pct")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("[plot_tip_pct_distribution] 저장 완료 : %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# 2) plot_tip_pct_interactive — Plotly 인터랙티브 차트 (그룹비교)
# ---------------------------------------------------------------------------
def plot_tip_pct_interactive(card: pd.DataFrame, output_path: Path) -> Path:
    """거리구간 x 혼잡구역별 평균 팁 비율을 비교하는 Plotly 인터랙티브 차트를 저장한다."""
    df = card.copy()
    df["distance_bin"] = pd.cut(df["trip_distance"], bins=[0, 1, 3, 5, 10, 20, 1000])

    grouped = (
        df.groupby(["distance_bin", "is_congestion"], observed=True)["tip_pct"].mean().reset_index()
    )
    grouped["distance_bin"] = grouped["distance_bin"].astype(str)
    grouped["혼잡구역"] = grouped["is_congestion"].map({True: "혼잡구역", False: "비혼잡구역"})

    fig = px.bar(
        grouped,
        x="distance_bin",
        y="tip_pct",
        color="혼잡구역",
        barmode="group",
        title="거리구간 x 혼잡구역별 평균 팁 비율",
        labels={"distance_bin": "거리구간(마일)", "tip_pct": "평균 팁 비율"},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info("[plot_tip_pct_interactive] 저장 완료 : %s", output_path)
    return output_path


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

    print("\n=== 2단계 : 차트 생성 ===\n")
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
    seaborn_path = plot_tip_pct_distribution(df, out_dir / "tip_by_hour_seaborn.png")
    plotly_path = plot_tip_pct_interactive(df, out_dir / "tip_by_distance_congestion_plotly.html")

    assert seaborn_path.exists(), "Checkpoint : Seaborn 차트 파일이 생성되어야 한다"
    assert plotly_path.exists(), "Checkpoint : Plotly 차트 파일이 생성되어야 한다"

    print(f"\n\nCheckpoint : 차트 2종 생성 완료 ({seaborn_path.name}, {plotly_path.name})")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
