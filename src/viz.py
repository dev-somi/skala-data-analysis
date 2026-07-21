"""2. 기본 EDA와 Seaborn·Plotly 시각화."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from config import (
    FIGURE_DIR,
    INTERACTIVE_DIR,
    PROCESSED_DATA_PATH,
    TABLE_DIR,
)
from src.common import build_group_summary, configure_korean_font, ensure_directories


def load_processed() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "전처리 데이터가 없습니다. 먼저 python3 -m src.clean 명령을 실행하세요."
        )
    return pd.read_parquet(PROCESSED_DATA_PATH)


def save_group_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries = {
        "time": build_group_summary(df, "time_group"),
        "distance": build_group_summary(df, "distance_group"),
        "fare": build_group_summary(df, "fare_group"),
        "passenger": build_group_summary(df, "passenger_group"),
    }

    for name, summary in summaries.items():
        summary.to_csv(
            TABLE_DIR / f"{name}_tip_summary.csv",
            index=False,
            encoding="utf-8-sig",
        )
    return summaries


def create_seaborn_chart(time_summary: pd.DataFrame) -> None:
    """과제 필수 Seaborn 정적 차트."""
    sns.set_theme(
        style="whitegrid",
        font="AppleGothic",
        rc={"axes.unicode_minus": False},
    )

    plt.figure(figsize=(9, 5))

    ax = sns.barplot(
        data=time_summary,
        x="time_group",
        y="tip_payment_rate_pct",
        color="#4C78A8",
    )
    ax.set_title("시간대별 신용카드 승객 팁 지급률")
    ax.set_xlabel("승차 시간대")
    ax.set_ylabel("팁 지급률 (%)")

    for patch in ax.patches:
        height = patch.get_height()
        ax.annotate(
            f"{height:.1f}%",
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "seaborn_time_tip_payment_rate.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def create_additional_static_charts(
    distance_summary: pd.DataFrame,
    fare_summary: pd.DataFrame,
    passenger_summary: pd.DataFrame,
) -> None:
    sns.set_theme(
        style="whitegrid",
        font="AppleGothic",
        rc={"axes.unicode_minus": False},
    )
    
    for summary, x_col, title, filename in [
        (
            distance_summary,
            "distance_group",
            "거리 구간별 평균 팁 비율",
            "distance_average_tip_rate.png",
        ),
        (
            fare_summary,
            "fare_group",
            "요금 구간별 평균 팁 비율",
            "fare_average_tip_rate.png",
        ),
        (
            passenger_summary,
            "passenger_group",
            "승객 수별 팁 지급률",
            "passenger_tip_payment_rate.png",
        ),
    ]:
        y_col = (
            "tip_payment_rate_pct"
            if x_col == "passenger_group"
            else "average_tip_rate_all_pct"
        )
        y_label = "팁 지급률 (%)" if "payment" in y_col else "평균 팁 비율 (%)"

        plt.figure(figsize=(10, 5))
        ax = sns.barplot(data=summary, x=x_col, y=y_col, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(y_label)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()


def create_plotly_chart(distance_summary: pd.DataFrame) -> None:
    """과제 필수 Plotly 인터랙티브 차트."""
    fig = px.bar(
        distance_summary,
        x="distance_group",
        y="tip_payment_rate_pct",
        hover_data={
            "trip_count": ":,",
            "average_tip_rate_all_pct": ":.2f",
            "average_tip_rate_payers_pct": ":.2f",
        },
        labels={
            "distance_group": "거리 구간",
            "tip_payment_rate_pct": "팁 지급률 (%)",
            "trip_count": "운행 건수",
            "average_tip_rate_all_pct": "전체 평균 팁 비율 (%)",
            "average_tip_rate_payers_pct": "팁 지급자 평균 팁 비율 (%)",
        },
        title="거리 구간별 팁 지급률",
    )
    fig.update_layout(xaxis_title="거리 구간", yaxis_title="팁 지급률 (%)")
    fig.write_html(
        INTERACTIVE_DIR / "plotly_distance_tip_payment_rate.html",
        include_plotlyjs="cdn",
    )


def create_correlation_scatter(df: pd.DataFrame) -> None:
    """표본 산점도로 요금과 팁 금액의 관계를 확인한다."""
    sample = df.sample(n=min(20_000, len(df)), random_state=42)
    fig = px.scatter(
        sample,
        x="fare_amount",
        y="tip_amount",
        color="time_group",
        opacity=0.35,
        labels={
            "fare_amount": "기본요금 ($)",
            "tip_amount": "팁 금액 ($)",
            "time_group": "시간대",
        },
        title="기본요금과 팁 금액의 관계 (표본)",
    )
    fig.write_html(
        INTERACTIVE_DIR / "plotly_fare_tip_scatter.html",
        include_plotlyjs="cdn",
    )


def main() -> None:
    ensure_directories()
    configure_korean_font()
    df = load_processed()
    summaries = save_group_summaries(df)

    create_seaborn_chart(summaries["time"])
    create_additional_static_charts(
        summaries["distance"],
        summaries["fare"],
        summaries["passenger"],
    )
    create_plotly_chart(summaries["distance"])
    create_correlation_scatter(df)

    print("\n[시간대별 분석]")
    print(summaries["time"].to_string(index=False))
    print("\n시각화 저장 완료")


if __name__ == "__main__":
    main()
