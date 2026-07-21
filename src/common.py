"""공통 유틸리티 함수."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

from config import FIGURE_DIR, INTERACTIVE_DIR, MODEL_DIR, OUTPUT_DIR, TABLE_DIR


def ensure_directories() -> None:
    """분석 산출물 폴더를 생성한다."""
    for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, INTERACTIVE_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def classify_time(hour: int) -> str:
    """승차 시각을 아침·낮·저녁·심야로 분류한다."""
    if 6 <= hour < 11:
        return "아침"
    if 11 <= hour < 17:
        return "낮"
    if 17 <= hour < 22:
        return "저녁"
    return "심야"


def save_json(data: dict, path: Path) -> None:
    """딕셔너리를 UTF-8 JSON으로 저장한다."""
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def build_group_summary(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """그룹별 팁 지급률과 평균 팁 비율을 집계한다.

    tip_payment_rate:
        그룹 전체 운행 중 tip_amount > 0인 운행의 비율
    average_tip_rate_all:
        팁 미지급(0%)을 포함한 전체 운행의 평균 팁 비율
    average_tip_rate_payers:
        실제 팁 지급 운행만 대상으로 한 평균 팁 비율
    """
    summary = (
        df.groupby(group_column, observed=False)
        .agg(
            trip_count=("tip_paid", "size"),
            tip_trip_count=("tip_paid", "sum"),
            tip_payment_rate=("tip_paid", "mean"),
            average_tip_amount_all=("tip_amount", "mean"),
            median_tip_amount_all=("tip_amount", "median"),
            average_tip_rate_all=("tip_rate", "mean"),
            median_tip_rate_all=("tip_rate", "median"),
        )
        .reset_index()
    )

    payer_summary = (
        df.loc[df["tip_paid"] == 1]
        .groupby(group_column, observed=False)
        .agg(
            average_tip_amount_payers=("tip_amount", "mean"),
            average_tip_rate_payers=("tip_rate", "mean"),
        )
        .reset_index()
    )

    summary = summary.merge(payer_summary, on=group_column, how="left")
    summary["tip_payment_rate_pct"] = summary["tip_payment_rate"] * 100
    summary["average_tip_rate_all_pct"] = summary["average_tip_rate_all"] * 100
    summary["average_tip_rate_payers_pct"] = (
        summary["average_tip_rate_payers"] * 100
    )
    return summary


def configure_korean_font() -> None:
    """운영체제에서 사용할 수 있는 한글 폰트를 자동 선택한다."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    
    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False

    candidates = [
        "AppleGothic",          # macOS
        "Malgun Gothic",        # Windows
        "NanumGothic",          # Linux/사용자 설치
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in candidates:
        if candidate in installed:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False
