"""
==============================================================================
 프로그램명 : viz
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   Seaborn 정적 차트 및 Plotly 인터랙티브 차트 생성 함수를 제공한다.
     1) calculate_tip_pct()            : tip_pct(팁 비율) 파생 컬럼 계산
     2) plot_tip_pct_distribution()    : Seaborn 정적 차트 (분포/상관관계/그룹비교)
     3) plot_tip_pct_interactive()     : Plotly 인터랙티브 차트

 [저장 위치]
   모든 차트는 root/data/external 폴더에 저장한다.
   (현재 파일 위치: root/src/viz.py 기준 parent.parent / data / external)

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  한세훈  plot_tip_pct_distribution / plot_tip_pct_interactive 구현
==============================================================================
"""

from pathlib import Path
 
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
 
sns.set_theme(style="whitegrid")
 
# 저장 위치: root/data/external  (현재 파일: root/src/viz.py)
SAVE_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
 
 
# --------------------------------------------------------------------------
# 0. 파생 컬럼 계산
# --------------------------------------------------------------------------
def calculate_tip_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    tip_pct(팁 비율, %) 파생 컬럼 계산
 
    tip_pct = tip_amount / fare_amount * 100
    fare_amount이 0 이하인 행은 계산 불가 -> NaN 처리
    """
    df = df.copy()
    valid = df["fare_amount"] > 0
    df["tip_pct"] = pd.NA
    df.loc[valid, "tip_pct"] = (
        df.loc[valid, "tip_amount"] / df.loc[valid, "fare_amount"] * 100
    )
    df["tip_pct"] = pd.to_numeric(df["tip_pct"], errors="coerce")
    return df
 
 
# --------------------------------------------------------------------------
# 1. Seaborn 정적 차트 (분포 / 상관관계 / 그룹비교)
# --------------------------------------------------------------------------
def plot_tip_pct_distribution(df: pd.DataFrame, save_dir: Path = SAVE_DIR, show: bool = True) -> dict:
    """
    Seaborn 정적 차트 3종 생성 및 저장
      1) tip_pct 분포 (히스토그램)
      2) 수치형 컬럼 상관관계 (히트맵)
      3) 결제수단(payment_type)별 tip_pct 그룹비교 (박스플롯)
 
    show=True(기본값)면 저장과 동시에 화면(예: 주피터 노트북)에도 출력한다.
 
    반환값: {차트명: 저장경로} dict
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    if "tip_pct" not in df.columns:
        df = calculate_tip_pct(df)
 
    saved_paths = {}
 
    # 1) tip_pct 분포
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = df["tip_pct"].dropna()
    low, high = plot_df.quantile([0.01, 0.99])
    sns.histplot(plot_df.clip(low, high), bins=50, kde=True, ax=ax, color="steelblue")
    ax.set_title("팁 비율(tip_pct) 분포")
    ax.set_xlabel("tip_pct (%)")
    ax.set_ylabel("건수")
    fig.tight_layout()
    path1 = save_dir / "tip_pct_distribution.png"
    fig.savefig(path1, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)
    saved_paths["distribution"] = path1
 
    # 2) 상관관계 히트맵
    num_cols = df.select_dtypes(include="number").columns
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("수치형 컬럼 상관관계")
    fig.tight_layout()
    path2 = save_dir / "tip_pct_correlation_heatmap.png"
    fig.savefig(path2, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)
    saved_paths["correlation"] = path2
 
    # 3) 결제수단별 tip_pct 그룹비교
    fig, ax = plt.subplots(figsize=(8, 5))
    group_df = df.dropna(subset=["tip_pct", "payment_type"])
    low, high = group_df["tip_pct"].quantile([0.01, 0.99])
    group_df = group_df.assign(tip_pct=group_df["tip_pct"].clip(low, high))
    sns.boxplot(x="payment_type", y="tip_pct", data=group_df, ax=ax, palette="Set2")
    ax.set_title("결제수단(payment_type)별 팁 비율 비교")
    ax.set_xlabel("payment_type")
    ax.set_ylabel("tip_pct (%)")
    fig.tight_layout()
    path3 = save_dir / "tip_pct_by_payment_type.png"
    fig.savefig(path3, dpi=100)
    if show:
        plt.show()
    else:
        plt.close(fig)
    saved_paths["group_comparison"] = path3
 
    print(f"[seaborn] 정적 차트 {len(saved_paths)}개 저장 완료 -> {save_dir}")
    for name, p in saved_paths.items():
        print(f"  - {name}: {p}")
 
    return saved_paths
 
 
# --------------------------------------------------------------------------
# 2. Plotly 인터랙티브 차트
# --------------------------------------------------------------------------
def plot_tip_pct_interactive(
    df: pd.DataFrame, save_dir: Path = SAVE_DIR, sample_n: int = 20_000, show: bool = True
) -> Path:
    """
    Plotly 인터랙티브 차트 생성 및 html로 저장
      - x축: trip_distance, y축: tip_pct, 색상: payment_type
      - 마우스 호버로 세부 값 확인 가능
 
    show=True(기본값)면 저장과 동시에 화면(예: 주피터 노트북)에도 출력한다.
 
    반환값: 저장된 html 파일 경로
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
 
    if "tip_pct" not in df.columns:
        df = calculate_tip_pct(df)
 
    plot_df = df.dropna(subset=["tip_pct", "trip_distance", "payment_type"])
    # 극단값 제외 (1~99 percentile)
    lo_d, hi_d = plot_df["trip_distance"].quantile([0.01, 0.99])
    lo_t, hi_t = plot_df["tip_pct"].quantile([0.01, 0.99])
    plot_df = plot_df[
        plot_df["trip_distance"].between(lo_d, hi_d)
        & plot_df["tip_pct"].between(lo_t, hi_t)
    ]
 
    if len(plot_df) > sample_n:
        plot_df = plot_df.sample(sample_n, random_state=42)
 
    fig = px.scatter(
        plot_df,
        x="trip_distance",
        y="tip_pct",
        color="payment_type",
        opacity=0.4,
        title="이동 거리 대비 팁 비율 (결제수단별)",
        labels={
            "trip_distance": "trip_distance (mile)",
            "tip_pct": "tip_pct (%)",
            "payment_type": "결제수단",
        },
    )
    fig.update_layout(title_x=0.5)
 
    path = save_dir / "tip_pct_interactive.html"
    fig.write_html(path)
 
    if show:
        fig.show()
 
    print(f"[plotly] 인터랙티브 차트 저장 완료 -> {path}")
    return path
 
 
# --------------------------------------------------------------------------
# 테스트용 로컬 실행
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from clean import (
        handle_missing_pandas,
        load_with_pandas,
        remove_outliers_iqr_pandas,
    )
 
    data, _ = load_with_pandas()
    data, _, _ = handle_missing_pandas(data)
    data, _, _ = remove_outliers_iqr_pandas(data)
 
    plot_tip_pct_distribution(data)
    plot_tip_pct_interactive(data)