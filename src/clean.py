"""
==============================================================================
 프로그램명 : clean
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
  [프로그램 설명]
   원본 parquet 데이터를 Pandas와 Polars 양쪽 방식으로 각각 로딩하고
   결과(로딩 시간, dtype, 행 수 등)를 비교한다.
     1) load_with_pandas()          : pandas.read_parquet 기반 로딩
     2) load_with_polars()          : polars.read_parquet 기반 로딩
     3) handle_missing_pandas()     : pandas 결측치 처리 (동일 규칙)
     4) handle_missing_polars()     : polars 결측치 처리 (동일 규칙)
     5) compare_loaders()           : 로딩 + 결측치 처리 결과 비교 (소요 시간·dtype·shape·결측치 건수 등)

  [결측치 처리 규칙] (pandas / polars 동일 적용)
   - tpep_pickup_datetime, tpep_dropoff_datetime : 결측 행 제거 (필수 컬럼)
   - passenger_count   : 최빈값(mode)으로 대체
   - RateCodeID        : 1 (Standard rate)로 대체
   - store_and_fwd_flag: 'N'으로 대체
   - 그 외 수치형 컬럼  : 중앙값(median)으로 대체
  [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  한세훈  load_with_pandas / load_with_polars / compare_loaders 구현
   v1.2  2026-07-21  한세훈  handle_missing_pandas / handle_missing_polars 추가, 비교 항목에 결측치 처리 반영
==============================================================================
"""
 
import time
from pathlib import Path
 
import pandas as pd
import polars as pl
 
# 프로젝트 루트 기준 데이터 경로
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "yellow_tripdata_2026-05.parquet"
 
USE_COLS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_longitude",
    "pickup_latitude",
    "RateCodeID",
    "store_and_fwd_flag",
    "dropoff_longitude",
    "dropoff_latitude",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
]
 
 
# --------------------------------------------------------------------------
# 1. Pandas 로딩
# --------------------------------------------------------------------------
def load_with_pandas(path: Path = DATA_PATH, use_cols: list = None) -> tuple[pd.DataFrame, float]:
    """pandas.read_parquet 기반 로딩.
    반환값: (DataFrame, 로딩 소요 시간(초))
    """
    use_cols = use_cols or USE_COLS
 
    # 실제 파일에 존재하는 컬럼만 필터링 (스키마 상이 대비)
    import pyarrow.parquet as pq
    available_cols = pq.ParquetFile(path).schema.names
    cols_to_read = [c for c in use_cols if c in available_cols]
 
    start = time.perf_counter()
    df = pd.read_parquet(path, columns=cols_to_read)
    elapsed = time.perf_counter() - start
 
    print(f"[pandas] 로딩 완료: {elapsed:.3f}초, shape={df.shape}")
    return df, elapsed
 
 
# --------------------------------------------------------------------------
# 2. Polars 로딩
# --------------------------------------------------------------------------
def load_with_polars(path: Path = DATA_PATH, use_cols: list = None) -> tuple[pl.DataFrame, float]:
    """polars.read_parquet 기반 로딩.
    반환값: (DataFrame, 로딩 소요 시간(초))
    """
    use_cols = use_cols or USE_COLS
 
    import pyarrow.parquet as pq
    available_cols = pq.ParquetFile(path).schema.names
    cols_to_read = [c for c in use_cols if c in available_cols]
 
    start = time.perf_counter()
    df = pl.read_parquet(path, columns=cols_to_read)
    elapsed = time.perf_counter() - start
 
    print(f"[polars] 로딩 완료: {elapsed:.3f}초, shape={df.shape}")
    return df, elapsed
 
 
# --------------------------------------------------------------------------
# 3. 결측치 처리 (pandas / polars 동일 규칙)
# --------------------------------------------------------------------------
REQUIRED_DATETIME_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]
MODE_FILL_COLS = ["passenger_count"]
CONST_FILL_MAP = {"RateCodeID": 1, "store_and_fwd_flag": "N"}
 
 
def handle_missing_pandas(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, float]:
    """pandas 결측치 처리.
    반환값: (처리된 DataFrame, 처리 리포트 dict, 소요 시간(초))
    """
    start = time.perf_counter()
 
    before_missing = int(df.isnull().sum().sum())
    n_before = len(df)
 
    df = df.copy()
 
    # 1) 필수 datetime 컬럼 결측 -> 행 제거
    dt_cols = [c for c in REQUIRED_DATETIME_COLS if c in df.columns]
    if dt_cols:
        df = df.dropna(subset=dt_cols)
 
    # 2) 최빈값 대체
    for col in MODE_FILL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
 
    # 3) 상수값 대체
    for col, val in CONST_FILL_MAP.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
 
    # 4) 그 외 수치형 컬럼 -> 중앙값 대체
    handled_cols = set(dt_cols) | set(MODE_FILL_COLS) | set(CONST_FILL_MAP)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        if col not in handled_cols and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
 
    after_missing = int(df.isnull().sum().sum())
    elapsed = time.perf_counter() - start
 
    report = {
        "n_rows_before": n_before,
        "n_rows_after": len(df),
        "rows_dropped": n_before - len(df),
        "missing_before": before_missing,
        "missing_after": after_missing,
    }
    print(f"[pandas] 결측치 처리 완료: {elapsed:.3f}초, {report}")
    return df, report, elapsed
 
 
def handle_missing_polars(df: pl.DataFrame) -> tuple[pl.DataFrame, dict, float]:
    """polars 결측치 처리 (pandas와 동일 규칙 적용).
    반환값: (처리된 DataFrame, 처리 리포트 dict, 소요 시간(초))
    """
    start = time.perf_counter()
 
    before_missing = int(sum(df.null_count().row(0)))
    n_before = df.shape[0]
 
    # 1) 필수 datetime 컬럼 결측 -> 행 제거
    dt_cols = [c for c in REQUIRED_DATETIME_COLS if c in df.columns]
    if dt_cols:
        df = df.drop_nulls(subset=dt_cols)
 
    # 2) 최빈값 대체
    for col in MODE_FILL_COLS:
        if col in df.columns and df[col].null_count() > 0:
            mode_val = df[col].drop_nulls().mode()
            if mode_val.len() > 0:
                df = df.with_columns(pl.col(col).fill_null(mode_val[0]))
 
    # 3) 상수값 대체
    for col, val in CONST_FILL_MAP.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).fill_null(val))
 
    # 4) 그 외 수치형 컬럼 -> 중앙값 대체
    handled_cols = set(dt_cols) | set(MODE_FILL_COLS) | set(CONST_FILL_MAP)
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    for col in num_cols:
        if col not in handled_cols and df[col].null_count() > 0:
            df = df.with_columns(pl.col(col).fill_null(df[col].median()))
 
    after_missing = int(sum(df.null_count().row(0)))
    elapsed = time.perf_counter() - start
 
    report = {
        "n_rows_before": n_before,
        "n_rows_after": df.shape[0],
        "rows_dropped": n_before - df.shape[0],
        "missing_before": before_missing,
        "missing_after": after_missing,
    }
    print(f"[polars] 결측치 처리 완료: {elapsed:.3f}초, {report}")
    return df, report, elapsed
 
 
# --------------------------------------------------------------------------
# 4. 이상치 제거 (IQR 방식, pandas / polars 동일 규칙)
# --------------------------------------------------------------------------
def calculate_iqr_bounds(series: pd.Series):
    """
    IQR 기준 이상치 범위 계산
 
    정상 범위:
    Q1 - 1.5 * IQR <= value <= Q3 + 1.5 * IQR
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
 
    lo = q1 - (1.5 * iqr)
    hi = q3 + (1.5 * iqr)
    return lo, hi
 
 
def remove_outliers_iqr_pandas(df: pd.DataFrame, cols: list = None) -> tuple[pd.DataFrame, dict, float]:
    """
    모든 숫자형 컬럼에 대해 IQR 방식으로 이상치 제거 (pandas)
    """
    start = time.perf_counter()
 
    n_before = len(df)
    cols = cols or df.select_dtypes(include="number").columns.tolist()
 
    df = df.copy()
    for col in cols:
        lo, hi = calculate_iqr_bounds(df[col])
        df = df[(df[col] >= lo) & (df[col] <= hi)]
 
    elapsed = time.perf_counter() - start
 
    report = {
        "n_rows_before": n_before,
        "n_rows_after": len(df),
        "rows_removed": n_before - len(df),
    }
    print(f"[pandas] 이상치 제거 완료: {elapsed:.3f}초, {report}")
    return df, report, elapsed
 
 
def calculate_iqr_bounds_polars(series: pl.Series):
    """
    IQR 기준 이상치 범위 계산 (polars)
 
    정상 범위:
    Q1 - 1.5 * IQR <= value <= Q3 + 1.5 * IQR
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
 
    lo = q1 - (1.5 * iqr)
    hi = q3 + (1.5 * iqr)
    return lo, hi
 
 
def remove_outliers_iqr_polars(df: pl.DataFrame, cols: list = None) -> tuple[pl.DataFrame, dict, float]:
    """
    모든 숫자형 컬럼에 대해 IQR 방식으로 이상치 제거 (polars)
    """
    start = time.perf_counter()
 
    n_before = df.shape[0]
    cols = cols or [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
 
    for col in cols:
        lo, hi = calculate_iqr_bounds_polars(df[col])
        df = df.filter((pl.col(col) >= lo) & (pl.col(col) <= hi))
 
    elapsed = time.perf_counter() - start
 
    report = {
        "n_rows_before": n_before,
        "n_rows_after": df.shape[0],
        "rows_removed": n_before - df.shape[0],
    }
    print(f"[polars] 이상치 제거 완료: {elapsed:.3f}초, {report}")
    return df, report, elapsed
 
 
# --------------------------------------------------------------------------
# 5. 비교
# --------------------------------------------------------------------------
def compare_loaders(path: Path = DATA_PATH, use_cols: list = None) -> pd.DataFrame:
    """pandas / polars의 로딩 + 결측치 처리 + 이상치 제거 결과를 동일 작업 내역으로 비교"""
    # 1) 로딩
    pdf, pandas_load_time = load_with_pandas(path, use_cols)
    pldf, polars_load_time = load_with_polars(path, use_cols)
 
    # 2) 결측치 처리 (동일 규칙)
    pdf, pandas_missing_report, pandas_missing_time = handle_missing_pandas(pdf)
    pldf, polars_missing_report, polars_missing_time = handle_missing_polars(pldf)
 
    # 3) 이상치 제거 (동일 규칙, IQR)
    pdf, pandas_outlier_report, pandas_outlier_time = remove_outliers_iqr_pandas(pdf)
    pldf, polars_outlier_report, polars_outlier_time = remove_outliers_iqr_polars(pldf)
 
    pandas_total = pandas_load_time + pandas_missing_time + pandas_outlier_time
    polars_total = polars_load_time + polars_missing_time + polars_outlier_time
 
    summary = pd.DataFrame(
        {
            "loader": ["pandas", "polars"],
            "load_time_sec": [round(pandas_load_time, 3), round(polars_load_time, 3)],
            "missing_handle_time_sec": [
                round(pandas_missing_time, 3),
                round(polars_missing_time, 3),
            ],
            "outlier_remove_time_sec": [
                round(pandas_outlier_time, 3),
                round(polars_outlier_time, 3),
            ],
            "total_time_sec": [round(pandas_total, 3), round(polars_total, 3)],
            "n_rows_after_missing": [
                pandas_missing_report["n_rows_after"],
                polars_missing_report["n_rows_after"],
            ],
            "n_rows_after_outlier": [
                pandas_outlier_report["n_rows_after"],
                polars_outlier_report["n_rows_after"],
            ],
            "rows_removed_outlier": [
                pandas_outlier_report["rows_removed"],
                polars_outlier_report["rows_removed"],
            ],
            "memory_MB": [
                round(pdf.memory_usage(deep=True).sum() / 1e6, 2),
                round(pldf.estimated_size("mb"), 2),
            ],
        }
    )
 
    print("=" * 60)
    print("Pandas vs Polars 로딩 + 결측치 처리 + 이상치 제거 비교")
    print("=" * 60)
    print(summary.to_string(index=False))
 
    print("\n[pandas dtypes]")
    print(pdf.dtypes)
    print("\n[polars dtypes]")
    print(pldf.schema)
 
    faster = "pandas" if pandas_total < polars_total else "polars"
    diff = abs(pandas_total - polars_total)
    print(f"\n>> {faster}가 전체 처리(로딩+결측치+이상치) 기준 {diff:.3f}초 더 빠름")
 
    return summary
 
 
if __name__ == "__main__":
    compare_loaders()