"""
==============================================================================
 프로그램명 : load
 작성자     : 광주 4반 박소미, 이다영
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   NYC Yellow Taxi Parquet 데이터를 Pandas와 Polars 방식으로 로딩하고
   로딩 시간, 행·열 크기, 메모리 사용량을 비교한다.

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 구조 작성
   v1.1  2026-07-21  이다영  Pandas·Polars 로딩 및 비교 로직 구현
==============================================================================
"""

from __future__ import annotations

import time

import pandas as pd
import polars as pl

from config import DATA_PATH, TABLE_DIR
from src.common import ensure_directories


def load_with_pandas() -> tuple[pd.DataFrame, float]:
    """Pandas로 원본 Parquet 데이터를 로딩하고 소요 시간을 반환한다."""
    start = time.perf_counter()
    dataframe = pd.read_parquet(DATA_PATH)
    elapsed_seconds = time.perf_counter() - start
    return dataframe, elapsed_seconds


def load_with_polars() -> tuple[pl.DataFrame, float]:
    """Polars로 원본 Parquet 데이터를 로딩하고 소요 시간을 반환한다."""
    start = time.perf_counter()
    dataframe = pl.read_parquet(DATA_PATH)
    elapsed_seconds = time.perf_counter() - start
    return dataframe, elapsed_seconds


def compare_loaders() -> dict[str, int | float | bool]:
    """Pandas와 Polars의 로딩 결과, 시간 및 메모리 사용량을 비교한다."""
    ensure_directories()

    pandas_df, pandas_seconds = load_with_pandas()
    polars_df, polars_seconds = load_with_polars()

    comparison = {
        "pandas_rows": len(pandas_df),
        "pandas_columns": pandas_df.shape[1],
        "pandas_load_seconds": round(pandas_seconds, 4),
        "pandas_memory_mb": round(
            pandas_df.memory_usage(deep=True).sum() / 1024**2,
            2,
        ),
        "polars_rows": polars_df.height,
        "polars_columns": polars_df.width,
        "polars_load_seconds": round(polars_seconds, 4),
        "polars_estimated_memory_mb": round(
            polars_df.estimated_size("mb"),
            2,
        ),
        "same_shape": pandas_df.shape == (polars_df.height, polars_df.width),
    }

    pd.DataFrame([comparison]).to_csv(
        TABLE_DIR / "pandas_polars_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return comparison


def main() -> None:
    """Pandas·Polars 로딩 비교 결과를 출력한다."""
    comparison = compare_loaders()

    print("\n[Pandas·Polars 로딩 비교]")
    print(pd.DataFrame([comparison]).to_string(index=False))


if __name__ == "__main__":
    main()
