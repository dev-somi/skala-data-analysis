"""
==============================================================================
 프로그램명 : load
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   원본 parquet 데이터를 Pandas와 Polars 양쪽 방식으로 각각 로딩하고
   결과(로딩 시간, dtype, 행 수 등)를 비교한다.
     1) load_with_pandas() : pandas.read_parquet 기반 로딩
     2) load_with_polars() : polars Lazy(scan_parquet + collect) 기반 로딩
     3) compare_loaders() : 두 결과 비교 (로딩 시간·shape·dtype 등)

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  로딩·비교 로직 구현
==============================================================================
"""

import logging
import time
from pathlib import Path

import pandas as pd
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = _PROJECT_ROOT / "data" / "raw" / "yellow_tripdata_2026-05.parquet"


# ---------------------------------------------------------------------------
# 1) load_with_pandas
# ---------------------------------------------------------------------------
def load_with_pandas(path: Path = DEFAULT_PATH) -> tuple[pd.DataFrame, float]:
    """pandas.read_parquet으로 전체 데이터를 로딩한다. (DataFrame, 소요시간초)를 반환한다."""
    tag = "[load_with_pandas]"
    t0 = time.perf_counter()
    df = pd.read_parquet(path)
    elapsed = time.perf_counter() - t0
    logger.info("%s %s, %.2f초", tag, df.shape, elapsed)
    return df, elapsed


# ---------------------------------------------------------------------------
# 2) load_with_polars
# ---------------------------------------------------------------------------
# WHY: Polars Eager(read_parquet)이 아니라 Lazy(scan_parquet + collect)를 쓴다.
# 지금은 단순 로딩뿐이라 체감 차이는 적지만, 이후 파이프라인에서 filter/select를
# 이어붙일 걸 감안해 쿼리 최적화가 가능한 Lazy 경로로 통일해둔다.
def load_with_polars(path: Path = DEFAULT_PATH) -> tuple[pl.DataFrame, float]:
    """polars Lazy(scan_parquet + collect)로 전체 데이터를 로딩한다.

    (DataFrame, 소요시간초)를 반환한다.
    """
    tag = "[load_with_polars]"
    t0 = time.perf_counter()
    df = pl.scan_parquet(path).collect()
    elapsed = time.perf_counter() - t0
    logger.info("%s %s, %.2f초", tag, df.shape, elapsed)
    return df, elapsed


# ---------------------------------------------------------------------------
# 3) compare_loaders
# ---------------------------------------------------------------------------
def compare_loaders(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    """Pandas/Polars 로딩 결과(shape·소요시간)를 한 표로 비교해 반환한다."""
    pdf, pandas_time = load_with_pandas(path)
    pldf, polars_time = load_with_polars(path)

    result = pd.DataFrame(
        [
            {
                "engine": "pandas",
                "rows": pdf.shape[0],
                "cols": pdf.shape[1],
                "elapsed_sec": pandas_time,
            },
            {
                "engine": "polars",
                "rows": pldf.shape[0],
                "cols": pldf.shape[1],
                "elapsed_sec": polars_time,
            },
        ]
    )
    logger.info("[compare_loaders] 결과:\n%s", result.to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== 1단계 : Pandas/Polars 로딩 비교 ===\n")
    result = compare_loaders()
    assert result["rows"].nunique() == 1, "Checkpoint : 두 엔진의 행 수가 일치해야 한다"

    print("\n\nCheckpoint : 로딩 비교 완료, 두 엔진 결과 일치")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
