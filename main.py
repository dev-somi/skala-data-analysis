"""
==============================================================================
 프로그램명 : main
------------------------------------------------------------------------------
 [프로그램 설명]
   src/ 각 모듈의 main()을 정해진 순서대로 한 번에 실행하는 통합 파이프라인.
     1) download : 원본 parquet 다운로드
     2) load     : Pandas/Polars 로딩 비교
     3) clean    : 결측치·중복 처리, 필터링, 파생변수
     4) schema   : Pydantic 스키마 검증
     5) viz      : Seaborn/Plotly 차트 생성
     6) stats    : 기술통계·상관계수·t-test
     7) ml       : sklearn Pipeline 학습·평가·저장
     8) report   : reports/report.md 자동 생성
==============================================================================
"""

import logging

from src import clean, download, load, ml, report, schema, stats, viz

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

STEPS = [
    ("1/8 데이터 다운로드", download.main),
    ("2/8 Pandas/Polars 로딩 비교", load.main),
    ("3/8 정제 (결측치·중복·필터링·파생변수)", clean.main),
    ("4/8 스키마 검증", schema.main),
    ("5/8 시각화", viz.main),
    ("6/8 통계 분석", stats.main),
    ("7/8 ML Pipeline", ml.main),
    ("8/8 리포트 생성", report.main),
]


def main() -> None:
    for label, step in STEPS:
        print(f"\n{'=' * 60}\n {label}\n{'=' * 60}")
        step()

    print("\n\nCheckpoint : 전체 파이프라인 8단계 완료")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("파이프라인 중단 : %s: %s", type(e).__name__, e)
        raise
