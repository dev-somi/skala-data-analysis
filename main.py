"""전체 분석 파이프라인 실행 진입점."""

from __future__ import annotations

import importlib


STEPS = [
    ("src.load", "Pandas·Polars 로딩 비교"),
    ("src.clean", "데이터 전처리"),
    ("src.viz", "EDA 및 시각화"),
    ("src.stats", "통계 분석"),
    ("src.ml", "ML Pipeline"),
    ("src.report", "report.md 자동 생성"),
]


def main() -> None:
    """전체 분석 단계를 순서대로 실행한다."""
    for module_name, description in STEPS:
        print("\n" + "=" * 80)
        print(f"[실행] {description}")
        print("=" * 80)

        module = importlib.import_module(module_name)
        module.main()

    print("\n전체 파이프라인 실행이 완료되었습니다.")


if __name__ == "__main__":
    main()
