"""5. 분석 결과를 읽어 report.md를 자동 생성한다."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from config import PROJECT_ROOT, TABLE_DIR
from src.common import ensure_directories


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dataframe_markdown(path: Path, columns: list[str] | None = None) -> str:
    if not path.exists():
        return "_결과 파일 없음_"
    df = pd.read_csv(path)
    if columns:
        available = [column for column in columns if column in df.columns]
        df = df[available]
    return df.to_markdown(index=False)


def format_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def main() -> None:
    ensure_directories()

    quality = read_json(TABLE_DIR / "data_quality_log.json")
    loading = (
        pd.read_csv(TABLE_DIR / "pandas_polars_comparison.csv").iloc[0].to_dict()
        if (TABLE_DIR / "pandas_polars_comparison.csv").exists()
        else {}
    )
    ttest = read_json(TABLE_DIR / "ttest_result.json")
    model = read_json(TABLE_DIR / "model_metrics.json")

    time_table = dataframe_markdown(
        TABLE_DIR / "time_tip_summary.csv",
        [
            "time_group",
            "trip_count",
            "tip_payment_rate_pct",
            "average_tip_rate_all_pct",
            "average_tip_rate_payers_pct",
        ],
    )
    distance_table = dataframe_markdown(
        TABLE_DIR / "distance_tip_summary.csv",
        [
            "distance_group",
            "trip_count",
            "tip_payment_rate_pct",
            "average_tip_rate_all_pct",
            "average_tip_rate_payers_pct",
        ],
    )
    fare_table = dataframe_markdown(
        TABLE_DIR / "fare_tip_summary.csv",
        [
            "fare_group",
            "trip_count",
            "tip_payment_rate_pct",
            "average_tip_rate_all_pct",
            "average_tip_rate_payers_pct",
        ],
    )
    passenger_table = dataframe_markdown(
        TABLE_DIR / "passenger_tip_summary.csv",
        [
            "passenger_group",
            "trip_count",
            "tip_payment_rate_pct",
            "average_tip_rate_all_pct",
            "average_tip_rate_payers_pct",
        ],
    )

    report = f"""# NYC Yellow Taxi 신용카드 승객 팁 분석

## 1. 분석 목적

2026년 5월 NYC Yellow Taxi 데이터에서 신용카드 결제 운행만 선택하여,
시간대·거리·요금·승객 수에 따른 팁 지급률과 평균 팁 비율을 분석한다.
또한 운행 조건만으로 승객의 팁 지급 여부를 예측하는 이진 분류 모델을 구축한다.

## 2. 핵심 정의

- 분석 대상: `payment_type == 1`
- 팁 지급 여부: `tip_paid = 1` if `tip_amount > 0`, otherwise `0`
- 팁 비율: `tip_rate = tip_amount / fare_amount`
- 주의: `tip_amount`, `tip_rate`, `total_amount`는 정답 정보가 포함될 수 있어 모델 입력에서 제외한다.

## 3. 데이터 준비 및 품질 처리

- 원본 행 수: {quality.get('raw_rows', 'N/A'):,}
- 중복 제거 후 행 수: {quality.get('rows_after_duplicate_removal', 'N/A'):,}
- 필수값 결측 제거 건수: {quality.get('removed_missing_required', 'N/A'):,}
- 대상 월 외 데이터 제거 건수: {quality.get('removed_outside_target_month', 'N/A'):,}
- 비신용카드 제거 건수: {quality.get('removed_non_credit_card', 'N/A'):,}
- 기본 이상치 제거 건수: {quality.get('removed_basic_outliers', 'N/A'):,}
- 팁 비율 이상치 제거 건수: {quality.get('removed_tip_rate_outliers', 'N/A'):,}
- 최종 분석 행 수: {quality.get('final_rows', 'N/A'):,}

이상치 기준:

- 거리: 0.1~30마일
- 기본요금: $3~$150
- 운행시간: 1~180분
- 승객 수: 1~6명
- 팁 금액: 0 이상
- 팁 비율: 0~100%

## 4. Pandas와 Polars 비교

- Pandas 로딩 시간: {loading.get('pandas_load_seconds', 'N/A')}초
- Polars 로딩 시간: {loading.get('polars_load_seconds', 'N/A')}초
- Pandas 메모리: {loading.get('pandas_memory_mb', 'N/A')}MB
- Polars 추정 메모리: {loading.get('polars_estimated_memory_mb', 'N/A')}MB
- 행·열 크기 일치 여부: {loading.get('same_shape', 'N/A')}

실행 환경에 따라 로딩 시간은 달라질 수 있으므로 절대적 우열보다는 동일 데이터가
두 라이브러리에서 일관되게 로딩되는지와 자원 사용 차이를 함께 확인한다.

## 5. 시간대별 분석

{time_table}

## 6. 거리별 분석

{distance_table}

## 7. 요금별 분석

{fare_table}

## 8. 승객 수별 분석

{passenger_table}

## 9. 통계 분석

Welch 독립표본 t-test로 단거리(3마일 이하)와 장거리(5마일 초과)의
평균 팁 비율을 비교했다.

- 단거리 평균 팁 비율: {format_pct(ttest.get('short_trip_mean_tip_rate'))}
- 장거리 평균 팁 비율: {format_pct(ttest.get('long_trip_mean_tip_rate'))}
- 평균 차이(장거리-단거리): {format_pct(ttest.get('mean_difference_long_minus_short'))}
- t 통계량: {ttest.get('t_statistic', 'N/A')}
- p-value: {ttest.get('p_value', 'N/A')}
- Cohen's d: {ttest.get('cohens_d', 'N/A')}
- 해석: {ttest.get('interpretation', 'N/A')}

표본이 매우 크면 작은 차이도 통계적으로 유의해질 수 있으므로,
p-value뿐 아니라 평균 차이와 효과크기를 함께 해석한다.

## 10. ML Pipeline

모델은 `LogisticRegression`을 사용한 이진 분류 Pipeline이다.

입력 변수:

- 운행 거리
- 기본요금
- 운행시간
- 승객 수
- 승차 시간의 순환형 변환(`hour_sin`, `hour_cos`)

평가 결과:

- Accuracy: {model.get('accuracy', 'N/A')}
- Precision: {model.get('precision', 'N/A')}
- Recall: {model.get('recall', 'N/A')}
- F1-score: {model.get('f1', 'N/A')}
- ROC-AUC: {model.get('roc_auc', 'N/A')}

클래스 불균형 가능성이 있으므로 Accuracy만으로 모델을 평가하지 않고
Precision, Recall, F1-score 및 ROC-AUC를 함께 확인한다.

## 11. 시각화 및 산출물

- Seaborn 정적 차트: `outputs/figures/seaborn_time_tip_payment_rate.png`
- Plotly 인터랙티브 차트: `outputs/interactive/plotly_distance_tip_payment_rate.html`
- 혼동행렬: `outputs/figures/confusion_matrix.png`
- 저장 모델: `models/tip_payment_logistic_pipeline.joblib`

## 12. 분석의 의미와 한계

이번 분석은 운행 조건에 따른 팁 행동을 설명하는 EDA인 동시에,
어떤 변수가 팁 지급 예측에 도움이 되는지 검토하는 과정이다.
다만 관찰 데이터의 상관관계만으로 인과관계를 주장할 수 없으며,
교통 상황, 운전자 서비스, 승객 특성 등 데이터에 없는 변수가 영향을 줄 수 있다.
또한 현금 팁은 정확히 기록되지 않을 수 있어 신용카드 결제로 분석 범위를 제한했다.
"""

    report_path = PROJECT_ROOT / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"report.md 생성 완료: {report_path}")


if __name__ == "__main__":
    main()
