"""4. 팁 지급 여부 예측 이진 분류 ML Pipeline."""
from __future__ import annotations

import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    FIGURE_DIR,
    MODEL_DIR,
    MODEL_SAMPLE_SIZE,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    TABLE_DIR,
    TEST_SIZE,
)
from src.common import configure_korean_font, ensure_directories, save_json


# 분석에서는 구간별 집계를 사용하지만 모델에는 원래 연속형 값을 사용한다.
NUMERIC_FEATURES = [
    "trip_distance",
    "fare_amount",
    "trip_duration_min",
    "passenger_count",
]

# 시간은 비선형·순환적 특성이 있으므로 sin/cos로 변환한다.
CYCLICAL_FEATURES = ["hour_sin", "hour_cos"]

TARGET = "tip_paid"

# 데이터 누수 방지를 위해 tip_amount, tip_rate, total_amount는 입력에서 제외한다.


def load_processed() -> pd.DataFrame:
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            "전처리 데이터가 없습니다. 먼저 python3 -m src.clean 명령을 실행하세요."
        )
    return pd.read_parquet(PROCESSED_DATA_PATH)


def prepare_model_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    model_df = df.copy()

    # 23시와 0시가 실제로 가깝다는 점을 반영하는 순환형 시간 특성
    model_df["hour_sin"] = np.sin(
        2 * np.pi * model_df["pickup_hour"] / 24
    )
    model_df["hour_cos"] = np.cos(
        2 * np.pi * model_df["pickup_hour"] / 24
    )

    feature_columns = NUMERIC_FEATURES + CYCLICAL_FEATURES
    model_df = model_df.dropna(subset=feature_columns + [TARGET]).copy()

    # 전체 EDA는 전 데이터를 사용하며, ML은 메모리·시간 안정을 위해 층화 표본을 사용한다.
    if len(model_df) > MODEL_SAMPLE_SIZE:
        sampled_parts = []
        for label, group in model_df.groupby(TARGET):
            target_n = max(
                1,
                round(MODEL_SAMPLE_SIZE * len(group) / len(model_df)),
            )
            sampled_parts.append(
                group.sample(
                    n=min(target_n, len(group)),
                    random_state=RANDOM_STATE,
                )
            )
        model_df = (
            pd.concat(sampled_parts)
            .sample(frac=1, random_state=RANDOM_STATE)
            .reset_index(drop=True)
        )

    X = model_df[feature_columns]
    y = model_df[TARGET].astype(int)
    return X, y


def build_pipeline() -> Pipeline:
    numeric_features = NUMERIC_FEATURES + CYCLICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return pipeline


def train_and_evaluate(df: pd.DataFrame) -> dict:
    X, y = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "LogisticRegression",
        "target": TARGET,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "input_features": list(X.columns),
        "excluded_leakage_features": [
            "tip_amount",
            "tip_rate",
            "tip_rate_pct",
            "tip_rate_total",
            "total_amount",
        ],
    }

    save_json(metrics, TABLE_DIR / "model_metrics.json")
    joblib.dump(model, MODEL_DIR / "tip_payment_logistic_pipeline.joblib")

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=["실제 미지급", "실제 지급"],
        columns=["예측 미지급", "예측 지급"],
    ).to_csv(
        TABLE_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    sns.set_theme(
    style="white",
    font="AppleGothic",
    rc={"axes.unicode_minus": False},
    )

    plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False  

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["예측 미지급", "예측 지급"],
        yticklabels=["실제 미지급", "실제 지급"],
    )
    plt.title("팁 지급 여부 예측 혼동행렬")
    plt.xlabel("예측값")
    plt.ylabel("실제값")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "confusion_matrix.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # 로지스틱 회귀 계수 저장: 양수일수록 팁 지급(1) 가능성을 높이는 방향
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_df = pd.DataFrame(
        {
            "feature": list(X.columns),
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)
    coefficient_df.to_csv(
        TABLE_DIR / "logistic_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics


def main() -> None:
    ensure_directories()
    configure_korean_font()
    df = load_processed()
    metrics = train_and_evaluate(df)

    print("\n[모델 평가]")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(
        f"\n모델 저장: {MODEL_DIR / 'tip_payment_logistic_pipeline.joblib'}"
    )


if __name__ == "__main__":
    main()
