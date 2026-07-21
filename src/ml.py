"""
==============================================================================
 프로그램명 : ml
 작성자     : 광주 4반 박소미
 작성일     : 2026-07-21
------------------------------------------------------------------------------
 [프로그램 설명]
   sklearn Pipeline 기반 전처리·모델 학습·평가 및 joblib 모델 저장을 담당한다.
     1) build_pipeline() : sklearn.pipeline.Pipeline 구성 (ColumnTransformer + 모델)
     2) train_and_evaluate() : 학습 후 평가 지표(정확도·F1·macro F1·클래스별 recall 등) 출력
     3) save_model() : joblib으로 models/ 에 모델 저장
     4) compute_mcfadden_r2() : LogisticRegression의 McFadden 의사결정계수 산출
        (분류 모델엔 일반 회귀의 결정계수가 없어, 대응 지표로 추가)
     5) run_target_pipeline() : 위 함수들을 묶어 타겟 하나(has_tip 또는 high_tip)에
        대해 LogisticRegression·RandomForest 학습·평가·저장을 한 번에 수행

 [변경 내역]
   v1.0  2026-07-21  박소미  최초 작성 (구조만, 로직 미구현)
   v1.1  2026-07-21  장상민  5개 함수 로직 구현
==============================================================================
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
RF_SAMPLE_SIZE = 300_000

CAT_FEATURES = ["VendorID", "RatecodeID"]
BIN_FEATURES = ["is_weekend", "is_congestion"]

# WHY: has_tip(지급여부)는 fare_amount를 피처로 써도 순환논리 문제가 없지만,
# high_tip(tip_pct>=20%)은 tip_pct = tip_amount/fare_amount로 정의되어 있어
# fare_amount를 피처로 넣으면 분모를 그대로 다시 넣는 셈이라 제외한다.
_BASE_NUM_FEATURES = ["trip_distance", "trip_duration_min", "pickup_hour", "passenger_count"]
TARGET_CONFIGS = {
    "has_tip": {
        "label": "팁 지급 여부 (has_tip)",
        "num_features": [*_BASE_NUM_FEATURES, "fare_amount"],
    },
    "high_tip": {
        "label": "팁 비율 20% 이상 (high_tip)",
        "num_features": _BASE_NUM_FEATURES,
    },
}


# ---------------------------------------------------------------------------
# 1) build_pipeline
# ---------------------------------------------------------------------------
def build_pipeline(
    model: ClassifierMixin, num_features: list[str], cat_features: list[str]
) -> Pipeline:
    """수치형 StandardScaler + 범주형 OneHotEncoder 전처리와 model을 묶은 Pipeline을 만든다."""
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder="passthrough",
    )
    return Pipeline([("prep", preprocess), ("model", model)])


# ---------------------------------------------------------------------------
# 2) train_and_evaluate
# ---------------------------------------------------------------------------
# WHY: accuracy만 보면 클래스 불균형(has_tip 베이스라인 90%대) 때문에 실제 성능을
# 과장해 보일 수 있어, macro F1과 클래스별(양성/음성) recall을 함께 산출한다.
def train_and_evaluate(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """pipe를 학습시키고 평가지표(accuracy/precision/recall/f1/macro_f1)를 dict로 반환한다."""
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    y_pred = pipe.predict(X_test)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall(양성)": recall_score(y_test, y_pred),
        "recall(음성)": recall_per_class[0],
        "f1(양성)": f1_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro"),
        "fit_time_sec": fit_time,
    }
    logger.info("[train_and_evaluate] %s", metrics)
    logger.info(
        "[train_and_evaluate] 클래스별 리포트:\n%s", classification_report(y_test, y_pred, digits=4)
    )
    return {
        "pipe": pipe,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


# ---------------------------------------------------------------------------
# 3) save_model
# ---------------------------------------------------------------------------
def save_model(pipe: Pipeline, path: Path) -> Path:
    """학습된 Pipeline을 joblib으로 path에 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)
    logger.info("[save_model] 저장 완료 : %s", path)
    return path


# ---------------------------------------------------------------------------
# 4) compute_mcfadden_r2
# ---------------------------------------------------------------------------
# WHY: LogisticRegression은 분류 모델이라 (수정된) 결정계수가 정의되지 않는다.
# 대신 로지스틱 회귀의 대응 지표인 McFadden 의사결정계수를 계산한다.
def compute_mcfadden_r2(pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """train 데이터 기준 McFadden 의사결정계수(R²)와 수정된 R²를 계산해 반환한다."""
    p_model = pipe.predict_proba(X_train)[:, 1]
    n = len(y_train)
    ll_model = -log_loss(y_train, p_model, normalize=False)

    p0 = y_train.mean()
    ll_null = np.sum(y_train * np.log(p0) + (1 - y_train) * np.log(1 - p0))

    k = pipe.named_steps["model"].coef_.shape[1]
    r2 = 1 - ll_model / ll_null
    adj_r2 = 1 - (ll_model - k) / ll_null

    logger.info("[compute_mcfadden_r2] R^2=%.4f, 수정된 R^2=%.4f (n=%d, k=%d)", r2, adj_r2, n, k)
    return {"n": n, "k": k, "r2": r2, "adj_r2": adj_r2}


# ---------------------------------------------------------------------------
# 5) run_target_pipeline
# ---------------------------------------------------------------------------
def run_target_pipeline(
    card: pd.DataFrame, target_col: str, num_features: list[str], models_dir: Path
) -> dict:
    """target_col(has_tip 또는 high_tip)에 대해 LogisticRegression·RandomForest를 학습·평가한다.

    joblib으로 저장한 뒤, 결과 dict(metrics_df, coef_df, mcfadden 등)를 반환한다.
    """
    X = card[num_features + CAT_FEATURES + BIN_FEATURES]
    y = card[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    lr_pipe = build_pipeline(LogisticRegression(max_iter=1000), num_features, CAT_FEATURES)
    lr_result = train_and_evaluate(lr_pipe, X_train, y_train, X_test, y_test)

    # WHY: RandomForest를 268만행 전체로 학습하면 시간이 오래 걸려, train 중 일부만
    # 층화추출해 학습한다 (예측·평가는 전체 test set에 대해 그대로 수행).
    rf_sample_n = min(RF_SAMPLE_SIZE, len(X_train))
    X_train_rf, _, y_train_rf, _ = train_test_split(
        X_train, y_train, train_size=rf_sample_n, stratify=y_train, random_state=RANDOM_STATE
    )
    rf_model = RandomForestClassifier(
        n_estimators=150, max_depth=12, n_jobs=-1, random_state=RANDOM_STATE
    )
    rf_pipe = build_pipeline(rf_model, num_features, CAT_FEATURES)
    rf_result = train_and_evaluate(rf_pipe, X_train_rf, y_train_rf, X_test, y_test)

    metrics_df = pd.DataFrame(
        [
            {"model": "LogisticRegression", **lr_result["metrics"]},
            {"model": "RandomForest", **rf_result["metrics"]},
        ]
    )

    feature_names = lr_pipe.named_steps["prep"].get_feature_names_out()
    coefs = lr_pipe.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {"feature": feature_names, "coef": coefs, "odds_ratio": np.exp(coefs)}
    ).sort_values("coef", ascending=False)

    mcfadden = compute_mcfadden_r2(lr_pipe, X_train, y_train)

    save_model(lr_pipe, models_dir / f"{target_col}_LogisticRegression.joblib")
    save_model(rf_pipe, models_dir / f"{target_col}_RandomForest.joblib")

    return {
        "target_col": target_col,
        "num_features": num_features,
        "baseline_acc": baseline_acc,
        "metrics_df": metrics_df,
        "coef_df": coef_df,
        "mcfadden": mcfadden,
        "pipe_lr": lr_pipe,
        "pipe_rf": rf_pipe,
    }


# ---------------------------------------------------------------------------
# 실행부
# ---------------------------------------------------------------------------
def main() -> None:
    from src.clean import add_time_features, add_tip_pct, filter_valid_trips
    from src.load import load_with_pandas

    print("\n=== 1단계 : 데이터 준비 ===\n")
    df, _ = load_with_pandas()
    df = filter_valid_trips(df)
    df = add_tip_pct(df)
    df = add_time_features(df)

    models_dir = Path(__file__).resolve().parent.parent / "models"

    print("\n=== 2단계 : 타겟별 ML Pipeline (has_tip, high_tip) ===\n")
    results = {}
    for target_col, cfg in TARGET_CONFIGS.items():
        print(f"--- {cfg['label']} ---")
        results[target_col] = run_target_pipeline(df, target_col, cfg["num_features"], models_dir)

    for target_col, res in results.items():
        best_acc = res["metrics_df"]["accuracy"].max()
        checkpoint_msg = f"Checkpoint : {target_col} 모델이 베이스라인보다 나아야 한다"
        assert best_acc > res["baseline_acc"], checkpoint_msg

    print(f"\n\nCheckpoint : has_tip/high_tip 두 타겟 모두 학습·평가·저장 완료 ({models_dir})")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        logger.error("검증 실패 : %s", e)
    except Exception as e:  # 예상치 못한 오류의 최종 방어선
        logger.error("예상치 못한 오류 : %s: %s", type(e).__name__, e)
