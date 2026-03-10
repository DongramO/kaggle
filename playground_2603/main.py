"""
skeleton 메인: data_loader → prepare_data → modeling 순서로 각 모듈 참조.
EDA는 run_eda.py로 별도 실행. run_eda=True 시 main에서도 호출 가능.
"""

from pathlib import Path
import json

from data_loader import load_all
from prepare_data import prepare_data, get_feature_columns
from modeling import run_ensemble_models

def main(
    data_dir: str | Path = "data",
    target_col: str | None = None,
    id_col: str | None = "id",
    run_eda: bool = False,
):
    """전체 파이프라인: 로드 → 전처리 → (선택) EDA → 모델링."""
    data_dir = Path(data_dir)

    # 1. data_loader
    train_df, test_df, submission_df = load_all(data_dir)

    # 2. prepare_data
    train_prepared = prepare_data(train_df, target_col=target_col)
    test_prepared = prepare_data(test_df, target_col=target_col)
    feature_cols = get_feature_columns(train_prepared, target_col=target_col, id_col=id_col)
    # 학습/평가/예측용 데이터
    X = train_prepared[feature_cols]
    y = train_prepared[target_col]
    X_test = test_prepared[feature_cols]

    # Optuna에서 저장한 하이퍼파라미터 로드 (키: catboost, lightgbm, xgboost)
    with open("best_hyperparameters.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    model_types = ["catboost", "lightgbm", "xgboost"]

    # 4. modeling: 모델별 학습 + 테스트 예측 앙상블
    ensemble_pred = run_ensemble_models(
        X=X,
        y=y,
        X_test=X_test,
        best_params_dict=best,
        model_types=model_types,
        weights=None,
    )
    submission_df[target_col] = ensemble_pred
    submission_df.to_csv("submission_ensemble.csv", index=False)

if __name__ == "__main__":
    main(data_dir="data", target_col="Churn", id_col="id")
