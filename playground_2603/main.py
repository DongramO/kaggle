"""
skeleton 메인: data_loader → prepare_data → modeling 순서로 각 모듈 참조.
EDA는 run_eda.py로 별도 실행. run_eda=True 시 main에서도 호출 가능.
"""

from pathlib import Path
import json

from data_loader import load_all
from prepare_data import prepare_data, get_feature_columns, filter_correlated_features
from modeling import run_ensemble_models, run_stacking_ensemble

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
    X, y, X_test = prepare_data(train_df, test_df, target_col=target_col)
    drop_cols = filter_correlated_features(X)
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    # Optuna에서 저장한 하이퍼파라미터 로드 (키: catboost, lightgbm, xgboost)
    with open("best_hyperparameters.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    model_types = ["catboost", "lightgbm", "xgboost"]

    # 4. modeling: 스태킹 앙상블 (Level0: tree+MLP, Level1: LogisticRegression)
    ensemble_pred = run_stacking_ensemble(
        X=X,
        y=y,
        X_test=X_test,
        best_params_dict=best,
        model_types=model_types,
        use_mlp=True,
    )
    submission_df[target_col] = ensemble_pred
    submission_df.to_csv("submission_ensemble.csv", index=False)

if __name__ == "__main__":
    main(data_dir="data", target_col="Churn", id_col="id")
