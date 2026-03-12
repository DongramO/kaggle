"""
skeleton 메인: data_loader → prepare_data → modeling 순서로 각 모듈 참조.
EDA는 run_eda.py로 별도 실행. run_eda=True 시 main에서도 호출 가능.
"""

from pathlib import Path
import json

from data_loader import load_all
from prepare_data import prepare_data, get_feature_columns, filter_correlated_features
from modeling import run_ensemble_models, run_stacking_ensemble, analyze_mlp_permutation_importance

def main(
    data_dir: str | Path = "data",
    target_col: str | None = None,
    id_col: str | None = "id",
    run_eda: bool = False,
):
    """전체 파이프라인: 로드 → 전처리 → (선택) EDA → 모델링."""
    data_dir = Path(data_dir)

    # 1. data_loader
    print("[1/5] 데이터 로딩 중...")
    train_df, test_df, submission_df = load_all(data_dir)
    print(f"      train: {train_df.shape}, test: {test_df.shape}")

    # 2. prepare_data
    print("[2/5] 데이터 전처리 중...")
    X, y, X_test = prepare_data(train_df, test_df, target_col=target_col)
    drop_cols = filter_correlated_features(X)
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)
    print(f"      features: {X.shape[1]}개 (correlated {len(drop_cols)}개 제거)")

    # Optuna에서 저장한 하이퍼파라미터 로드 (키: catboost, lightgbm, xgboost)
    print("[3/5] 하이퍼파라미터 로드 중...")
    with open("best_hyperparameters.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    model_types = ["catboost", "lightgbm", "xgboost"]

    # 4. modeling: 스태킹 앙상블 (Level0: tree+MLP, Level1: LogisticRegression)
    print("[4/5] 스태킹 앙상블 학습 중...")
    ensemble_pred, mlp_artifacts = run_stacking_ensemble(
        X=X,
        y=y,
        X_test=X_test,
        best_params_dict=best,
        model_types=model_types,
        use_mlp=True,
    )

    # 5. submission 저장 (분석 전에 먼저 저장)
    print("[5/5] submission 저장 중...")
    submission_df[target_col] = ensemble_pred
    submission_df.to_csv("submission_ensemble.csv", index=False)
    print("      submission_ensemble.csv saved")

    # 6. MLP permutation importance 분석 (submission 저장 후 실행)
    if mlp_artifacts is not None:
        analyze_mlp_permutation_importance(
            mlp=mlp_artifacts["mlp"],
            scaler=mlp_artifacts["scaler"],
            X_val=X.iloc[mlp_artifacts["val_idx"]],
            y_val=y.iloc[mlp_artifacts["val_idx"]],
            tree_fi_dict=mlp_artifacts["fi_accumulator"],
            save_path="mlp_vs_tree_importance.png",
        )

if __name__ == "__main__":
    main(data_dir="data", target_col="Churn", id_col="id")
