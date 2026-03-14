"""
skeleton 메인: data_loader → prepare_data → modeling 순서로 각 모듈 참조.
EDA는 run_eda.py로 별도 실행. run_eda=True 시 main에서도 호출 가능.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data_loader import load_all
from prepare_data import prepare_data, filter_correlated_features
from modeling import run_stacking_ensemble, train_model, diagnose_stacking
from history import save_auc_log, save_fi_history, plot_fi_history


def main(
    data_dir: str | Path = "data",
    target_col: str = "Churn",
    id_col: str = "id",
):
    """전체 파이프라인: 로드 → 전처리 → 모델링 → 저장."""
    data_dir = Path(data_dir)

    print("[1/5] 데이터 로딩 중...")
    train_df, test_df, submission_df = load_all(data_dir)
    print(f"      train: {train_df.shape}, test: {test_df.shape}")

    print("[2/5] 데이터 전처리 중...")
    X, y, X_test = prepare_data(train_df, test_df, target_col=target_col)
    # drop_cols = filter_correlated_features(X)
    # X = X.drop(columns=drop_cols)
    # X_test = X_test.drop(columns=drop_cols)

    print("[3/5] 하이퍼파라미터 로드 중...")
    with open(Path(__file__).parent / "best_hyperparameters.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    model_types = ["catboost", "lightgbm", "xgboost"]

    print("[4/5] 스태킹 앙상블 학습 중...")
    ensemble_pred, _, fi_accumulator, oof_preds, meta_model = run_stacking_ensemble(
        X=X,
        y=y,
        X_test=X_test,
        best_params_dict=best,
        model_types=model_types,
        use_mlp=False,
        use_tabnet=False,
        use_ft_transformer=False,
    )

    print("[5/5] submission 저장 중...")
    submission_df[target_col] = ensemble_pred
    submission_df.to_csv("submission_ensemble.csv", index=False)
    print("      submission_ensemble.csv saved")

    auc_dict = {m: roc_auc_score(y, oof_preds[m]) for m in oof_preds}
    meta_oof = meta_model.predict_proba(
        np.column_stack([oof_preds[m] for m in oof_preds])
    )[:, 1]
    auc_dict["ensemble"] = roc_auc_score(y, meta_oof)
    save_auc_log(auc_dict, run_type=f"ensemble:{'+'.join(model_types)}")

    diagnose_stacking(oof_preds, y, meta_model, n_train=len(X))

    save_fi_history(fi_accumulator)
    plot_fi_history()


def run_single_model(model_type: str, data_dir: str | Path = "data", target_col: str = "Churn"):
    """단일 모델로 OOF AUC 확인 + submission 저장. 일반화 성능 개별 파악용."""
    data_dir = Path(data_dir)
    train_df, test_df, submission_df = load_all(data_dir)
    X, y, X_test = prepare_data(train_df, test_df, target_col=target_col)

    with open(Path(__file__).parent / "best_hyperparameters.json", "r", encoding="utf-8") as f:
        best = json.load(f)

    print(f"\n=== {model_type} 단독 실행 ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y.iloc[train_idx]
        model = train_model(X_tr, y_tr, model_type, best[model_type])
        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_test)[:, 1] / 5

    out_path = Path(__file__).parent / f"submission_{model_type}.csv"
    submission_df[target_col] = test_pred
    submission_df.to_csv(out_path, index=False)
    print(f"저장: {out_path}")

    save_auc_log({model_type: roc_auc_score(y, oof)}, run_type=f"single:{model_type}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("catboost", "lightgbm", "xgboost"):
        run_single_model(sys.argv[1], data_dir="data", target_col="Churn")
    else:
        main(data_dir="data", target_col="Churn", id_col="id")
