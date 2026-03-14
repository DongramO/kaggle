"""
skeleton 메인: data_loader → prepare_data → modeling 순서로 각 모듈 참조.
EDA는 run_eda.py로 별도 실행. run_eda=True 시 main에서도 호출 가능.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from data_loader import load_all
from prepare_data import prepare_data, filter_correlated_features, FE_FLAGS_PER_MODEL, apply_model_fe
from modeling import run_stacking_ensemble, train_model, diagnose_stacking
from history import save_auc_log, save_fi_history, plot_fi_history


def _auto_commit_if_best(current_auc: float, run_label: str) -> None:
    """현재 ensemble OOF AUC가 역대 최고면 자동 git commit."""
    auc_log_path = Path(__file__).parent / "auc_history.json"
    if not auc_log_path.exists():
        return

    with open(auc_log_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    prev_best = max(
        (entry["auc"].get("ensemble", 0.0) for entry in log[:-1]),
        default=0.0,
    )

    if current_auc <= prev_best:
        print(f"\n[Auto-commit] 스킵 (현재 {current_auc:.5f} ≤ 역대 최고 {prev_best:.5f})")
        return

    print(f"\n[Auto-commit] 신기록! {prev_best:.5f} → {current_auc:.5f}")
    msg = f"신기록 OOF AUC {current_auc:.5f} ({run_label})"

    repo_root = Path(__file__).parent.parent
    try:
        subprocess.run(
            ["git", "add",
             "playground_2603/auc_history.json",
             "playground_2603/fi_history.json",
             "playground_2603/best_hyperparameters.json",
             "playground_2603/main.py",
             "playground_2603/modeling.py",
             "playground_2603/prepare_data.py",
             "playground_2603/history.py",
             "playground_2603/run_optuna.py",
             ],
            cwd=repo_root, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_root, check=True,
        )
        print(f"[Auto-commit] 완료: {msg}")
    except subprocess.CalledProcessError as e:
        print(f"[Auto-commit] 실패: {e}")


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
    drop_cols = filter_correlated_features(X)
    X = X.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    print("[3/5] 하이퍼파라미터 로드 중...")
    with open(Path(__file__).parent / "best_hyperparameters.json", "r", encoding="utf-8") as f:
        best = json.load(f)
    model_types = ["catboost", "lightgbm", "xgboost"]

    print("[4/5] 스태킹 앙상블 학습 중...")
    X_per_model, X_test_per_model = {}, {}
    for mt in model_types:
        X_fe      = apply_model_fe(X,      FE_FLAGS_PER_MODEL.get(mt, {}))
        X_test_fe = apply_model_fe(X_test, FE_FLAGS_PER_MODEL.get(mt, {}))
        drop_fe = filter_correlated_features(X_fe)
        X_per_model[mt]      = X_fe.drop(columns=drop_fe)
        X_test_per_model[mt] = X_test_fe.drop(columns=[c for c in drop_fe if c in X_test_fe.columns])
        added   = [c for c in X_per_model[mt].columns if c not in X.columns]
        removed = [c for c in drop_fe if c in X.columns]
        print(f"      [{mt}] FE 추가: {added if added else '없음'} | 상관 제거: {removed if removed else '없음'}")

    ensemble_pred, _, fi_accumulator, oof_preds, meta_model = run_stacking_ensemble(
        X=X,
        y=y,
        X_test=X_test,
        best_params_dict=best,
        model_types=model_types,
        use_mlp=False,
        use_tabnet=False,
        use_ft_transformer=False,
        X_per_model=X_per_model,
        X_test_per_model=X_test_per_model,
        seeds=[42, 0, 7],
        n_folds=5,
        compute_perm_importance=False,
        meta_method="ridge",
    )

    print("[5/5] submission 저장 중...")
    submission_df[target_col] = ensemble_pred
    submission_df.to_csv("submission_ensemble.csv", index=False)
    print("      submission_ensemble.csv saved")

    auc_dict = {m: roc_auc_score(y, oof_preds[m]) for m in oof_preds}
    meta_oof = np.clip(
        meta_model.predict(np.column_stack([oof_preds[m] for m in oof_preds])),
        0.0, 1.0,
    )
    auc_dict["ensemble"] = roc_auc_score(y, meta_oof)
    run_label = f"ensemble:{'+'.join(model_types)}"
    save_auc_log(auc_dict, run_type=run_label)

    diagnose_stacking(oof_preds, y, meta_model, n_train=len(X))

    save_fi_history(fi_accumulator)
    plot_fi_history()

    _auto_commit_if_best(auc_dict["ensemble"], run_label)


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
