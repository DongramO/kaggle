"""
Optuna 하이퍼파라미터 최적화 (독립 실행)
- python optuna.py 로 별도 실행
- skeleton의 data_loader, prepare_data만 참조
"""

import json
from pathlib import Path

import optuna
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from data_loader import load_all
from prepare_data import prepare_data, get_feature_columns


# ========== 설정 (실행 시 수정) ==========
DATA_DIR = Path(__file__).parent / "data"
TARGET_COL = "Churn"
ID_COL = "id"
N_TRIALS = 30
N_FOLDS = 5
RANDOM_STATE = 42
OUTPUT_PATH = Path(__file__).parent / "best_hyperparameters.json"
MODEL_TYPES = ["lightgbm", "catboost", "xgboost"]

def _get_data():
    """데이터 로드 및 전처리."""
    train_df, _, _ = load_all(DATA_DIR)
    train_prepared = prepare_data(train_df, target_col=TARGET_COL)
    feature_cols = get_feature_columns(train_prepared, target_col=TARGET_COL, id_col=ID_COL)

    X = train_prepared[feature_cols]
    y = train_prepared[TARGET_COL]

    return X, y


def create_objective_lgbm(X, y):
    """LightGBM용 Optuna objective 함수."""

    def objective(trial):
        import lightgbm as lgb

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "verbosity": -1,
        }

        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=N_FOLDS, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


def create_objective_catboost(X, y):
    """CatBoost용 Optuna objective 함수."""

    def objective(trial):
        from catboost import CatBoostClassifier

        params = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": RANDOM_STATE,
            "verbose": 0,
        }

        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X, y, cv=N_FOLDS, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


def create_objective_xgb(X, y):
    """XGBoost용 Optuna objective 함수."""

    def objective(trial):
        import xgboost as xgb

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-6, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }

        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=N_FOLDS, scoring="roc_auc", n_jobs=-1)
        return scores.mean()

    return objective


# 모델별 Optuna objective 함수만 매핑. GPU/verbose/bootstrap_type 등은 best_hyperparameters.json에서 관리.
OBJECTIVE_FNS = {
    "lightgbm": create_objective_lgbm,
    "catboost": create_objective_catboost,
    "xgboost": create_objective_xgb,
}


def run_optuna():
    """Optuna 최적화 실행 및 결과 저장. model_types 미지정 시 MODEL_TYPES 사용."""
    model_types = MODEL_TYPES
    print("데이터 로드 중...")
    X, y = _get_data()
    print(f"X: {X.shape}, y: {y.shape}")

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    for model_type in model_types:
        if model_type not in OBJECTIVE_FNS:
            print(f"건너뜀(미지원): {model_type}")
            continue
        obj_fn = OBJECTIVE_FNS[model_type]
        print(f"\n=== {model_type} Optuna 시작 (n_trials={N_TRIALS}) ===")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=5),
        )
        study.optimize(obj_fn(X, y), n_trials=N_TRIALS, show_progress_bar=True)
        # 기존 JSON 값(bootstrap_type, task_type 등) 유지하고 Optuna 결과만 갱신
        existing = all_params.get(model_type, {})
        all_params[model_type] = {**existing, **study.best_params}
        print(f"{model_type} Best AUC: {study.best_value:.4f}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_optuna()
