"""
Optuna 하이퍼파라미터 최적화 (독립 실행)
- python run_optuna.py 로 별도 실행
- skeleton의 data_loader, prepare_data만 참조
"""

import json
from pathlib import Path

import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

from data_loader import load_all
from prepare_data import prepare_data, get_feature_columns, FE_FLAGS_PER_MODEL, apply_model_fe, filter_correlated_features


# ========== 문제별 수정 영역 ==========
DATA_DIR = Path(__file__).parent / "data"
TARGET_COL = "target"
ID_COL = "id"
N_TRIALS = 30
N_FOLDS = 5
RANDOM_STATE = 42
OUTPUT_PATH = Path(__file__).parent / "best_hyperparameters.json"
MODEL_TYPES = ["xgboost", "catboost", "lightgbm"]
# ======================================


def _get_data(model_type: str):
    """데이터 로드 및 전처리 — main.py와 동일한 파이프라인 적용.
    1) 기본 피처 상관 필터
    2) 모델별 FE 적용
    3) FE 포함 상관 필터
    """
    train_df, test_df, _ = load_all(DATA_DIR)
    X, y, _ = prepare_data(train_df, test_df, target_col=TARGET_COL)

    drop_base = filter_correlated_features(X)
    X = X.drop(columns=drop_base)

    X = apply_model_fe(X, FE_FLAGS_PER_MODEL.get(model_type, {}))

    drop_fe = filter_correlated_features(X)
    X = X.drop(columns=drop_fe)

    return X, y


def create_objective_lgbm(X, y):
    """LightGBM용 Optuna objective 함수."""

    def objective(trial):
        import lightgbm as lgb

        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 300, 1000),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":         trial.suggest_int("num_leaves", 20, 80),
            "min_child_samples":  trial.suggest_int("min_child_samples", 20, 100),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha":          trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 0.5, 10.0, log=True),
            "random_state": RANDOM_STATE,
            "verbosity": -1,
            "device": "gpu",
        }

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    return objective


def create_objective_catboost(X, y):
    """CatBoost용 Optuna objective 함수."""

    def objective(trial):
        from catboost import CatBoostClassifier

        params = {
            "iterations":       trial.suggest_int("iterations", 500, 1500),
            "learning_rate":    trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth":            trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg":      trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength":  trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "random_state": RANDOM_STATE,
            "verbose": 0,
            "task_type": "GPU",
            "bootstrap_type": "Bernoulli",
        }

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        model = CatBoostClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    return objective


def create_objective_xgb(X, y):
    """XGBoost용 Optuna objective 함수."""

    def objective(trial):
        import xgboost as xgb

        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 1500),
            "max_depth":         trial.suggest_int("max_depth", 3, 6),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 5, 60),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "colsample_bynode":  trial.suggest_float("colsample_bynode", 0.3, 0.8),
            "gamma":             trial.suggest_float("gamma", 0.0, 10.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 20.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
            "random_state": RANDOM_STATE,
            "verbosity": 0,
            "tree_method": "hist",
            "device": "cuda",
        }

        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        return scores.mean()

    return objective


OBJECTIVE_FNS = {
    "lightgbm": create_objective_lgbm,
    "catboost": create_objective_catboost,
    "xgboost": create_objective_xgb,
}


def run_optuna():
    """Optuna 최적화 실행 및 결과 저장."""
    model_types = MODEL_TYPES

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            all_params = json.load(f)
    else:
        all_params = {}

    for model_type in model_types:
        if model_type not in OBJECTIVE_FNS:
            print(f"건너뜀(미지원): {model_type}")
            continue

        print(f"\n데이터 로드 중... [{model_type}]")
        X, y = _get_data(model_type)
        print(f"X: {X.shape}, y: {y.shape}")

        obj_fn = OBJECTIVE_FNS[model_type]
        print(f"=== {model_type} Optuna 시작 (n_trials={N_TRIALS}) ===")
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
