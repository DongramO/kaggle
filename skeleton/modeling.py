"""
모델링 모듈 (독립 스켈레톤)
- 학습 / 예측 / 평가 진입점
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

def train_model(X: pd.DataFrame, y: pd.Series, model_type: str, best_params: dict):
    """
    모델 타입과 하이퍼파라미터에 따라 모델을 생성·학습.
    GPU/로그 관련 옵션은 best_hyperparameters.json에서 제어한다.
    """
    params = dict(best_params)

    if model_type == "catboost":
        model = CatBoostClassifier(**params)
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(**params)
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X, y)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """예측."""
    return model.predict(X)


def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, metric: str = "accuracy") -> float:
    """평가 지표 계산. metric: 'accuracy' | 'auc'."""
    

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "auc":
        return float(roc_auc_score(y_true, y_pred))
    return float(accuracy_score(y_true, y_pred))


def run_ensemble_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    best_params_dict: dict,
    model_types: list[str],
    weights: dict[str, float] | None = None,
) -> np.ndarray:
    """
    여러 모델을 학습하고 테스트 예측을 가중 평균 앙상블.
    best_params_dict: {model_type: params} 형태.
    weights=None이면 OOF AUC에 비례해 자동 가중치 적용.
    """
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    test_pred_weighted = np.zeros(len(X_test))
    oof_ensemble_weighted = np.zeros(len(X))
    weight_sum = 0.0
    use_auc_weights = weights is None
    if use_auc_weights:
        weights = {}

    for model_type in model_types:
        best_params = best_params_dict[model_type]
        oof_proba = np.zeros(len(X))
        fold_test_pred = np.zeros(len(X_test))

        for train_idx, valid_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            model = train_model(X_tr, y_tr, model_type, best_params)

            if hasattr(model, "predict_proba"):
                oof_proba[valid_idx] = model.predict_proba(X_val)[:, 1]
                fold_test_pred += model.predict_proba(X_test)[:, 1] / n_folds
            else:
                oof_proba[valid_idx] = predict(model, X_val)
                fold_test_pred += predict(model, X_test) / n_folds

        auc_score = evaluate(y, oof_proba, metric="auc")
        print(f"{model_type} OOF AUC: {auc_score:.4f}")

        if use_auc_weights:
            w = max(1e-6, auc_score)
            weights[model_type] = w
        else:
            w = weights.get(model_type, 1.0)
        test_pred_weighted += w * fold_test_pred
        oof_ensemble_weighted += w * oof_proba
        weight_sum += w

    if weight_sum == 0.0:
        raise ValueError("No predictions were generated for ensemble.")

    if use_auc_weights:
        total = sum(weights.values())
        normalized = {k: v / total for k, v in weights.items()}
        print("Ensemble weights (OOF AUC 비율):", {k: f"{v:.3f}" for k, v in normalized.items()})

    ensemble_oof = oof_ensemble_weighted / weight_sum
    ensemble_auc = evaluate(y, ensemble_oof, metric="auc")
    print(f"Ensemble OOF AUC: {ensemble_auc:.4f}")

    return test_pred_weighted / weight_sum
