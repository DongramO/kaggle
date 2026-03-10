"""
모델링 모듈 (독립 스켈레톤)
- 학습 / 예측 / 평가 진입점
"""

import pandas as pd
import numpy as np


def train_model(X: pd.DataFrame, y: pd.Series, model=None):
    """
    모델 학습.
    model이 None이면 기본 스켈레톤만 반환 (실제 모델은 외부에서 주입).
    """
    if model is None:
        return None
    model.fit(X, y)
    return model


def predict(model, X: pd.DataFrame) -> np.ndarray:
    """예측."""
    if model is None:
        return np.zeros(len(X))
    return model.predict(X)


def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray, metric: str = "accuracy") -> float:
    """평가 지표 계산 (스켈레톤: metric에 따라 확장)."""
    from sklearn.metrics import accuracy_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    return float(accuracy_score(y_true, y_pred))
