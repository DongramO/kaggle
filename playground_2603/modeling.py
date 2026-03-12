"""
모델링 모듈 (독립 스켈레톤)
- 학습 / 예측 / 평가 진입점
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from pytorch_tabnet.tab_model import TabNetClassifier
import matplotlib.pyplot as plt


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


def train_mlp(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray):
    """
    StandardScaler 적용 후 MLP 학습.
    스케일러와 모델을 튜플로 반환해 추론 시 재사용.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        learning_rate_init=1e-3,
    )
    mlp.fit(X_tr_scaled, y_tr)
    return mlp, scaler, X_val_scaled


def train_tabnet(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """
    StandardScaler 적용 후 TabNet 학습.
    스케일러와 모델을 반환해 추론 시 재사용.
    """
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    tabnet = TabNetClassifier(
        n_d=32, n_a=32,
        n_steps=3,
        gamma=1.3,
        momentum=0.02,
        optimizer_params={"lr": 2e-3},
        scheduler_params={"step_size": 50, "gamma": 0.9},
        scheduler_fn=None,
        mask_type="sparsemax",
        verbose=0,
        seed=42,
    )
    tabnet.fit(
        X_tr_scaled, y_tr.astype(int),
        eval_set=[(X_val_scaled, y_val.astype(int))],
        eval_metric=["auc"],
        max_epochs=200,
        patience=20,
        batch_size=1024,
        virtual_batch_size=256,
    )
    return tabnet, scaler, X_val_scaled


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


def get_feature_importance(model, feature_names: list[str], model_type: str) -> pd.DataFrame:
    """모델에서 feature importance를 추출해 DataFrame으로 반환."""
    if model_type == "catboost":
        importances = model.get_feature_importance()
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return pd.DataFrame()

    return pd.DataFrame({"feature": feature_names, "importance": importances})


def plot_feature_importance(fi_dict: dict[str, pd.DataFrame], top_n: int = 30, save_path: str = "feature_importance.png"):
    """모델별 feature importance를 subplot으로 시각화."""
    n_models = len(fi_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 10))
    if n_models == 1:
        axes = [axes]

    for ax, (model_type, fi_df) in zip(axes, fi_dict.items()):
        fi_sorted = fi_df.sort_values("importance", ascending=False).head(top_n)
        ax.barh(fi_sorted["feature"][::-1], fi_sorted["importance"][::-1])
        ax.set_title(f"{model_type} Feature Importance (Top {top_n})")
        ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Feature importance plot saved: {save_path}")


def analyze_mlp_permutation_importance(
    mlp: MLPClassifier,
    scaler: StandardScaler,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    tree_fi_dict: dict[str, pd.DataFrame],
    top_n: int = 20,
    save_path: str = "mlp_permutation_importance.png",
):
    """
    MLP의 permutation importance를 계산하고 트리 모델과 비교 시각화.
    트리에서 낮은데 MLP에서 높은 feature → 비선형 패턴 후보.
    """
    feature_names = list(X_val.columns)
    X_val_scaled = scaler.transform(X_val.values)

    result = permutation_importance(mlp, X_val_scaled, y_val, n_repeats=10, random_state=42, scoring="roc_auc")
    mlp_fi = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n=== MLP Permutation Importance (Top 20) ===")
    print(mlp_fi.head(top_n).to_string(index=False))

    # 트리 평균 importance와 비교 (각 모델 정규화 후 평균)
    tree_avg = None
    for fi_df in tree_fi_dict.values():
        fi_norm = fi_df.set_index("feature")["importance"]
        fi_norm = fi_norm / (fi_norm.sum() + 1e-9)
        tree_avg = fi_norm if tree_avg is None else tree_avg.add(fi_norm, fill_value=0)
    tree_avg = (tree_avg / len(tree_fi_dict)).reset_index()
    tree_avg.columns = ["feature", "tree_avg"]

    if tree_avg is not None:

        mlp_norm = mlp_fi.copy()
        mlp_norm["importance"] = mlp_norm["importance"] / (mlp_norm["importance"].sum() + 1e-9)
        comparison = mlp_norm.merge(tree_avg, on="feature")
        comparison["nonlinear_signal"] = comparison["importance"] - comparison["tree_avg"]
        comparison = comparison.sort_values("nonlinear_signal", ascending=False)

        print("\n=== 비선형 패턴 후보 (MLP importance - Tree avg importance) ===")
        print("▲ 양수: MLP가 트리보다 중요하게 본 feature (비선형 관계 가능성)")
        print("▼ 음수: 트리가 MLP보다 중요하게 본 feature (선형/분기로 충분)")
        print(comparison[["feature", "importance", "tree_avg", "nonlinear_signal"]].head(top_n).to_string(index=False))

        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        fi_sorted = mlp_fi.head(top_n)
        axes[0].barh(fi_sorted["feature"][::-1], fi_sorted["importance"][::-1])
        axes[0].set_title(f"MLP Permutation Importance (Top {top_n})")
        axes[0].set_xlabel("Mean AUC decrease")

        comp_sorted = comparison.head(top_n)
        colors = ["steelblue" if v >= 0 else "salmon" for v in comp_sorted["nonlinear_signal"][::-1]]
        axes[1].barh(comp_sorted["feature"][::-1], comp_sorted["nonlinear_signal"][::-1], color=colors)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_title("비선형 신호 (MLP - Tree avg)")
        axes[1].set_xlabel("Importance difference (normalized)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"MLP vs Tree comparison plot saved: {save_path}")

    return mlp_fi


def run_ensemble_models(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    best_params_dict: dict,
    model_types: list[str],
    weights: dict[str, float] | None = None,
    show_feature_importance: bool = True,
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

    feature_names = list(X.columns)
    fi_accumulator: dict[str, pd.DataFrame] = {}

    for model_type in model_types:
        best_params = best_params_dict[model_type]
        oof_proba = np.zeros(len(X))
        fold_test_pred = np.zeros(len(X_test))
        fi_sum = np.zeros(len(feature_names))

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

            fi_df = get_feature_importance(model, feature_names, model_type)
            if not fi_df.empty:
                fi_sum += fi_df["importance"].values

        auc_score = evaluate(y, oof_proba, metric="auc")
        print(f"{model_type} OOF AUC: {auc_score:.4f}")

        fi_mean = pd.DataFrame({"feature": feature_names, "importance": fi_sum / n_folds})
        fi_accumulator[model_type] = fi_mean.sort_values("importance", ascending=False).reset_index(drop=True)

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

    if show_feature_importance and fi_accumulator:
        print("\n=== Feature Importance (Top 20) ===")
        for model_type, fi_df in fi_accumulator.items():
            print(f"\n[{model_type}]")
            print(fi_df.head(20).to_string(index=False))
        plot_feature_importance(fi_accumulator, top_n=30, save_path="feature_importance.png")

    return test_pred_weighted / weight_sum


def run_stacking_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    best_params_dict: dict,
    model_types: list[str],
    use_mlp: bool = True,
    use_tabnet: bool = True,
    show_feature_importance: bool = True,
) -> np.ndarray:
    """
    스태킹 앙상블:
    - Level 0: CatBoost / LightGBM / XGBoost / MLP / TabNet (OOF 예측)
    - Level 1: Logistic Regression (메타 모델)
    MLP permutation importance와 트리 importance 비교로 비선형 패턴 분석.
    """
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    feature_names = list(X.columns)

    neural_types = (["mlp"] if use_mlp else []) + (["tabnet"] if use_tabnet else [])
    all_model_types = model_types + neural_types
    oof_preds = {m: np.zeros(len(X)) for m in all_model_types}
    test_preds = {m: np.zeros(len(X_test)) for m in all_model_types}
    fi_accumulator: dict[str, pd.DataFrame] = {}

    # MLP permutation importance용으로 마지막 fold val 보관
    last_mlp_model, last_mlp_scaler, last_val_idx = None, None, None

    # --- Level 0: 트리 모델 ---
    for model_type in model_types:
        best_params = best_params_dict[model_type]
        fi_sum = np.zeros(len(feature_names))

        for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
            print(f"  [{model_type}] fold {fold_i}/{n_folds} 학습 중...", flush=True)
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, _ = y.iloc[train_idx], y.iloc[valid_idx]

            model = train_model(X_tr, y_tr, model_type, best_params)
            oof_preds[model_type][valid_idx] = model.predict_proba(X_val)[:, 1]
            test_preds[model_type] += model.predict_proba(X_test)[:, 1] / n_folds

            fi_df = get_feature_importance(model, feature_names, model_type)
            if not fi_df.empty:
                fi_sum += fi_df["importance"].values

        auc_score = evaluate(y, oof_preds[model_type], metric="auc")
        print(f"[Level 0] {model_type} OOF AUC: {auc_score:.4f}")

        fi_mean = pd.DataFrame({"feature": feature_names, "importance": fi_sum / n_folds})
        fi_accumulator[model_type] = fi_mean.sort_values("importance", ascending=False).reset_index(drop=True)

    # --- Level 0: MLP ---
    if use_mlp:
        for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
            print(f"  [mlp] fold {fold_i}/{n_folds} 학습 중...", flush=True)
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, _ = y.iloc[train_idx], y.iloc[valid_idx]

            mlp, scaler, X_val_scaled = train_mlp(X_tr.values, y_tr.values, X_val.values)
            oof_preds["mlp"][valid_idx] = mlp.predict_proba(X_val_scaled)[:, 1]
            test_preds["mlp"] += mlp.predict_proba(scaler.transform(X_test.values))[:, 1] / n_folds
            last_mlp_model, last_mlp_scaler, last_val_idx = mlp, scaler, valid_idx

        mlp_auc = evaluate(y, oof_preds["mlp"], metric="auc")
        print(f"[Level 0] mlp OOF AUC: {mlp_auc:.4f}")

    # --- Level 0: TabNet ---
    if use_tabnet:
        for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
            print(f"  [tabnet] fold {fold_i}/{n_folds} 학습 중...", flush=True)
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

            tabnet, scaler, X_val_scaled = train_tabnet(
                X_tr.values, y_tr.values, X_val.values, y_val.values
            )
            oof_preds["tabnet"][valid_idx] = tabnet.predict_proba(X_val_scaled)[:, 1]
            test_preds["tabnet"] += tabnet.predict_proba(
                scaler.transform(X_test.values).astype(np.float32)
            )[:, 1] / n_folds

        tabnet_auc = evaluate(y, oof_preds["tabnet"], metric="auc")
        print(f"[Level 0] tabnet OOF AUC: {tabnet_auc:.4f}")

    # --- Level 1: 메타 모델 ---
    meta_X = np.column_stack([oof_preds[m] for m in all_model_types])
    meta_X_test = np.column_stack([test_preds[m] for m in all_model_types])

    meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    meta_model.fit(meta_X, y)
    meta_oof_proba = meta_model.predict_proba(meta_X)[:, 1]
    stacking_auc = evaluate(y, meta_oof_proba, metric="auc")
    print(f"\n[Level 1] Stacking OOF AUC: {stacking_auc:.4f}")

    meta_weights = dict(zip(all_model_types, meta_model.coef_[0]))
    print("Meta model coefficients:", {k: f"{v:.4f}" for k, v in meta_weights.items()})

    # Feature importance 출력 및 시각화
    if show_feature_importance and fi_accumulator:
        print("\n=== Feature Importance (Top 20) ===")
        for mt, fi_df in fi_accumulator.items():
            print(f"\n[{mt}]")
            print(fi_df.head(20).to_string(index=False))
        plot_feature_importance(fi_accumulator, top_n=30, save_path="feature_importance.png")

    final_pred = meta_model.predict_proba(meta_X_test)[:, 1]

    mlp_artifacts = None
    if use_mlp and last_mlp_model is not None:
        mlp_artifacts = {
            "mlp": last_mlp_model,
            "scaler": last_mlp_scaler,
            "val_idx": last_val_idx,
            "fi_accumulator": fi_accumulator,
        }

    return final_pred, mlp_artifacts
