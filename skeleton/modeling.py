"""
모델링 모듈 (독립 스켈레톤)
- 학습 / 예측 / 평가 / 스태킹 앙상블 진입점
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.inspection import permutation_importance
from pytorch_tabnet.tab_model import TabNetClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False


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


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: tuple, dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hidden_sizes: tuple = (256, 128, 64),
    dropout: float = 0.3,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 15,
    batch_size: int = 512,
):
    """
    PyTorch MLP 학습 (BatchNorm + Dropout + CosineAnnealing + EarlyStopping).
    RobustScaler로 전처리해 이상치 영향을 줄임.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_s).to(device),
        torch.from_numpy(y_tr.astype(np.float32)).to(device),
    )
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    model = _MLP(X_tr_s.shape[1], hidden_sizes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    X_val_t = torch.from_numpy(X_val_s).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        for X_batch, y_batch in tr_loader:
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, scaler, X_val_s, device


class _FeatureTokenizer(nn.Module):
    """각 피처를 독립적인 d_token 차원 벡터로 변환 + CLS 토큰 추가."""
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        self.cls    = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.normal_(self.cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        cls    = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class _FTTransformer(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021)."""
    def __init__(self, n_features: int, d_token: int = 64, n_heads: int = 8,
                 n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = _FeatureTokenizer(n_features, d_token)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_token * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.transformer(self.tokenizer(x))
        return self.head(out[:, 0]).squeeze(1)


def train_ft_transformer(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    d_token: int = 32,
    n_heads: int = 4,
    n_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 512,
):
    """FT-Transformer 학습. RobustScaler 전처리 + EarlyStopping."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  [ft_transformer] device: {device} | cuda available: {torch.cuda.is_available()}")

    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    tr_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr_s).to(device),
            torch.from_numpy(y_tr.astype(np.float32)).to(device),
        ),
        batch_size=batch_size, shuffle=True,
    )
    X_val_t = torch.from_numpy(X_val_s).to(device)
    y_val_t = torch.from_numpy(y_val.astype(np.float32)).to(device)

    model     = _FTTransformer(X_tr_s.shape[1], d_token, n_heads, n_layers, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    best_val_loss, best_state, wait = float("inf"), None, 0

    for _ in range(max_epochs):
        model.train()
        for X_b, y_b in tr_loader:
            optimizer.zero_grad()
            criterion(model(X_b), y_b).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, scaler, X_val_s, device


class _TorchWrapper:
    """PyTorch 모델을 predict_proba 인터페이스로 감싸는 공통 wrapper."""
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def fit(self, X, y):
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(X.astype(np.float32)).to(self.device)
            proba = torch.sigmoid(self.model(t)).cpu().numpy()
        return np.column_stack([1 - proba, proba])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return roc_auc_score(y, self.predict_proba(X)[:, 1])


_TorchMLPWrapper = _TorchWrapper


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


def plot_tree_permutation_importance(
    perm_fi: dict[str, pd.DataFrame],
    builtin_fi: dict[str, pd.DataFrame],
    top_n: int = 30,
    save_path: str = "permutation_importance.png",
) -> None:
    """
    트리 모델별 permutation importance(val 기준)와 내장 importance 비교 시각화.
    val AUC 기준이므로 음수 = 해당 피처 제거 시 오히려 성능 향상 → 제거 후보.
    """
    model_types = list(perm_fi.keys())
    fig, axes = plt.subplots(len(model_types), 2, figsize=(16, 6 * len(model_types)))
    if len(model_types) == 1:
        axes = [axes]

    print("\n========== Permutation Importance (val 기준) ==========")
    for i, mt in enumerate(model_types):
        perm_df = perm_fi[mt].sort_values("importance", ascending=False)
        print(f"\n[{mt}] Top {top_n}")
        print(perm_df.head(top_n).to_string(index=False))

        neg = perm_df[perm_df["importance"] < 0]
        if not neg.empty:
            print(f"  ⚠ AUC 감소 유발 피처 (제거 후보): {neg['feature'].tolist()}")

        ax = axes[i][0]
        top = perm_df.head(top_n)
        colors = ["steelblue" if v >= 0 else "salmon" for v in top["importance"][::-1]]
        ax.barh(top["feature"][::-1], top["importance"][::-1], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"[{mt}] Permutation Importance (val AUC 기준)")
        ax.set_xlabel("Mean AUC change")

        ax = axes[i][1]
        if mt in builtin_fi:
            bi = builtin_fi[mt].head(top_n)
            ax.barh(bi["feature"][::-1], bi["importance"][::-1], color="mediumseagreen")
            ax.set_title(f"[{mt}] Built-in Importance")
            ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n저장: {save_path}")


class _SimpleAverager:
    """predict_proba 없이 단순 평균을 반환하는 메타모델 호환 wrapper."""
    def fit(self, X, y):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.mean(axis=1)

    @property
    def coef_(self):
        return None


def run_stacking_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    best_params_dict: dict,
    model_types: list[str],
    use_mlp: bool = True,
    use_tabnet: bool = True,
    use_ft_transformer: bool = False,
    show_feature_importance: bool = True,
    X_per_model: dict[str, pd.DataFrame] | None = None,
    X_test_per_model: dict[str, pd.DataFrame] | None = None,
    seeds: list[int] = [42],
    n_folds: int = 5,
    compute_perm_importance: bool = False,
    meta_method: str = "ridge",
) -> np.ndarray:
    """
    멀티시드 스태킹 앙상블:
    - Level 0: CatBoost / LightGBM / XGBoost (각 seed × n_folds로 OOF 평균)
    - Level 1: meta_method="ridge" → RidgeCV | "average" → 단순 평균
    """
    neural_types = (["mlp"] if use_mlp else []) + (["tabnet"] if use_tabnet else []) + (["ft_transformer"] if use_ft_transformer else [])
    all_model_types = model_types + neural_types

    seeds_oof: dict[str, list[np.ndarray]] = {m: [] for m in all_model_types}
    test_preds: dict[str, np.ndarray] = {m: np.zeros(len(X_test)) for m in all_model_types}
    fi_sums: dict[str, np.ndarray] = {}
    fi_feature_names: dict[str, list[str]] = {}
    fi_counts: dict[str, int] = {mt: 0 for mt in model_types}
    perm_fi_sums: dict[str, np.ndarray] = {}
    perm_fi_counts: dict[str, int] = {mt: 0 for mt in model_types}

    last_mlp_model, last_mlp_scaler, last_val_idx = None, None, None

    for seed in seeds:
        print(f"\n{'='*20} Seed {seed} {'='*20}")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        seed_oof = {m: np.zeros(len(X)) for m in all_model_types}

        # --- Level 0: 트리 모델 ---
        for model_type in model_types:
            best_params = best_params_dict[model_type]
            X_m      = X_per_model.get(model_type, X)           if X_per_model      else X
            X_test_m = X_test_per_model.get(model_type, X_test) if X_test_per_model else X_test
            model_feature_names = list(X_m.columns)

            if model_type not in fi_sums:
                fi_sums[model_type] = np.zeros(len(model_feature_names))
                fi_feature_names[model_type] = model_feature_names
                perm_fi_sums[model_type] = np.zeros(len(model_feature_names))

            fold_train_aucs, fold_val_aucs = [], []

            for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
                print(f"  [seed={seed}][{model_type}] fold {fold_i}/{n_folds}", flush=True)
                X_tr, X_val = X_m.iloc[train_idx], X_m.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

                model = train_model(X_tr, y_tr, model_type, best_params)
                val_proba = model.predict_proba(X_val)[:, 1]
                seed_oof[model_type][valid_idx] = val_proba
                test_preds[model_type] += model.predict_proba(X_test_m)[:, 1] / (n_folds * len(seeds))

                train_auc = evaluate(y_tr, model.predict_proba(X_tr)[:, 1], metric="auc")
                val_auc   = evaluate(y_val, val_proba, metric="auc")
                fold_train_aucs.append(train_auc)
                fold_val_aucs.append(val_auc)
                print(f"    train AUC: {train_auc:.4f} | val AUC: {val_auc:.4f} | gap: {train_auc - val_auc:+.4f}")

                fi_df = get_feature_importance(model, model_feature_names, model_type)
                if not fi_df.empty:
                    fi_sums[model_type] += fi_df["importance"].values
                    fi_counts[model_type] += 1

                if compute_perm_importance and seed == seeds[-1] and fold_i == n_folds:
                    auc_scorer = lambda est, Xv, yv: roc_auc_score(yv, est.predict_proba(Xv)[:, 1])
                    pi = permutation_importance(
                        model, X_val, y_val,
                        n_repeats=5, random_state=42, scoring=auc_scorer,
                    )
                    perm_fi_sums[model_type] += pi.importances_mean
                    perm_fi_counts[model_type] += 1

            auc_score = evaluate(y, seed_oof[model_type], metric="auc")
            avg_gap   = np.mean(fold_train_aucs) - np.mean(fold_val_aucs)
            print(f"[seed={seed}][Level 0] {model_type} OOF AUC: {auc_score:.4f} | avg gap: {avg_gap:+.4f}"
                  + (" ⚠ 과적합 의심" if avg_gap > 0.01 else ""))

        # --- Level 0: MLP ---
        if use_mlp:
            for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
                print(f"  [seed={seed}][mlp] fold {fold_i}/{n_folds}", flush=True)
                X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
                mlp_raw, scaler, X_val_scaled, device = train_mlp(X_tr.values, y_tr.values, X_val.values, y_val.values)
                mlp = _TorchMLPWrapper(mlp_raw, device)
                seed_oof["mlp"][valid_idx] = mlp.predict_proba(X_val_scaled)[:, 1]
                test_preds["mlp"] += mlp.predict_proba(scaler.transform(X_test.values).astype(np.float32))[:, 1] / (n_folds * len(seeds))
                last_mlp_model, last_mlp_scaler, last_val_idx = mlp, scaler, valid_idx
            print(f"[seed={seed}][Level 0] mlp OOF AUC: {evaluate(y, seed_oof['mlp'], metric='auc'):.4f}")

        # --- Level 0: TabNet ---
        if use_tabnet:
            for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
                print(f"  [seed={seed}][tabnet] fold {fold_i}/{n_folds}", flush=True)
                X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
                tabnet, scaler, X_val_scaled = train_tabnet(X_tr.values, y_tr.values, X_val.values, y_val.values)
                seed_oof["tabnet"][valid_idx] = tabnet.predict_proba(X_val_scaled)[:, 1]
                test_preds["tabnet"] += tabnet.predict_proba(scaler.transform(X_test.values).astype(np.float32))[:, 1] / (n_folds * len(seeds))
            print(f"[seed={seed}][Level 0] tabnet OOF AUC: {evaluate(y, seed_oof['tabnet'], metric='auc'):.4f}")

        # --- Level 0: FT-Transformer ---
        if use_ft_transformer:
            for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
                print(f"  [seed={seed}][ft_transformer] fold {fold_i}/{n_folds}", flush=True)
                X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
                ft_raw, scaler, X_val_scaled, device = train_ft_transformer(X_tr.values, y_tr.values, X_val.values, y_val.values)
                ft = _TorchWrapper(ft_raw, device)
                seed_oof["ft_transformer"][valid_idx] = ft.predict_proba(X_val_scaled)[:, 1]
                test_preds["ft_transformer"] += ft.predict_proba(scaler.transform(X_test.values).astype(np.float32))[:, 1] / (n_folds * len(seeds))
            print(f"[seed={seed}][Level 0] ft_transformer OOF AUC: {evaluate(y, seed_oof['ft_transformer'], metric='auc'):.4f}")

        for m in all_model_types:
            seeds_oof[m].append(seed_oof[m].copy())

    # 시드 평균 OOF
    oof_preds = {m: np.mean(seeds_oof[m], axis=0) for m in all_model_types}

    print(f"\n{'='*20} 시드 평균 OOF AUC {'='*20}")
    for m in all_model_types:
        print(f"  {m}: {evaluate(y, oof_preds[m], metric='auc'):.4f}")

    # FI 평균
    fi_accumulator: dict[str, pd.DataFrame] = {}
    for mt in model_types:
        cnt = fi_counts[mt]
        if cnt > 0:
            fi_mean = pd.DataFrame({"feature": fi_feature_names[mt], "importance": fi_sums[mt] / cnt})
            fi_accumulator[mt] = fi_mean.sort_values("importance", ascending=False).reset_index(drop=True)

    # Permutation importance 집계 및 시각화
    if compute_perm_importance:
        perm_fi_accumulator: dict[str, pd.DataFrame] = {}
        for mt in model_types:
            cnt = perm_fi_counts[mt]
            if cnt > 0:
                perm_mean = pd.DataFrame({
                    "feature": fi_feature_names[mt],
                    "importance": perm_fi_sums[mt] / cnt,
                })
                perm_fi_accumulator[mt] = perm_mean.sort_values("importance", ascending=False).reset_index(drop=True)
        plot_tree_permutation_importance(perm_fi_accumulator, fi_accumulator)
        from history import save_perm_fi
        save_perm_fi(perm_fi_accumulator)

    # --- Level 1: 메타 모델 ---
    meta_X      = np.column_stack([oof_preds[m] for m in all_model_types])
    meta_X_test = np.column_stack([test_preds[m] for m in all_model_types])

    if meta_method == "average":
        meta_model = _SimpleAverager()
        meta_model.fit(meta_X, y)
        print(f"\n[Level 1] Simple Average (모델 수: {len(all_model_types)})")
    else:
        alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        meta_model = RidgeCV(alphas=alphas, cv=5)
        meta_model.fit(meta_X, y)
        print(f"\n[Level 1] Ridge best alpha: {meta_model.alpha_:.4f}")
        print("Meta model coefficients:", {k: f"{v:.4f}" for k, v in zip(all_model_types, np.ravel(meta_model.coef_))})

    meta_oof_proba = np.clip(meta_model.predict(meta_X), 0.0, 1.0)
    stacking_auc = evaluate(y, meta_oof_proba, metric="auc")
    print(f"[Level 1] Ensemble OOF AUC: {stacking_auc:.4f}")

    if show_feature_importance and fi_accumulator:
        print("\n=== Feature Importance (Top 20) ===")
        for mt, fi_df in fi_accumulator.items():
            print(f"\n[{mt}]")
            print(fi_df.head(20).to_string(index=False))
        plot_feature_importance(fi_accumulator, top_n=30, save_path="feature_importance.png")

    final_pred = np.clip(meta_model.predict(meta_X_test), 0.0, 1.0)

    mlp_artifacts = None
    if use_mlp and last_mlp_model is not None:
        mlp_artifacts = {"mlp": last_mlp_model, "scaler": last_mlp_scaler, "val_idx": last_val_idx}

    return final_pred, mlp_artifacts, fi_accumulator, oof_preds, meta_model


def diagnose_stacking(
    oof_preds: dict[str, np.ndarray],
    y: pd.Series,
    meta_model,
    n_train: int,
    save_path: str = "stacking_diagnosis.png",
) -> None:
    """
    스태킹 앙상블 진단:
    1. 모델별 OOF AUC 비교
    2. OOF 예측값 간 상관관계 (다양성 확인)
    3. 메타 모델 계수 (실제 기여 방향)
    """
    model_types = list(oof_preds.keys())

    auc_scores = {m: roc_auc_score(y, oof_preds[m]) for m in model_types}
    oof_df = pd.DataFrame(oof_preds)
    corr_matrix = oof_df.corr()
    coef = dict(zip(model_types, np.ravel(meta_model.coef_))) if meta_model.coef_ is not None else {}

    print("\n========== Stacking Diagnosis ==========")
    print(f"학습 데이터 크기: {n_train:,}행")
    print("\n[OOF AUC per model]")
    for m, auc in sorted(auc_scores.items(), key=lambda x: -x[1]):
        print(f"  {m:12s}: {auc:.4f}")
    if coef:
        print("\n[Meta model coefficients]")
        for m, c in sorted(coef.items(), key=lambda x: -x[1]):
            direction = "▲ 기여" if c > 0 else "▼ 역기여"
            print(f"  {m:12s}: {c:+.4f}  {direction}")
    else:
        print("\n[Meta model] Simple Average (계수 없음)")
    print("\n[OOF 예측 상관관계] — 1에 가까울수록 중복, 낮을수록 다양성 있음")
    print(corr_matrix.round(3).to_string())

    for m in ["mlp", "tabnet"]:
        if m not in auc_scores:
            continue
        tree_avg_auc = np.mean([v for k, v in auc_scores.items() if k not in ("mlp", "tabnet")])
        auc_gap = auc_scores[m] - tree_avg_auc
        avg_corr = corr_matrix[m].drop(m).mean()
        print(f"\n[{m} 진단]")
        print(f"  AUC gap vs tree avg : {auc_gap:+.4f}  {'(트리보다 낮음)' if auc_gap < 0 else '(트리와 대등 이상)'}")
        print(f"  트리 모델과 평균 상관: {avg_corr:.3f}  {'(다양성 낮음 — 앙상블 기여 의문)' if avg_corr > 0.95 else '(다양성 있음)'}")
        if coef.get(m, 0) < 0:
            print(f"  메타 계수 음수 → 메타 모델이 {m}을 역방향으로 보정 중 (노이즈 가능성)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    names = list(auc_scores.keys())
    vals = [auc_scores[n] for n in names]
    colors = ["steelblue" if n not in ("mlp", "tabnet") else "coral" for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylim(min(vals) - 0.005, max(vals) + 0.005)
    ax.set_title("OOF AUC per Model")
    ax.set_ylabel("AUC")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0005, f"{v:.4f}", ha="center", fontsize=9)

    ax = axes[1]
    import seaborn as sns
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=0.8, vmax=1.0,
                ax=ax, cbar=True, square=True)
    ax.set_title("OOF Prediction Correlation\n(다양성: 낮을수록 좋음)")

    ax = axes[2]
    if coef:
        coef_names = list(coef.keys())
        coef_vals = [coef[n] for n in coef_names]
        colors_c = ["steelblue" if v >= 0 else "salmon" for v in coef_vals]
        ax.barh(coef_names, coef_vals, color=colors_c)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Meta Model Coefficients\n(음수 = 역기여)")
        ax.set_xlabel("Coefficient")
    else:
        ax.text(0.5, 0.5, "Simple Average\n(계수 없음)", ha="center", va="center", fontsize=13)
        ax.set_title("Meta Model")

    plt.suptitle(f"Stacking Diagnosis  |  train {n_train:,}행", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n진단 차트 저장: {save_path}")
    print("=" * 40)
