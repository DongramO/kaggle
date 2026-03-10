"""
MLP 기반 '분석 초점' 수치화 모듈

예측용이 아닌 분석용: MLP가 학습한 가중치를 이용해
- 어떤 특성에 초점을 두는지(특성 중요도)
- 어떤 특성 쌍의 상호작용이 강한지
를 수치화하고, 트리 모델 중요도와 비교해 "어디에 초점을 맞출지" 제안.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor


def _fit_mlp_for_analysis(X: np.ndarray, y: np.ndarray, task_type: str,
                          random_state: int = 42, max_iter: int = 500,
                          hidden_layer_sizes: Tuple[int, ...] = (256, 128)
                          ) -> Tuple[object, StandardScaler]:
    """분석용 MLP 1회 학습. (model, scaler) 반환."""
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cls = MLPRegressor if task_type == 'regression' else MLPClassifier
    model = cls(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    model.fit(X_scaled, y)
    return model, scaler


def mlp_feature_importance(model, feature_names: List[str]) -> pd.DataFrame:
    """
    MLP 1층 가중치 기반 특성 중요도 수치화.
    반환: Feature, Importance, Importance_pct, Rank
    """
    if not hasattr(model, 'coefs_') or model.coefs_ is None or len(model.coefs_) == 0:
        n = len(feature_names)
        return pd.DataFrame({
            'Feature': feature_names,
            'Importance': np.ones(n) / n,
            'Importance_pct': 100.0 / n,
            'Rank': list(range(1, n + 1)),
        })
    W = model.coefs_[0]  # (n_features, n_hidden)
    imp = np.abs(W).sum(axis=1)
    total = imp.sum()
    imp_pct = (100.0 * imp / total) if total > 0 else np.zeros_like(imp)
    order = np.argsort(-imp)
    df = pd.DataFrame({
        'Feature': [feature_names[i] for i in order],
        'Importance': imp[order],
        'Importance_pct': imp_pct[order],
        'Rank': np.arange(1, len(order) + 1),
    })
    return df


def mlp_interaction_strength(model, feature_names: List[str],
                             top_k: int = 50) -> pd.DataFrame:
    """
    MLP 1층 가중치로 특성 쌍 (i,j)의 상호작용 강도 수치화.
    같은 은닉 뉴런으로 들어가는 |W[i,h]|*|W[j,h]| 합으로 근사.
    반환: Feature_i, Feature_j, InteractionStrength, Rank
    """
    if not hasattr(model, 'coefs_') or model.coefs_ is None or len(model.coefs_) == 0:
        return pd.DataFrame(columns=['Feature_i', 'Feature_j', 'InteractionStrength', 'Rank'])
    W = model.coefs_[0]  # (n_features, n_hidden)
    n = W.shape[0]
    strength = np.zeros((n, n))
    for h in range(W.shape[1]):
        strength += np.outer(np.abs(W[:, h]), np.abs(W[:, h]))
    np.fill_diagonal(strength, 0)
    i_idx, j_idx = np.unravel_index(np.argsort(-strength.ravel()), strength.shape)
    top = min(top_k, (strength > 0).sum())
    rows = []
    for t in range(top):
        i, j = int(i_idx.flat[t]), int(j_idx.flat[t])
        if strength[i, j] <= 0:
            break
        rows.append({
            'Feature_i': feature_names[i],
            'Feature_j': feature_names[j],
            'InteractionStrength': strength[i, j],
            'Rank': t + 1,
        })
    return pd.DataFrame(rows)


def average_tree_importance(model_importances: Dict[str, pd.DataFrame],
                            feature_col: str = 'Feature',
                            importance_col: str = 'Importance') -> pd.DataFrame:
    """
    여러 트리 모델(CatBoost, LightGBM, XGBoost)의 특성 중요도를 정규화 후 평균해
    하나의 Tree Importance 테이블로 반환. mlp_vs_tree_focus에서 '트리 전체' 대표값으로 사용.
    """
    if not model_importances:
        return pd.DataFrame()
    series_list = []
    for model_type, df in model_importances.items():
        if df is None or len(df) == 0 or feature_col not in df.columns or importance_col not in df.columns:
            continue
        df = df[[feature_col, importance_col]].copy()
        total = df[importance_col].sum()
        if total <= 0:
            continue
        df = df.set_index(feature_col)
        df = (df[importance_col] / total).rename(model_type)
        series_list.append(df)
    if not series_list:
        return pd.DataFrame()
    combined = pd.concat(series_list, axis=1)
    avg = combined.mean(axis=1, skipna=True)
    out = pd.DataFrame({feature_col: avg.index.tolist(), importance_col: avg.values})
    out = out.sort_values(importance_col, ascending=False).reset_index(drop=True)
    return out


def compare_with_tree_importance(mlp_importance_df: pd.DataFrame,
                                 tree_importance_df: pd.DataFrame,
                                 feature_col: str = 'Feature',
                                 importance_col: str = 'Importance'
                                 ) -> pd.DataFrame:
    """
    MLP 특성 중요도와 트리 특성 중요도(단일) 병합, Focus_Type 부여.
    """
    mlp = mlp_importance_df[[feature_col, importance_col]].copy()
    mlp.columns = [feature_col, 'MLP_Importance']
    mlp['Rank_MLP'] = range(1, len(mlp) + 1)
    tree = tree_importance_df[[feature_col, importance_col]].copy()
    tree.columns = [feature_col, 'Tree_Importance']
    tree['Rank_Tree'] = range(1, len(tree) + 1)
    merged = mlp.merge(tree, on=feature_col, how='outer')
    n_mlp = len(mlp_importance_df)
    n_tree = len(tree_importance_df)
    merged['Rank_MLP'] = merged['Rank_MLP'].fillna(n_mlp + 1)
    merged['Rank_Tree'] = merged['Rank_Tree'].fillna(n_tree + 1)
    merged['MLP_Importance'] = merged['MLP_Importance'].fillna(0)
    merged['Tree_Importance'] = merged['Tree_Importance'].fillna(0)
    merged['Focus_Type'] = 'both_low'
    merged.loc[merged['Rank_Tree'] <= max(1, n_tree * 0.3), 'Focus_Type'] = 'tree_high'
    merged.loc[merged['Rank_MLP'] <= max(1, n_mlp * 0.3), 'Focus_Type'] = 'MLP_high'
    merged.loc[(merged['Rank_MLP'] <= max(1, n_mlp * 0.3)) & (merged['Rank_Tree'] <= max(1, n_tree * 0.3)), 'Focus_Type'] = 'both_high'
    merged['Rank_MLP'] = merged['Rank_MLP'].astype(int)
    merged['Rank_Tree'] = merged['Rank_Tree'].astype(int)
    return merged.sort_values('MLP_Importance', ascending=False).reset_index(drop=True)


def compare_with_tree_importance_per_model(
    mlp_importance_df: pd.DataFrame,
    model_importances: Dict[str, pd.DataFrame],
    feature_col: str = 'Feature',
    importance_col: str = 'Importance',
) -> pd.DataFrame:
    """
    MLP와 모델별 트리(CatBoost, LightGBM, XGBoost) importance를 병합.
    컬럼: Feature, MLP_Importance, Rank_MLP, catboost_Importance, Rank_catboost, lightgbm_Importance, ...
    Focus_Type: tree_high = 해당 특성이 트리 중 하나라도 상위 30%, both_high = MLP·트리 둘 다 상위.
    """
    mlp = mlp_importance_df[[feature_col, importance_col]].copy()
    mlp.columns = [feature_col, 'MLP_Importance']
    mlp['Rank_MLP'] = range(1, len(mlp) + 1)
    n_mlp = len(mlp)
    merged = mlp.copy()

    tree_rank_cols = []
    n_per_model = {}
    for model_type, df in (model_importances or {}).items():
        if df is None or len(df) == 0 or feature_col not in df.columns or importance_col not in df.columns:
            continue
        tree_df = df[[feature_col, importance_col]].copy()
        tree_df.columns = [feature_col, f'{model_type}_Importance']
        tree_df[f'Rank_{model_type}'] = range(1, len(tree_df) + 1)
        n_per_model[model_type] = len(tree_df)
        merged = merged.merge(tree_df, on=feature_col, how='outer')
        tree_rank_cols.append(f'Rank_{model_type}')

    for c in merged.columns:
        if c == feature_col:
            continue
        if 'Importance' in c:
            merged[c] = merged[c].fillna(0)

    merged['Rank_MLP'] = merged['Rank_MLP'].fillna(n_mlp + 1)
    for model_type, n in n_per_model.items():
        rcol = f'Rank_{model_type}'
        if rcol in merged.columns:
            merged[rcol] = merged[rcol].fillna(n + 1)

    merged['Focus_Type'] = 'both_low'
    tree_high = pd.Series(False, index=merged.index)
    for model_type, n in n_per_model.items():
        rcol = f'Rank_{model_type}'
        if rcol in merged.columns:
            tree_high = tree_high | (merged[rcol] <= max(1, n * 0.3))
    merged.loc[tree_high, 'Focus_Type'] = 'tree_high'
    merged.loc[merged['Rank_MLP'] <= max(1, n_mlp * 0.3), 'Focus_Type'] = 'MLP_high'
    merged.loc[(merged['Rank_MLP'] <= max(1, n_mlp * 0.3)) & tree_high, 'Focus_Type'] = 'both_high'

    for c in ['Rank_MLP'] + tree_rank_cols:
        if c in merged.columns:
            merged[c] = merged[c].astype(int)
    return merged.sort_values('MLP_Importance', ascending=False).reset_index(drop=True)


def analyze_mlp_focus(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str = 'classification',
    feature_names: Optional[List[str]] = None,
    random_state: int = 42,
    save_dir: Optional[str] = None,
    tree_importance_df: Optional[pd.DataFrame] = None,
    model_importances: Optional[Dict[str, pd.DataFrame]] = None,
    top_n: int = 30,
    top_interactions: int = 50,
) -> Dict[str, pd.DataFrame]:
    """
    MLP로 '어디에 초점을 맞출지' 수치화 후 CSV 저장 및 DataFrame 반환.

    - model_importances가 있으면 모델별(CatBoost, LightGBM, XGBoost) importance 컬럼으로 저장.
    - tree_importance_df만 있으면 기존처럼 단일 Tree_Importance로 비교.
    """
    X = np.asarray(X_train)
    y = np.asarray(y_train).ravel()
    if feature_names is not None:
        names = feature_names
    elif hasattr(X_train, 'columns'):
        names = list(X_train.columns)
    else:
        names = [f'F{i}' for i in range(X.shape[1])]

    model, scaler = _fit_mlp_for_analysis(X, y, task_type, random_state=random_state)
    feat_df = mlp_feature_importance(model, names)
    inter_df = mlp_interaction_strength(model, names, top_k=top_interactions)

    result = {'feature_importance': feat_df, 'interaction_strength': inter_df}

    if model_importances and len(model_importances) > 0:
        focus_df = compare_with_tree_importance_per_model(feat_df, model_importances)
        result['focus_vs_tree'] = focus_df
    elif tree_importance_df is not None and len(tree_importance_df) > 0:
        focus_df = compare_with_tree_importance(feat_df, tree_importance_df)
        result['focus_vs_tree'] = focus_df

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(os.path.join(save_dir, 'mlp_feature_focus.csv'), index=False, encoding='utf-8-sig')
        inter_df.to_csv(os.path.join(save_dir, 'mlp_interaction_strength.csv'), index=False, encoding='utf-8-sig')
        if 'focus_vs_tree' in result:
            result['focus_vs_tree'].to_csv(os.path.join(save_dir, 'mlp_vs_tree_focus.csv'), index=False, encoding='utf-8-sig')

    return result
