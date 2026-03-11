"""
EDA 모듈
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all")


def missing_report(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def run_eda(df: pd.DataFrame, target_col: str | None = None) -> dict:
    result = {
        "shape": df.shape,
        "describe": summary(df),
        "missing": missing_report(df),
        "dtypes": df.dtypes,
    }
    if target_col and target_col in df.columns:
        result["target_distribution"] = df[target_col].value_counts()
    return result


def correlation_analysis(
    df: pd.DataFrame,
    target_col: str | None = None,
    high_corr_threshold: float = 0.9,
    low_target_corr_threshold: float = 0.01,
    save_dir: str | None = None,
) -> dict:
    numeric_df = df.select_dtypes(include=[np.number])
    feature_df = numeric_df.drop(columns=[target_col]) if target_col in numeric_df.columns else numeric_df

    corr_matrix = feature_df.corr(method='pearson')
    cols = corr_matrix.columns.tolist()

    high_corr_pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = abs(corr_matrix.loc[cols[i], cols[j]])
            if corr_val >= high_corr_threshold:
                high_corr_pairs.append({
                    'feature_1': cols[i],
                    'feature_2': cols[j],
                    'correlation': round(corr_matrix.loc[cols[i], cols[j]], 4),
                })

    drop_candidates = []
    low_target_corr = []
    target_corr = None

    if target_col and target_col in numeric_df.columns:
        target_corr = feature_df.corrwith(numeric_df[target_col]).abs().sort_values(ascending=False)
        low_target_corr = target_corr[target_corr < low_target_corr_threshold].index.tolist()

        for pair in high_corr_pairs:
            f1, f2 = pair['feature_1'], pair['feature_2']
            drop_feature = f2 if target_corr.get(f1, 0) >= target_corr.get(f2, 0) else f1
            if drop_feature not in drop_candidates:
                drop_candidates.append(drop_feature)

        for feat in low_target_corr:
            if feat not in drop_candidates:
                drop_candidates.append(feat)
    else:
        for pair in high_corr_pairs:
            if pair['feature_2'] not in drop_candidates:
                drop_candidates.append(pair['feature_2'])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _save_heatmap(corr_matrix, save_dir)
        if target_corr is not None:
            _save_target_corr_bar(target_corr, target_col, save_dir)

    return {
        'corr_matrix': corr_matrix,
        'high_corr_pairs': high_corr_pairs,
        'low_target_corr': low_target_corr,
        'drop_candidates': drop_candidates,
        'target_corr': target_corr,
    }


def _save_heatmap(corr_matrix: pd.DataFrame, save_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(corr_matrix) * 0.6), max(8, len(corr_matrix) * 0.5)))
    sns.heatmap(
        corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, vmin=-1, vmax=1, ax=ax,
        annot_kws={'size': 7}, linewidths=0.3,
    )
    ax.set_title('Feature Correlation Matrix (Pearson)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _save_target_corr_bar(target_corr: pd.Series, target_col: str, save_dir: str) -> None:
    sorted_corr = target_corr.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_corr) * 0.35)))
    colors = ['steelblue' if v >= 0.01 else 'lightcoral' for v in sorted_corr]
    ax.barh(sorted_corr.index, sorted_corr.values, color=colors)
    ax.axvline(x=0.01, color='red', linestyle='--', linewidth=1, label='threshold=0.01')
    ax.set_xlabel(f'|Correlation| with {target_col}')
    ax.set_title(f'Feature Correlation with Target ({target_col})')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'target_correlation_bar.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
