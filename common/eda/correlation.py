"""
상관관계 분석 모듈
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass


def _set_korean_font() -> None:
    """한글이 깨지지 않도록 폰트 설정 (heatmap/plot 제목·축 라벨용)."""
    plt.rcParams['axes.unicode_minus'] = False
    korean_candidates = ['Malgun Gothic', 'AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic', 'NanumBarunGothic']
    available = {f.name for f in fm.fontManager.ttflist}
    for name in korean_candidates:
        if name in available:
            plt.rcParams['font.family'] = name
            return
    for name in korean_candidates:
        for av in available:
            if name.lower() in av.lower():
                plt.rcParams['font.family'] = av
                return


@dataclass
class CorrelationConfig:
    """상관관계 분석 설정 클래스"""
    method: str = 'pearson'  # 'pearson', 'spearman', 'kendall'
    threshold: float = 0.7  # 높은 상관관계 임계값
    figsize: Tuple[int, int] = (15, 12)
    top_n: int = 20  # 타겟 상관관계 상위 N개
    save_path: Optional[str] = None


def calculate_correlation_matrix(
    df: pd.DataFrame,
    method: str = 'pearson',
    numeric_only: bool = True
) -> pd.DataFrame:
    """
    상관관계 행렬 계산
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    method : str
        상관관계 계산 방법 ('pearson', 'spearman', 'kendall', 기본값: 'pearson')
    numeric_only : bool
        수치형 컬럼만 사용 여부 (기본값: True)
    
    Returns:
    --------
    pd.DataFrame: 상관관계 행렬
    """
    if numeric_only:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols].corr(method=method)
    else:
        return df.corr(method=method)


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    config: Optional[CorrelationConfig] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    상관관계 히트맵 시각화
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        상관관계 행렬
    config : CorrelationConfig, optional
        설정 (None이면 기본값 사용)
    title : str, optional
        그래프 제목 (None이면 기본 제목 사용)
    
    Returns:
    --------
    plt.Figure: 생성된 figure 객체
    """
    if config is None:
        config = CorrelationConfig()
    _set_korean_font()

    fig, ax = plt.subplots(figsize=config.figsize)
    mask = None
    if config.threshold is not None and config.threshold > 0:
        mask = np.abs(corr_matrix) < config.threshold
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        mask=mask,
        ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    
    plot_title = title if title else f"특성 간 상관관계 히트맵 ({config.method})"
    ax.set_title(plot_title, fontsize=16, pad=20)
    plt.tight_layout()
    
    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches='tight')
    
    return fig


def find_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
    exclude_diagonal: bool = True
) -> pd.DataFrame:
    """
    높은 상관관계 특성 쌍 탐색
    
    Parameters:
    -----------
    corr_matrix : pd.DataFrame
        상관관계 행렬
    threshold : float
        상관관계 임계값 (기본값: 0.7)
    exclude_diagonal : bool
        대각선 제외 여부 (기본값: True)
    
    Returns:
    --------
    pd.DataFrame: 높은 상관관계 특성 쌍 (Feature_1, Feature_2, Correlation, Abs_Correlation)
    """
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1 if exclude_diagonal else 0).astype(bool)
    )
    
    high_corr_pairs = []
    for col in upper_triangle.columns:
        for idx in upper_triangle.index:
            corr_value = upper_triangle.loc[idx, col]
            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                high_corr_pairs.append({
                    'Feature_1': idx,
                    'Feature_2': col,
                    'Correlation': corr_value,
                    'Abs_Correlation': abs(corr_value)
                })
    
    if high_corr_pairs:
        return pd.DataFrame(high_corr_pairs).sort_values('Abs_Correlation', ascending=False)
    return pd.DataFrame(columns=['Feature_1', 'Feature_2', 'Correlation', 'Abs_Correlation'])


def plot_correlation_with_target(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[CorrelationConfig] = None
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    타겟 변수와의 상관관계 시각화
    
    Parameters:
    -----------
    X : pd.DataFrame
        특성 데이터프레임
    y : pd.Series
        타겟 변수
    config : CorrelationConfig, optional
        설정 (None이면 기본값 사용)
    
    Returns:
    --------
    tuple: (figure, correlation_dataframe)
    """
    if config is None:
        config = CorrelationConfig()
    _set_korean_font()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    correlations = []
    for col in numeric_cols:
        corr = X[col].corr(y)
        correlations.append({
            'Feature': col,
            'Correlation': corr,
            'Abs_Correlation': abs(corr)
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False).head(config.top_n)
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(corr_df) * 0.4)))
    colors = ['red' if x < 0 else 'blue' for x in corr_df['Correlation']]
    ax.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
    ax.set_xlabel('상관계수', fontsize=12)
    ax.set_title(f'타겟 변수와의 상관관계 (상위 {config.top_n}개)', fontsize=14)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if config.save_path:
        plt.savefig(config.save_path, dpi=300, bbox_inches='tight')
    
    return fig, corr_df


def calculate_vif(
    X: pd.DataFrame,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    VIF (Variance Inflation Factor) 계산 - 다중공선성 측정
    
    Parameters:
    -----------
    X : pd.DataFrame
        특성 데이터프레임
    target_col : str, optional
        타겟 컬럼 (제외할 컬럼)
    
    Returns:
    --------
    pd.DataFrame: VIF 결과 (Feature, VIF)
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        raise ImportError(
            "statsmodels가 설치되어 있지 않습니다. "
            "VIF 계산을 위해 'pip install statsmodels'를 실행해주세요."
        )
    
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        numeric_cols = [col for col in numeric_cols if col != target_col]
    
    if len(numeric_cols) == 0:
        return pd.DataFrame(columns=['Feature', 'VIF'])
    
    X_numeric = X[numeric_cols].dropna()
    
    if len(X_numeric) == 0:
        return pd.DataFrame(columns=['Feature', 'VIF'])
    
    vif_data = []
    for i, col in enumerate(numeric_cols):
        try:
            vif_value = variance_inflation_factor(X_numeric.values, i)
            vif_data.append({'Feature': col, 'VIF': vif_value})
        except:
            vif_data.append({'Feature': col, 'VIF': np.nan})
    
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    return vif_df


def analyze_correlation(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    config: Optional[CorrelationConfig] = None,
    include_vif: bool = False
) -> Dict:
    """
    종합 상관관계 분석
    
    Parameters:
    -----------
    X : pd.DataFrame
        특성 데이터프레임
    y : pd.Series, optional
        타겟 변수
    config : CorrelationConfig, optional
        설정 (None이면 기본값 사용)
    include_vif : bool
        VIF 계산 포함 여부 (기본값: False)
    
    Returns:
    --------
    dict: 분석 결과 딕셔너리
        - correlation_matrix: 상관관계 행렬
        - high_correlations: 높은 상관관계 특성 쌍
        - target_correlations: 타겟 상관관계 (y가 제공된 경우)
        - vif: VIF 결과 (include_vif=True인 경우)
    """
    if config is None:
        config = CorrelationConfig()
    
    results = {}
    
    # 상관관계 행렬 계산
    corr_matrix = calculate_correlation_matrix(X, method=config.method)
    results['correlation_matrix'] = corr_matrix
    
    # 높은 상관관계 탐색
    high_corr = find_high_correlations(corr_matrix, threshold=config.threshold)
    results['high_correlations'] = high_corr
    
    # 타겟 상관관계 (y가 제공된 경우)
    if y is not None:
        _, target_corr = plot_correlation_with_target(X, y, config)
        results['target_correlations'] = target_corr
    
    # VIF 계산 (요청된 경우)
    if include_vif:
        try:
            vif_df = calculate_vif(X)
            results['vif'] = vif_df
        except Exception as e:
            results['vif'] = None
            print(f"⚠️ VIF 계산 실패: {e}")
    
    return results


def compare_correlations(
    X_before: pd.DataFrame,
    X_after: pd.DataFrame,
    y: Optional[pd.Series] = None,
    config: Optional[CorrelationConfig] = None
) -> Dict:
    """
    Feature Engineering 전후 상관관계 비교
    
    Parameters:
    -----------
    X_before : pd.DataFrame
        Feature Engineering 전 데이터
    X_after : pd.DataFrame
        Feature Engineering 후 데이터
    y : pd.Series, optional
        타겟 변수
    config : CorrelationConfig, optional
        설정 (None이면 기본값 사용)
    
    Returns:
    --------
    dict: 비교 결과 딕셔너리
        - before: 전 데이터 분석 결과
        - after: 후 데이터 분석 결과
        - new_features: 새로 생성된 특성 리스트
    """
    if config is None:
        config = CorrelationConfig()
    
    # 전 데이터 분석
    results_before = analyze_correlation(X_before, y, config, include_vif=False)
    
    # 후 데이터 분석
    results_after = analyze_correlation(X_after, y, config, include_vif=False)
    
    # 새로 생성된 특성 찾기
    before_cols = set(X_before.columns)
    after_cols = set(X_after.columns)
    new_features = sorted(list(after_cols - before_cols))
    
    return {
        'before': results_before,
        'after': results_after,
        'new_features': new_features
    }
