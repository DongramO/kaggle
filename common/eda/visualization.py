"""
시각화 모듈
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass, field


@dataclass
class PlotConfig:
    """플롯 설정 클래스"""
    figsize: Tuple[int, int] = (15, 8)
    title: str = ""
    show_stats: bool = False
    show_outliers: bool = True
    max_outliers: int = 10
    grid: bool = True
    style: str = "default"
    
    def to_dict(self) -> dict:
        """딕셔너리로 변환"""
        return {
            'figsize': self.figsize,
            'title': self.title,
            'show_stats': self.show_stats,
            'show_outliers': self.show_outliers,
            'max_outliers': self.max_outliers,
            'grid': self.grid,
            'style': self.style
        }


def _get_decimals(value_range: float) -> int:
    """값 범위에 따라 소수점 자릿수 결정"""
    if value_range >= 100:
        return 0
    elif value_range >= 10:
        return 1
    elif value_range >= 1:
        return 2
    else:
        return 3


def _calculate_iqr_stats(data: pd.Series) -> Dict:
    """IQR 통계 계산"""
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    median = data.median()
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    
    return {
        'q1': q1,
        'q3': q3,
        'median': median,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_pct': len(outliers) / len(data) * 100 if len(data) > 0 else 0
    }


def plot_boxplot(df: pd.DataFrame, columns: List[str], 
                 config: Optional[PlotConfig] = None) -> Tuple[plt.Figure, Dict]:
    """
    박스플롯 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    columns : List[str]
        시각화할 컬럼 리스트
    config : PlotConfig, optional
        플롯 설정 (None이면 기본값 사용)
    
    Returns:
    --------
    tuple: (figure, stats_dict)
    """
    if config is None:
        config = PlotConfig()
    
    stats_dict = {}
    valid_data = []
    valid_cols = []
    
    for col in columns:
        data = df[col].dropna()
        if len(data) > 0:
            valid_data.append(data)
            valid_cols.append(col)
            stats_dict[col] = _calculate_iqr_stats(data)
    
    if len(valid_data) == 0:
        raise ValueError("시각화할 데이터가 없습니다.")
    
    n_cols = len(valid_cols)
    n_rows = (n_cols + 2) // 3
    n_cols_plot = min(3, n_cols)
    fig_height = config.figsize[1] * n_rows / 2 if n_rows > 1 else config.figsize[1]
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(config.figsize[0], fig_height))
    
    if n_cols == 1:
        axes_list = [axes]
    elif n_rows == 1:
        axes_list = list(axes) if isinstance(axes, np.ndarray) else [axes]
    else:
        axes_list = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [ax for row in axes for ax in row]
    
    for idx, (data, col) in enumerate(zip(valid_data, valid_cols)):
        if idx >= len(axes_list):
            continue
        
        ax = axes_list[idx]
        stats = stats_dict[col]
        
        ax.boxplot([data], vert=True, patch_artist=True,
                   showmeans=True, meanline=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   meanprops=dict(color='green', linewidth=2, linestyle='--'),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5))
        
        ax.axhline(y=stats['lower_bound'], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axhline(y=stats['upper_bound'], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        
        if config.show_outliers and len(stats['outliers']) > 0:
            outliers = stats['outliers'].sort_values()
            lower_outliers = outliers[outliers < stats['lower_bound']].head(config.max_outliers).values
            upper_outliers = outliers[outliers > stats['upper_bound']].tail(config.max_outliers).values
            
            if len(lower_outliers) > 0:
                ax.scatter([1.05] * len(lower_outliers), lower_outliers, 
                          color='red', s=30, alpha=0.6, zorder=5)
            if len(upper_outliers) > 0:
                ax.scatter([1.05] * len(upper_outliers), upper_outliers, 
                          color='red', s=30, alpha=0.6, zorder=5)
        
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(col, fontsize=13, fontweight='bold')
        if config.grid:
            ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticks([])
    
    for idx in range(n_cols, len(axes_list)):
        axes_list[idx].axis('off')
    
    if config.title:
        fig.suptitle(config.title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig, stats_dict


def plot_histogram(df: pd.DataFrame, columns: List[str], 
                   bins: Optional[Union[int, List]] = None,
                   config: Optional[PlotConfig] = None) -> plt.Figure:
    """
    히스토그램 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    columns : List[str]
        시각화할 컬럼 리스트
    bins : int or List, optional
        히스토그램 구간 (None이면 자동 설정)
    config : PlotConfig, optional
        플롯 설정
    
    Returns:
    --------
    plt.Figure: 생성된 figure 객체
    """
    if config is None:
        config = PlotConfig()
    
    n_cols = len(columns)
    if n_cols == 0:
        raise ValueError("columns 리스트가 비어있습니다.")
    
    n_rows = (n_cols + 2) // 3
    n_cols_plot = min(3, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=config.figsize)
    
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'{col}\n(No data)', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col, fontsize=10, fontweight='bold')
            continue
        
        if bins is None:
            bins_col = 10
        elif isinstance(bins, int):
            bins_col = bins
        else:
            bins_col = bins
        
        data_binned = pd.cut(data, bins=bins_col, include_lowest=True, duplicates='drop')
        counts = data_binned.value_counts().sort_index()
        
        labels = [str(interval) for interval in counts.index]
        values = counts.values
        
        bars = ax.barh(range(len(labels)), values, color='crimson', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Frequency", fontsize=9)
        ax.set_title(col, fontsize=11, fontweight='bold')
        if config.grid:
            ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    if config.title:
        fig.suptitle(config.title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def plot_categorical(df: pd.DataFrame, columns: List[str],
                    config: Optional[PlotConfig] = None) -> plt.Figure:
    """
    범주형 변수 분포 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    columns : List[str]
        시각화할 범주형 컬럼 리스트
    config : PlotConfig, optional
        플롯 설정
    
    Returns:
    --------
    plt.Figure: 생성된 figure 객체
    """
    if config is None:
        config = PlotConfig()
    
    n_cols = len(columns)
    if n_cols == 0:
        raise ValueError("columns 리스트가 비어있습니다.")
    
    n_rows = (n_cols + 2) // 3
    n_cols_plot = min(3, n_cols)
    fig_height = config.figsize[1] * n_rows / 2 if n_rows > 1 else config.figsize[1]
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(config.figsize[0], fig_height))
    
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.tolist() if isinstance(axes, np.ndarray) else list(axes)
    else:
        axes = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [ax for row in axes for ax in row]
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        data = df[col].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'{col}\n(No data)', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(col, fontsize=11, fontweight='bold')
            continue
        
        value_counts = data.value_counts()
        labels = value_counts.index.tolist()
        values = value_counts.values
        total = values.sum()
        percentages = (values / total * 100).round(2)
        
        bars = ax.barh(range(len(labels)), values, color='steelblue', alpha=0.7, 
                      edgecolor='black', linewidth=1.2)
        ax.set_yticks(range(len(labels)))
        display_labels = [str(label) if len(str(label)) <= 30 else str(label)[:27] + '...' 
                         for label in labels]
        ax.set_yticklabels(display_labels, fontsize=9)
        ax.set_xlabel("Frequency", fontsize=10)
        ax.set_ylabel("Category", fontsize=10)
        ax.set_title(col, fontsize=12, fontweight='bold', pad=10)
        if config.grid:
            ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if config.show_stats:
            for i, (bar, value, pct) in enumerate(zip(bars, values, percentages)):
                ax.text(bar.get_width() + total * 0.01, i, 
                       f'{value:,} ({pct}%)', 
                       va='center', fontsize=9, fontweight='bold')
    
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    title = config.title if config.title else "Categorical Variables Distribution"
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig


def plot_histogram_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                           config: Optional[PlotConfig] = None,
                           bins: int = 30, alpha: float = 0.7,
                           side_by_side: bool = True) -> Tuple[plt.Figure, Dict]:
    """
    그룹별 히스토그램 비교 시각화
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    value_col : str
        분포를 확인할 수치형 컬럼명
    group_col : str
        그룹을 나누는 범주형 컬럼명
    config : PlotConfig, optional
        플롯 설정
    bins : int
        히스토그램 구간 개수 (기본값: 30)
    alpha : float
        투명도 (기본값: 0.7)
    side_by_side : bool
        True: 나란히 비교, False: 겹쳐서 비교 (기본값: True)
    
    Returns:
    --------
    tuple: (figure, stats_dict)
    """
    if config is None:
        config = PlotConfig()
    
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' 컬럼이 데이터프레임에 없습니다.")
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' 컬럼이 데이터프레임에 없습니다.")
    
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) == 0:
        raise ValueError(f"'{group_col}' 컬럼에 데이터가 없습니다.")
    
    stats_dict = {}
    data_by_group = {}
    for group in groups:
        group_data = df[df[group_col] == group][value_col].dropna()
        data_by_group[group] = group_data
        
        if len(group_data) > 0:
            stats_dict[group] = {
                'count': len(group_data),
                'mean': group_data.mean(),
                'median': group_data.median(),
                'std': group_data.std(),
                'min': group_data.min(),
                'max': group_data.max()
            }
    
    if side_by_side:
        fig, axes = plt.subplots(1, len(groups), figsize=config.figsize, sharey=True)
        if len(groups) == 1:
            axes = [axes]
        
        for idx, group in enumerate(groups):
            ax = axes[idx]
            group_data = data_by_group[group]
            
            if len(group_data) > 0:
                ax.hist(group_data, bins=bins, alpha=alpha, edgecolor='black', 
                       color=plt.cm.Set2(idx), label=str(group))
                
                mean_val = stats_dict[group]['mean']
                median_val = stats_dict[group]['median']
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {median_val:.2f}')
                
                ax.set_xlabel(value_col, fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{group_col} = {group}', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                if config.grid:
                    ax.grid(alpha=0.3, linestyle='--')
            else:
                ax.text(0.5, 0.5, f'{group_col} = {group}\n(No data)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{group_col} = {group}', fontsize=12, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=config.figsize)
        colors = plt.cm.Set2(range(len(groups)))
        
        for idx, group in enumerate(groups):
            group_data = data_by_group[group]
            if len(group_data) > 0:
                ax.hist(group_data, bins=bins, alpha=alpha, edgecolor='black',
                       color=colors[idx], label=f'{group_col} = {group}')
        
        for idx, group in enumerate(groups):
            if group in stats_dict:
                mean_val = stats_dict[group]['mean']
                ax.axvline(mean_val, color=colors[idx], linestyle='--', 
                          linewidth=2, alpha=0.8, label=f'{group} Mean: {mean_val:.2f}')
        
        ax.set_xlabel(value_col, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        title = config.title if config.title else f'{value_col} Distribution by {group_col}'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        if config.grid:
            ax.grid(alpha=0.3, linestyle='--')
    
    title = config.title if config.title else f'{value_col} Distribution by {group_col}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.0 if side_by_side else 0.98)
    plt.tight_layout()
    
    return fig, stats_dict


# 하위 호환성을 위한 별칭
visualization_boxplot_iqr_multiple = plot_boxplot
visualization_bar_multiple = plot_histogram
visualization_categorical_bar = plot_categorical
visualization_histogram_by_group = plot_histogram_by_group
