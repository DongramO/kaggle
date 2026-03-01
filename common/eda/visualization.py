"""
시각화 모듈
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """플롯 공통 설정. 함수별 옵션(이상치 표시 등)은 각 함수 인자로."""
    figsize: Tuple[int, int] = (15, 8)
    title: str = ""
    grid: bool = True

    def to_dict(self) -> dict:
        return {'figsize': self.figsize, 'title': self.title, 'grid': self.grid}


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
    
    mean_val = data.mean() if len(data) > 0 else np.nan
    return {
        'q1': q1,
        'q3': q3,
        'median': median,
        'mean': mean_val,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers,
        'outlier_count': len(outliers),
        'outlier_pct': len(outliers) / len(data) * 100 if len(data) > 0 else 0
    }


def plot_boxplot(df: pd.DataFrame, columns: List[str],
                 config: Optional[PlotConfig] = None,
                 show_outliers: bool = True,
                 max_outliers: int = 10) -> Tuple[plt.Figure, Dict]:
    """수치형 컬럼 박스플롯. IQR·이상치 표시. 반환: (figure, stats_dict)."""
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
        
        if show_outliers and len(stats['outliers']) > 0:
            outliers = stats['outliers'].sort_values()
            lower_outliers = outliers[outliers < stats['lower_bound']].head(max_outliers).values
            upper_outliers = outliers[outliers > stats['upper_bound']].tail(max_outliers).values
            
            if len(lower_outliers) > 0:
                ax.scatter([1.05] * len(lower_outliers), lower_outliers, 
                          color='red', s=30, alpha=0.6, zorder=5)
            if len(upper_outliers) > 0:
                ax.scatter([1.05] * len(upper_outliers), upper_outliers, 
                          color='red', s=30, alpha=0.6, zorder=5)
        
        q1, q3, med, mean = stats['q1'], stats['q3'], stats['median'], stats['mean']
        for y_val, label in [(q1, f'Q1:{q1:.2f}'), (med, f'Med:{med:.2f}'), (q3, f'Q3:{q3:.2f}'), (mean, f'Mean:{mean:.2f}')]:
            ax.text(1.02, y_val, label, fontsize=7, va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.9))
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


def plot_boxplot_by_group(df: pd.DataFrame, value_col: str, group_col: str,
                          config: Optional[PlotConfig] = None,
                          show_outliers: bool = True,
                          max_outliers: int = 10) -> Tuple[plt.Figure, Dict]:
    """그룹별 박스플롯. value_col 분포를 group_col 카테고리별로 그린다. value_col, group_col은 호출 시 지정. 반환: (figure, stats_dict)."""
    if config is None:
        config = PlotConfig()

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' 컬럼이 데이터프레임에 없습니다.")
    if group_col not in df.columns:
        raise ValueError(f"'{group_col}' 컬럼이 데이터프레임에 없습니다.")

    groups = sorted(df[group_col].dropna().unique())
    if len(groups) == 0:
        raise ValueError(f"'{group_col}' 컬럼에 데이터가 없습니다.")

    data_by_group = []
    labels = []
    stats_dict = {}
    for group in groups:
        series = df[df[group_col] == group][value_col].dropna()
        if len(series) > 0:
            data_by_group.append(series.values)
            labels.append(str(group))
            stats_dict[group] = _calculate_iqr_stats(series)
        else:
            data_by_group.append(np.array([]))
            labels.append(str(group))

    if len(data_by_group) == 0:
        raise ValueError("시각화할 데이터가 없습니다.")

    fig, ax = plt.subplots(figsize=config.figsize)
    bp = ax.boxplot(
        data_by_group,
        labels=labels,
        vert=True,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
        meanprops=dict(color='green', linewidth=2, linestyle='--'),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
    )

    if show_outliers:
        for idx, group in enumerate(groups):
            if group not in stats_dict or not stats_dict[group]:
                continue
            stats = stats_dict[group]
            outliers = stats.get('outliers', pd.Series(dtype=float))
            if len(outliers) > 0:
                x_pos = idx + 1
                lower_outliers = outliers[outliers < stats['lower_bound']].head(max_outliers).values
                upper_outliers = outliers[outliers > stats['upper_bound']].tail(max_outliers).values
                if len(lower_outliers) > 0:
                    ax.scatter([x_pos + 0.05] * len(lower_outliers), lower_outliers,
                               color='red', s=30, alpha=0.6, zorder=5)
                if len(upper_outliers) > 0:
                    ax.scatter([x_pos + 0.05] * len(upper_outliers), upper_outliers,
                               color='red', s=30, alpha=0.6, zorder=5)

    for idx, group in enumerate(groups):
        if group not in stats_dict or not stats_dict[group]:
            continue
        s = stats_dict[group]
        q1, q3, med, mean = s['q1'], s['q3'], s['median'], s['mean']
        x_pos = idx + 1
        for y_val, label in [(q1, f'Q1:{q1:.2f}'), (med, f'Med:{med:.2f}'), (q3, f'Q3:{q3:.2f}'), (mean, f'Mean:{mean:.2f}')]:
            ax.text(x_pos + 0.02, y_val, label, fontsize=6, va='center', ha='left',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='wheat', alpha=0.9))

    ax.set_ylabel(value_col, fontsize=11)
    ax.set_xlabel(group_col, fontsize=11)
    title = config.title if config.title else f'{value_col} by {group_col}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    if config.grid:
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig, stats_dict


def plot_histogram(df: pd.DataFrame, columns: List[str],
                   bins: Optional[Union[int, List]] = None,
                   config: Optional[PlotConfig] = None) -> plt.Figure:
    """수치형 컬럼 구간별 빈도(가로 막대). bins 미지정 시 10. 반환: figure."""
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
                     config: Optional[PlotConfig] = None,
                     show_stats: bool = False) -> plt.Figure:
    """범주형 컬럼 카테고리별 빈도(가로 막대). 반환: figure."""
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
        
        if show_stats:
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
    """그룹별 수치형 분포 비교. side_by_side: True 나란히 / False 겹침. 반환: (figure, stats_dict)."""
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
visualization_boxplot_by_group = plot_boxplot_by_group
visualization_bar_multiple = plot_histogram
visualization_categorical_bar = plot_categorical
visualization_histogram_by_group = plot_histogram_by_group
