import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _get_decimals(value_range):
    """값 범위에 따라 적절한 소수점 자리수 반환"""
    if value_range >= 100:
        return 0
    elif value_range >= 10:
        return 1
    elif value_range >= 1:
        return 2
    else:
        return 3


def _plot_outliers(ax, outlier_values, x_pos, decimals, max_display=10):
    """이상치를 그래프에 표시하는 헬퍼 함수"""
    if len(outlier_values) == 0:
        return
    
    ax.scatter([x_pos] * len(outlier_values), outlier_values, 
               color='red', s=30, alpha=0.6, zorder=5, marker='o')
    
    if len(outlier_values) <= 5:
        for val in outlier_values:
            ax.text(x_pos + 0.03, val, f'{val:.{decimals}f}',
                   fontsize=7, verticalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7, edgecolor='red'))
    elif len(outlier_values) <= max_display:
        ax.text(x_pos + 0.03, outlier_values[0], f'{outlier_values[0]:.{decimals}f}',
               fontsize=7, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7, edgecolor='red'))
        ax.text(x_pos + 0.03, outlier_values[-1], f'{outlier_values[-1]:.{decimals}f}',
               fontsize=7, verticalalignment='center',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7, edgecolor='red'))
        mid_idx = len(outlier_values) // 2
        ax.text(x_pos + 0.03, outlier_values[mid_idx], f'...{len(outlier_values)}개',
               fontsize=6, verticalalignment='center', style='italic',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.5))


def visualization_boxplot_iqr_multiple(df, columns, title="", 
                                       figsize=(15, 8), show_stats=True, grid_layout=True,
                                       show_outlier_values=True, max_outliers_display=10):

    stats_dict = {}
    data_list = []
    labels_list = []
    
    for col in columns:
        data_clean = df[col].dropna()
        if len(data_clean) > 0:
            data_list.append(data_clean)
            labels_list.append(col)
            
            q1 = data_clean.quantile(0.25)
            q3 = data_clean.quantile(0.75)
            median = data_clean.median()
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data_clean[(data_clean < lower_bound) | (data_clean > upper_bound)]
            
            stats_dict[col] = {
                'q1': q1,
                'q3': q3,
                'median': median,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': outliers,
                'outlier_count': len(outliers),
                'outlier_percentage': round(len(outliers) / len(data_clean) * 100, 2) if len(data_clean) > 0 else 0
            }
    
    # 통계 정보 출력
    if show_stats:
        print("\n" + "=" * 80)
        print(f"IQR 통계 정보: {title}")
        print("=" * 80)
        for col in columns:
            if col in stats_dict:
                stats = stats_dict[col]
                print(f"\n[{col}]")
                print(f"  Q1: {stats['q1']:.2f} | Median: {stats['median']:.2f} | Q3: {stats['q3']:.2f}")
                print(f"  IQR: {stats['iqr']:.2f}")
                print(f"  하한 경계: {stats['lower_bound']:.2f} | 상한 경계: {stats['upper_bound']:.2f}")
                print(f"  이상치 개수: {stats['outlier_count']:,}개 ({stats['outlier_percentage']:.2f}%)")
        print("=" * 80 + "\n")
    
    n_cols = len(labels_list)
    if n_cols == 0:
        raise ValueError("시각화할 데이터가 없습니다.")
    
    if grid_layout:
        n_rows = (n_cols + 2) // 3
        n_cols_plot = min(3, n_cols)
        fig_height = figsize[1] * n_rows / 2 if n_rows > 1 else figsize[1]
        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(figsize[0], fig_height))
        
        if n_cols == 1:
            axes_list = [axes]
        elif n_rows == 1:
            axes_list = list(axes) if isinstance(axes, (np.ndarray, list)) else [axes]
        else:
            axes_list = axes.flatten().tolist() if isinstance(axes, np.ndarray) else [ax for row in axes for ax in row]
        
        for idx, (data_clean, col) in enumerate(zip(data_list, labels_list)):
            if idx >= len(axes_list):
                continue
                
            ax = axes_list[idx]
            
            if col in stats_dict:
                stats = stats_dict[col]
                q1 = stats['q1']
                q3 = stats['q3']
                lower_bound = stats['lower_bound']
                upper_bound = stats['upper_bound']
                actual_min = data_clean.min()
                actual_max = data_clean.max()
                
                ax.boxplot([data_clean], vert=True, patch_artist=True,
                          showmeans=True, meanline=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2),
                          meanprops=dict(color='green', linewidth=2, linestyle='--'),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5))
                
                ax.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                
                decimals = _get_decimals(actual_max - actual_min)
                
                x_pos = 1
                data_min = min(actual_min, lower_bound) if lower_bound < actual_min else actual_min
                data_max = max(actual_max, upper_bound) if upper_bound > actual_max else actual_max
                
                data_range = data_max - data_min
                if data_range <= 0:
                    data_range = max(abs(data_max), abs(data_min)) if data_max != data_min else 1
                    y_min_plot = data_min - data_range * 0.1
                    y_max_plot = data_max + data_range * 0.1
                else:
                    y_min_plot = data_min - data_range * 0.15
                    y_max_plot = data_max + data_range * 0.15
                
                ax.set_ylim(y_min_plot, y_max_plot)
                y_range = y_max_plot - y_min_plot
                
                y_bottom = y_min_plot + y_range * 0.12
                y_top = y_max_plot - y_range * 0.12
                available_range = y_top - y_bottom
                
                text_y_min = y_bottom + available_range * 0.1
                text_y_q1 = y_bottom + available_range * 0.35
                text_y_q3 = y_bottom + available_range * 0.65
                text_y_max = y_bottom + available_range * 0.9
                
                ax.text(x_pos + 0.08, text_y_min, f'Min: {actual_min:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='blue', linewidth=2))
                ax.text(x_pos + 0.08, text_y_q1, f'Q1: {q1:.{decimals}f}', 
                       verticalalignment='center', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='darkgreen', linewidth=2))
                ax.text(x_pos + 0.08, text_y_q3, f'Q3: {q3:.{decimals}f}', 
                       verticalalignment='center', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='darkgreen', linewidth=2))
                ax.text(x_pos + 0.08, text_y_max, f'Max: {actual_max:.{decimals}f}', 
                       verticalalignment='top', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='blue', linewidth=2))
                
                if y_range > 0:
                    num_ticks = 10
                    tick_spacing = y_range / num_ticks
                    if tick_spacing > 0:
                        try:
                            magnitude = 10 ** (math.floor(math.log10(abs(tick_spacing))))
                            tick_spacing = round(tick_spacing / magnitude) * magnitude
                        except (ValueError, ZeroDivisionError):
                            tick_spacing = y_range / 10
                    else:
                        tick_spacing = y_range / 10
                    
                    if tick_spacing > 0:
                        start = math.ceil(y_min_plot / tick_spacing) * tick_spacing
                        end = math.floor(y_max_plot / tick_spacing) * tick_spacing
                        y_ticks = [start + i * tick_spacing for i in range(int((end - start) / tick_spacing) + 1)]
                        
                        if len(y_ticks) >= 5:
                            ax.set_yticks(y_ticks)
                            decimals_y = _get_decimals(abs(tick_spacing))
                            ax.set_yticklabels([f'{tick:.{decimals_y}f}' for tick in y_ticks], fontsize=9)
                
                if show_outlier_values and len(stats['outliers']) > 0:
                    outliers = stats['outliers'].sort_values()
                    lower_outliers = outliers[outliers < lower_bound].head(max_outliers_display).values
                    upper_outliers = outliers[outliers > upper_bound].tail(max_outliers_display).values
                    x_pos_outlier = 1.05
                    
                    if len(lower_outliers) > 0:
                        _plot_outliers(ax, lower_outliers, x_pos_outlier, decimals, max_outliers_display)
                    if len(upper_outliers) > 0:
                        _plot_outliers(ax, upper_outliers, x_pos_outlier, decimals, max_outliers_display)
                
                ax.minorticks_on()
                ax.tick_params(axis='y', which='minor', length=2, width=0.5)
                ax.grid(axis='y', which='minor', alpha=0.15, linestyle=':')
                ax.set_ylabel('Value', fontsize=11)
                ax.set_title(str(col), fontsize=13, fontweight='bold', pad=15, loc='center')
                ax.grid(axis='y', which='major', alpha=0.3, linestyle='--')
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel('')
        
        for idx in range(n_cols, len(axes_list)):
            axes_list[idx].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        
    else:
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.boxplot(data_list, vert=True, patch_artist=True,
                  showmeans=True, meanline=True,
                  labels=labels_list,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2),
                  meanprops=dict(color='green', linewidth=2, linestyle='--'),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5))
        
        for idx, (data_clean, col) in enumerate(zip(data_list, labels_list)):
            if col in stats_dict:
                stats = stats_dict[col]
                q1 = stats['q1']
                q3 = stats['q3']
                actual_min = data_clean.min()
                actual_max = data_clean.max()
                x_pos = idx + 1
                decimals = _get_decimals(actual_max - actual_min)
                
                ax.text(x_pos, actual_min, f' Min\n{actual_min:.{decimals}f}', 
                       verticalalignment='top', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue'))
                ax.text(x_pos, q1, f' Q1\n{q1:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='darkgreen'))
                ax.text(x_pos, q3, f' Q3\n{q3:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='darkgreen'))
                ax.text(x_pos, actual_max, f' Max\n{actual_max:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue'))
                
                if show_outlier_values and len(stats['outliers']) > 0:
                    outliers = stats['outliers'].sort_values()
                    lower_bound = stats['lower_bound']
                    upper_bound = stats['upper_bound']
                    lower_outliers = outliers[outliers < lower_bound].head(max_outliers_display).values
                    upper_outliers = outliers[outliers > upper_bound].tail(max_outliers_display).values
                    x_pos_outlier = x_pos + 0.08
                    
                    if len(lower_outliers) > 0:
                        ax.scatter([x_pos_outlier] * len(lower_outliers), lower_outliers, 
                                 color='red', s=20, alpha=0.6, zorder=5, marker='o')
                        if len(lower_outliers) <= 5:
                            for val in lower_outliers:
                                ax.text(x_pos_outlier + 0.05, val, f'{val:.{decimals}f}',
                                       fontsize=6, verticalalignment='center',
                                       bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', alpha=0.7, edgecolor='red'))
                    
                    if len(upper_outliers) > 0:
                        ax.scatter([x_pos_outlier] * len(upper_outliers), upper_outliers, 
                                 color='red', s=20, alpha=0.6, zorder=5, marker='o')
                        if len(upper_outliers) <= 5:
                            for val in upper_outliers:
                                ax.text(x_pos_outlier + 0.05, val, f'{val:.{decimals}f}',
                                       fontsize=6, verticalalignment='center',
                                       bbox=dict(boxstyle='round,pad=0.15', facecolor='yellow', alpha=0.7, edgecolor='red'))
        
        if len(data_list) > 0:
            all_values = []
            all_bounds = []
            for data_clean in data_list:
                all_values.extend([data_clean.min(), data_clean.max()])
                q1 = data_clean.quantile(0.25)
                q3 = data_clean.quantile(0.75)
                iqr = q3 - q1
                all_bounds.extend([q1 - 1.5 * iqr, q3 + 1.5 * iqr])
            
            y_min = min(min(all_values), min(all_bounds)) * 0.95
            y_max = max(max(all_values), max(all_bounds)) * 1.05
            y_range = y_max - y_min
            
            if y_range > 0:
                num_ticks = 12
                tick_spacing = y_range / num_ticks
                magnitude = 10 ** (math.floor(math.log10(tick_spacing)))
                tick_spacing = round(tick_spacing / magnitude) * magnitude
                
                start = math.ceil(y_min / tick_spacing) * tick_spacing
                end = math.floor(y_max / tick_spacing) * tick_spacing
                y_ticks = [start + i * tick_spacing for i in range(int((end - start) / tick_spacing) + 1)]
            
                if len(y_ticks) < 5:
                    tick_spacing = tick_spacing / 2
                    start = math.ceil(y_min / tick_spacing) * tick_spacing
                    end = math.floor(y_max / tick_spacing) * tick_spacing
                    y_ticks = [start + i * tick_spacing for i in range(int((end - start) / tick_spacing) + 1)]
                
                ax.set_yticks(y_ticks)
                decimals = _get_decimals(abs(tick_spacing))
                ax.set_yticklabels([f'{tick:.{decimals}f}' for tick in y_ticks], fontsize=10)
            else:
                ax.tick_params(axis='y', labelsize=10)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            
            ax.minorticks_on()
            ax.tick_params(axis='y', which='minor', length=3, width=1)
            ax.grid(axis='y', which='minor', alpha=0.2, linestyle=':')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', which='major', alpha=0.3, linestyle='--')
        plt.tight_layout()
    
    return fig, stats_dict


def visualization_bar_multiple(df, columns, bins=None, title='',
                               figsize=(15, 10), num_bins=10):
    n_cols = len(columns)
    if n_cols == 0:
        raise ValueError("columns 리스트가 비어있습니다.")
    
    n_rows = (n_cols + 2) // 3
    n_cols_plot = min(3, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=figsize)
    
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
            col_min = data.min()
            col_max = data.max()
            step = (col_max - col_min) / num_bins
            auto_bins = [col_min + i * step for i in range(num_bins + 1)]
            auto_bins = [round(b, 2) for b in auto_bins]
            auto_bins[-1] = col_max
            bins_col = auto_bins
        elif isinstance(bins, int):
            bins_col = bins
        else:
            bins_col = bins
        
        data_binned = pd.cut(data, bins=bins_col, include_lowest=True, duplicates='drop')
        counts = data_binned.value_counts().sort_index()
        
        labels = []
        for i, interval in enumerate(counts.index):
            left = interval.left
            right = interval.right
            if pd.notna(left) and pd.notna(right):
                labels.append(f"{int(left)}-{int(right)}" if i == 0 else f"{int(left)+1}-{int(right)}")
            else:
                labels.append(str(interval))
        
        values = counts.values
        total = values.sum()
        percentages = (values / total * 100).round(2)
        
        bars = ax.barh(range(len(labels)), values, color='crimson', alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Frequency", fontsize=9)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if len(labels) <= 15:
            for i, (bar, value, pct) in enumerate(zip(bars, values, percentages)):
                ax.text(bar.get_width() + total * 0.01, i, 
                       f'{value:,}\n({pct}%)', 
                       va='center', fontsize=7, fontweight='bold')
    
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


def visualization_categorical_bar(df, columns, title='', figsize=(15, 10), show_percentage=True):
    """
    범주형 변수들의 각 카테고리별 비율을 bar 그래프로 시각화하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        데이터프레임
    columns : list
        시각화할 범주형 컬럼명 리스트
    title : str
        차트 제목 (기본값: "")
    figsize : tuple
        차트 크기 (기본값: (15, 10))
    show_percentage : bool
        바 위에 비율(%) 표시 여부 (기본값: True)
    """
    n_cols = len(columns)
    if n_cols == 0:
        raise ValueError("columns 리스트가 비어있습니다.")
    
    n_rows = (n_cols + 2) // 3
    n_cols_plot = min(3, n_cols)
    fig_height = figsize[1] * n_rows / 2 if n_rows > 1 else figsize[1]
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(figsize[0], fig_height))
    
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
        
        bars = ax.barh(range(len(labels)), values, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)
        ax.set_yticks(range(len(labels)))
        display_labels = [str(label) if len(str(label)) <= 30 else str(label)[:27] + '...' for label in labels]
        ax.set_yticklabels(display_labels, fontsize=9)
        ax.set_xlabel("Frequency", fontsize=10)
        ax.set_ylabel("Category", fontsize=10)
        ax.set_title(col, fontsize=12, fontweight='bold', pad=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        if show_percentage:
            for i, (bar, value, pct) in enumerate(zip(bars, values, percentages)):
                ax.text(bar.get_width() + total * 0.01, i, 
                       f'{value:,} ({pct}%)', 
                       va='center', fontsize=9, fontweight='bold')
    
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title if title else "Categorical Variables Distribution", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    return fig


def visualization_histogram_by_group(df, value_col, group_col, title='', 
                                     figsize=(14, 6), bins=30, alpha=0.7, 
                                     side_by_side=True, show_statistics=True):
    """
    그룹별 수치형 변수의 히스토그램을 비교 시각화하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        데이터프레임
    value_col : str
        분포를 확인할 수치형 컬럼명 (예: 'exam_score')
    group_col : str
        그룹을 나누는 범주형 컬럼명 (예: 'internet_access')
    title : str
        차트 제목 (기본값: "")
    figsize : tuple
        차트 크기 (기본값: (14, 6))
    bins : int
        히스토그램 구간 개수 (기본값: 30)
    alpha : float
        투명도 (기본값: 0.7)
    side_by_side : bool
        True: 나란히 비교, False: 겹쳐서 비교 (기본값: True)
    show_statistics : bool
        통계 정보 표시 여부 (기본값: True)
    """
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
        fig, axes = plt.subplots(1, len(groups), figsize=figsize, sharey=True)
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
                
                if show_statistics:
                    stats_text = f"Count: {stats_dict[group]['count']:,}\n"
                    stats_text += f"Mean: {stats_dict[group]['mean']:.2f}\n"
                    stats_text += f"Median: {stats_dict[group]['median']:.2f}\n"
                    stats_text += f"Std: {stats_dict[group]['std']:.2f}\n"
                    stats_text += f"Min: {stats_dict[group]['min']:.2f}\n"
                    stats_text += f"Max: {stats_dict[group]['max']:.2f}"
                    
                    ax.text(0.65, 0.95, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel(value_col, fontsize=11)
                ax.set_ylabel('Frequency', fontsize=11)
                ax.set_title(f'{group_col} = {group}', fontsize=12, fontweight='bold')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(alpha=0.3, linestyle='--')
            else:
                ax.text(0.5, 0.5, f'{group_col} = {group}\n(No data)', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{group_col} = {group}', fontsize=12, fontweight='bold')
    else:
        fig, ax = plt.subplots(figsize=figsize)
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
        ax.set_title(title if title else f'{value_col} Distribution by {group_col}', 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.3, linestyle='--')
    
    if show_statistics:
        print("\n" + "=" * 80)
        print(f"{value_col} 통계 정보 (by {group_col})")
        print("=" * 80)
        for group in groups:
            if group in stats_dict:
                stats = stats_dict[group]
                print(f"\n[{group_col} = {group}]")
                print(f"  Count: {stats['count']:,}")
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Median: {stats['median']:.2f}")
                print(f"  Std: {stats['std']:.2f}")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
        print("=" * 80 + "\n")
    
    fig.suptitle(title if title else f'{value_col} Distribution by {group_col}', 
                 fontsize=14, fontweight='bold', y=1.0 if side_by_side else 0.98)
    plt.tight_layout()
    
    return fig, stats_dict


if __name__ == "__main__":
    from dataload import load_data, data_classifier
    
    df_train, df_test, df_sub = load_data()
    numeric_cols, categorical_cols, df_exam_score = data_classifier(df_train)
    
    num_bins_numeric = 10
    numeric_cols_to_visualize = [col for col in numeric_cols if col != 'id']
    
    if len(numeric_cols_to_visualize) > 0:
        fig_multi = visualization_bar_multiple(
            df_train,
            numeric_cols_to_visualize,
            bins=None,
            title=f"Numeric Columns Distribution (Bins: {num_bins_numeric})",
            figsize=(18, 6 * ((len(numeric_cols_to_visualize) + 2) // 3)),
            num_bins=num_bins_numeric
        )
        plt.show()
    
    numeric_cols_to_analyze = [col for col in numeric_cols if col != 'id']
    
    if len(numeric_cols_to_analyze) > 0:
        fig_iqr_multi, stats_iqr_multi = visualization_boxplot_iqr_multiple(
            df_train,
            numeric_cols_to_analyze,
            title="",
            figsize=(max(15, len(numeric_cols_to_analyze) * 2), 8),
            show_stats=True
        )
        plt.show()
    
    if len(categorical_cols) > 0:
        fig_cat = visualization_categorical_bar(
            df_train,
            categorical_cols,
            title="Categorical Variables Distribution",
            figsize=(18, 6 * ((len(categorical_cols) + 2) // 3)),
            show_percentage=True
        )
        plt.show()
    
    if 'internet_access' in df_train.columns and 'exam_score' in df_train.columns:
        fig_hist, stats_hist = visualization_histogram_by_group(
            df_train,
            value_col='exam_score',
            group_col='internet_access',
            title='Exam Score Distribution by Internet Access',
            figsize=(14, 6),
            bins=30,
            alpha=0.7,
            side_by_side=True,
            show_statistics=True
        )
        plt.show()
