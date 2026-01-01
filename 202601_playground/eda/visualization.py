import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualization_boxplot_iqr_multiple(df, columns, title="", 
                                       figsize=(15, 8), show_stats=True, grid_layout=True):
    """
    여러 수치형 컬럼을 grid 형태의 subplot에 boxplot으로 시각화하는 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        데이터프레임
    columns : list
        시각화할 컬럼명 리스트 (예: ['col1', 'col2', 'col3'])
    title : str
        차트 제목 (기본값: "Multiple Columns: IQR Outlier Detection")
    figsize : tuple
        차트 크기 (기본값: (15, 8))
    show_stats : bool
        IQR 통계 정보 출력 여부 (기본값: True)
    grid_layout : bool
        True: 각 컬럼을 개별 subplot으로 grid 형태로 배치 (기본값: True)
        False: 모든 컬럼을 하나의 subplot에 배치
    """
    stats_dict = {}
    
    # 데이터 준비
    data_list = []
    labels_list = []
    
    for col in columns:
        data_clean = df[col].dropna()
        if len(data_clean) > 0:
            data_list.append(data_clean)
            labels_list.append(col)
            
            # 각 컬럼별 IQR 통계 계산
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
        # Grid 형태: 각 컬럼을 개별 subplot으로
        n_rows = (n_cols + 2) // 3  # 3열 기준으로 행 개수 계산
        n_cols_plot = min(3, n_cols)  # 최대 3열
        
        # figsize 조정 (행 개수에 맞게)
        fig_height = figsize[1] * n_rows / 2 if n_rows > 1 else figsize[1]
        fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=(figsize[0], fig_height))
        
        # axes를 1차원 배열로 변환 (subplot이 1개일 때 대비)
        if n_cols == 1:
            axes_list = [axes]
        elif n_rows == 1:
            # 1행인 경우
            if isinstance(axes, np.ndarray):
                axes_list = axes.tolist()
            elif isinstance(axes, list):
                axes_list = axes
            else:
                axes_list = [axes]
        else:
            # 여러 행인 경우
            if isinstance(axes, np.ndarray):
                axes_list = axes.flatten().tolist()
            else:
                # 이미 list이거나 다른 형태인 경우
                try:
                    axes_list = np.array(axes).flatten().tolist()
                except:
                    axes_list = [ax for row in axes for ax in row] if isinstance(axes[0], (list, np.ndarray)) else list(axes)
        
        # axes 개수 확인 및 디버깅
        if len(axes_list) < n_cols:
            raise ValueError(f"Subplot 개수({len(axes_list)})가 컬럼 개수({n_cols})보다 적습니다.")
        
        # 디버깅 정보 출력
        print(f"Debug: n_cols={n_cols}, len(axes_list)={len(axes_list)}, len(data_list)={len(data_list)}, len(labels_list)={len(labels_list)}")
        print(f"Debug: labels_list={labels_list}")
        
        # 각 컬럼에 대해 개별 subplot에 boxplot 그리기
        for idx, (data_clean, col) in enumerate(zip(data_list, labels_list)):
            if idx >= len(axes_list):
                print(f"Warning: 컬럼 '{col}'를 위한 subplot이 없습니다. (idx: {idx}, axes 개수: {len(axes_list)})")
                continue
                
            ax = axes_list[idx]
            
            # 디버깅: 마지막 항목인지 확인
            if idx == len(data_list) - 1:
                print(f"Debug: 마지막 항목 처리 중 - col='{col}', idx={idx}, ax={ax}")
                print(f"Debug: col in stats_dict? {col in stats_dict}, stats_dict keys: {list(stats_dict.keys())}")
            
            # stats_dict에 없으면 통계 계산
            if col not in stats_dict:
                print(f"Warning: '{col}'가 stats_dict에 없습니다. 통계를 다시 계산합니다.")
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
            
            if col in stats_dict:
                stats = stats_dict[col]
                q1 = stats['q1']
                q3 = stats['q3']
                lower_bound = stats['lower_bound']
                upper_bound = stats['upper_bound']
                actual_min = data_clean.min()
                actual_max = data_clean.max()
                
                # Boxplot 그리기
                box_plot = ax.boxplot([data_clean], vert=True, patch_artist=True,
                                      showmeans=True, meanline=True,
                                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                                      medianprops=dict(color='red', linewidth=2),
                                      meanprops=dict(color='green', linewidth=2, linestyle='--'),
                                      whiskerprops=dict(color='black', linewidth=1.5),
                                      capprops=dict(color='black', linewidth=1.5))
                
                # 이상치 경계선 표시
                ax.axhline(y=lower_bound, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                ax.axhline(y=upper_bound, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                
                # 소수점 자리수 결정
                value_range = actual_max - actual_min
                if value_range >= 100:
                    decimals = 0
                elif value_range >= 10:
                    decimals = 1
                elif value_range >= 1:
                    decimals = 2
                else:
                    decimals = 3
                
                # Q1, Q3, min, max 값 표시
                x_pos = 1
                
                # y축 범위를 먼저 계산 (텍스트 공간 포함)
                # 데이터 범위 계산
                data_min = min(actual_min, lower_bound) if lower_bound < actual_min else actual_min
                data_max = max(actual_max, upper_bound) if upper_bound > actual_max else actual_max
                
                # 안전한 범위 계산 (0으로 나누기 방지)
                data_range = data_max - data_min
                if data_range <= 0:
                    data_range = max(abs(data_max), abs(data_min)) if data_max != data_min else 1
                    y_min_plot = data_min - data_range * 0.1
                    y_max_plot = data_max + data_range * 0.1
                else:
                    y_min_plot = data_min - data_range * 0.15
                    y_max_plot = data_max + data_range * 0.15
                
                # y축 범위 먼저 설정 (텍스트가 잘리지 않도록)
                ax.set_ylim(y_min_plot, y_max_plot)
                y_range = y_max_plot - y_min_plot
                
                # 텍스트 위치 조정 - 단순하고 확실한 방법
                # y축 범위를 5개 영역으로 나누어 각 값을 배치
                text_y_positions = []
                text_labels = []
                
                # 하단부터 상단까지 균등하게 분배 (4개 값)
                y_bottom = y_min_plot + y_range * 0.12  # 하단 여백
                y_top = y_max_plot - y_range * 0.12     # 상단 여백
                available_range = y_top - y_bottom
                
                # 각 텍스트의 y 위치 (하단부터 균등 분배)
                text_y_min = y_bottom + available_range * 0.1
                text_y_q1 = y_bottom + available_range * 0.35
                text_y_q3 = y_bottom + available_range * 0.65
                text_y_max = y_bottom + available_range * 0.9
                
                # Min 값 표시 (가장 아래)
                ax.text(x_pos + 0.08, text_y_min, f'Min: {actual_min:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='blue', linewidth=2))
                
                # Q1 값 표시
                ax.text(x_pos + 0.08, text_y_q1, f'Q1: {q1:.{decimals}f}', 
                       verticalalignment='center', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='darkgreen', linewidth=2))
                
                # Q3 값 표시
                ax.text(x_pos + 0.08, text_y_q3, f'Q3: {q3:.{decimals}f}', 
                       verticalalignment='center', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='darkgreen', linewidth=2))
                
                # Max 값 표시 (가장 위)
                ax.text(x_pos + 0.08, text_y_max, f'Max: {actual_max:.{decimals}f}', 
                       verticalalignment='top', horizontalalignment='left',
                       fontsize=10, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95, edgecolor='blue', linewidth=2))
                
                if y_range > 0:
                    import math
                    num_ticks = 10
                    tick_spacing = y_range / num_ticks
                    
                    # 0으로 나누기 방지
                    if tick_spacing > 0:
                        try:
                            magnitude = 10 ** (math.floor(math.log10(abs(tick_spacing))))
                            tick_spacing = round(tick_spacing / magnitude) * magnitude
                        except (ValueError, ZeroDivisionError):
                            tick_spacing = y_range / 10
                    else:
                        tick_spacing = y_range / 10
                    
                    y_ticks = []
                    if tick_spacing > 0:
                        start = math.ceil(y_min_plot / tick_spacing) * tick_spacing
                        end = math.floor(y_max_plot / tick_spacing) * tick_spacing
                        current = start
                        while current <= end:
                            y_ticks.append(current)
                            current += tick_spacing
                        
                        # 최소 5개 이상의 눈금이 있으면 설정
                        if len(y_ticks) >= 5:
                            ax.set_yticks(y_ticks)
                            
                            if abs(tick_spacing) >= 1:
                                decimals_y = 0
                            elif abs(tick_spacing) >= 0.1:
                                decimals_y = 1
                            elif abs(tick_spacing) >= 0.01:
                                decimals_y = 2
                            else:
                                decimals_y = 3
                            
                            ax.set_yticklabels([f'{tick:.{decimals_y}f}' for tick in y_ticks], fontsize=9)
                
                # 보조 눈금 추가
                ax.minorticks_on()
                ax.tick_params(axis='y', which='minor', length=2, width=0.5)
                ax.grid(axis='y', which='minor', alpha=0.15, linestyle=':')
                
                # 레이블 및 제목 설정
                ax.set_ylabel('Value', fontsize=11)
                # 제목 명확하게 설정 - col 변수가 제대로 전달되도록 보장
                col_name = str(col) if col else 'Unknown'
                ax.set_title(col_name, fontsize=13, fontweight='bold', pad=15, loc='center')
                ax.grid(axis='y', which='major', alpha=0.3, linestyle='--')
                # x축 레이블 제거 (boxplot에서는 불필요)
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel('')  # x축 레이블 명시적으로 제거
            else:
                # stats_dict에 없는 경우에도 기본 subplot 표시
                ax.text(0.5, 0.5, f'{col}\n(No statistics)', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=12)
                ax.set_title(str(col), fontsize=13, fontweight='bold', pad=15)
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 사용하지 않는 subplot 숨기기 (n_cols부터 끝까지, 마지막 항목은 제외)
        # n_cols는 실제 사용하는 subplot 개수이므로, n_cols 이상부터 숨김
        for idx in range(n_cols, len(axes_list)):
            if idx < len(axes_list):
                axes_list[idx].axis('off')
        
        # 디버깅: 마지막 항목 확인
        print(f"Debug: 마지막 항목 확인 - n_cols={n_cols}, len(axes_list)={len(axes_list)}")
        if n_cols > 0 and n_cols - 1 < len(axes_list):
            last_ax = axes_list[n_cols - 1]
            print(f"Debug: 마지막 axes 상태 - visible={last_ax.get_visible()}, title={last_ax.get_title()}")
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # 상단 여백 확보
        
    else:
        # 기존 방식: 모든 컬럼을 하나의 subplot에
        fig, ax = plt.subplots(figsize=figsize)
        
        # Boxplot 그리기
        box_plot = ax.boxplot(data_list, vert=True, patch_artist=True,
                              showmeans=True, meanline=True,
                              labels=labels_list,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              meanprops=dict(color='green', linewidth=2, linestyle='--'),
                              whiskerprops=dict(color='black', linewidth=1.5),
                              capprops=dict(color='black', linewidth=1.5))
        
        # 각 컬럼에 대해 Q1, Q3, min, max 값 표시
        for idx, (data_clean, col) in enumerate(zip(data_list, labels_list)):
            if col in stats_dict:
                stats = stats_dict[col]
                q1 = stats['q1']
                q3 = stats['q3']
                actual_min = data_clean.min()
                actual_max = data_clean.max()
                
                # x 위치 (boxplot 인덱스는 1부터 시작)
                x_pos = idx + 1
                
                # 소수점 자리수 결정
                value_range = actual_max - actual_min
                if value_range >= 100:
                    decimals = 0
                elif value_range >= 10:
                    decimals = 1
                elif value_range >= 1:
                    decimals = 2
                else:
                    decimals = 3
                
                # Min 값 표시 (whisker 아래쪽)
                ax.text(x_pos, actual_min, f' Min\n{actual_min:.{decimals}f}', 
                       verticalalignment='top', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue'))
                
                # Q1 값 표시 (box 하단)
                ax.text(x_pos, q1, f' Q1\n{q1:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='darkgreen'))
                
                # Q3 값 표시 (box 상단)
                ax.text(x_pos, q3, f' Q3\n{q3:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='darkgreen'))
                
                # Max 값 표시 (whisker 위쪽)
                ax.text(x_pos, actual_max, f' Max\n{actual_max:.{decimals}f}', 
                       verticalalignment='bottom', horizontalalignment='center',
                       fontsize=8, fontweight='bold', color='blue',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='blue'))
        
        # y축 눈금 세밀하게 설정
        if len(data_list) > 0:
            # 모든 데이터의 최소/최대값 계산
            all_values = []
            all_bounds = []
            for data_clean in data_list:
                all_values.extend([data_clean.min(), data_clean.max()])
                # 각 데이터의 IQR 경계도 고려
                q1 = data_clean.quantile(0.25)
                q3 = data_clean.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                all_bounds.extend([lower_bound, upper_bound])
            
            y_min = min(min(all_values), min(all_bounds)) * 0.95
            y_max = max(max(all_values), max(all_bounds)) * 1.05
            y_range = y_max - y_min
            
            if y_range > 0:
                # 눈금 간격 계산 (10-15개 정도의 눈금 표시)
                num_ticks = 12
                tick_spacing = y_range / num_ticks
                
                # 적절한 간격으로 반올림
                import math
                magnitude = 10 ** (math.floor(math.log10(tick_spacing)))
                tick_spacing = round(tick_spacing / magnitude) * magnitude
                
                # 눈금 위치 생성
                y_ticks = []
                start = math.ceil(y_min / tick_spacing) * tick_spacing
                end = math.floor(y_max / tick_spacing) * tick_spacing
                current = start
                while current <= end:
                    y_ticks.append(current)
                    current += tick_spacing
            
                if len(y_ticks) < 5:
                    # 눈금이 너무 적으면 더 세밀하게
                    tick_spacing = tick_spacing / 2
                    start = math.ceil(y_min / tick_spacing) * tick_spacing
                    end = math.floor(y_max / tick_spacing) * tick_spacing
                    y_ticks = []
                    current = start
                    while current <= end:
                        y_ticks.append(current)
                        current += tick_spacing
                
                ax.set_yticks(y_ticks)
                
                # 소수점 자리수 결정
                if tick_spacing >= 1:
                    decimals = 0
                elif tick_spacing >= 0.1:
                    decimals = 1
                elif tick_spacing >= 0.01:
                    decimals = 2
                else:
                    decimals = 3
                
                # y축 레이블 포맷팅
                ax.set_yticklabels([f'{tick:.{decimals}f}' for tick in y_ticks], fontsize=10)
            else:
                # 범위가 0이면 자동 설정
                ax.tick_params(axis='y', labelsize=10)
                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
                # 소수점 자리수 표시
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            
            # 보조 눈금(minor ticks) 추가
            ax.minorticks_on()
            ax.tick_params(axis='y', which='minor', length=3, width=1)
            ax.grid(axis='y', which='minor', alpha=0.2, linestyle=':')
        
        # X축 레이블 회전 (긴 이름 대비)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # 레이블 및 제목 설정
        ax.set_ylabel('Value', fontsize=12)
        ax.set_xlabel('Columns', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', which='major', alpha=0.3, linestyle='--')
        
        # 레이아웃 조정
        plt.tight_layout()
    
    return fig, stats_dict


def visualization_bar_multiple(df, columns, bins=None, title='',
                               figsize=(15, 10), num_bins=10):
    n_cols = len(columns)
    if n_cols == 0:
        raise ValueError("columns 리스트가 비어있습니다.")
    
    # subplot 개수 계산 (3열 기준)
    n_rows = (n_cols + 2) // 3  # 올림 계산
    n_cols_plot = min(3, n_cols)  # 최대 3열
    
    fig, axes = plt.subplots(n_rows, n_cols_plot, figsize=figsize)
    
    # axes를 1차원 배열로 변환 (subplot이 1개일 때 대비)
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
        
        # 구간 설정
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
        
        # 구간 분할
        data_binned = pd.cut(data, bins=bins_col, include_lowest=True, duplicates='drop')
        counts = data_binned.value_counts().sort_index()
        
        # 레이블 생성
        labels = []
        for i, interval in enumerate(counts.index):
            left = interval.left
            right = interval.right
            if pd.notna(left) and pd.notna(right):
                if i == 0:
                    labels.append(f"{int(left)}-{int(right)}")
                else:
                    labels.append(f"{int(left)+1}-{int(right)}")
            else:
                labels.append(str(interval))
        
        values = counts.values
        total = values.sum()
        percentages = (values / total * 100).round(2)
        
        # 바 차트 그리기
        bars = ax.barh(range(len(labels)), values, color='crimson', alpha=0.7, edgecolor='black')
        
        # Y축 레이블 설정
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Frequency", fontsize=9)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # 바 위에 비율 표시 (공간이 충분할 때만)
        if len(labels) <= 15:
            for i, (bar, value, pct) in enumerate(zip(bars, values, percentages)):
                x_pos = bar.get_width()
                ax.text(x_pos + total * 0.01, i, 
                       f'{value:,}\n({pct}%)', 
                       va='center', fontsize=7, fontweight='bold')
    
    # 사용하지 않는 subplot 숨기기
    for idx in range(n_cols, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig


# 사용 예시
if __name__ == "__main__":
    # dataload 모듈에서 데이터 로드
    from dataload import load_data, data_classifier
    
    # 데이터 로드
    df_train, df_test, df_sub = load_data()
    
    # exam_score 컬럼의 분포 시각화 예시
    numeric_cols, categorical_cols, df_exam_score = data_classifier(df_train)
    
    # exam_score의 범위를 확인하고 적절한 구간 설정
    score_min = df_exam_score.min()
    score_max = df_exam_score.max()
    
    # 구간 개수 설정
    num_bins_numeric = 10  # 원하는 구간 개수를 여기에 입력 (예: 5, 10, 20 등)
    
    # id와 exam_score는 제외 (id는 불필요, exam_score는 이미 시각화됨)
    # numeric_cols_to_visualize = [col for col in numeric_cols if col != 'exam_score']
    numeric_cols_to_visualize = [col for col in numeric_cols if col != 'id']  # id 제외, 모든 수치형 컬럼 시각화
    
    # 여러 컬럼을 하나의 subplot에 시각화
    if len(numeric_cols_to_visualize) > 0:
        print(f"\n{numeric_cols_to_visualize} 컬럼들을 하나의 subplot에 시각화 중...")
        fig_multi = visualization_bar_multiple(
            df_train,
            numeric_cols_to_visualize,
            bins=None,  # 자동 구간 생성
            title=f"Numeric Columns Distribution (Bins: {num_bins_numeric})",
            figsize=(18, 6 * ((len(numeric_cols_to_visualize) + 2) // 3)),
            num_bins=num_bins_numeric
        )
        plt.show()
        print("모든 수치형 컬럼 시각화 완료!")
    
    print("\n" + "=" * 80)
    print("=" * 80)
    
    # IQR을 활용한 이상치 탐지 (Boxplot) - 여러 컬럼을 하나의 subplot에
    print("\n" + "=" * 80)
    print("IQR을 활용한 이상치 탐지 (Boxplot - Multiple Columns)")
    print("=" * 80)
    
    # id와 exam_score는 제외 (id는 불필요, exam_score는 이미 시각화됨)
    # numeric_cols_to_analyze = [col for col in numeric_cols if col != 'exam_score']
    numeric_cols_to_analyze = [col for col in numeric_cols if col != 'id']  # id 제외, 모든 수치형 컬럼 분석
    
    # 여러 컬럼을 하나의 boxplot에 시각화
    if len(numeric_cols_to_analyze) > 0:
        fig_iqr_multi, stats_iqr_multi = visualization_boxplot_iqr_multiple(
            df_train,
            numeric_cols_to_analyze,
            title="",
            figsize=(max(15, len(numeric_cols_to_analyze) * 2), 8),
            show_stats=True
        )
        plt.show()
        print("모든 수치형 컬럼 이상치 탐지 완료!")
    
    print("=" * 80)
    print("=" * 80)
