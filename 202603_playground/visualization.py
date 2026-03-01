"""
202603_playground EDA 시각화
common.eda.visualization, common.eda.correlation 사용.
실행: python visualization.py
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

from config import PROJECT_ROOT, ID_COL, TARGET_COL
from common.data.loader import load_data
from common.eda import get_numeric_and_categorical
from common.eda.visualization import (
    PlotConfig,
    plot_boxplot,
    plot_histogram,
    plot_categorical,
    plot_histogram_by_group,
    plot_boxplot_by_group,
)
from common.eda.correlation import (
    calculate_correlation_matrix,
    plot_correlation_heatmap,
    plot_correlation_with_target,
    CorrelationConfig,
)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional


def run_eda_visualization(
    df: Optional[pd.DataFrame] = None,
    data_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    id_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> None:
    """
    EDA 시각화: boxplot, histogram, 범주형 분포, 상관계수·타겟 상관.
    그룹별 히스토그램은 필요 시 common.eda.visualization.plot_histogram_by_group를 value_col, group_col 지정해 직접 호출.
    df 지정 시 해당 데이터로 시각화(특성 공학 후 가능). df가 None이면 data_dir에서 로드.
    """
    if df is None:
        data_dir = PROJECT_ROOT
        if not os.path.isdir(data_dir):
            data_dir = PROJECT_ROOT
        print("📂 데이터 로드 중...")
        df, _, _ = load_data(data_dir=data_dir)
    else:
        df = df.copy()

    output_dir = output_dir or os.path.join(PROJECT_ROOT, 'eda_results')
    os.makedirs(output_dir, exist_ok=True)
    id_col = id_col if id_col is not None else ID_COL
    target_col = target_col if target_col is not None else TARGET_COL

    plot_config = PlotConfig()
    corr_config = CorrelationConfig()
    PLOTS_PER_FILE = 9  # boxplot, histogram 한 파일당 최대 컬럼 수

    numeric_cols, categorical_cols = get_numeric_and_categorical(
        df, id_col=id_col,
    )
    print(f"  수치형: {len(numeric_cols)}개, 범주형: {len(categorical_cols)}개")

    # Boxplot: 컬럼이 많으면 9개씩 끊어서 여러 PNG로 저장
    if numeric_cols:
        print("📊 Boxplot 생성 중...")
        for i in range(0, len(numeric_cols), PLOTS_PER_FILE):
            chunk = numeric_cols[i : i + PLOTS_PER_FILE]
            fig, _ = plot_boxplot(df, chunk, config=plot_config)
            page = i // PLOTS_PER_FILE + 1
            if len(numeric_cols) <= PLOTS_PER_FILE:
                path = os.path.join(output_dir, 'boxplot.png')
            else:
                path = os.path.join(output_dir, f'boxplot_{page}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        if len(numeric_cols) > PLOTS_PER_FILE:
            print(f"  → {len(numeric_cols)}개 컬럼을 {(len(numeric_cols) - 1) // PLOTS_PER_FILE + 1}개 파일로 저장")

    # Histogram: 컬럼이 많으면 9개씩 끊어서 여러 PNG로 저장
    if numeric_cols:
        print("📊 Histogram 생성 중...")
        for i in range(0, len(numeric_cols), PLOTS_PER_FILE):
            chunk = numeric_cols[i : i + PLOTS_PER_FILE]
            fig = plot_histogram(df, chunk, bins=10, config=plot_config)
            page = i // PLOTS_PER_FILE + 1
            if len(numeric_cols) <= PLOTS_PER_FILE:
                path = os.path.join(output_dir, 'histogram.png')
            else:
                path = os.path.join(output_dir, f'histogram_{page}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        if len(numeric_cols) > PLOTS_PER_FILE:
            print(f"  → {len(numeric_cols)}개 컬럼을 {(len(numeric_cols) - 1) // PLOTS_PER_FILE + 1}개 파일로 저장")
    
    
    # grouping histogram
    # fig, _ = plot_histogram_by_group(
    #     df, value_col='exam_score', group_col='gender',
    #     config=PlotConfig(), bins=20, side_by_side=True,
    # )
    # fig.savefig('eda_results/histogram3.png', dpi=150, bbox_inches='tight')
    # plt.close(fig)

    # fig, _ = plot_histogram_by_group(
    #     df, value_col='study_hours', group_col='sleep_quality',
    #     config=PlotConfig(), bins=20, side_by_side=True,
    # )
    
    # fig, stats = plot_boxplot_by_group(
    #     df, value_col='exam_score', group_col='course',
    #     config=PlotConfig(),
    # )
    # fig.savefig(os.path.join(output_dir, 'boxplot_by_gender.png'), dpi=150, bbox_inches='tight')
    # plt.close(fig)

    # fig.savefig('eda_results/histogram2.png', dpi=150, bbox_inches='tight')
    # plt.close(fig)


    # Categorical
    if categorical_cols:
        print("📊 범주형 분포 생성 중...")
        fig = plot_categorical(df, categorical_cols, config=plot_config)
        fig.savefig(os.path.join(output_dir, 'categorical.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 상관계수: 수치형 + 타겟(범주형이면 0/1 등 수치로 변환 후 포함)
    if target_col and target_col in df.columns and len(numeric_cols) >= 1:
        y_series = df[target_col]
        if not pd.api.types.is_numeric_dtype(y_series):
            y_numeric = pd.Series(pd.Categorical(y_series).codes, index=df.index, name=target_col)
        else:
            y_numeric = y_series
        corr_df = df[numeric_cols].copy()
        corr_df[target_col] = y_numeric
        if len(corr_df.columns) >= 2:
            print("📊 상관계수 히트맵 생성 중 (타겟 포함)...")
            corr_matrix = calculate_correlation_matrix(
                corr_df, method=corr_config.method, numeric_only=True
            )
            fig = plot_correlation_heatmap(corr_matrix, config=corr_config)
            fig.savefig(
                os.path.join(output_dir, 'correlation_heatmap.png'),
                dpi=150, bbox_inches='tight',
            )
            plt.close(fig)

            print("📊 타겟 상관관계 생성 중...")
            fig, _ = plot_correlation_with_target(
                df[numeric_cols], y_numeric, config=corr_config,
            )
            fig.savefig(
                os.path.join(output_dir, 'correlation_with_target.png'),
                dpi=150, bbox_inches='tight',
            )
            plt.close(fig)
    elif len(numeric_cols) >= 2:
        print("📊 상관계수 히트맵 생성 중...")
        corr_matrix = calculate_correlation_matrix(
            df[numeric_cols], method=corr_config.method, numeric_only=True
        )
        fig = plot_correlation_heatmap(corr_matrix, config=corr_config)
        fig.savefig(
            os.path.join(output_dir, 'correlation_heatmap.png'),
            dpi=150, bbox_inches='tight',
        )
        plt.close(fig)

    print(f"✅ 시각화 저장 완료: {output_dir}")


if __name__ == "__main__":
    run_eda_visualization()
