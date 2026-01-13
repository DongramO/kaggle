"""
특성 간 상관관계 및 다중공선성 분석 스크립트
전처리 전후 데이터 모두 분석 가능
"""
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.family'] = 'AppleGothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

# 데이터 로드 및 전처리 함수 import
try:
    from modeling.train import prepare_data
except ImportError:
    print("⚠️ modeling.train에서 prepare_data를 import할 수 없습니다.")
    raise

try:
    from eda.dataload import load_data
except ImportError:
    print("⚠️ eda.dataload에서 load_data를 import할 수 없습니다.")
    print("   기본 데이터 로드 함수를 사용합니다.")
    
    def load_data():
        """기본 데이터 로드"""
        if os.path.exists('data/train.csv'):
            return pd.read_csv('data/train.csv'), pd.read_csv('data/test.csv'), pd.read_csv('data/sample_submission.csv')
        elif os.path.exists('../data/train.csv'):
            return pd.read_csv('../data/train.csv'), pd.read_csv('../data/test.csv'), pd.read_csv('../data/sample_submission.csv')
        else:
            raise FileNotFoundError("데이터 파일을 찾을 수 없습니다.")

def is_kaggle_environment():
    return os.path.exists('/kaggle/input')

# 분석 함수들
try:
    from eda.correlation_analysis import (
        calculate_correlation_matrix,
        plot_correlation_heatmap,
        find_high_correlations,
        calculate_vif,
        analyze_multicollinearity,
        plot_correlation_with_target
    )
except ImportError:
    print("⚠️ correlation_analysis.py를 찾을 수 없습니다.")
    print("   인라인으로 분석 함수를 사용합니다.")
    
    # 인라인 분석 함수들 (간단 버전)
    def calculate_correlation_matrix(X_train, method='pearson'):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        return X_train[numeric_cols].corr(method=method)
    
    def plot_correlation_heatmap(corr_matrix, figsize=(15, 12), save_path=None, title="특성 간 상관관계 히트맵", threshold=None):
        plt.figure(figsize=figsize)
        mask = None
        if threshold is not None:
            mask = np.abs(corr_matrix) < threshold
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   vmin=-1, vmax=1, square=True, linewidths=0.5, mask=mask)
        plt.title(title, fontsize=16, pad=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 히트맵 저장: {save_path}")
        plt.show()
    
    def find_high_correlations(corr_matrix, threshold=0.7, exclude_diagonal=True):
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
    
    def calculate_vif(X_train, target_col=None):
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
            if target_col and target_col in X_train.columns:
                X = X_train.drop(columns=[target_col])
            else:
                X = X_train.copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X_numeric = X[numeric_cols].dropna()
            if len(X_numeric) == 0:
                return pd.DataFrame()
            vif_data = []
            for i in range(len(X_numeric.columns)):
                try:
                    vif_value = variance_inflation_factor(X_numeric.values, i)
                    vif_data.append({'Feature': X_numeric.columns[i], 'VIF': vif_value})
                except:
                    vif_data.append({'Feature': X_numeric.columns[i], 'VIF': np.nan})
            vif_df = pd.DataFrame(vif_data)
            def classify_multicollinearity(vif):
                if pd.isna(vif):
                    return '계산 불가'
                elif vif < 5:
                    return '낮음 (양호)'
                elif vif < 10:
                    return '보통 (주의)'
                else:
                    return '높음 (문제)'
            vif_df['Multicollinearity_Level'] = vif_df['VIF'].apply(classify_multicollinearity)
            return vif_df.sort_values('VIF', ascending=False, na_last=True)
        except ImportError:
            print("⚠️ statsmodels가 설치되어 있지 않습니다. pip install statsmodels")
            return pd.DataFrame()
    
    def analyze_multicollinearity(X_train, target_col=None, vif_threshold=10.0, corr_threshold=0.8):
        results = {}
        corr_matrix = calculate_correlation_matrix(X_train)
        high_corr = find_high_correlations(corr_matrix, threshold=corr_threshold)
        results['correlation_matrix'] = corr_matrix
        results['high_correlations'] = high_corr
        try:
            results['vif'] = calculate_vif(X_train, target_col)
        except:
            results['vif'] = None
        return results
    
    def plot_correlation_with_target(X_train, y_train, top_n=20, save_path=None):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        correlations = []
        for col in numeric_cols:
            corr = X_train[col].corr(y_train)
            correlations.append({'Feature': col, 'Correlation': corr, 'Abs_Correlation': abs(corr)})
        corr_df = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False).head(top_n)
        plt.figure(figsize=(10, max(8, len(corr_df) * 0.4)))
        colors = ['red' if x < 0 else 'blue' for x in corr_df['Correlation']]
        plt.barh(corr_df['Feature'], corr_df['Correlation'], color=colors)
        plt.xlabel('상관계수', fontsize=12)
        plt.title(f'타겟 변수와의 상관관계 (상위 {top_n}개)', fontsize=14)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        return corr_df


def analyze_preprocessed_data():
    """
    전처리된 데이터로 상관관계 및 다중공선성 분석
    실제 모델 학습에 사용되는 데이터와 동일한 전처리 과정을 거친 데이터 분석
    """
    print("="*60)
    print("📊 전처리된 데이터 상관관계 분석")
    print("="*60)
    
    # 출력 디렉토리
    output_dir = 'eda_results'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df_train, df_test, df_sub = load_data()
    
    # 전처리 (모델 학습과 동일한 과정)
    print("\n🔧 데이터 전처리 중...")
    ENCODING_CONFIG = {
        'onehot_cols': ['gender', 'course', 'internet_access', 'study_method'],
        'ordinal_cols': ['exam_difficulty', 'facility_rating', 'sleep_quality'],
        'onehot_params': {'handle_unknown': 'ignore'},
        'ordinal_params': {'handle_unknown': 'use_encoded_value', 'unknown_value': -1},
        'drop_original': True
    }
    
    X_train, y_train, X_test, categorical_cols, numeric_cols, encoder, encoded_cols_tag = prepare_data(
        df_train, df_test, 
        target_col='exam_score', 
        use_feature_engineering=True,
        encoding_config=ENCODING_CONFIG
    )
    
    print(f"\n📊 전처리된 데이터 정보:")
    print(f"   총 특성 수: {len(X_train.columns)}")
    print(f"   학습 데이터 크기: {X_train.shape}")
    
    # 1. 상관관계 히트맵 (전처리된 데이터)
    print(f"\n{'='*60}")
    print("1️⃣ 전처리된 데이터 상관관계 히트맵")
    print(f"{'='*60}")
    
    corr_matrix = calculate_correlation_matrix(X_train, method='pearson')
    print(f"   상관관계 행렬 크기: {corr_matrix.shape}")
    
    # 전체 히트맵
    plot_correlation_heatmap(
        corr_matrix,
        figsize=(20, 16),
        save_path=f"{output_dir}/correlation_heatmap_preprocessed.png",
        title="전처리된 데이터 특성 간 상관관계 히트맵"
    )
    
    # 2. 높은 상관관계 특성 쌍
    print(f"\n{'='*60}")
    print("2️⃣ 높은 상관관계 특성 쌍")
    print(f"{'='*60}")
    
    for threshold in [0.95, 0.9, 0.8, 0.7]:
        high_corr = find_high_correlations(corr_matrix, threshold=threshold)
        print(f"\n   |상관계수| >= {threshold}: {len(high_corr)}개 쌍")
        if len(high_corr) > 0:
            print("   상위 10개:")
            for idx, row in high_corr.head(10).iterrows():
                print(f"     {row['Feature_1']:30s} ↔ {row['Feature_2']:30s}: {row['Correlation']:7.4f}")
    
    # 저장
    high_corr_all = find_high_correlations(corr_matrix, threshold=0.7)
    if len(high_corr_all) > 0:
        high_corr_all.to_csv(
            f"{output_dir}/high_correlations_preprocessed.csv", 
            index=False, 
            encoding='utf-8-sig'
        )
        print(f"\n   ✅ 저장: {output_dir}/high_correlations_preprocessed.csv")
    
    # 3. 다중공선성 분석 (VIF)
    print(f"\n{'='*60}")
    print("3️⃣ 다중공선성 분석 (VIF)")
    print(f"{'='*60}")
    
    analysis_results = analyze_multicollinearity(X_train, target_col=None)
    
    if analysis_results.get('vif') is not None and len(analysis_results['vif']) > 0:
        vif_df = analysis_results['vif']
        vif_df.to_csv(
            f"{output_dir}/vif_preprocessed.csv", 
            index=False, 
            encoding='utf-8-sig'
        )
        print(f"   ✅ VIF 결과 저장: {output_dir}/vif_preprocessed.csv")
        
        # 높은 VIF 특성 출력
        high_vif = vif_df[vif_df['VIF'] >= 10]
        if len(high_vif) > 0:
            print(f"\n   ⚠️ 높은 VIF 특성 (VIF >= 10): {len(high_vif)}개")
            for idx, row in high_vif.iterrows():
                print(f"     {row['Feature']:40s}: VIF = {row['VIF']:8.2f} ({row['Multicollinearity_Level']})")
        else:
            print("\n   ✅ VIF >= 10인 특성이 없습니다. 다중공선성 문제가 적습니다.")
        
        # VIF 시각화
        plt.figure(figsize=(12, max(10, len(vif_df) * 0.25)))
        vif_sorted = vif_df.sort_values('VIF', ascending=True).tail(30)
        colors = ['green' if v < 5 else 'orange' if v < 10 else 'red' 
                 for v in vif_sorted['VIF']]
        plt.barh(vif_sorted['Feature'], vif_sorted['VIF'], color=colors)
        plt.xlabel('VIF (Variance Inflation Factor)', fontsize=12)
        plt.title('다중공선성 분석 - VIF (상위 30개)', fontsize=14, pad=20)
        plt.axvline(x=5, color='orange', linestyle='--', linewidth=1.5, label='VIF = 5 (주의)')
        plt.axvline(x=10, color='red', linestyle='--', linewidth=1.5, label='VIF = 10 (문제)')
        plt.legend(fontsize=10)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vif_preprocessed.png", dpi=300, bbox_inches='tight')
        print(f"   ✅ VIF 그래프 저장: {output_dir}/vif_preprocessed.png")
        plt.show()
    
    # 4. 타겟 변수와의 상관관계
    print(f"\n{'='*60}")
    print("4️⃣ 타겟 변수와의 상관관계")
    print(f"{'='*60}")
    
    target_corr = plot_correlation_with_target(
        X_train, y_train, top_n=30,
        save_path=f"{output_dir}/target_correlation_preprocessed.png"
    )
    target_corr.to_csv(
        f"{output_dir}/target_correlations_preprocessed.csv", 
        index=False, 
        encoding='utf-8-sig'
    )
    print(f"   ✅ 타겟 상관관계 저장: {output_dir}/target_correlations_preprocessed.csv")
    
    print(f"\n상위 10개 특성:")
    for idx, row in target_corr.head(10).iterrows():
        print(f"   {row['Feature']:40s}: {row['Correlation']:7.4f}")
    
    # 5. 모델별 특성 구성 분석
    print(f"\n{'='*60}")
    print("5️⃣ 모델별 특성 구성 분석")
    print(f"{'='*60}")
    
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    
    # CatBoost 특성
    catboost_features = [col for col in X_train.columns if col not in encoded_cols]
    print(f"\n   CatBoost 사용 특성: {len(catboost_features)}개")
    print(f"     - 원본 범주형 포함: {len([c for c in categorical_cols if c in catboost_features])}개")
    print(f"     - 수치형 + FE 특성: {len([c for c in catboost_features if c not in categorical_cols])}개")
    
    # LightGBM/XGBoost 특성
    lgbm_xgb_features = [col for col in X_train.columns if col not in categorical_cols]
    print(f"\n   LightGBM/XGBoost 사용 특성: {len(lgbm_xgb_features)}개")
    print(f"     - Ordinal 인코딩: {len(encoded_cols)}개")
    print(f"     - One-Hot 인코딩: {len([c for c in lgbm_xgb_features if any(col in c for col in ENCODING_CONFIG.get('onehot_cols', []))])}개")
    print(f"     - 수치형 + FE 특성: {len([c for c in lgbm_xgb_features if c not in encoded_cols and not any(col in c for col in ENCODING_CONFIG.get('onehot_cols', []))])}개")
    
    # 모델별 상관관계 분석
    print(f"\n   CatBoost 특성 간 상관관계 분석...")
    catboost_corr = calculate_correlation_matrix(X_train[catboost_features])
    catboost_high_corr = find_high_correlations(catboost_corr, threshold=0.7)
    print(f"     높은 상관관계 (|r| >= 0.7): {len(catboost_high_corr)}개 쌍")
    if len(catboost_high_corr) > 0:
        catboost_high_corr.to_csv(
            f"{output_dir}/catboost_high_correlations.csv",
            index=False,
            encoding='utf-8-sig'
        )
    
    print(f"\n   LightGBM/XGBoost 특성 간 상관관계 분석...")
    lgbm_xgb_corr = calculate_correlation_matrix(X_train[lgbm_xgb_features])
    lgbm_xgb_high_corr = find_high_correlations(lgbm_xgb_corr, threshold=0.7)
    print(f"     높은 상관관계 (|r| >= 0.7): {len(lgbm_xgb_high_corr)}개 쌍")
    if len(lgbm_xgb_high_corr) > 0:
        lgbm_xgb_high_corr.to_csv(
            f"{output_dir}/lgbm_xgb_high_correlations.csv",
            index=False,
            encoding='utf-8-sig'
        )
    
    print(f"\n{'='*60}")
    print("✅ 분석 완료!")
    print(f"   결과 저장 위치: {output_dir}/")
    print(f"{'='*60}")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_all,
        'vif': analysis_results.get('vif'),
        'target_correlations': target_corr
    }


if __name__ == "__main__":
    # 전처리된 데이터 분석 실행
    results = analyze_preprocessed_data()
