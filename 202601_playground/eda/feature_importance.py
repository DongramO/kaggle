"""
Feature Importance 분석 및 시각화 모듈
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 사용할 수 없습니다.")


def analyze_permutation_importance(trainer, X_train: pd.DataFrame, y_train: pd.Series,
                                   categorical_cols: List[str], encoded_cols_tag: str = '_encoded',
                                   top_n: int = 20, n_repeats: int = 10, 
                                   save_dir: str = 'feature_importance_results'):
    """
    Permutation Importance 분석 및 시각화
    
    Parameters:
    -----------
    trainer : ModelTrainer
        학습된 모델을 포함한 ModelTrainer 객체
    X_train : pd.DataFrame
        학습 데이터 (검증용)
    y_train : pd.Series
        타겟 데이터 (검증용)
    categorical_cols : List[str]
        범주형 컬럼 리스트
    encoded_cols_tag : str
        인코딩된 컬럼 태그
    top_n : int
        상위 N개 특성만 표시
    n_repeats : int
        Permutation 반복 횟수
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    dict or None
        모델별 Permutation Importance 딕셔너리
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 건너뜁니다.")
        return None
    
    # 저장 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 Permutation Importance 분석")
    print(f"{'='*60}")
    
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    all_importances = {}
    
    # 모델별로 Permutation Importance 추출
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        if model_type not in trainer.models:
            print(f"⚠️ {model_type} 모델이 없습니다. 건너뜁니다.")
            continue
        
        print(f"\n🔍 {model_type.upper()} Permutation Importance 추출 중...")
        
        # 모델별 특성 선택
        if model_type == 'catboost':
            feature_cols = [col for col in X_train.columns if col not in encoded_cols]
        else:
            feature_cols = [col for col in X_train.columns if col not in categorical_cols]
        
        try:
            importance_df = trainer.get_permutation_importance(
                model_type=model_type,
                X=X_train[feature_cols],
                y=y_train,
                feature_names=feature_cols,
                n_repeats=n_repeats,
                random_state=trainer.random_state
            )
            
            if len(importance_df) > 0:
                all_importances[model_type] = importance_df
                
                # CSV 저장
                csv_path = os.path.join(save_dir, f'{model_type}_permutation_importance.csv')
                importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"   ✅ 저장: {csv_path}")
                
                # 상위 N개 출력
                print(f"\n   상위 {top_n}개 특성:")
                for idx, row in importance_df.head(top_n).iterrows():
                    print(f"     {idx+1:2d}. {row['Feature']:40s}: {row['Importance']:8.4f} (std: {row['Std']:.4f})")
            else:
                print(f"   ⚠️ Permutation Importance를 추출할 수 없습니다.")
        except Exception as e:
            print(f"   ⚠️ {model_type} Permutation Importance 추출 실패: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_importances) == 0:
        print("\n⚠️ 추출된 Permutation Importance가 없습니다.")
        return None
    
    # 시각화
    print(f"\n📈 Permutation Importance 시각화 중...")
    
    # 1. 모델별 상위 N개 특성 비교
    _plot_model_comparison(all_importances, top_n, save_dir, suffix='permutation')
    
    # 2. 모든 모델의 상위 N개 특성을 하나의 그래프로 (누적 막대)
    if len(all_importances) > 1:
        _plot_combined_comparison(all_importances, top_n, save_dir, suffix='permutation')
    
    # 3. 공통 중요 특성 찾기 및 저장
    if len(all_importances) > 1:
        _find_and_save_common_features(all_importances, top_n, save_dir, suffix='permutation')
    
    print(f"\n{'='*60}")
    print(f"✅ Permutation Importance 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return all_importances


def analyze_feature_importance(trainer, X_train: pd.DataFrame, 
                               categorical_cols: List[str], encoded_cols_tag: str = '_encoded',
                               top_n: int = 20, save_dir: str = 'feature_importance_results'):
    """
    Feature Importance 분석 및 시각화
    
    Parameters:
    -----------
    trainer : ModelTrainer
        학습된 모델을 포함한 ModelTrainer 객체
    X_train : pd.DataFrame
        학습 데이터 (특성 이름 추출용)
    categorical_cols : List[str]
        범주형 컬럼 리스트
    encoded_cols_tag : str
        인코딩된 컬럼 태그
    top_n : int
        상위 N개 특성만 표시
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    dict or None
        모델별 Feature Importance 딕셔너리, 시각화 불가능한 경우 None
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 건너뜁니다.")
        return None
    
    # 저장 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 Feature Importance 분석")
    print(f"{'='*60}")
    
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    all_importances = {}
    
    # 모델별로 Feature Importance 추출
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        if model_type not in trainer.models:
            print(f"⚠️ {model_type} 모델이 없습니다. 건너뜁니다.")
            continue
        
        print(f"\n🔍 {model_type.upper()} Feature Importance 추출 중...")
        
        # 모델별 특성 선택
        if model_type == 'catboost':
            feature_cols = [col for col in X_train.columns if col not in encoded_cols]
        else:
            feature_cols = [col for col in X_train.columns if col not in categorical_cols]
        
        try:
            importance_df = trainer.get_feature_importance(
                model_type=model_type,
                feature_names=feature_cols,
                average_across_folds=True
            )
            
            if len(importance_df) > 0:
                all_importances[model_type] = importance_df
                
                # CSV 저장
                csv_path = os.path.join(save_dir, f'{model_type}_feature_importance.csv')
                importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"   ✅ 저장: {csv_path}")
                
                # 상위 N개 출력
                print(f"\n   상위 {top_n}개 특성:")
                for idx, row in importance_df.head(top_n).iterrows():
                    print(f"     {idx+1:2d}. {row['Feature']:40s}: {row['Importance']:8.4f} (std: {row['Std']:.4f})")
            else:
                print(f"   ⚠️ Feature Importance를 추출할 수 없습니다.")
        except Exception as e:
            print(f"   ⚠️ {model_type} Feature Importance 추출 실패: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_importances) == 0:
        print("\n⚠️ 추출된 Feature Importance가 없습니다.")
        return None
    
    # 시각화
    print(f"\n📈 Feature Importance 시각화 중...")
    
    # 1. 모델별 상위 N개 특성 비교
    _plot_model_comparison(all_importances, top_n, save_dir)
    
    # 2. 모든 모델의 상위 N개 특성을 하나의 그래프로 (누적 막대)
    if len(all_importances) > 1:
        _plot_combined_comparison(all_importances, top_n, save_dir)
    
    # 3. 공통 중요 특성 찾기 및 저장
    if len(all_importances) > 1:
        _find_and_save_common_features(all_importances, top_n, save_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Feature Importance 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return all_importances


def _plot_model_comparison(all_importances: Dict[str, pd.DataFrame], top_n: int, save_dir: str, suffix: str = ''):
    """
    모델별 상위 N개 특성 비교 그래프 생성
    
    Parameters:
    -----------
    all_importances : dict
        모델별 Feature Importance DataFrame 딕셔너리
    top_n : int
        상위 N개 특성
    save_dir : str
        저장 디렉토리
    """
    fig, axes = plt.subplots(len(all_importances), 1, figsize=(12, 5 * len(all_importances)))
    if len(all_importances) == 1:
        axes = [axes]
    
    for idx, (model_type, importance_df) in enumerate(all_importances.items()):
        top_features = importance_df.head(top_n)
        
        axes[idx].barh(range(len(top_features)), top_features['Importance'], 
                      xerr=top_features['Std'], capsize=3, alpha=0.7)
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['Feature'])
        axes[idx].set_xlabel('Importance', fontsize=12)
        axes[idx].set_title(f'{model_type.upper()} - Top {top_n} Features', fontsize=14, pad=10)
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    filename = f'feature_importance_comparison.png' if suffix == '' else f'permutation_importance_comparison.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장: {plot_path}")
    plt.close()


def _plot_combined_comparison(all_importances: Dict[str, pd.DataFrame], top_n: int, save_dir: str, suffix: str = ''):
    """
    모든 모델의 상위 N개 특성을 하나의 그래프로 비교 (누적 막대)
    
    Parameters:
    -----------
    all_importances : dict
        모델별 Feature Importance DataFrame 딕셔너리
    top_n : int
        상위 N개 특성
    save_dir : str
        저장 디렉토리
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 공통 상위 특성 찾기
    common_features = set()
    for importance_df in all_importances.values():
        common_features.update(importance_df.head(top_n)['Feature'].tolist())
    
    # 각 모델의 중요도 데이터 준비
    plot_data = []
    for model_type, importance_df in all_importances.items():
        for feature in common_features:
            if feature in importance_df['Feature'].values:
                row = importance_df[importance_df['Feature'] == feature].iloc[0]
                plot_data.append({
                    'Feature': feature,
                    'Model': model_type.upper(),
                    'Importance': row['Importance']
                })
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        pivot_df = plot_df.pivot(index='Feature', columns='Model', values='Importance').fillna(0)
        pivot_df = pivot_df.sort_values(by=pivot_df.columns[0], ascending=True).tail(top_n)
        
        pivot_df.plot(kind='barh', ax=ax, width=0.8, alpha=0.8)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance Comparison (Top {top_n} Features)', fontsize=14, pad=20)
        ax.legend(title='Model', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = f'feature_importance_combined.png' if suffix == '' else f'permutation_importance_combined.png'
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 저장: {plot_path}")
        plt.close()


def _find_and_save_common_features(all_importances: Dict[str, pd.DataFrame], top_n: int, save_dir: str, suffix: str = ''):
    """
    공통 중요 특성 찾기 및 저장
    
    Parameters:
    -----------
    all_importances : dict
        모델별 Feature Importance DataFrame 딕셔너리
    top_n : int
        상위 N개 특성
    save_dir : str
        저장 디렉토리
    """
    print(f"\n🔍 공통 중요 특성 분석...")
    
    # 각 모델의 상위 N개 특성 집합
    top_features_sets = {}
    for model_type, importance_df in all_importances.items():
        top_features = set(importance_df.head(top_n)['Feature'].values)
        top_features_sets[model_type] = top_features
    
    # 교집합 찾기
    common_features = set.intersection(*top_features_sets.values())
    
    if common_features:
        print(f"   모든 모델에서 상위 {top_n}개에 포함된 특성: {len(common_features)}개")
        
        # 공통 특성의 평균 중요도 계산
        common_importance = {}
        for feature in common_features:
            avg_importance = np.mean([
                all_importances[model_type][
                    all_importances[model_type]['Feature'] == feature
                ]['Importance'].values[0]
                for model_type in all_importances.keys()
                if feature in all_importances[model_type]['Feature'].values
            ])
            common_importance[feature] = avg_importance
        
        # 중요도 순으로 정렬
        sorted_common = sorted(common_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n   공통 중요 특성 (중요도 순):")
        for i, (feature, avg_imp) in enumerate(sorted_common, 1):
            print(f"     {i:2d}. {feature:40s}: 평균 중요도 {avg_imp:8.4f}")
        
        # CSV로 저장
        common_df = pd.DataFrame([
            {'Feature': feat, 'Average_Importance': imp} 
            for feat, imp in sorted_common
        ])
        filename = 'common_important_features.csv' if suffix == '' else 'common_important_features_permutation.csv'
        csv_path = os.path.join(save_dir, filename)
        common_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 저장: {csv_path}")
    else:
        print(f"   공통 중요 특성이 없습니다.")
