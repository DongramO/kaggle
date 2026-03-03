"""
Feature Importance 분석 및 시각화 모듈

주요 기능:
1. 모델별 Feature Importance 분석 (일반/permutation)
2. 앙상블 Feature Importance 분석
3. 모델별 vs 앙상블 비교 분석
4. 시각화 및 결과 저장
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 사용할 수 없습니다.")


# ============================================================================
# 유틸리티 함수 (내부 사용)
# ============================================================================

def _set_korean_font() -> None:
    """한글 폰트 설정 (시각화용)"""
    if not VISUALIZATION_AVAILABLE:
        return
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


def _get_model_feature_cols(model_type: str, X_train: pd.DataFrame, 
                            categorical_cols: List[str], encoded_cols: List[str]) -> List[str]:
    """
    모델별 사용 가능한 특성 컬럼 선택
    
    CatBoost: 인코딩된 컬럼 제외 (원본 범주형 사용)
    LightGBM/XGBoost: 범주형 컬럼 제외 (인코딩된 컬럼만 사용)
    """
    if model_type == 'catboost':
        return [col for col in X_train.columns if col not in encoded_cols]
    else:
        return [col for col in X_train.columns if col not in categorical_cols]


def _extract_model_importance(trainer, model_type: str, X_train: pd.DataFrame,
                              categorical_cols: List[str], encoded_cols: List[str],
                              use_permutation: bool = False, y_train: Optional[pd.Series] = None,
                              n_repeats: int = 10) -> Optional[pd.DataFrame]:
    """
    단일 모델의 Feature Importance 추출
    
    use_permutation=True: Permutation Importance (더 정확하지만 느림)
    use_permutation=False: 일반 Feature Importance (빠름)
    """
    feature_cols = _get_model_feature_cols(model_type, X_train, categorical_cols, encoded_cols)
    
    try:
        if use_permutation:
            if y_train is None:
                return None
            return trainer.get_permutation_importance(
                model_type=model_type,
                X=X_train[feature_cols],
                y=y_train,
                feature_names=feature_cols,
                n_repeats=n_repeats,
                random_state=trainer.random_state
            )
        else:
            return trainer.get_feature_importance(
                model_type=model_type,
                feature_names=feature_cols,
                average_across_folds=True
            )
    except Exception as e:
        print(f"   ⚠️ {model_type} Feature Importance 추출 실패: {e}")
        return None


def _extract_all_model_importances(trainer, X_train: pd.DataFrame, categorical_cols: List[str],
                                   encoded_cols_tag: str, use_permutation: bool = False,
                                   y_train: Optional[pd.Series] = None, n_repeats: int = 10) -> Dict[str, pd.DataFrame]:
    """
    모든 모델의 Feature Importance 추출 (CatBoost, LightGBM, XGBoost)
    
    Returns:
        {model_type: importance_df} 형태의 딕셔너리
    """
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    all_importances = {}
    
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        if model_type not in trainer.models:
            continue
        
        importance_df = _extract_model_importance(
            trainer, model_type, X_train, categorical_cols, encoded_cols,
            use_permutation, y_train, n_repeats
        )
        
        if importance_df is not None and len(importance_df) > 0:
            all_importances[model_type] = importance_df
    
    return all_importances


def _save_and_print_importance(importance_df: pd.DataFrame, model_type: str,
                               save_dir: str, top_n: int, suffix: str = '') -> None:
    """Importance 결과를 CSV로 저장하고 상위 N개를 콘솔에 출력"""
    filename = f'{model_type}_{suffix}_importance.csv' if suffix else f'{model_type}_feature_importance.csv'
    csv_path = os.path.join(save_dir, filename)
    importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 저장: {csv_path}")
    
    print(f"\n   상위 {top_n}개 특성:")
    for idx, row in importance_df.head(top_n).iterrows():
        std_str = f" (std: {row['Std']:.4f})" if 'Std' in row else ""
        print(f"     {idx+1:2d}. {row['Feature']:40s}: {row['Importance']:8.4f}{std_str}")


# ============================================================================
# Public API: Feature Importance 분석 함수들
# ============================================================================

def analyze_feature_importance(trainer, X_train: pd.DataFrame, 
                               categorical_cols: List[str], encoded_cols_tag: str = '_encoded',
                               top_n: int = 20, save_dir: str = 'feature_importance_results'):
    """
    모델별 Feature Importance 분석 및 시각화
    
    각 모델(CatBoost, LightGBM, XGBoost)의 Feature Importance를 추출하고,
    CSV 저장, 콘솔 출력, 시각화를 수행합니다.
    
    Parameters:
    -----------
    trainer : ModelTrainer
        학습된 모델을 포함한 ModelTrainer 객체
    X_train : pd.DataFrame
        학습 데이터
    categorical_cols : List[str]
        범주형 컬럼 리스트
    encoded_cols_tag : str
        인코딩된 컬럼 태그 (기본값: '_encoded')
    top_n : int
        상위 N개 특성만 표시 (기본값: 20)
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    dict or None: {model_type: importance_df} 형태의 딕셔너리
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 건너뜁니다.")
        return None
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 Feature Importance 분석")
    print(f"{'='*60}")
    
    all_importances = _extract_all_model_importances(
        trainer, X_train, categorical_cols, encoded_cols_tag, use_permutation=False
    )
    
    if len(all_importances) == 0:
        print("\n⚠️ 추출된 Feature Importance가 없습니다.")
        return None
    
    for model_type, importance_df in all_importances.items():
        print(f"\n🔍 {model_type.upper()} Feature Importance 추출 중...")
        _save_and_print_importance(importance_df, model_type, save_dir, top_n)
    
    if VISUALIZATION_AVAILABLE:
        print(f"\n📈 Feature Importance 시각화 중...")
        _plot_model_comparison(all_importances, top_n, save_dir)
        if len(all_importances) > 1:
            _plot_combined_comparison(all_importances, top_n, save_dir)
            _find_and_save_common_features(all_importances, top_n, save_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ Feature Importance 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return all_importances


def analyze_permutation_importance(trainer, X_train: pd.DataFrame, y_train: pd.Series,
                                   categorical_cols: List[str], encoded_cols_tag: str = '_encoded',
                                   top_n: int = 20, n_repeats: int = 10, 
                                   save_dir: str = 'feature_importance_results'):
    """
    Permutation Importance 분석 및 시각화
    
    Permutation Importance는 특성을 무작위로 섞어서 성능 저하를 측정하는 방법으로,
    일반 Feature Importance보다 더 정확하지만 계산 시간이 오래 걸립니다.
    
    Parameters:
    -----------
    trainer : ModelTrainer
        학습된 모델을 포함한 ModelTrainer 객체
    X_train : pd.DataFrame
        학습 데이터
    y_train : pd.Series
        타겟 데이터
    categorical_cols : List[str]
        범주형 컬럼 리스트
    encoded_cols_tag : str
        인코딩된 컬럼 태그
    top_n : int
        상위 N개 특성만 표시
    n_repeats : int
        Permutation 반복 횟수 (기본값: 10, 높을수록 정확하지만 느림)
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    dict or None: {model_type: importance_df} 형태의 딕셔너리
    """
    if not VISUALIZATION_AVAILABLE:
        print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 건너뜁니다.")
        return None
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 Permutation Importance 분석")
    print(f"{'='*60}")
    
    all_importances = _extract_all_model_importances(
        trainer, X_train, categorical_cols, encoded_cols_tag,
        use_permutation=True, y_train=y_train, n_repeats=n_repeats
    )
    
    if len(all_importances) == 0:
        print("\n⚠️ 추출된 Permutation Importance가 없습니다.")
        return None
    
    for model_type, importance_df in all_importances.items():
        print(f"\n🔍 {model_type.upper()} Permutation Importance 추출 중...")
        _save_and_print_importance(importance_df, model_type, save_dir, top_n, suffix='permutation')
    
    if VISUALIZATION_AVAILABLE:
        print(f"\n📈 Permutation Importance 시각화 중...")
        _plot_model_comparison(all_importances, top_n, save_dir, suffix='permutation')
        if len(all_importances) > 1:
            _plot_combined_comparison(all_importances, top_n, save_dir, suffix='permutation')
            _find_and_save_common_features(all_importances, top_n, save_dir, suffix='permutation')
    
    print(f"\n{'='*60}")
    print(f"✅ Permutation Importance 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return all_importances


def analyze_ensemble_feature_importance(trainer, ensemble, X_train: pd.DataFrame,
                                       categorical_cols: List[str], encoded_cols_tag: str = '_encoded',
                                       top_n: int = 20, save_dir: str = 'feature_importance_results'):
    """
    앙상블 Feature Importance 분석 및 시각화
    
    각 모델의 Feature Importance에 앙상블 가중치를 적용하여
    최종 앙상블 모델의 Feature Importance를 계산합니다.
    
    Parameters:
    -----------
    trainer : ModelTrainer
        학습된 모델을 포함한 ModelTrainer 객체
    ensemble : EnsembleModel
        학습된 앙상블 모델 객체 (weights 속성 필요)
    X_train : pd.DataFrame
        학습 데이터
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
    pd.DataFrame or None: 앙상블 Feature Importance DataFrame
    """
    if ensemble.weights is None:
        print("⚠️ 앙상블 모델이 학습되지 않았습니다.")
        return None
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 앙상블 Feature Importance 분석")
    print(f"{'='*60}")
    
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    model_importances = {}
    all_features = set()
    
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        if model_type not in trainer.models or model_type not in ensemble.weights:
            continue
        
        importance_df = _extract_model_importance(
            trainer, model_type, X_train, categorical_cols, encoded_cols, use_permutation=False
        )
        
        if importance_df is not None and len(importance_df) > 0:
            model_importances[model_type] = importance_df
            all_features.update(importance_df['Feature'].tolist())
    
    if len(model_importances) == 0:
        print("⚠️ 추출된 Feature Importance가 없습니다.")
        return None
    
    print(f"\n🔍 앙상블 가중치 적용 중...")
    print(f"   앙상블 가중치: {ensemble.weights}")
    
    ensemble_importance = {}
    for feature in all_features:
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_type, importance_df in model_importances.items():
            weight = ensemble.weights[model_type]
            feature_row = importance_df[importance_df['Feature'] == feature]
            
            if len(feature_row) > 0:
                weighted_sum += weight * feature_row['Importance'].values[0]
                total_weight += weight
        
        if total_weight > 0:
            ensemble_importance[feature] = weighted_sum / total_weight
    
    ensemble_df = pd.DataFrame([
        {'Feature': feat, 'Importance': imp}
        for feat, imp in ensemble_importance.items()
    ]).sort_values('Importance', ascending=False)
    
    if ensemble_df['Importance'].sum() > 0:
        ensemble_df['Importance'] = ensemble_df['Importance'] / ensemble_df['Importance'].sum()
    
    csv_path = os.path.join(save_dir, 'ensemble_feature_importance.csv')
    ensemble_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 저장: {csv_path}")
    
    print(f"\n   상위 {top_n}개 특성:")
    for idx, row in ensemble_df.head(top_n).iterrows():
        print(f"     {idx+1:2d}. {row['Feature']:40s}: {row['Importance']:8.4f}")
    
    if VISUALIZATION_AVAILABLE:
        print(f"\n📈 앙상블 Feature Importance 시각화 중...")
        _plot_ensemble_importance(ensemble_df, top_n, save_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ 앙상블 Feature Importance 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return ensemble_df


def compare_model_and_ensemble_importance(model_importances: Dict[str, pd.DataFrame],
                                          ensemble_importance: pd.DataFrame,
                                          top_n: int = 20,
                                          save_dir: str = 'feature_importance_results'):
    """
    모델별 Feature Importance와 앙상블 Feature Importance 비교 분석
    
    각 모델의 중요도와 앙상블 중요도를 한 테이블로 비교하여
    어떤 모델이 어떤 특성을 중요하게 보는지 확인할 수 있습니다.
    
    Parameters:
    -----------
    model_importances : Dict[str, pd.DataFrame]
        analyze_feature_importance()의 반환값
    ensemble_importance : pd.DataFrame
        analyze_ensemble_feature_importance()의 반환값
    top_n : int
        상위 N개 특성만 표시
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    pd.DataFrame: 비교 결과 DataFrame
    """
    if model_importances is None or len(model_importances) == 0:
        print("⚠️ 모델별 Feature Importance가 없습니다.")
        return None
    
    if ensemble_importance is None or len(ensemble_importance) == 0:
        print("⚠️ 앙상블 Feature Importance가 없습니다.")
        return None
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 모델별 vs 앙상블 Feature Importance 비교")
    print(f"{'='*60}")
    
    all_features = set(ensemble_importance['Feature'].tolist())
    for model_df in model_importances.values():
        all_features.update(model_df['Feature'].tolist())
    
    comparison_data = []
    for feature in all_features:
        row = {'Feature': feature}
        
        ensemble_row = ensemble_importance[ensemble_importance['Feature'] == feature]
        row['Ensemble'] = ensemble_row['Importance'].values[0] if len(ensemble_row) > 0 else 0.0
        
        for model_type, model_df in model_importances.items():
            model_row = model_df[model_df['Feature'] == feature]
            row[model_type.upper()] = model_row['Importance'].values[0] if len(model_row) > 0 else 0.0
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('Ensemble', ascending=False)
    
    csv_path = os.path.join(save_dir, 'model_vs_ensemble_comparison.csv')
    comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 저장: {csv_path}")
    
    print(f"\n   상위 {top_n}개 특성 비교:")
    print(f"   {'Feature':<40} {'Ensemble':>12} ", end='')
    for model_type in model_importances.keys():
        print(f"{model_type.upper():>12} ", end='')
    print()
    print("   " + "-" * (40 + 12 * (len(model_importances) + 1)))
    
    for idx, row in comparison_df.head(top_n).iterrows():
        print(f"   {row['Feature']:<40} {row['Ensemble']:>12.4f} ", end='')
        for model_type in model_importances.keys():
            print(f"{row[model_type.upper()]:>12.4f} ", end='')
        print()
    
    if VISUALIZATION_AVAILABLE:
        print(f"\n📈 비교 시각화 중...")
        _plot_model_vs_ensemble_comparison(comparison_df, model_importances, top_n, save_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ 비교 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return comparison_df


# ============================================================================
# 시각화 함수들 (내부 사용)
# ============================================================================

def _plot_model_comparison(all_importances: Dict[str, pd.DataFrame], top_n: int, save_dir: str, suffix: str = ''):
    """모델별 상위 N개 특성 비교 막대 그래프"""
    _set_korean_font()
    fig, axes = plt.subplots(len(all_importances), 1, figsize=(12, 5 * len(all_importances)))
    if len(all_importances) == 1:
        axes = [axes]
    
    for idx, (model_type, importance_df) in enumerate(all_importances.items()):
        top_features = importance_df.head(top_n)
        has_std = 'Std' in top_features.columns
        
        if has_std:
            axes[idx].barh(range(len(top_features)), top_features['Importance'], 
                          xerr=top_features['Std'], capsize=3, alpha=0.7)
        else:
            axes[idx].barh(range(len(top_features)), top_features['Importance'], alpha=0.7)
        
        axes[idx].set_yticks(range(len(top_features)))
        axes[idx].set_yticklabels(top_features['Feature'])
        axes[idx].set_xlabel('Importance', fontsize=12)
        axes[idx].set_title(f'{model_type.upper()} - Top {top_n} Features', fontsize=14, pad=10)
        axes[idx].grid(axis='x', alpha=0.3)
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    filename = 'feature_importance_comparison.png' if suffix == '' else 'permutation_importance_comparison.png'
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장: {plot_path}")
    plt.close()


def _plot_combined_comparison(all_importances: Dict[str, pd.DataFrame], top_n: int, save_dir: str, suffix: str = ''):
    """모든 모델의 상위 N개 특성을 하나의 그래프로 비교 (누적 막대)"""
    _set_korean_font()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    common_features = set()
    for importance_df in all_importances.values():
        common_features.update(importance_df.head(top_n)['Feature'].tolist())
    
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
        filename = 'feature_importance_combined.png' if suffix == '' else 'permutation_importance_combined.png'
        plot_path = os.path.join(save_dir, filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 저장: {plot_path}")
        plt.close()


def _find_and_save_common_features(all_importances: Dict[str, pd.DataFrame], top_n: int, save_dir: str, suffix: str = ''):
    """모든 모델에서 공통으로 중요하게 평가된 특성 찾기 및 저장"""
    print(f"\n🔍 공통 중요 특성 분석...")
    
    top_features_sets = {}
    for model_type, importance_df in all_importances.items():
        top_features_sets[model_type] = set(importance_df.head(top_n)['Feature'].values)
    
    common_features = set.intersection(*top_features_sets.values())
    
    if common_features:
        print(f"   모든 모델에서 상위 {top_n}개에 포함된 특성: {len(common_features)}개")
        
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
        
        sorted_common = sorted(common_importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n   공통 중요 특성 (중요도 순):")
        for i, (feature, avg_imp) in enumerate(sorted_common, 1):
            print(f"     {i:2d}. {feature:40s}: 평균 중요도 {avg_imp:8.4f}")
        
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


def _plot_ensemble_importance(ensemble_df: pd.DataFrame, top_n: int, save_dir: str):
    """앙상블 Feature Importance 막대 그래프"""
    _set_korean_font()
    top_features = ensemble_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    ax.barh(range(len(top_features)), top_features['Importance'], alpha=0.7, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Ensemble Feature Importance - Top {top_n} Features', fontsize=14, pad=10)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'ensemble_feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장: {plot_path}")
    plt.close()


def _plot_model_vs_ensemble_comparison(comparison_df: pd.DataFrame,
                                       model_importances: Dict[str, pd.DataFrame],
                                       top_n: int, save_dir: str):
    """모델별 vs 앙상블 비교 시각화 (막대 그래프 + 히트맵)"""
    _set_korean_font()
    top_features = comparison_df.head(top_n)
    
    # 막대 그래프
    fig, ax = plt.subplots(figsize=(14, max(10, top_n * 0.5)))
    
    x = np.arange(len(top_features))
    width = 0.8 / (len(model_importances) + 1)
    
    ax.barh(x, top_features['Ensemble'], width, label='Ensemble', alpha=0.8, color='steelblue')
    
    colors = ['coral', 'lightgreen', 'gold', 'plum', 'skyblue']
    offset = width
    for idx, (model_type, _) in enumerate(model_importances.items()):
        color = colors[idx % len(colors)]
        ax.barh(x + offset, top_features[model_type.upper()], width, 
                label=model_type.upper(), alpha=0.8, color=color)
        offset += width
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Model vs Ensemble Feature Importance Comparison (Top {top_n})', fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'model_vs_ensemble_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장: {plot_path}")
    plt.close()
    
    # 히트맵
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    
    plot_data = top_features[['Ensemble'] + [m.upper() for m in model_importances.keys()]].T
    plot_data.columns = top_features['Feature'].tolist()
    
    sns.heatmap(plot_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Importance'}, ax=ax, linewidths=0.5)
    ax.set_title(f'Feature Importance Heatmap (Top {top_n})', fontsize=14, pad=20)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'model_vs_ensemble_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 저장: {plot_path}")
    plt.close()
