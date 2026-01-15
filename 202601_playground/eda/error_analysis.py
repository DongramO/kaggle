"""
오차 분석 모듈 - 오차가 큰 샘플 식별 및 분석
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ matplotlib/seaborn이 설치되어 있지 않아 시각화를 사용할 수 없습니다.")


def find_high_error_samples(y_true: pd.Series, y_pred: np.ndarray, 
                            X_original: Optional[pd.DataFrame] = None,
                            top_n: int = 100, 
                            error_threshold: Optional[float] = None,
                            error_type: str = 'absolute') -> pd.DataFrame:
    """
    오차가 큰 샘플 식별
    
    Parameters:
    -----------
    y_true : pd.Series
        실제 타겟 값
    y_pred : np.ndarray
        예측 값
    X_original : pd.DataFrame, optional
        원본 특성 데이터 (저장용)
    top_n : int
        상위 N개 오차가 큰 샘플 (기본값: 100)
    error_threshold : float, optional
        오차 임계값 (이 값보다 큰 샘플만 선택)
    error_type : str
        오차 계산 방법 ('absolute', 'squared', 'percentage')
        
    Returns:
    --------
    pd.DataFrame
        오차가 큰 샘플 정보 (인덱스, 실제값, 예측값, 오차 등)
    """
    # 오차 계산
    if error_type == 'absolute':
        errors = np.abs(y_true.values - y_pred)
    elif error_type == 'squared':
        errors = (y_true.values - y_pred) ** 2
    elif error_type == 'percentage':
        errors = np.abs((y_true.values - y_pred) / (y_true.values + 1e-10)) * 100
    else:
        raise ValueError(f"Unknown error_type: {error_type}. Use 'absolute', 'squared', or 'percentage'")
    
    # 결과 DataFrame 생성
    error_df = pd.DataFrame({
        'index': y_true.index,
        'actual': y_true.values,
        'predicted': y_pred,
        'error': errors,
        'residual': y_true.values - y_pred
    })
    
    # 오차 임계값 필터링
    if error_threshold is not None:
        error_df = error_df[error_df['error'] >= error_threshold]
    
    # 상위 N개 선택
    error_df = error_df.nlargest(top_n, 'error').reset_index(drop=True)
    
    # 원본 데이터 추가 (있는 경우)
    if X_original is not None:
        # 인덱스로 조인
        error_df = error_df.merge(
            X_original.reset_index(),
            left_on='index',
            right_on='index',
            how='left'
        )
    
    return error_df


def analyze_high_error_samples(trainer, X_train: pd.DataFrame, y_train: pd.Series,
                               ensemble_pred: Optional[np.ndarray] = None,
                               top_n: int = 100,
                               error_threshold: Optional[float] = None,
                               save_dir: str = 'error_analysis_results'):
    """
    오차가 큰 샘플 분석 및 저장
    
    Parameters:
    -----------
    trainer : ModelTrainer
        학습된 모델을 포함한 ModelTrainer 객체
    X_train : pd.DataFrame
        학습 데이터 (원본 특성)
    y_train : pd.Series
        타겟 데이터
    ensemble_pred : np.ndarray, optional
        앙상블 OOF 예측값 (없으면 개별 모델별로 분석)
    top_n : int
        상위 N개 오차가 큰 샘플
    error_threshold : float, optional
        오차 임계값 (RMSE 단위)
    save_dir : str
        결과 저장 디렉토리
        
    Returns:
    --------
    dict
        모델별 오차가 큰 샘플 딕셔너리
    """
    # 저장 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("📊 오차가 큰 샘플 분석")
    print(f"{'='*60}")
    
    all_error_samples = {}
    
    # 앙상블 예측이 있으면 앙상블 기준으로 분석
    if ensemble_pred is not None:
        print(f"\n🔍 앙상블 모델 오차 분석...")
        
        # 앙상블 오차 계산
        ensemble_rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_train, ensemble_pred)
        
        print(f"   앙상블 RMSE: {ensemble_rmse:.4f}")
        print(f"   앙상블 MAE: {ensemble_mae:.4f}")
        
        # 오차 임계값 설정 (없으면 평균 + 2*표준편차)
        if error_threshold is None:
            errors = np.abs(y_train.values - ensemble_pred)
            error_threshold = np.mean(errors) + 2 * np.std(errors)
            print(f"   자동 설정된 오차 임계값: {error_threshold:.4f}")
        
        # 오차가 큰 샘플 찾기
        error_samples = find_high_error_samples(
            y_train, ensemble_pred, X_original=X_train,
            top_n=top_n, error_threshold=error_threshold
        )
        
        all_error_samples['ensemble'] = error_samples
        
        # 저장
        csv_path = os.path.join(save_dir, 'high_error_samples_ensemble.csv')
        error_samples.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 저장: {csv_path} ({len(error_samples)}개 샘플)")
        
        # 통계 정보 출력
        print(f"\n   상위 10개 오차가 큰 샘플:")
        for idx, row in error_samples.head(10).iterrows():
            print(f"     샘플 {row['index']:6d}: 실제={row['actual']:6.2f}, 예측={row['predicted']:6.2f}, 오차={row['error']:6.2f}")
    
    # 개별 모델별 분석
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        if model_type not in trainer.oof_predictions:
            continue
        
        print(f"\n🔍 {model_type.upper()} 모델 오차 분석...")
        
        oof_pred = trainer.oof_predictions[model_type]
        
        # 모델별 오차 계산
        model_rmse = np.sqrt(mean_squared_error(y_train, oof_pred))
        model_mae = mean_absolute_error(y_train, oof_pred)
        
        print(f"   {model_type} RMSE: {model_rmse:.4f}")
        print(f"   {model_type} MAE: {model_mae:.4f}")
        
        # 오차 임계값 설정
        if error_threshold is None:
            errors = np.abs(y_train.values - oof_pred)
            threshold = np.mean(errors) + 2 * np.std(errors)
        else:
            threshold = error_threshold
        
        # 오차가 큰 샘플 찾기
        error_samples = find_high_error_samples(
            y_train, oof_pred, X_original=X_train,
            top_n=top_n, error_threshold=threshold
        )
        
        all_error_samples[model_type] = error_samples
        
        # 저장
        csv_path = os.path.join(save_dir, f'high_error_samples_{model_type}.csv')
        error_samples.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"   ✅ 저장: {csv_path} ({len(error_samples)}개 샘플)")
    
    # 시각화
    if VISUALIZATION_AVAILABLE and len(all_error_samples) > 0:
        print(f"\n📈 오차 분석 시각화 중...")
        _visualize_error_analysis(all_error_samples, y_train, save_dir)
    
    # 공통 오차가 큰 샘플 찾기
    if len(all_error_samples) > 1:
        _find_common_high_error_samples(all_error_samples, save_dir)
    
    print(f"\n{'='*60}")
    print(f"✅ 오차 분석 완료!")
    print(f"   결과 저장 위치: {save_dir}/")
    print(f"{'='*60}")
    
    return all_error_samples


def _visualize_error_analysis(all_error_samples: Dict[str, pd.DataFrame], 
                              y_train: pd.Series, save_dir: str):
    """오차 분석 시각화"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 오차 분포 히스토그램
        ax = axes[0, 0]
        for model_type, error_df in all_error_samples.items():
            ax.hist(error_df['error'], bins=30, alpha=0.5, label=model_type, density=True)
        ax.set_xlabel('Error', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Error Distribution (High Error Samples)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. 실제값 vs 예측값 산점도 (앙상블만)
        if 'ensemble' in all_error_samples:
            ax = axes[0, 1]
            error_df = all_error_samples['ensemble']
            ax.scatter(error_df['actual'], error_df['predicted'], 
                      alpha=0.5, s=20, c=error_df['error'], cmap='Reds')
            ax.plot([y_train.min(), y_train.max()], 
                   [y_train.min(), y_train.max()], 'r--', lw=2)
            ax.set_xlabel('Actual', fontsize=12)
            ax.set_ylabel('Predicted', fontsize=12)
            ax.set_title('Actual vs Predicted (High Error Samples)', fontsize=14)
            ax.grid(alpha=0.3)
        
        # 3. 잔차 플롯
        if 'ensemble' in all_error_samples:
            ax = axes[1, 0]
            error_df = all_error_samples['ensemble']
            ax.scatter(error_df['predicted'], error_df['residual'], 
                      alpha=0.5, s=20)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Residual', fontsize=12)
            ax.set_title('Residual Plot (High Error Samples)', fontsize=14)
            ax.grid(alpha=0.3)
        
        # 4. 오차 통계 비교
        ax = axes[1, 1]
        model_names = []
        error_means = []
        error_stds = []
        for model_type, error_df in all_error_samples.items():
            model_names.append(model_type)
            error_means.append(error_df['error'].mean())
            error_stds.append(error_df['error'].std())
        
        x_pos = np.arange(len(model_names))
        ax.bar(x_pos, error_means, yerr=error_stds, alpha=0.7, capsize=5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names)
        ax.set_ylabel('Mean Error', fontsize=12)
        ax.set_title('Error Statistics by Model', fontsize=14)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'error_analysis_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ 저장: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"   ⚠️ 시각화 실패: {e}")


def _find_common_high_error_samples(all_error_samples: Dict[str, pd.DataFrame], save_dir: str):
    """모든 모델에서 공통으로 오차가 큰 샘플 찾기"""
    print(f"\n🔍 공통 오차가 큰 샘플 분석...")
    
    # 각 모델의 상위 오차 샘플 인덱스
    error_indices_sets = {}
    for model_type, error_df in all_error_samples.items():
        # 상위 50개만 고려
        top_indices = set(error_df.head(50)['index'].values)
        error_indices_sets[model_type] = top_indices
    
    # 교집합 찾기
    if len(error_indices_sets) > 1:
        common_indices = set.intersection(*error_indices_sets.values())
        
        if len(common_indices) > 0:
            print(f"   모든 모델에서 공통으로 오차가 큰 샘플: {len(common_indices)}개")
            
            # 공통 샘플 정보 수집
            common_samples = []
            for idx in common_indices:
                sample_info = {'index': idx}
                for model_type, error_df in all_error_samples.items():
                    row = error_df[error_df['index'] == idx]
                    if len(row) > 0:
                        sample_info[f'{model_type}_error'] = row.iloc[0]['error']
                        sample_info[f'{model_type}_actual'] = row.iloc[0]['actual']
                        sample_info[f'{model_type}_predicted'] = row.iloc[0]['predicted']
                common_samples.append(sample_info)
            
            common_df = pd.DataFrame(common_samples)
            csv_path = os.path.join(save_dir, 'common_high_error_samples.csv')
            common_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"   ✅ 저장: {csv_path}")
        else:
            print(f"   공통 오차가 큰 샘플이 없습니다.")
    else:
        print(f"   모델이 1개뿐이어서 공통 샘플 분석을 건너뜁니다.")
