"""
메인 실행 파일
모델 학습 및 앙상블을 실행합니다.
"""
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from modeling.train import main

if __name__ == "__main__":
    print("="*60)
    print("🎯 Kaggle Competition - 모델 학습 및 앙상블")
    print("="*60)
    
    # ========== 설정 옵션 ==========
    # GPU 사용 여부 설정
    USE_GPU = True  # GPU를 사용하려면 True로 변경
    USE_OPTUNA = False  # True로 설정하면 Optuna 최적화 실행 (시간이 오래 걸림)
    N_TRIALS = 50  # Optuna 시도 횟수 (USE_OPTUNA=True일 때만 사용)
    OPTUNA_SAMPLE_SIZE = None  # 예: 50000 (5만 개만 사용)
    USE_SAVED_PARAMS = None  # None으로 설정하면 자동 감지
    USE_PERMUTATION_IMPORTANCE = False  # True로 설정하면 Permutation Importance 분석 실행 (시간이 오래 걸림)
    
    ENCODING_CONFIG = {
        'onehot_cols': ['gender', 'course', 'internet_access', 'study_method'],
        'ordinal_cols': ['facility_rating', 'sleep_quality', 'exam_difficulty'],
        'onehot_params': {
            'handle_unknown': 'ignore',
            'drop': None  # 첫 번째 카테고리 제거 (다중공선성 방지)
        },
        'ordinal_params': {'handle_unknown': 'use_encoded_value', 'unknown_value': -1},
        'drop_original': True
    }

    PARAMS_FILEPATH = os.path.join(project_root, 'best_hyperparameters.json')
    
    # ========== 실행 ==========
    results, ensemble_pred, submission = main(
        use_optuna=USE_OPTUNA,
        n_trials=N_TRIALS,
        use_saved_params=USE_SAVED_PARAMS,
        params_filepath=PARAMS_FILEPATH,
        use_gpu=USE_GPU,
        optuna_sample_size=OPTUNA_SAMPLE_SIZE,
        encoding_config=ENCODING_CONFIG,
        use_permutation_importance=USE_PERMUTATION_IMPORTANCE
    )
    
    print("\n" + "="*60)
    print("✅ 모든 작업이 완료되었습니다!")
    print("📤 제출 파일이 프로젝트 루트에 저장되었습니다.")
    print("="*60)

