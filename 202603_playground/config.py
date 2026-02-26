"""
프로젝트별 설정 파일
각 playground 프로젝트마다 이 파일을 수정하여 사용
"""
import os

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== 데이터 설정 ==========
TARGET_COL = 'exam_score'  # 타겟 컬럼 이름 (프로젝트별로 변경)
ID_COL = 'id'  # ID 컬럼 이름

# ========== 인코딩 설정 ==========
ENCODING_CONFIG = {
    'onehot_cols': ['gender', 'course', 'internet_access', 'study_method'],
    'ordinal_cols': ['facility_rating', 'sleep_quality', 'exam_difficulty'],
    'onehot_params': {
        'handle_unknown': 'ignore',
        'drop': None  # 첫 번째 카테고리 제거 (다중공선성 방지)
    },
    'ordinal_params': {
        'handle_unknown': 'use_encoded_value', 
        'unknown_value': -1,
        'category_orders': {
            'sleep_quality': ['poor', 'average', 'good'],
            'facility_rating': ['low', 'medium', 'high'],
            'exam_difficulty': ['easy', 'moderate', 'hard']
        }
    },
    'drop_original': True
}

# ========== Feature Engineering 설정 ==========
FEATURE_ENGINEERING_CONFIG = {
    'clip_outliers': {
        'flag': True,
        'clip_rules': {
            'study_hours': 0.99,
            'class_attendance': 0.99,
            'sleep_hours': 0.99,
            'age': 0.99,
        }
    },
    'create_interactions_before_encoding': {
        'flag': True,
        'feature_pairs': [
            ('class_attendance', 'study_hours'),
            ('study_hours', 'sleep_hours'),
        ],
        'operations': ['multiply', 'add']
    },
    'create_interactions_after_encoding': {
        'flag': True,
        'feature_pairs': [
            ('class_attendance', 'sleep_quality_encoded'),
            ('study_hours', 'sleep_quality_encoded'),
        ],
        'operations': ['multiply', 'multiply']
    },
    'create_ratios': {
        'flag': True,
        'numerator_cols': ['study_hours', 'class_attendance'],
        'denominator_cols': ['study_hours_add_sleep_hours', 'study_hours'],
        'ratio_feature_names': "_ratio"
    },
    'create_categorical_interactions': {
        'flag': True,
        'categorical_pairs': [
            ('facility_rating', 'exam_difficulty'),
            ('sleep_quality', 'exam_difficulty'),
        ],
        'separator': '_'
    },
    'create_statistical_features': {
        'flag': True,
        'feature_groups': [
            ['study_hours', 'class_attendance', 'sleep_hours'],
            ['age', 'study_hours', 'class_attendance'],
        ],
        'statistics': ['mean'],
    }
}

# ========== 모델 학습 설정 ==========
TASK_TYPE = 'regression'  # 'regression' or 'classification'
N_FOLDS = 5
RANDOM_STATE = 42
USE_GPU = True

# ========== 하이퍼파라미터 최적화 설정 ==========
USE_OPTUNA = False  # True로 설정하면 Optuna 최적화 실행 (시간이 오래 걸림)
N_TRIALS = 50  # Optuna 시도 횟수
OPTUNA_SAMPLE_SIZE = None  # 예: 50000 (5만 개만 사용)
USE_SAVED_PARAMS = None  # None으로 설정하면 자동 감지
USE_PERMUTATION_IMPORTANCE = False  # True로 설정하면 Permutation Importance 분석 실행

# ========== 파일 경로 ==========
PARAMS_FILEPATH = os.path.join(PROJECT_ROOT, 'best_hyperparameters.json')
SUBMISSION_FILEPATH = os.path.join(PROJECT_ROOT, 'submission.csv')
SUMMARY_FILEPATH = os.path.join(PROJECT_ROOT, 'training_summary.txt')
