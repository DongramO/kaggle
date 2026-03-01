"""
프로젝트별 설정 파일
각 playground 프로젝트마다 이 파일을 수정하여 사용
"""
import os

# 프로젝트 루트 경로
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ========== 데이터 설정 ==========
ID_COL = 'id'  # ID 컬럼 이름
TARGET_COL = 'Churn'  # 타겟 컬럼 (대회별 변경)

# ========== 인코딩 설정 ==========
ENCODING_CONFIG = {
    'onehot_cols': ['gender', 'InternetService', 'PaymentMethod', 'Contract'],
    'ordinal_cols': ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling'],
    'onehot_params': {
        'handle_unknown': 'ignore',
        'drop': None  # 첫 번째 카테고리 제거 (다중공선성 방지)
    },
    'ordinal_params': {
        'handle_unknown': 'use_encoded_value', 
        'unknown_value': -1,
        'category_orders': {
            '': ['No', 'Yes'],
            '': ['No', 'Yes'],
            '': ['No', 'Yes'],
            '': ['No', 'No phone service', 'Yes'],
            '': ['No', 'No Internet service', 'Yes'],
            '': ['No', 'No Internet service', 'Yes'],
            '': ['No', 'No Internet service', 'Yes'],
            '': ['No', 'No Internet service', 'Yes'],
            '': ['No', 'No Internet service', 'Yes'],
            '': ['No', 'No Internet service', 'Yes'],
            '': ['No', 'Yes'],
        }
    },
    'drop_original': True
}

# ========== Feature Engineering 설정 (대회별 컬럼에 맞게 수정) ==========
FEATURE_ENGINEERING_CONFIG = {
    'change_to_numeric': [
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ],
    'clip_outliers': {
        'flag': True,
        'clip_rules': {
            'tenure': 0.99,
            'MonthlyCharges': 0.99,
            'TotalCharges': 0.99,
        }
    },
    # 'create_interactions_before_encoding': {
    #     'flag': True,
    #     'feature_pairs': [
    #         ('tenure', 'MonthlyCharges'),
    #         ('MonthlyCharges', 'TotalCharges'),
    #     ],
    #     'operations': ['multiply', 'add']
    # },
    # 'create_interactions_after_encoding': {
    #     'flag': True,
    #     'feature_pairs': [],
    #     'operations': []
    # },
    # 'create_ratios': {
    #     'flag': False,
    #     'numerator_cols': [],
    #     'denominator_cols': [],
    #     'ratio_feature_names': '_ratio'
    # },
    # 'create_categorical_interactions': {
    #     'flag': False,
    #     'categorical_pairs': [],
    #     'separator': '_'
    # },
    # 'create_statistical_features': {
    #     'flag': True,
    #     'feature_groups': [
    #         ['tenure', 'MonthlyCharges', 'TotalCharges'],
    #     ],
    #     'statistics': ['mean'],
    # }
}

# ========== 모델 학습 설정 ==========
TASK_TYPE = 'classification'  # 'regression' | 'classification'
SCORING_METRIC = 'auc'  # 문제별 평가·제출 형식: 회귀=rmse|mae|r2, 분류=auc|logloss|accuracy
N_FOLDS = 1
RANDOM_STATE = 42
USE_GPU = True

# ========== 하이퍼파라미터 최적화 설정 ==========
USE_OPTUNA = False  # True로 설정하면 Optuna 최적화 실행 (시간이 오래 걸림)
N_TRIALS = 50  # Optuna 시도 횟수
OPTUNA_SAMPLE_SIZE = None  # 예: 50000 (5만 개만 사용)
USE_SAVED_PARAMS = None  # None으로 설정하면 자동 감지
USE_PERMUTATION_IMPORTANCE = False  # True로 설정하면 Permutation Importance 분석 실행

# ========== 파일 경로 ==========
# Optuna 저장 파일: 이 playground 폴더 아래 best_hyperparameters.json
PARAMS_FILEPATH = os.path.join(PROJECT_ROOT, 'best_hyperparameters.json')
SUBMISSION_FILEPATH = os.path.join(PROJECT_ROOT, 'submission.csv')
SUMMARY_FILEPATH = os.path.join(PROJECT_ROOT, 'training_summary.txt')
