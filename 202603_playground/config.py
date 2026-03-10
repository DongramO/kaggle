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
# ========== 모델 학습 설정 ==========
TASK_TYPE = 'classification'  # 'regression' | 'classification'
SCORING_METRIC = 'auc'  # 문제별 평가·제출 형식: 회귀=rmse|mae|r2, 분류=auc|logloss|accuracy
N_FOLDS = 5  # 본 학습(K-Fold)용 Fold 수
RANDOM_STATE = 42
USE_GPU = True
# 트리 모델(CatBoost, LightGBM, XGBoost) early stopping: validation 성능이 N round 개선 없으면 학습 중단
EARLY_STOPPING_ROUNDS = 100

# ========== 모델 설정 ==========
# 'mlp' 추가 시 MLP(딥러닝)로 특성 간 비선형 관계 보완, 트리 모델과 앙상블
MODEL_TYPES = ['catboost', 'lightgbm', 'xgboost']  # 사용하려는 모델 리스트

# ========== 모델별 Feature Engineering 사용 설정 ==========
USE_FEATURE_ENGINEERING_FOR_MODELS = {
    'catboost': True,
    'lightgbm': True,
    'xgboost': True,
    'mlp': True   # MLP도 수치형(인코딩) 특성만 사용
}

# ========== 앙상블 설정 ==========
ENSEMBLE_METHOD = 'ridge_meta'  # 'weighted_average' | 'simple_average' | 'ridge_meta'
RIDGE_ALPHA = 1.0  # ridge_meta 사용 시 Ridge 정규화 강도 (Optuna 미사용 시 적용)
OPTUNA_RIDGE_ALPHA = True  # True면 Ridge alpha를 Optuna로 최적화
RIDGE_ALPHA_N_TRIALS = 20  # Ridge alpha Optuna 시도 횟수

# ========== 하이퍼파라미터 최적화 설정 ==========
USE_OPTUNA = False  # True로 설정하면 Optuna 최적화 실행 (시간이 오래 걸림)
N_TRIALS = 50  # Optuna 시도 횟수
OPTUNA_N_FOLDS = 3  # Optuna 전용 Fold 수 (기본은 본 학습과 동일, 필요 시 N_FOLDS와 다르게 설정)
# Optuna 시 샘플 수. None이면 전량 사용(느림). 5만~10만 권장: trial 속도와 대표성 균형
OPTUNA_SAMPLE_SIZE = 100000  # 예: None(전량), 50000, 80000, 100000
USE_SAVED_PARAMS = True  # None으로 설정하면 자동 감지
USE_PERMUTATION_IMPORTANCE = False  # True로 설정하면 Permutation Importance 분석 실행
# MLP로 '분석 초점' 수치화 (예측용 아님): 특성 중요도·특성 쌍 상호작용·트리와 비교해 어디에 초점 맞출지 제안
USE_MLP_FOCUS_ANALYSIS = False
# True면 모델 학습/앙상블/제출 건너뛰고 전처리 후 MLP 분석만 실행 (트리 비교 없음)
USE_MLP_FOCUS_ONLY = False

# ========== 파일 경로 ==========
# Optuna 저장 파일: 이 playground 폴더 아래 best_hyperparameters.json
PARAMS_FILEPATH = os.path.join(PROJECT_ROOT, 'best_hyperparameters.json')
SUBMISSION_FILEPATH = os.path.join(PROJECT_ROOT, 'submission.csv')
SUMMARY_FILEPATH = os.path.join(PROJECT_ROOT, 'training_summary.txt')


# ========== 인코딩 설정 ==========
ENCODING_CONFIG = {
    'onehot_cols': [
        'gender', 'PhoneService',
        'Dependents', 'PaperlessBilling',
        'PaymentMethod', 
        'Partner', 'Internet'
    ],
    'ordinal_cols': [

    ],
    'onehot_params': {
        'handle_unknown': 'ignore',
        'drop': 'first'  # 첫 번째 카테고리 제거 (다중공선성 방지)
    },
    'ordinal_params': {
        'handle_unknown': 'use_encoded_value', 
        'unknown_value': -1,
        'category_orders': {
            # 'Contract': ['Month-to-month', 'One year', 'Two year'],
        }
    },
    'drop_original': False
}

# ========== Feature Engineering 설정 (대회별 컬럼에 맞게 수정) ==========
FEATURE_ENGINEERING_CONFIG = {
    'target_encoding': {
        'flag': False,  # True로 두면 PaymentMethod, InternetService 등에 K-Fold Smooth TE 적용
        # 타겟 인코딩을 적용할 컬럼 (범주형 위주 권장)
        'cols': [
            'PaymentMethod',
            'InternetService',
        ],
        'n_splits': N_FOLDS,
        'smoothing': 10.0,
    },
    'clip_outliers': {
        'flag': True,
        'clip_rules': {
            'tenure': 0.99,
            'MonthlyCharges': 0.99,
            'TotalCharges': 0.99,
        }
    },
    'change_to_numeric': [
        'Contract',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 
    ],
    'convert_ordered_categorical_to_numeric': {
        'flag': True,
        # Partner_numeric: Cat/LGB/XGB 모두 importance 극소(0.0002~0.0005) → 제거
        'mappings': {
            'Contract': {'Month-to-month': 1, 'One year': 12, 'Two year': 24},
            'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 0},    
            'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 0},
            'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 0},
        }
    },
    'create_interactions_before_encoding': {
        'flag': True,
        'feature_pairs': [
            # ('MonthlyCharges', 'MonthlyCharges_mean'),
            # ('TotalCharges', 'TotalCharges_mean'),
            ('PhoneService_Yes_encoded', 'MultipleLines_numeric'),
            ('Dependents_Yes_encoded', 'MonthlyCharges'),
        ],
        'operations': [
            # 'subtract', 'subtract',
            'multiply', 'multiply',
        ]
    },
    'create_interactions_after_encoding': {
        'flag': True,
        # Contract_numeric 활용: 계약×tenure, tenure-계약, MonthlyCharges×계약
        'feature_pairs': [
            ('MonthlyCharges', 'Contract_numeric'),
            ('has_internet', 'OnlineSecurity_numeric'),
            ('has_internet', 'OnlineBackup_numeric'),
            ('has_internet', 'DeviceProtection_numeric'),
            ('has_internet', 'TechSupport_numeric'),
            ('has_internet', 'StreamingTV_numeric'),
            ('has_internet', 'StreamingMovies_numeric'),
        ],
        'operations': [
            'multiply',
            'multiply', 'multiply','multiply',
            'multiply', 'multiply', 'multiply',
        ]
    },
    'create_ratios': {
        'flag': True,
        # tenure/Contract_numeric(계약 대비 가입 기간), TotalCharges/(MonthlyCharges*Contract_numeric)
        'numerator_cols': ['tenure', 'TotalCharges'],
        'denominator_cols': ['Contract_numeric', 'MonthlyCharges_x_Contract_numeric_FE'],
        'ratio_feature_names': '_ratio'
    },
    # tenure, MonthlyCharges, TotalCharges 세 수치형 간 사칙연산(쌍별 add/mul/sub/div 양방향) 신규 특성
    'create_tenure_monthly_total_arithmetic': {
        'flag': False,
        'feature_pairs': [
            ('tenure', 'MonthlyCharges'), ('tenure', 'MonthlyCharges'), ('tenure', 'MonthlyCharges'), ('tenure', 'MonthlyCharges'),
            ('MonthlyCharges', 'tenure'), ('MonthlyCharges', 'tenure'),
            ('tenure', 'TotalCharges'), ('tenure', 'TotalCharges'), ('tenure', 'TotalCharges'), ('tenure', 'TotalCharges'),
            ('TotalCharges', 'tenure'), ('TotalCharges', 'tenure'),
            ('MonthlyCharges', 'TotalCharges'), ('MonthlyCharges', 'TotalCharges'), ('MonthlyCharges', 'TotalCharges'), ('MonthlyCharges', 'TotalCharges'),
            ('TotalCharges', 'MonthlyCharges'), ('TotalCharges', 'MonthlyCharges'),
        ],
        'operations': [
            'add', 'multiply', 'subtract', 'divide',
            'subtract', 'divide',
            'add', 'multiply', 'subtract', 'divide',
            'subtract', 'divide',
            'add', 'multiply', 'subtract', 'divide',
            'subtract', 'divide',
        ],
    },
    # tenure_years(가입 년수), tenure_segment(0,12,24,48,72 구간) — main.py에서 직접 생성
    'create_tenure_derived': {
        'flag': True,
        'tenure_segment_bins': [0, 12, 24, 48, 72],  # 구간 우측 경계. 72 초과는 마지막 구간(3)으로 처리
    },
    # 'create_categorical_interactions': {
    #     'flag': False,
    #     'categorical_pairs': [],
    #     'separator': '_'
    # },
    'transform_numeric_features': {
        'flag': False,
        'columns': ['tenure'],
        'transformations': ['log'],
        'separator': '_'
    },
    'create_statistical_features': {
        'flag': False,
        'feature_groups': [
            ['tenure'],
            ['MonthlyCharges'],
            ['TotalCharges'],
        ],
        'statistics': ['mean','mean','mean'],
    }
}