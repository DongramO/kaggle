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
N_FOLDS = 5
RANDOM_STATE = 42
USE_GPU = True

# ========== 모델 설정 ==========
MODEL_TYPES = ['catboost','lightgbm','xgboost']  # 사용할 모델 리스트: 'catboost', 'lightgbm', 'xgboost' 중 선택

# ========== 모델별 Feature Engineering 사용 설정 ==========
USE_FEATURE_ENGINEERING_FOR_MODELS = {
    'catboost': True,  # CatBoost는 feature engineering 특성 사용 안 함
    'lightgbm': True,   # LightGBM은 feature engineering 특성 사용
    'xgboost': True     # XGBoost는 feature engineering 특성 사용
}

# ========== 앙상블 설정 ==========
ENSEMBLE_METHOD = 'weighted_average'  # 'weighted_average' | 'simple_average' | 'ridge_meta'
RIDGE_ALPHA = 1.0  # ridge_meta 사용 시 Ridge 정규화 강도 (Optuna 미사용 시 적용)
OPTUNA_RIDGE_ALPHA = True  # True면 Ridge alpha를 Optuna로 최적화
RIDGE_ALPHA_N_TRIALS = 20  # Ridge alpha Optuna 시도 횟수

# ========== 하이퍼파라미터 최적화 설정 ==========
USE_OPTUNA = False  # True로 설정하면 Optuna 최적화 실행 (시간이 오래 걸림)
N_TRIALS = 50  # Optuna 시도 횟수
OPTUNA_SAMPLE_SIZE = None  # 예: 50000 (5만 개만 사용)
USE_SAVED_PARAMS = True  # None으로 설정하면 자동 감지
USE_PERMUTATION_IMPORTANCE = False  # True로 설정하면 Permutation Importance 분석 실행

# ========== 파일 경로 ==========
# Optuna 저장 파일: 이 playground 폴더 아래 best_hyperparameters.json
PARAMS_FILEPATH = os.path.join(PROJECT_ROOT, 'best_hyperparameters.json')
SUBMISSION_FILEPATH = os.path.join(PROJECT_ROOT, 'submission.csv')
SUMMARY_FILEPATH = os.path.join(PROJECT_ROOT, 'training_summary.txt')


# ========== 인코딩 설정 ==========
ENCODING_CONFIG = {
    'onehot_cols': ['gender', 'InternetService', 'PaymentMethod'],
    'ordinal_cols': ['PhoneService', 'PaperlessBilling', 'Contract'],
    'onehot_params': {
        'handle_unknown': 'ignore',
        'drop': 'first'  # 첫 번째 카테고리 제거 (다중공선성 방지)
    },
    'ordinal_params': {
        'handle_unknown': 'use_encoded_value', 
        'unknown_value': -1,
        'category_orders': {
            '': ['No', 'Yes'],
            '': ['No', 'Yes'],
            '': ['Month-to-month', 'One Year', 'Two Year'],
        }
    },
    'drop_original': False
}

# ========== Feature Engineering 설정 (대회별 컬럼에 맞게 수정) ==========
FEATURE_ENGINEERING_CONFIG = {
    'clip_outliers': {
        'flag': True,
        'clip_rules': {
            'tenure': 0.99,
            'MonthlyCharges': 0.99,
            'TotalCharges': 0.99,
        }
    },
    'change_to_numeric': [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ],
    'convert_ordered_categorical_to_numeric': {
        'flag': True,
        'mappings': {
            'Partner': {
                'No': 0,
                'Yes': 1,
            },
            'Dependents': {
                'No': 0,
                'Yes': 1,
            },
            'Contract': {
                'Month-to-month': 1,
                'One year': 12,
                'Two year': 24
            },
            'MultipleLines': {
                'No': 0,
                'No phone service': 0,
                'Yes': 1,
            },
            'OnlineSecurity': {
                'No': 0,
                'No internet service': 0,
                'Yes': 1,
            },
            'OnlineBackup': {
                'No': 0,
                'No internet service': 0,
                'Yes': 1,
            },
            'DeviceProtection': {
                'No': 0,
                'No internet service': 0,
                'Yes': 1,
            },
            'TechSupport': {
                'No': 0,
                'No internet service': 0,
                'Yes': 1,
            },
            'StreamingTV': {
                'No': 0,
                'No internet service': 0,
                'Yes': 1,
            },
            'StreamingMovies': {
                'No': 0,
                'No internet service': 0,
                'Yes': 1,
            }
        }
    },
    'create_interactions_before_encoding': {
        'flag': True,
        'feature_pairs': [
            ('MonthlyCharges', 'MonthlyCharges_mean'),
            ('TotalCharges', 'TotalCharges_mean'),
            ('PhoneService', 'MultipleLines_numeric'),
            ('InternetService', 'OnlineSecurity_numeric'),
            ('InternetService', 'OnlineBackup_numeric'),
            ('InternetService', 'DeviceProtection_numeric'),
            ('InternetService', 'TechSupport_numeric'),
            ('InternetService', 'StreamingTV_numeric'),
            ('InternetService', 'StreamingMovies_numeric'),
            ('SeniorCitizen', 'MonthlyCharges'),
            ('Dependents_numeric', 'MonthlyCharges'),
        ],
        'operations': [
            'subtract', 'subtract', 'multiply', 'multiply', 'multiply', 
            'multiply', 'multiply', 'multiply', 'multiply', 'multiply', 'multiply'
        ]
    },
    'create_interactions_after_encoding': {
        'flag': True,
        'feature_pairs': [
            ('InternetService_Fiber optic_encoded', 'MonthlyCharges'),
            ('InternetService_No_encoded', 'MonthlyCharges'),
        ],
        'operations': ['multiply', 'multiply']
    },
    'create_ratios': {
        'flag': True,
        'numerator_cols': ['TotalCharges', 'tenure' , 'MonthlyCharges'],
        'denominator_cols': ['tenure', 'Contract_numeric', 'TotalCharges'],
        'ratio_feature_names': '_ratio'
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
        'flag': True,
        'feature_groups': [
            ['tenure', 'MonthlyCharges', 'TotalCharges'],
        ],
        'statistics': ['mean'],
    }
}