"""
하이퍼파라미터 최적화 및 관리 모듈
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# 모델 라이브러리
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

# 평가 및 검증
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score

# 하이퍼파라미터 최적화
try:
    import optuna
    from optuna import Trial
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna가 설치되어 있지 않습니다. 하이퍼파라미터 최적화를 사용할 수 없습니다.")

# GPU 관련 유틸리티 함수
def get_gpu_device():
    """GPU 디바이스 번호 반환"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if result.returncode == 0:
            # 첫 번째 GPU 사용 (0번)
            return 0
    except:
        pass
    return None


class HyperparameterOptimizer:
    """Optuna를 사용한 하이퍼파라미터 최적화 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, use_gpu: bool = False):
        """
        Parameters:
        -----------
        task_type : str
            작업 타입 ('regression' or 'classification')
        random_state : int
            랜덤 시드
        use_gpu : bool
            GPU 사용 여부
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna가 설치되어 있지 않습니다. 'pip install optuna'로 설치해주세요.")
        
        self.task_type = task_type
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.best_params = {}
        
    def optimize_catboost(self, X_train, y_train, n_trials: int = 50, 
                         cat_features: Optional[List] = None, **kwargs):
        """
        CatBoost 하이퍼파라미터 최적화
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            학습 데이터
        y_train : pd.Series or np.ndarray
            타겟 데이터
        n_trials : int
            Optuna 시도 횟수
        cat_features : List, optional
            범주형 특성 리스트
        **kwargs
            추가 파라미터
            
        Returns:
        --------
        dict
            최적화된 하이퍼파라미터
        """
        def objective(trial):
            # bootstrap_type을 먼저 선택
            bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
            
            if self.task_type == 'regression':
                params = {
                    # CatBoost 최적화: 더 많은 iterations와 낮은 learning_rate 범위
                    'iterations': trial.suggest_int('iterations', 500, 2000),  # 범위 증가: 500-2000
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # 0.3 -> 0.15로 제한 (CatBoost 권장 범위)
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'bootstrap_type': bootstrap_type,
                    'random_state': self.random_state,
                    'verbose': False,
                    'allow_writing_files': False,
                }
                # bagging_temperature는 bootstrap_type='Bayesian'일 때만 사용 가능
                if bootstrap_type == 'Bayesian':
                    params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
                # GPU 설정 추가
                if self.use_gpu:
                    gpu_device = get_gpu_device()
                    if gpu_device is not None:
                        params['task_type'] = 'GPU'
                        params['devices'] = str(gpu_device)
                model = CatBoostRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                params = {
                    # CatBoost 최적화: 더 많은 iterations와 낮은 learning_rate 범위
                    'iterations': trial.suggest_int('iterations', 500, 2000),  # 범위 증가: 500-2000
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # 0.3 -> 0.15로 제한 (CatBoost 권장 범위)
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'bootstrap_type': bootstrap_type,
                    'random_state': self.random_state,
                    'verbose': False,
                    'allow_writing_files': False,
                }
                # bagging_temperature는 bootstrap_type='Bayesian'일 때만 사용 가능
                if bootstrap_type == 'Bayesian':
                    params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
                # GPU 설정 추가
                if self.use_gpu:
                    gpu_device = get_gpu_device()
                    if gpu_device is not None:
                        params['task_type'] = 'GPU'
                        params['devices'] = str(gpu_device)
                model = CatBoostClassifier(**params)
                scoring = 'roc_auc'
            
            # K-Fold 교차 검증 (Optuna 최적화 시 속도 개선을 위해 3-fold 사용)
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # 5 -> 3으로 축소
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)  # 5 -> 3으로 축소
            
            # CatBoost 범주형 특성 처리
            # DataFrame인 경우 CatBoost가 자동으로 범주형을 감지하지만,
            # 명시적으로 전달하기 위해 수동으로 CV 수행
            if cat_features is not None and isinstance(X_train, pd.DataFrame):
                cat_indices = [X_train.columns.get_loc(col) for col in cat_features if col in X_train.columns]
                # 수동으로 CV 수행하여 cat_features 전달
                cv_scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_train_fold = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
                    X_val_fold = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                    y_train_fold = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                    y_val_fold = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
                    
                    # 각 fold마다 새 모델 생성 (params는 이미 GPU 설정 포함)
                    model_fold = CatBoostRegressor(**params) if self.task_type == 'regression' else CatBoostClassifier(**params)
                    model_fold.fit(X_train_fold, y_train_fold, cat_features=cat_indices, verbose=False)
                    y_pred = model_fold.predict(X_val_fold)
                    
                    if self.task_type == 'regression':
                        score = -mean_squared_error(y_val_fold, y_pred)  # neg_mean_squared_error
                    else:
                        score = roc_auc_score(y_val_fold, y_pred)
                    cv_scores.append(score)
                scores = np.array(cv_scores)
            else:
                # 범주형 특성이 없거나 자동 감지 가능한 경우
                # n_jobs=1로 설정하여 Optuna와의 충돌 방지
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            
            if self.task_type == 'regression':
                return -scores.mean()  # RMSE는 최소화
            else:
                return scores.mean()  # AUC는 최대화
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 진행 상황 출력을 위한 콜백
        def callback(study, trial):
            # 1% 간격 또는 첫 번째 trial에서 출력
            interval = max(1, n_trials // 100)
            if (trial.number + 1) % interval == 0 or trial.number == 0:
                best_value = study.best_value if study.best_value is not None else float('inf')
                current_value = trial.value if trial.value is not None else 'N/A'
                if isinstance(current_value, (int, float)):
                    print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                          f"Best Score: {best_value:.6f} | "
                          f"Current Score: {current_value:.6f}")
                else:
                    print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                          f"Best Score: {best_value:.6f} | "
                          f"Current Score: {current_value}")
        
        print(f"  🔄 Optuna 최적화 시작 (총 {n_trials} trials)...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])
        
        best_value = study.best_value if study.best_value is not None else float('inf')
        print(f"  ✅ 최적화 완료! Best Score: {best_value:.6f}")
        self.best_params['catboost'] = study.best_params
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, n_trials: int = 50, **kwargs):
        """
        LightGBM 하이퍼파라미터 최적화
        """
        def objective(trial):
            if self.task_type == 'regression':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbosity': -1,
                }
                # GPU 설정 추가
                if self.use_gpu:
                    gpu_device = get_gpu_device()
                    if gpu_device is not None:
                        params['device'] = 'gpu'
                        params['gpu_platform_id'] = 0
                        params['gpu_device_id'] = gpu_device
                model = lgb.LGBMRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'lambda_l1': trial.suggest_float('lambda_l1', 1e-3, 10.0, log=True),
                    'lambda_l2': trial.suggest_float('lambda_l2', 1e-3, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbosity': -1,
                }
                # GPU 설정 추가
                if self.use_gpu:
                    gpu_device = get_gpu_device()
                    if gpu_device is not None:
                        params['device'] = 'gpu'
                        params['gpu_platform_id'] = 0
                        params['gpu_device_id'] = gpu_device
                model = lgb.LGBMClassifier(**params)
                scoring = 'roc_auc'
            
            # K-Fold 교차 검증 (Optuna 최적화 시 속도 개선을 위해 3-fold 사용)
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # 5 -> 3으로 축소
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)  # 5 -> 3으로 축소
            
            # n_jobs=1로 설정하여 Optuna와의 충돌 방지
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            else:
                return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 진행 상황 출력을 위한 콜백
        def callback(study, trial):
            # 1% 간격 또는 첫 번째 trial에서 출력
            interval = max(1, n_trials // 100)
            if (trial.number + 1) % interval == 0 or trial.number == 0:
                best_value = study.best_value if study.best_value is not None else float('inf')
                current_value = trial.value if trial.value is not None else 'N/A'
                if isinstance(current_value, (int, float)):
                    print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                          f"Best Score: {best_value:.6f} | "
                          f"Current Score: {current_value:.6f}")
                else:
                    print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                          f"Best Score: {best_value:.6f} | "
                          f"Current Score: {current_value}")
        
        print(f"  🔄 Optuna 최적화 시작 (총 {n_trials} trials)...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])
        
        best_value = study.best_value if study.best_value is not None else float('inf')
        print(f"  ✅ 최적화 완료! Best Score: {best_value:.6f}")
        self.best_params['lightgbm'] = study.best_params
        return study.best_params
    
    def optimize_xgboost(self, X_train, y_train, n_trials: int = 50, **kwargs):
        """
        XGBoost 하이퍼파라미터 최적화
        """
        def objective(trial):
            if self.task_type == 'regression':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbosity': 0,
                }
                # GPU 설정 추가 (XGBoost 2.0+에서는 tree_method='hist'와 device 파라미터 사용)
                if self.use_gpu:
                    gpu_device = get_gpu_device()
                    if gpu_device is not None:
                        params['tree_method'] = 'hist'  # gpu_hist는 더 이상 지원되지 않음
                        params['device'] = f'cuda:{gpu_device}'
                    else:
                        params['tree_method'] = 'hist'
                else:
                    params['tree_method'] = 'hist'
                model = xgb.XGBRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                    'random_state': self.random_state,
                    'verbosity': 0,
                }
                # GPU 설정 추가 (XGBoost 2.0+에서는 tree_method='hist'와 device 파라미터 사용)
                if self.use_gpu:
                    gpu_device = get_gpu_device()
                    if gpu_device is not None:
                        params['tree_method'] = 'hist'  # gpu_hist는 더 이상 지원되지 않음
                        params['device'] = f'cuda:{gpu_device}'
                    else:
                        params['tree_method'] = 'hist'
                else:
                    params['tree_method'] = 'hist'
                model = xgb.XGBClassifier(**params)
                scoring = 'roc_auc'
            
            # K-Fold 교차 검증 (Optuna 최적화 시 속도 개선을 위해 3-fold 사용)
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)  # 5 -> 3으로 축소
            else:
                cv = KFold(n_splits=3, shuffle=True, random_state=self.random_state)  # 5 -> 3으로 축소
            
            # n_jobs=1로 설정하여 Optuna와의 충돌 방지
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            else:
                return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 진행 상황 출력을 위한 콜백
        def callback(study, trial):
            # 1% 간격 또는 첫 번째 trial에서 출력
            interval = max(1, n_trials // 100)
            if (trial.number + 1) % interval == 0 or trial.number == 0:
                best_value = study.best_value if study.best_value is not None else float('inf')
                current_value = trial.value if trial.value is not None else 'N/A'
                if isinstance(current_value, (int, float)):
                    print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                          f"Best Score: {best_value:.6f} | "
                          f"Current Score: {current_value:.6f}")
                else:
                    print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                          f"Best Score: {best_value:.6f} | "
                          f"Current Score: {current_value}")
        
        print(f"  🔄 Optuna 최적화 시작 (총 {n_trials} trials)...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])
        
        best_value = study.best_value if study.best_value is not None else float('inf')
        print(f"  ✅ 최적화 완료! Best Score: {best_value:.6f}")
        self.best_params['xgboost'] = study.best_params
        return study.best_params


def optimize_hyperparameters(X_train, y_train, categorical_cols, task_type='regression',
                              n_trials=50, random_state=42, use_saved_params=False, 
                              params_filepath=None, encoded_cols_tag='_encoded', use_gpu=False,
                              sample_size=None):
    """
    하이퍼파라미터 최적화 또는 저장된 파라미터 로드
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        학습 데이터
    y_train : pd.Series
        타겟 데이터
    categorical_cols : List[str]
        범주형 컬럼 리스트
    task_type : str
        작업 타입 ('regression' or 'classification')
    n_trials : int
        Optuna 시도 횟수
    random_state : int
        랜덤 시드
    use_saved_params : bool
        저장된 파라미터 사용 여부
    params_filepath : str
        파라미터 파일 경로
    encoded_cols_tag : str
        인코딩된 컬럼 태그
    use_gpu : bool
        GPU 사용 여부
    sample_size : int, optional
        샘플 크기 (속도 개선용)
        
    Returns:
    --------
    dict
        모델별 최적화된 하이퍼파라미터
    """
    # use_saved_params가 True이고 파일이 존재하면 저장된 파라미터 사용
    # 단, use_optuna가 True인 경우는 Optuna 최적화를 실행 (use_saved_params 무시)
    # use_optuna는 optimize_hyperparameters 함수에서 받지 않으므로, 
    # 호출하는 쪽에서 use_saved_params=False로 전달해야 함
    if use_saved_params and params_filepath and os.path.exists(params_filepath):
        print(f"\n📂 저장된 하이퍼파라미터 불러오기: {params_filepath}")
        best_params = load_hyperparameters(params_filepath)
        return best_params
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna가 설치되어 있지 않습니다. 'pip install optuna'로 설치해주세요.")
    
    print(f"\n{'='*60}")
    print("🔍 Optuna를 사용한 하이퍼파라미터 최적화 시작")
    if use_gpu:
        print("🚀 GPU를 사용하여 최적화를 진행합니다.")
    
    # 데이터 샘플링 (속도 개선)
    if sample_size is not None and len(X_train) > sample_size:
        print(f"⚡ 속도 개선을 위해 데이터 샘플링: {len(X_train)} -> {sample_size}개")
        if task_type == 'classification':
            from sklearn.model_selection import train_test_split
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=sample_size, 
                stratify=y_train, random_state=random_state
            )
        else:
            # Regression의 경우 단순 샘플링
            sample_idx = np.random.RandomState(seed=random_state).choice(
                len(X_train), size=sample_size, replace=False
            )
            X_train_sample = X_train.iloc[sample_idx] if isinstance(X_train, pd.DataFrame) else X_train[sample_idx]
            y_train_sample = y_train.iloc[sample_idx] if isinstance(y_train, pd.Series) else y_train[sample_idx]
        print(f"   샘플링된 데이터로 최적화를 진행합니다.")
    else:
        X_train_sample = X_train
        y_train_sample = y_train
    
    print(f"{'='*60}")
    
    optimizer = HyperparameterOptimizer(task_type=task_type, random_state=random_state, use_gpu=use_gpu)
    best_params = {}
    
    # 모델별로 사용할 컬럼 준비 (샘플링된 데이터 사용)
    encoded_cols = [col for col in X_train_sample.columns if col.endswith(encoded_cols_tag)]
    
    # 각 모델별로 최적화
    for model_type in ['xgboost', 'catboost', 'lightgbm']:
        print(f"\n{'='*60}")
        print(f"🔍 {model_type.upper()} 하이퍼파라미터 최적화 (n_trials={n_trials})")
        print(f"{'='*60}")
        
        # 모델별로 사용할 컬럼 선택
        if model_type == 'catboost':
            # CatBoost: 원본 범주형 컬럼 + 조합 범주형 컬럼 사용 (인코딩된 컬럼 제외)
            feature_cols = [col for col in X_train_sample.columns if col not in encoded_cols]
            X_train_model = X_train_sample[feature_cols].copy()
            
            # 실제 데이터프레임에서 모든 범주형 컬럼 동적으로 찾기 (조합 특성 포함)
            all_categorical_cols = X_train_model.select_dtypes(include=['object', 'category']).columns.tolist()
            print(f"  CatBoost: 범주형 컬럼 {len(all_categorical_cols)}개 사용 (원본 + 조합 특성 포함)")
            if len(categorical_cols) > 0:
                print(f"    원본 범주형: {len(categorical_cols)}개")
                print(f"    조합 특성 포함: {len(all_categorical_cols) - len([c for c in categorical_cols if c in all_categorical_cols])}개")
            
            params = optimizer.optimize_catboost(
                X_train_model, y_train_sample,
                n_trials=n_trials,
                cat_features=all_categorical_cols if all_categorical_cols else None
            )
        elif model_type == 'lightgbm':
            # LightGBM: 인코딩된 컬럼 + 수치형 컬럼만 사용 (모든 범주형 컬럼 제외)
            all_categorical_cols = X_train_sample.select_dtypes(include=['object', 'category']).columns.tolist()
            feature_cols = [col for col in X_train_sample.columns if col not in all_categorical_cols]
            X_train_model = X_train_sample[feature_cols].copy()
            print(f"  LightGBM: 인코딩된 컬럼 {len(encoded_cols)}개 + 수치형 컬럼 사용 (모든 범주형 컬럼 제외)")
            
            params = optimizer.optimize_lightgbm(
                X_train_model, y_train_sample,
                n_trials=n_trials
            )
        elif model_type == 'xgboost':
            # XGBoost: 인코딩된 컬럼 + 수치형 컬럼만 사용 (모든 범주형 컬럼 제외)
            all_categorical_cols = X_train_sample.select_dtypes(include=['object', 'category']).columns.tolist()
            feature_cols = [col for col in X_train_sample.columns if col not in all_categorical_cols]
            X_train_model = X_train_sample[feature_cols].copy()
            print(f"  XGBoost: 인코딩된 컬럼 {len(encoded_cols)}개 + 수치형 컬럼 사용 (모든 범주형 컬럼 제외)")
            
            params = optimizer.optimize_xgboost(
                X_train_model, y_train_sample,
                n_trials=n_trials
            )
        
        best_params[model_type] = params
        
        print(f"\n✅ {model_type.upper()} 최적화 완료!")
        print(f"📊 최적 파라미터:")
        for key, value in sorted(params.items()):
            print(f"  {key:25s}: {value}")
        print()  # 빈 줄 추가
    
    # 최적화된 파라미터 저장
    if params_filepath:
        print(f"\n💾 Optuna 최적화된 하이퍼파라미터 저장 중...")
        save_hyperparameters(
            best_params,
            params_filepath,
            task_type=task_type,
            additional_info={
                'n_trials': n_trials,
                'random_state': random_state,
                'optimization_method': 'Optuna'
            }
        )
    
    return best_params


def save_hyperparameters(best_params: Dict[str, dict], filepath: str, 
                         task_type: str = 'regression', additional_info: Optional[dict] = None):
    """
    최적화된 하이퍼파라미터를 JSON 파일로 저장
    
    Parameters:
    -----------
    best_params : dict
        {model_type: params} 형태의 딕셔너리
    filepath : str
        저장할 파일 경로
    task_type : str
        작업 타입
    additional_info : dict, optional
        추가 정보 (CV 점수, 날짜 등)
    """
    save_data = {
        'task_type': task_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': best_params
    }
    
    if additional_info:
        save_data['additional_info'] = additional_info
    
    # 디렉토리가 없으면 생성
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 하이퍼파라미터 저장 완료: {filepath}")


def load_hyperparameters(filepath: str) -> Dict[str, dict]:
    """
    저장된 하이퍼파라미터를 JSON 파일에서 불러오기
    
    Parameters:
    -----------
    filepath : str
        파일 경로
        
    Returns:
    --------
    dict
        {model_type: params} 형태의 딕셔너리
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📂 하이퍼파라미터 불러오기 완료: {filepath}")
    print(f"  Task Type: {data.get('task_type', 'unknown')}")
    print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
    
    return data.get('hyperparameters', {})
