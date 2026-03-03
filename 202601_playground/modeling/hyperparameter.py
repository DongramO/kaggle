"""
하이퍼파라미터 최적화 및 관리 모듈
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna가 설치되어 있지 않습니다. 하이퍼파라미터 최적화를 사용할 수 없습니다.")


def get_gpu_device():
    """GPU 디바이스 번호 반환"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
        if result.returncode == 0:
            return 0
    except:
        pass
    return None


def _get_cv_splitter(task_type: str, random_state: int):
    """교차 검증 splitter 반환"""
    if task_type == 'classification':
        return StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    return KFold(n_splits=3, shuffle=True, random_state=random_state)


def _create_optimization_callback(n_trials: int):
    """최적화 진행 상황 출력 콜백 생성"""
    def callback(study, trial):
        interval = max(1, n_trials // 100)
        if (trial.number + 1) % interval == 0 or trial.number == 0:
            best_value = study.best_value if study.best_value is not None else float('inf')
            current_value = trial.value if trial.value is not None else 'N/A'
            if isinstance(current_value, (int, float)):
                print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                      f"Best Score: {best_value:.6f} | Current Score: {current_value:.6f}")
            else:
                print(f"  Trial {trial.number + 1}/{n_trials} 완료 | "
                      f"Best Score: {best_value:.6f} | Current Score: {current_value}")
    return callback


def _run_optimization(study, objective, n_trials: int, model_name: str):
    """최적화 실행 및 결과 반환"""
    callback = _create_optimization_callback(n_trials)
    print(f"  🔄 Optuna 최적화 시작 (총 {n_trials} trials)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])
    best_value = study.best_value if study.best_value is not None else float('inf')
    print(f"  ✅ 최적화 완료! Best Score: {best_value:.6f}")
    return study.best_params


class HyperparameterOptimizer:
    """Optuna를 사용한 하이퍼파라미터 최적화 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, use_gpu: bool = False):
        """하이퍼파라미터 최적화 클래스 초기화"""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna가 설치되어 있지 않습니다. 'pip install optuna'로 설치해주세요.")
        
        self.task_type = task_type
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.best_params = {}
    
    def optimize_catboost(self, X_train, y_train, n_trials: int = 50, 
                         cat_features: Optional[List] = None, **kwargs):
        """CatBoost 하이퍼파라미터 최적화"""
        def objective(trial):
            bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
            
            params = {
                'iterations': trial.suggest_int('iterations', 500, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                'random_strength': trial.suggest_float('random_strength', 0, 1),
                'bootstrap_type': bootstrap_type,
                'random_state': self.random_state,
                'verbose': False,
                'allow_writing_files': False,
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),
            }
            
            if bootstrap_type == 'Bayesian':
                params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)
            
            if self.use_gpu:
                gpu_device = get_gpu_device()
                if gpu_device is not None:
                    params['task_type'] = 'GPU'
                    params['devices'] = str(gpu_device)
            
            if self.task_type == 'regression':
                model = CatBoostRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                model = CatBoostClassifier(**params)
                scoring = 'roc_auc'
            
            cv = _get_cv_splitter(self.task_type, self.random_state)
            
            if cat_features is not None and isinstance(X_train, pd.DataFrame):
                cat_indices = [X_train.columns.get_loc(col) for col in cat_features if col in X_train.columns]
                cv_scores = []
                for train_idx, val_idx in cv.split(X_train, y_train):
                    X_train_fold = X_train.iloc[train_idx] if isinstance(X_train, pd.DataFrame) else X_train[train_idx]
                    X_val_fold = X_train.iloc[val_idx] if isinstance(X_train, pd.DataFrame) else X_train[val_idx]
                    y_train_fold = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
                    y_val_fold = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
                    
                    model_fold = CatBoostRegressor(**params) if self.task_type == 'regression' else CatBoostClassifier(**params)
                    model_fold.fit(X_train_fold, y_train_fold, cat_features=cat_indices, verbose=False)
                    y_pred = model_fold.predict(X_val_fold)
                    
                    if self.task_type == 'regression':
                        score = -mean_squared_error(y_val_fold, y_pred)
                    else:
                        score = roc_auc_score(y_val_fold, y_pred)
                    cv_scores.append(score)
                scores = np.array(cv_scores)
            else:
                scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        best_params = _run_optimization(study, objective, n_trials, 'catboost')
        self.best_params['catboost'] = best_params
        return best_params
    
    def optimize_lightgbm(self, X_train, y_train, n_trials: int = 50, **kwargs):
        """LightGBM 하이퍼파라미터 최적화"""
        def objective(trial):
            if self.task_type == 'regression':
                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
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
                    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),
                }
                model = lgb.LGBMRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
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
                    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),
                }
                model = lgb.LGBMClassifier(**params)
                scoring = 'roc_auc'
            
            if self.use_gpu:
                gpu_device = get_gpu_device()
                if gpu_device is not None:
                    params['device'] = 'gpu'
                    params['gpu_platform_id'] = 0
                    params['gpu_device_id'] = gpu_device
            
            cv = _get_cv_splitter(self.task_type, self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        best_params = _run_optimization(study, objective, n_trials, 'lightgbm')
        self.best_params['lightgbm'] = best_params
        return best_params
    
    def optimize_xgboost(self, X_train, y_train, n_trials: int = 50, **kwargs):
        """XGBoost 하이퍼파라미터 최적화"""
        def objective(trial):
            if self.task_type == 'regression':
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
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
                    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),
                }
                model = xgb.XGBRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
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
                    'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 200),
                }
                model = xgb.XGBClassifier(**params)
                scoring = 'roc_auc'
            
            params['tree_method'] = 'hist'
            if self.use_gpu:
                gpu_device = get_gpu_device()
                if gpu_device is not None:
                    params['device'] = f'cuda:{gpu_device}'
            
            cv = _get_cv_splitter(self.task_type, self.random_state)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        best_params = _run_optimization(study, objective, n_trials, 'xgboost')
        self.best_params['xgboost'] = best_params
        return best_params


def optimize_hyperparameters(X_train, y_train, categorical_cols, task_type='regression',
                              n_trials=50, random_state=42, use_saved_params=False, 
                              params_filepath=None, encoded_cols_tag='_encoded', use_gpu=False,
                              sample_size=None):
    """하이퍼파라미터 최적화 또는 저장된 파라미터 로드"""
    if use_saved_params and params_filepath and os.path.exists(params_filepath):
        print(f"\n📂 저장된 하이퍼파라미터 불러오기: {params_filepath}")
        return load_hyperparameters(params_filepath)
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna가 설치되어 있지 않습니다. 'pip install optuna'로 설치해주세요.")
    
    print(f"\n{'='*60}")
    print("🔍 Optuna를 사용한 하이퍼파라미터 최적화 시작")
    if use_gpu:
        print("🚀 GPU를 사용하여 최적화를 진행합니다.")
    
    if sample_size is not None and len(X_train) > sample_size:
        print(f"⚡ 속도 개선을 위해 데이터 샘플링: {len(X_train)} -> {sample_size}개")
        if task_type == 'classification':
            from sklearn.model_selection import train_test_split
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=sample_size, 
                stratify=y_train, random_state=random_state
            )
        else:
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
    encoded_cols = [col for col in X_train_sample.columns if col.endswith(encoded_cols_tag)]
    
    for model_type in ['xgboost', 'catboost', 'lightgbm']:
        print(f"\n{'='*60}")
        print(f"🔍 {model_type.upper()} 하이퍼파라미터 최적화 (n_trials={n_trials})")
        print(f"{'='*60}")
        
        if model_type == 'catboost':
            feature_cols = [col for col in X_train_sample.columns if col not in encoded_cols]
            X_train_model = X_train_sample[feature_cols].copy()
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
            all_categorical_cols = X_train_sample.select_dtypes(include=['object', 'category']).columns.tolist()
            feature_cols = [col for col in X_train_sample.columns if col not in all_categorical_cols]
            X_train_model = X_train_sample[feature_cols].copy()
            print(f"  LightGBM: 인코딩된 컬럼 {len(encoded_cols)}개 + 수치형 컬럼 사용 (모든 범주형 컬럼 제외)")
            
            params = optimizer.optimize_lightgbm(X_train_model, y_train_sample, n_trials=n_trials)
        elif model_type == 'xgboost':
            all_categorical_cols = X_train_sample.select_dtypes(include=['object', 'category']).columns.tolist()
            feature_cols = [col for col in X_train_sample.columns if col not in all_categorical_cols]
            X_train_model = X_train_sample[feature_cols].copy()
            print(f"  XGBoost: 인코딩된 컬럼 {len(encoded_cols)}개 + 수치형 컬럼 사용 (모든 범주형 컬럼 제외)")
            
            params = optimizer.optimize_xgboost(X_train_model, y_train_sample, n_trials=n_trials)
        
        best_params[model_type] = params
        
        print(f"\n✅ {model_type.upper()} 최적화 완료!")
        print(f"📊 최적 파라미터:")
        for key, value in sorted(params.items()):
            print(f"  {key:25s}: {value}")
        print()
    
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
    """최적화된 하이퍼파라미터를 JSON 파일로 저장"""
    save_data = {
        'task_type': task_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': best_params
    }
    
    if additional_info:
        save_data['additional_info'] = additional_info
    
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 하이퍼파라미터 저장 완료: {filepath}")


def load_hyperparameters(filepath: str) -> Dict[str, dict]:
    """저장된 하이퍼파라미터를 JSON 파일에서 불러오기"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📂 하이퍼파라미터 불러오기 완료: {filepath}")
    print(f"  Task Type: {data.get('task_type', 'unknown')}")
    print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
    
    return data.get('hyperparameters', {})
