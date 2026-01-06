import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# 모델 라이브러리
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

# 평가 및 검증
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss

# 하이퍼파라미터 최적화
import optuna
from optuna import Trial


def check_gpu_availability():
    """
    GPU 사용 가능 여부 확인
    
    Returns:
    --------
    bool
        GPU 사용 가능 여부
    """
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False


def get_gpu_device():
    """
    사용 가능한 GPU 디바이스 번호 반환
    
    Returns:
    --------
    str or None
        GPU 디바이스 번호 ('0', '1', ...) 또는 None
    """
    if check_gpu_availability():
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                # 첫 번째 GPU 사용
                return '0'
        except:
            pass
    return None


class BaseModel:
    """모든 모델의 기본 클래스"""
    
    def __init__(self, model_type: str, task_type: str = 'regression', random_state: int = 42):
        """
        Parameters:
        -----------
        model_type : str
            모델 타입 ('catboost', 'lightgbm', 'xgboost')
        task_type : str
            작업 타입 ('regression' or 'classification')
        random_state : int
            랜덤 시드
        """
        self.model_type = model_type
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """모델 학습"""
        raise NotImplementedError
        
    def predict(self, X):
        """예측"""
        raise NotImplementedError
        
    def get_feature_importance(self):
        """특성 중요도 반환"""
        return self.feature_importance_


class CatBoostModel(BaseModel):
    """CatBoost 모델 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, 
                 cat_features: Optional[List] = None, use_gpu: bool = False, **kwargs):
        super().__init__('catboost', task_type, random_state)
        self.cat_features = cat_features
        self.params = kwargs
        
        # 기본 파라미터 설정
        default_params = {
            'random_state': random_state,
            'verbose': False,
            'allow_writing_files': False,
        }
        
        # GPU 설정
        if use_gpu:
            gpu_device = get_gpu_device()
            if gpu_device is not None:
                default_params['task_type'] = 'GPU'
                default_params['devices'] = gpu_device
                print(f"✅ CatBoost: GPU 사용 설정 (Device: {gpu_device})")
            else:
                print("⚠️ CatBoost: GPU를 사용할 수 없습니다. CPU로 학습합니다.")
        
        if task_type == 'regression':
            default_params.update({
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE',
            })
        else:
            default_params.update({
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
            })
        
        default_params.update(self.params)
        self.params = default_params
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """CatBoost 모델 학습"""
        # CatBoost는 범주형 특성을 자동으로 처리
        if isinstance(X_train, pd.DataFrame):
            if self.cat_features is None:
                # 범주형 특성 자동 감지
                self.cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            cat_indices = [X_train.columns.get_loc(col) for col in self.cat_features if col in X_train.columns]
        else:
            cat_indices = self.cat_features if self.cat_features else None
        
        # 모델 생성
        if self.task_type == 'regression':
            self.model = CatBoostRegressor(**self.params)
        else:
            self.model = CatBoostClassifier(**self.params)
        
        # 학습
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_indices,
                **kwargs
            )
        else:
            self.model.fit(
                X_train, y_train,
                cat_features=cat_indices,
                **kwargs
            )
        
        # 특성 중요도 저장
        self.feature_importance_ = self.model.feature_importances_
        
        return self
        
    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM 모델 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, 
                 use_gpu: bool = False, **kwargs):
        super().__init__('lightgbm', task_type, random_state)
        self.params = kwargs
        
        # 기본 파라미터 설정
        default_params = {
            'random_state': random_state,
            'verbosity': -1,
        }
        
        # GPU 설정
        if use_gpu:
            gpu_device = get_gpu_device()
            if gpu_device is not None:
                default_params['device'] = 'gpu'
                default_params['gpu_platform_id'] = 0
                default_params['gpu_device_id'] = int(gpu_device)
                print(f"✅ LightGBM: GPU 사용 설정 (Device: {gpu_device})")
            else:
                print("⚠️ LightGBM: GPU를 사용할 수 없습니다. CPU로 학습합니다.")
        
        if task_type == 'regression':
            default_params.update({
                'objective': 'regression',
                'metric': 'rmse',
            })
        else:
            default_params.update({
                'objective': 'binary',
                'metric': 'binary_logloss',
            })
        
        default_params.update(self.params)
        self.params = default_params
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """LightGBM 모델 학습"""
        # 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                **kwargs
            )
        else:
            self.model = lgb.train(
                self.params,
                train_data,
                **kwargs
            )
        
        # 특성 중요도 저장
        self.feature_importance_ = self.model.feature_importance(importance_type='gain')
        
        return self
        
    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost 모델 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, 
                 use_gpu: bool = False, **kwargs):
        super().__init__('xgboost', task_type, random_state)
        self.params = kwargs
        
        # 기본 파라미터 설정
        default_params = {
            'random_state': random_state,
            'verbosity': 0,
        }
        
        # GPU 설정
        self.use_gpu = False  # 실제 GPU 사용 여부 플래그
        if use_gpu:
            gpu_device = get_gpu_device()
            if gpu_device is not None:
                # GPU 설정 시도 (실제 지원 여부는 fit에서 확인)
                default_params['tree_method'] = 'gpu_hist'
                # XGBoost 3.1+에서는 device에 'cuda:0' 형식으로 GPU 번호 포함
                default_params['device'] = f'cuda:{gpu_device}'
                print(f"🔄 XGBoost: GPU 사용 시도 (Device: cuda:{gpu_device})")
                print("   (GPU 지원이 없으면 자동으로 CPU로 전환됩니다)")
            else:
                print("⚠️ XGBoost: GPU 디바이스를 찾을 수 없습니다. CPU로 학습합니다.")
                default_params['tree_method'] = 'hist'
        else:
            # CPU 기본값 설정
            default_params['tree_method'] = 'hist'
        
        if task_type == 'regression':
            default_params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
            })
        else:
            default_params.update({
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
            })
        
        default_params.update(self.params)
        self.params = default_params
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """XGBoost 모델 학습"""
        # 데이터셋 생성
        train_data = xgb.DMatrix(X_train, label=y_train)
        
        # GPU 사용 시도, 실패하면 CPU로 자동 전환
        try:
            if X_val is not None and y_val is not None:
                val_data = xgb.DMatrix(X_val, label=y_val)
                watchlist = [(train_data, 'train'), (val_data, 'val')]
                self.model = xgb.train(
                    self.params,
                    train_data,
                    evals=watchlist,
                    **kwargs
                )
            else:
                self.model = xgb.train(
                    self.params,
                    train_data,
                    **kwargs
                )
        except Exception as e:
            error_msg = str(e).lower()
            # GPU 관련 오류인 경우 CPU로 전환
            if 'gpu' in error_msg or 'gpu_hist' in error_msg or 'cuda' in error_msg:
                print(f"⚠️ XGBoost GPU 학습 실패: {str(e)[:100]}")
                print("🔄 CPU로 자동 전환하여 학습합니다...")
                # CPU 파라미터로 변경
                cpu_params = self.params.copy()
                cpu_params['tree_method'] = 'hist'
                if 'device' in cpu_params:
                    del cpu_params['device']
                
                if X_val is not None and y_val is not None:
                    val_data = xgb.DMatrix(X_val, label=y_val)
                    watchlist = [(train_data, 'train'), (val_data, 'val')]
                    self.model = xgb.train(
                        cpu_params,
                        train_data,
                        evals=watchlist,
                        **kwargs
                    )
                else:
                    self.model = xgb.train(
                        cpu_params,
                        train_data,
                        **kwargs
                    )
                self.use_gpu = False
            else:
                # GPU와 무관한 오류는 그대로 전파
                raise
        
        # 특성 중요도 저장
        self.feature_importance_ = np.array(list(self.model.get_score(importance_type='gain').values()))
        
        return self
        
    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        test_data = xgb.DMatrix(X)
        return self.model.predict(test_data)


class HyperparameterOptimizer:
    """Optuna를 사용한 하이퍼파라미터 최적화 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42):
        """
        Parameters:
        -----------
        task_type : str
            작업 타입 ('regression' or 'classification')
        random_state : int
            랜덤 시드
        """
        self.task_type = task_type
        self.random_state = random_state
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
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        
        def objective(trial):
            if self.task_type == 'regression':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                    'random_state': self.random_state,
                    'verbose': False,
                    'allow_writing_files': False,
                }
                model = CatBoostRegressor(**params)
                scoring = 'neg_mean_squared_error'
            else:
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 2000),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 100, log=True),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'bootstrap_type': trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS']),
                    'random_state': self.random_state,
                    'verbose': False,
                    'allow_writing_files': False,
                }
                model = CatBoostClassifier(**params)
                scoring = 'roc_auc'
            
            # K-Fold 교차 검증
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            if self.task_type == 'regression':
                return -scores.mean()  # RMSE는 최소화
            else:
                return scores.mean()  # AUC는 최대화
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['catboost'] = study.best_params
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, n_trials: int = 50, **kwargs):
        """
        LightGBM 하이퍼파라미터 최적화
        """
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        
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
                model = lgb.LGBMClassifier(**params)
                scoring = 'roc_auc'
            
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            else:
                return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['lightgbm'] = study.best_params
        return study.best_params
    
    def optimize_xgboost(self, X_train, y_train, n_trials: int = 50, **kwargs):
        """
        XGBoost 하이퍼파라미터 최적화
        """
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
        
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
                model = xgb.XGBClassifier(**params)
                scoring = 'roc_auc'
            
            if self.task_type == 'classification':
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            
            if self.task_type == 'regression':
                return -scores.mean()
            else:
                return scores.mean()
        
        study = optuna.create_study(
            direction='minimize' if self.task_type == 'regression' else 'maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params


class ModelTrainer:
    """모델 학습 및 관리 클래스"""
    
    def __init__(self, task_type: str = 'regression', random_state: int = 42, use_gpu: bool = False):
        """
        Parameters:
        -----------
        task_type : str
            작업 타입 ('regression' or 'classification')
        random_state : int
            랜덤 시드
        use_gpu : bool
            GPU 사용 여부 (기본값: False)
        """
        self.task_type = task_type
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.models = {}
        self.cv_scores = {}
        self.oof_predictions = {}
        self.best_params = {}  # 최적화된 하이퍼파라미터 저장
        
        # GPU 사용 가능 여부 확인 및 출력
        if use_gpu:
            if check_gpu_availability():
                print("🚀 GPU 사용 가능: 모델 학습에 GPU를 사용합니다.")
            else:
                print("⚠️ GPU를 사용할 수 없습니다. CPU로 학습합니다.")
                self.use_gpu = False
        
    def create_model(self, model_type: str, cat_features: Optional[List] = None, **kwargs):
        """
        모델 생성
        
        Parameters:
        -----------
        model_type : str
            모델 타입 ('catboost', 'lightgbm', 'xgboost')
        cat_features : List, optional
            범주형 특성 리스트 (CatBoost용)
        **kwargs
            모델별 하이퍼파라미터
        """
        if model_type == 'catboost':
            return CatBoostModel(
                task_type=self.task_type,
                random_state=self.random_state,
                cat_features=cat_features,
                use_gpu=self.use_gpu,
                **kwargs
            )
        elif model_type == 'lightgbm':
            return LightGBMModel(
                task_type=self.task_type,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                **kwargs
            )
        elif model_type == 'xgboost':
            return XGBoostModel(
                task_type=self.task_type,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train_with_cv(self, X, y, model_type: str, n_folds: int = 5, 
                     cat_features: Optional[List] = None, **model_params):
        """
        K-Fold 교차 검증으로 모델 학습
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            학습 데이터
        y : pd.Series or np.ndarray
            타겟 데이터
        model_type : str
            모델 타입 ('catboost', 'lightgbm', 'xgboost')
        n_folds : int
            K-Fold 개수
        cat_features : List, optional
            범주형 특성 리스트
        **model_params
            모델별 하이퍼파라미터
            
        Returns:
        --------
        dict
            학습된 모델들, OOF 예측, CV 점수
        """
        # K-Fold 설정
        if self.task_type == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        models = []
        oof_preds = np.zeros(len(X))
        fold_scores = []
        
        print(f"\n{'='*60}")
        print(f"🚀 {model_type.upper()} 모델 학습 시작 (K-Fold={n_folds})")
        print(f"{'='*60}")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            print(f"\n📊 Fold {fold}/{n_folds} 학습 중...")
            
            X_train, X_val = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], \
                            X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
            y_train, y_val = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx], \
                            y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]
            
            # 모델 생성 및 학습
            model = self.create_model(model_type, cat_features=cat_features, **model_params)
            
            if model_type == 'lightgbm':
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    num_boost_round=model_params.get('num_boost_round', 1000),
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
                )
            elif model_type == 'xgboost':
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    num_boost_round=model_params.get('num_boost_round', 1000),
                    early_stopping_rounds=100,
                    verbose_eval=False
                )
            else:  # catboost
                model.fit(
                    X_train, y_train,
                    X_val=X_val, y_val=y_val,
                    early_stopping_rounds=100
                )
            
            # 검증 예측
            val_pred = model.predict(X_val)
            oof_preds[val_idx] = val_pred
            
            # 평가 점수 계산
            if self.task_type == 'regression':
                score = np.sqrt(mean_squared_error(y_val, val_pred))
                print(f"  Fold {fold} RMSE: {score:.4f}")
            else:
                score = roc_auc_score(y_val, val_pred)
                print(f"  Fold {fold} AUC: {score:.4f}")
            
            fold_scores.append(score)
            models.append(model)
        
        # 전체 CV 점수
        if self.task_type == 'regression':
            cv_score = np.sqrt(mean_squared_error(y, oof_preds))
            print(f"\n✅ CV RMSE: {cv_score:.4f} (std: {np.std(fold_scores):.4f})")
        else:
            cv_score = roc_auc_score(y, oof_preds)
            print(f"\n✅ CV AUC: {cv_score:.4f} (std: {np.std(fold_scores):.4f})")
        
        self.models[model_type] = models
        self.oof_predictions[model_type] = oof_preds
        self.cv_scores[model_type] = {
            'mean': cv_score,
            'std': np.std(fold_scores),
            'fold_scores': fold_scores
        }
        
        return {
            'models': models,
            'oof_predictions': oof_preds,
            'cv_score': cv_score,
            'fold_scores': fold_scores
        }
    
    def predict_test(self, X_test, model_type: str):
        """
        테스트 데이터 예측 (앙상블: 모든 fold 모델의 평균)
        
        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray
            테스트 데이터
        model_type : str
            모델 타입
            
        Returns:
        --------
        np.ndarray
            예측 결과
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        
        models = self.models[model_type]
        predictions = np.array([model.predict(X_test) for model in models])
        return predictions.mean(axis=0)


class EnsembleModel:
    """앙상블 모델 클래스"""
    
    def __init__(self, task_type: str = 'regression'):
        """
        Parameters:
        -----------
        task_type : str
            작업 타입 ('regression' or 'classification')
        """
        self.task_type = task_type
        self.weights = None
        
    def fit(self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray, 
            method: str = 'weighted_average', optimize: bool = True):
        """
        앙상블 모델 학습
        
        Parameters:
        -----------
        predictions_dict : dict
            {model_name: predictions} 형태의 딕셔너리
        y_true : np.ndarray
            실제 타겟 값
        method : str
            앙상블 방법 ('weighted_average', 'simple_average', 'stacking')
        optimize : bool
            가중치 최적화 여부
        """
        self.method = method
        
        if method == 'simple_average':
            self.weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict.keys()}
        
        elif method == 'weighted_average':
            if optimize:
                self.weights = self._optimize_weights(predictions_dict, y_true)
            else:
                # 동일 가중치로 시작
                self.weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict.keys()}
        
        elif method == 'stacking':
            # 간단한 선형 스태킹 (실제로는 메타 모델이 필요하지만 여기서는 가중 평균으로 대체)
            self.weights = self._optimize_weights(predictions_dict, y_true)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _optimize_weights(self, predictions_dict: Dict[str, np.ndarray], 
                         y_true: np.ndarray) -> Dict[str, float]:
        """
        가중치 최적화 (scipy.optimize 사용)
        """
        from scipy.optimize import minimize
        
        model_names = list(predictions_dict.keys())
        predictions_list = [predictions_dict[name] for name in model_names]
        
        def objective(weights):
            """최적화 목적 함수"""
            weighted_pred = np.zeros_like(predictions_list[0])
            for pred, weight in zip(predictions_list, weights):
                weighted_pred += weight * pred
            
            if self.task_type == 'regression':
                return np.sqrt(mean_squared_error(y_true, weighted_pred))
            else:
                return -roc_auc_score(y_true, weighted_pred)  # 최대화를 위해 음수
        
        # 제약 조건: 가중치 합 = 1, 모든 가중치 >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(model_names))]
        
        # 초기값: 동일 가중치
        initial_weights = np.array([1.0 / len(model_names)] * len(model_names))
        
        # 최적화
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        weights_dict = {name: weight for name, weight in zip(model_names, result.x)}
        
        print(f"\n📊 최적화된 가중치:")
        for name, weight in weights_dict.items():
            print(f"  {name}: {weight:.4f}")
        
        return weights_dict
    
    def predict(self, predictions_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        앙상블 예측
        
        Parameters:
        -----------
        predictions_dict : dict
            {model_name: predictions} 형태의 딕셔너리
            
        Returns:
        --------
        np.ndarray
            앙상블 예측 결과
        """
        if self.weights is None:
            raise ValueError("Ensemble model must be fitted first")
        
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        
        for name, pred in predictions_dict.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred


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
    import json
    from datetime import datetime
    
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
    import json
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📂 하이퍼파라미터 불러오기 완료: {filepath}")
    print(f"  Task Type: {data.get('task_type', 'unknown')}")
    print(f"  Timestamp: {data.get('timestamp', 'unknown')}")
    
    return data.get('hyperparameters', {})


def evaluate_model(y_true, y_pred, task_type: str = 'regression'):
    """
    모델 평가
    
    Parameters:
    -----------
    y_true : np.ndarray
        실제 타겟 값
    y_pred : np.ndarray
        예측 값
    task_type : str
        작업 타입
        
    Returns:
    --------
    dict
        평가 지표 딕셔너리
    """
    if task_type == 'regression':
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    else:
        return {
            'AUC': roc_auc_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, (y_pred > 0.5).astype(int)),
            'LogLoss': log_loss(y_true, y_pred)
        }

