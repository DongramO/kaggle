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
from sklearn.inspection import permutation_importance



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
        
        # GPU 설정 (XGBoost 2.0+에서는 tree_method='hist'와 device 파라미터 사용)
        self.use_gpu = False  # 실제 GPU 사용 여부 플래그
        if use_gpu:
            gpu_device = get_gpu_device()
            if gpu_device is not None:
                # GPU 설정 시도 (실제 지원 여부는 fit에서 확인)
                default_params['tree_method'] = 'hist'  # gpu_hist는 더 이상 지원되지 않음
                # XGBoost 2.0+에서는 device에 'cuda:0' 형식으로 GPU 번호 포함
                default_params['device'] = f'cuda:{gpu_device}'
            else:
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
        # kwargs에서 random_state 제거 (중복 전달 방지)
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'random_state'}
        
        if model_type == 'catboost':
            return CatBoostModel(
                task_type=self.task_type,
                random_state=self.random_state,
                cat_features=cat_features,
                use_gpu=self.use_gpu,
                **kwargs_clean
            )
        elif model_type == 'lightgbm':
            return LightGBMModel(
                task_type=self.task_type,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                **kwargs_clean
            )
        elif model_type == 'xgboost':
            return XGBoostModel(
                task_type=self.task_type,
                random_state=self.random_state,
                use_gpu=self.use_gpu,
                **kwargs_clean
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def train_with_cv(self, X, y, model_type: str, n_folds: int = 5, 
                     cat_features: Optional[List] = None, **model_params):

        # K-Fold 설정
        if self.task_type == 'classification':
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        models = []
        oof_preds = np.zeros(len(X))
        fold_scores = []
        
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

        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        
        models = self.models[model_type]
        predictions = np.array([model.predict(X_test) for model in models])
        return predictions.mean(axis=0)
    
    def get_feature_importance(self, model_type: str, feature_names: List[str], 
                              average_across_folds: bool = True) -> pd.DataFrame:

        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        
        models = self.models[model_type]
        n_folds = len(models)
        
        # 각 Fold의 feature importance 추출
        fold_importances = []
        for fold_idx, model in enumerate(models):
            try:
                # 각 모델 클래스의 get_feature_importance 메서드 사용
                if model_type == 'catboost':
                    importance = model.get_feature_importance()
                elif model_type == 'lightgbm':
                    importance = model.get_feature_importance()
                elif model_type == 'xgboost':
                    # XGBoost는 BaseModel의 get_feature_importance() 사용 (파라미터 없음)
                    importance = model.get_feature_importance()
                else:
                    importance = None
                
                # importance 길이 확인
                if importance is not None:
                    if len(importance) != len(feature_names):
                        print(f"⚠️ Warning: Feature importance length ({len(importance)}) != feature names length ({len(feature_names)}) for {model_type} Fold {fold_idx + 1}")
                        # 길이가 맞지 않으면 0으로 패딩
                        if len(importance) < len(feature_names):
                            importance = np.pad(importance, (0, len(feature_names) - len(importance)), 'constant')
                        elif len(importance) > len(feature_names):
                            importance = importance[:len(feature_names)] # 너무 길면 자르기
                    
                    # 정규화 (합이 1이 되도록)
                    if np.sum(importance) > 0:
                        importance = importance / np.sum(importance)
                    fold_importances.append(importance)
            except Exception as e:
                print(f"⚠️ Fold {fold_idx + 1}의 feature importance 추출 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(fold_importances) == 0:
            return pd.DataFrame()
        
        fold_importances = np.array(fold_importances)
        
        if average_across_folds:
            # Fold별 평균
            mean_importance = np.mean(fold_importances, axis=0)
            std_importance = np.std(fold_importances, axis=0)
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_importance,
                'Std': std_importance
            }).sort_values('Importance', ascending=False)
        else:
            # 각 Fold별 중요도
            importance_dict = {'Feature': feature_names}
            for fold_idx in range(len(fold_importances)):
                importance_dict[f'Fold_{fold_idx+1}'] = fold_importances[fold_idx]
            importance_df = pd.DataFrame(importance_dict)
            # 평균과 표준편차도 추가
            importance_df['Mean'] = importance_df[[f'Fold_{i+1}' for i in range(len(fold_importances))]].mean(axis=1)
            importance_df['Std'] = importance_df[[f'Fold_{i+1}' for i in range(len(fold_importances))]].std(axis=1)
            importance_df = importance_df.sort_values('Mean', ascending=False)
        
        return importance_df
    
    def get_permutation_importance(self, model_type: str, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                                   feature_names: Optional[List[str]] = None,
                                   n_repeats: int = 10, random_state: Optional[int] = None,
                                   scoring: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
        """
        Permutation Importance 계산
        
        Parameters:
        -----------
        model_type : str
            모델 타입 ('catboost', 'lightgbm', 'xgboost')
        X : pd.DataFrame
            검증 데이터 (특성)
        y : Union[pd.Series, np.ndarray]
            검증 데이터 (타겟)
        feature_names : List[str], optional
            특성 이름 리스트 (None이면 X.columns 사용)
        n_repeats : int
            Permutation 반복 횟수 (기본값: 10)
        random_state : int, optional
            랜덤 시드
        scoring : str, optional
            평가 지표 ('neg_mean_squared_error', 'roc_auc' 등)
            None이면 task_type에 따라 자동 선택
        n_jobs : int
            병렬 처리 작업 수 (기본값: -1, 모든 CPU 사용)
            
        Returns:
        --------
        pd.DataFrame
            Permutation Importance 결과 (Feature, Importance, Std)
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        models = self.models[model_type]
        n_folds = len(models)
        
        # scoring 자동 설정
        if scoring is None:
            if self.task_type == 'regression':
                scoring = 'neg_mean_squared_error'
            else:
                scoring = 'roc_auc'
        
        # 각 Fold의 permutation importance 계산
        fold_importances = []
        
        print(f"\n📊 {model_type.upper()} Permutation Importance 계산 중...")
        print(f"   반복 횟수: {n_repeats}, Fold 수: {n_folds}")
        
        for fold_idx, model in enumerate(models, 1):
            print(f"   Fold {fold_idx}/{n_folds} 처리 중...", end='\r')
            
            try:
                # 모델의 predict 메서드가 있는지 확인
                # CatBoost, LightGBM, XGBoost는 모두 sklearn API를 따름
                if hasattr(model, 'predict'):
                    # permutation_importance 계산
                    perm_result = permutation_importance(
                        model, X, y,
                        n_repeats=n_repeats,
                        random_state=random_state if random_state is not None else self.random_state,
                        scoring=scoring,
                        n_jobs=n_jobs
                    )
                    
                    # Permutation importance는 원래 점수 - permuted 점수
                    # neg_mean_squared_error의 경우: 더 큰 값(덜 음수)이 더 중요함
                    importances = perm_result.importances_mean
                    if scoring.startswith('neg_'):
                        # 음수 지표는 반전 (더 큰 값이 더 중요)
                        importances = -importances
                    
                    stds = perm_result.importances_std
                    
                    fold_importances.append({
                        'importances': importances,
                        'stds': stds
                    })
                else:
                    print(f"   ⚠️ Fold {fold_idx}: 모델에 predict 메서드가 없습니다.")
            except Exception as e:
                print(f"   ⚠️ Fold {fold_idx} Permutation Importance 계산 실패: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print()  # 줄바꿈
        
        if len(fold_importances) == 0:
            print("   ⚠️ Permutation Importance를 계산할 수 없습니다.")
            return pd.DataFrame()
        
        # Fold별 평균 및 표준편차 계산
        all_importances = np.array([fold['importances'] for fold in fold_importances])
        all_stds = np.array([fold['stds'] for fold in fold_importances])
        
        mean_importance = np.mean(all_importances, axis=0)
        std_importance = np.std(all_importances, axis=0)
        
        # 특성 이름과 중요도 길이 확인
        if len(feature_names) != len(mean_importance):
            print(f"   ⚠️ Warning: Feature names length ({len(feature_names)}) != importance length ({len(mean_importance)})")
            if len(feature_names) < len(mean_importance):
                feature_names = feature_names + [f'Feature_{i}' for i in range(len(feature_names), len(mean_importance))]
            else:
                feature_names = feature_names[:len(mean_importance)]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_importance,
            'Std': std_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"   ✅ Permutation Importance 계산 완료!")
        print(f"   상위 10개 특성:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"     {row['Feature']:40s}: {row['Importance']:8.4f} (std: {row['Std']:.4f})")
        
        return importance_df


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
        self.ensemble_score = None
        
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
        
        # 앙상블 점수 계산 (가중치 기반으로 직접 계산)
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        for name, pred in predictions_dict.items():
            if self.weights and name in self.weights:
                ensemble_pred += self.weights[name] * pred
        
        if self.task_type == 'regression':
            self.ensemble_score = np.sqrt(mean_squared_error(y_true, ensemble_pred))
        else:
            self.ensemble_score = roc_auc_score(y_true, ensemble_pred)
    
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
        bounds = [(0.33, 1) for _ in range(len(model_names))]
        
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

