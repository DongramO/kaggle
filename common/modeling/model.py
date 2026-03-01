"""
GBDT 모델 래퍼 (CatBoost, LightGBM, XGBoost) 및 K-Fold 학습/앙상블 유틸.
"""
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Union

warnings.filterwarnings('ignore')

from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.inspection import permutation_importance

from ..utils.helpers import setup_gpu_params, check_gpu_availability


def _task_params(task_type: str, library: str) -> dict:
    """회귀/분류에 따른 loss, objective, metric 등 공통 파라미터."""
    if task_type == 'regression':
        if library == 'catboost':
            return {'loss_function': 'RMSE', 'eval_metric': 'RMSE'}
        if library == 'lightgbm':
            return {'objective': 'regression', 'metric': 'rmse'}
        return {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
    
    # classification
    if library == 'catboost':
        return {'loss_function': 'Logloss', 'eval_metric': 'Logloss'}
    if library == 'lightgbm':
        return {'objective': 'binary', 'metric': 'binary_logloss'}
    return {'objective': 'binary:logistic', 'eval_metric': 'logloss'}


class BaseModel:
    """CatBoost/LightGBM/XGBoost 공통 인터페이스."""

    def __init__(self, model_type: str, task_type: str = 'regression', random_state: int = 42):
        self.model_type = model_type
        self.task_type = task_type
        self.random_state = random_state
        self.model = None
        self.feature_importance_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        """분류 시 양성 클래스 확률(ROC AUC용). 회귀에서는 predict와 동일."""
        return self.predict(X)

    def get_feature_importance(self):
        return self.feature_importance_


class CatBoostModel(BaseModel):
    """CatBoost 래퍼. 범주형 컬럼 자동 감지 지원."""

    def __init__(self, task_type: str = 'regression', random_state: int = 42,
                 cat_features: Optional[List] = None, use_gpu: bool = False, **kwargs):
        super().__init__('catboost', task_type, random_state)
        self.cat_features = cat_features
        default = {
            'random_state': random_state,
            'verbose': False,
            'allow_writing_files': False,
            **_task_params(task_type, 'catboost'),
        }
        
        gpu_params = setup_gpu_params(use_gpu, 'catboost')
        if gpu_params:
            default.update(gpu_params)
            print(f"✅ CatBoost: GPU 사용 (Device: {gpu_params.get('devices', 'N/A')})")
        elif use_gpu:
            print("⚠️ CatBoost: GPU 불가, CPU로 학습합니다.")
        default.update(kwargs)
        self.params = default

    def _cat_indices(self, X_train):
        if not isinstance(X_train, pd.DataFrame):
            return self.cat_features
        if self.cat_features is None:
            self.cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        return [X_train.columns.get_loc(c) for c in self.cat_features if c in X_train.columns]

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        fit_params = self.params.copy()
        if 'early_stopping_rounds' in fit_params:
            kwargs['early_stopping_rounds'] = fit_params.pop('early_stopping_rounds')
        cls = CatBoostRegressor if self.task_type == 'regression' else CatBoostClassifier
        self.model = cls(**fit_params)
        fit_kw = dict(cat_features=self._cat_indices(X_train), **kwargs)
        if X_val is not None and y_val is not None:
            fit_kw['eval_set'] = (X_val, y_val)
        self.model.fit(X_train, y_train, **fit_kw)
        self.feature_importance_ = self.model.feature_importances_
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        if self.task_type == 'classification':
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """LightGBM 래퍼."""

    def __init__(self, task_type: str = 'regression', random_state: int = 42,
                 use_gpu: bool = False, **kwargs):
        super().__init__('lightgbm', task_type, random_state)
        default = {
            'random_state': random_state,
            'verbosity': -1,
            **_task_params(task_type, 'lightgbm'),
        }
        gpu_params = setup_gpu_params(use_gpu, 'lightgbm')
        if gpu_params:
            default.update(gpu_params)
            print(f"✅ LightGBM: GPU 사용 (Device: {gpu_params.get('gpu_device_id', 'N/A')})")
        elif use_gpu:
            print("⚠️ LightGBM: GPU 불가, CPU로 학습합니다.")
        default.update(kwargs)
        self.params = default

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        fit_params = self.params.copy()
        es_rounds = fit_params.pop('early_stopping_rounds', None)
        if es_rounds and X_val is not None and y_val is not None:
            kwargs.setdefault('callbacks', [])
            if not any('early_stopping' in str(type(cb)).lower() for cb in kwargs['callbacks']):
                kwargs['callbacks'].append(lgb.early_stopping(stopping_rounds=es_rounds, verbose=False))
        train_data = lgb.Dataset(X_train, label=y_train)
        valid = [train_data]
        if X_val is not None and y_val is not None:
            valid.append(lgb.Dataset(X_val, label=y_val, reference=train_data))
        self.model = lgb.train(fit_params, train_data, valid_sets=valid, valid_names=['train', 'val'][:len(valid)], **kwargs)
        self.feature_importance_ = self.model.feature_importance(importance_type='gain')
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        """LightGBM binary는 predict()가 이미 확률 반환."""
        return self.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost 래퍼. GPU 실패 시 자동으로 CPU로 전환."""

    def __init__(self, task_type: str = 'regression', random_state: int = 42,
                 use_gpu: bool = False, **kwargs):
        super().__init__('xgboost', task_type, random_state)
        default = {
            'random_state': random_state,
            'verbosity': 0,
            **_task_params(task_type, 'xgboost'),
        }
        gpu_params = setup_gpu_params(use_gpu, 'xgboost')
        default.update(gpu_params)
        self.use_gpu = use_gpu and 'device' in gpu_params
        default.update(kwargs)
        self.params = default

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        fit_params = self.params.copy()
        if 'early_stopping_rounds' in fit_params:
            kwargs['early_stopping_rounds'] = fit_params.pop('early_stopping_rounds')
        dtrain = xgb.DMatrix(X_train, label=y_train)
        watchlist = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            watchlist.append((xgb.DMatrix(X_val, label=y_val), 'val'))
        try:
            self.model = xgb.train(fit_params, dtrain, evals=watchlist, **kwargs)
        except Exception as e:
            if 'gpu' in str(e).lower() or 'cuda' in str(e).lower():
                fit_params = {**fit_params, 'tree_method': 'hist'}
                fit_params.pop('device', None)
                self.model = xgb.train(fit_params, dtrain, evals=watchlist, **kwargs)
                self.use_gpu = False
            else:
                raise
        self.feature_importance_ = np.array(list(self.model.get_score(importance_type='gain').values()))
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(xgb.DMatrix(X))

    def predict_proba(self, X):
        """XGBoost binary:logistic은 predict()가 이미 확률 반환."""
        return self.predict(X)


def _compute_score(y_true, y_pred, task_type: str, metric: str):
    """문제별 평가 지표 계산. (score, metric_name) 반환. metric='auto'면 task 기본값 사용."""
    if task_type == 'regression':
        if metric in ('auto', 'rmse'):
            return np.sqrt(mean_squared_error(y_true, y_pred)), "RMSE"
        if metric == 'mae':
            return mean_absolute_error(y_true, y_pred), "MAE"
        if metric == 'r2':
            return r2_score(y_true, y_pred), "R2"
        return np.sqrt(mean_squared_error(y_true, y_pred)), "RMSE"
    # classification
    if metric in ('auto', 'auc'):
        return roc_auc_score(y_true, y_pred), "AUC"
    if metric == 'logloss':
        return log_loss(y_true, y_pred), "LogLoss"
    if metric == 'accuracy':
        return accuracy_score(y_true, (np.asarray(y_pred) > 0.5).astype(int)), "Accuracy"
    return roc_auc_score(y_true, y_pred), "AUC"


class ModelTrainer:
    """K-Fold CV 학습, OOF 예측, 특성 중요도/permutation importance 제공. 문제별 평가·출력 형식은 scoring_metric으로 선택."""

    def __init__(self, task_type: str = 'regression', random_state: int = 42, use_gpu: bool = False,
                 scoring_metric: Optional[str] = None):
        """
        task_type: 'regression' | 'classification'
        scoring_metric: 평가·제출에 사용할 지표. None이면 'auto'(회귀=RMSE, 분류=AUC).
          회귀: 'rmse', 'mae', 'r2' / 분류: 'auc', 'logloss', 'accuracy'
        """
        self.task_type = task_type
        self.scoring_metric = (scoring_metric or 'auto').lower()
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.models = {}
        self.cv_scores = {}
        self.oof_predictions = {}
        self.best_params = {}
        if use_gpu and not check_gpu_availability():
            print("⚠️ GPU를 사용할 수 없습니다. CPU로 학습합니다.")
            self.use_gpu = False
        elif use_gpu:
            print("🚀 GPU 사용 가능: 모델 학습에 GPU를 사용합니다.")

    def _use_probability(self):
        """분류 시 확률 출력 여부. AUC/LogLoss면 확률, Accuracy면 라벨."""
        if self.task_type != 'classification':
            return False
        return self.scoring_metric in ('auto', 'auc', 'logloss')

    def _predict(self, model, X):
        """평가/제출용: scoring_metric에 따라 회귀=예측값, 분류=확률 또는 라벨."""
        if self.task_type == 'regression':
            return model.predict(X)
        return model.predict_proba(X) if self._use_probability() else model.predict(X)

    def _score(self, y_true, y_pred):
        """설정된 scoring_metric으로 (score, metric_name) 반환."""
        return _compute_score(y_true, y_pred, self.task_type, self.scoring_metric)

    def _fit_fold(self, model, model_type: str, X_train, y_train, X_val, y_val, n_rounds: int):
        """Fold 1개 학습. model_type별 early stopping 호출 통일."""
        if model_type == 'lightgbm':
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, num_boost_round=n_rounds,
                      callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
        elif model_type == 'xgboost':
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, num_boost_round=n_rounds,
                      early_stopping_rounds=100, verbose_eval=False)
        else:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=100)

    def create_model(self, model_type: str, cat_features: Optional[List] = None, **kwargs):
        """model_type에 따라 CatBoost/LightGBM/XGBoost 인스턴스 생성."""
        kwargs_clean = {k: v for k, v in kwargs.items() if k != 'random_state'}
        common = dict(task_type=self.task_type, random_state=self.random_state, use_gpu=self.use_gpu, **kwargs_clean)
        if model_type == 'catboost':
            return CatBoostModel(cat_features=cat_features, **common)
        if model_type == 'lightgbm':
            return LightGBMModel(**common)
        if model_type == 'xgboost':
            return XGBoostModel(**common)
        raise ValueError(f"Unknown model_type: {model_type}")
    
    def train_with_cv(self, X, y, model_type: str, n_folds: int = 5,
                      cat_features: Optional[List] = None, **model_params):
        """K-Fold CV로 학습 후 OOF 예측·CV 점수 반환."""
        kf = (StratifiedKFold if self.task_type == 'classification' else KFold)(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )
        models, oof_preds, fold_scores = [], np.zeros(len(X)), []
        n_rounds = model_params.get('num_boost_round', 1000)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
            print(f"\n📊 Fold {fold}/{n_folds} 학습 중...")
            X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
            X_val = X.iloc[val_idx] if isinstance(X, pd.DataFrame) else X[val_idx]
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            y_val = y.iloc[val_idx] if isinstance(y, pd.Series) else y[val_idx]

            model = self.create_model(model_type, cat_features=cat_features, **model_params)
            self._fit_fold(model, model_type, X_train, y_train, X_val, y_val, n_rounds)

            val_pred = self._predict(model, X_val)
            oof_preds[val_idx] = val_pred
            score, metric = self._score(y_val, val_pred)
            print(f"  Fold {fold} {metric}: {score:.4f}")
            fold_scores.append(score)
            models.append(model)

        cv_score, _ = self._score(y, oof_preds)
        print(f"\n✅ CV {metric}: {cv_score:.4f} (std: {np.std(fold_scores):.4f})")
        self.models[model_type] = models
        self.oof_predictions[model_type] = oof_preds
        self.cv_scores[model_type] = {'mean': cv_score, 'std': np.std(fold_scores), 'fold_scores': fold_scores}
        return {'models': models, 'oof_predictions': oof_preds, 'cv_score': cv_score, 'fold_scores': fold_scores}
    
    def predict_test(self, X_test, model_type: str):
        """테스트 예측. scoring_metric에 따라 회귀=예측값, 분류=확률 또는 라벨. K-Fold 평균."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        models = self.models[model_type]
        predictions = np.array([self._predict(m, X_test) for m in models])
        return predictions.mean(axis=0)
    
    def get_feature_importance(self, model_type: str, feature_names: List[str],
                              average_across_folds: bool = True) -> pd.DataFrame:
        """Fold별 특성 중요도 수집 후 평균 또는 Fold별 컬럼으로 반환."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        models = self.models[model_type]
        fold_importances = []
        for fold_idx, model in enumerate(models):
            try:
                importance = model.get_feature_importance()
                if importance is None:
                    continue
                if len(importance) != len(feature_names):
                    if len(importance) < len(feature_names):
                        importance = np.pad(importance, (0, len(feature_names) - len(importance)), 'constant')
                    else:
                        importance = importance[:len(feature_names)]
                if np.sum(importance) > 0:
                    importance = importance / np.sum(importance)
                fold_importances.append(importance)
            except Exception as e:
                print(f"⚠️ Fold {fold_idx + 1} feature importance 추출 실패: {e}")
                continue
        if not fold_importances:
            return pd.DataFrame()
        fold_importances = np.array(fold_importances)
        if average_across_folds:
            return pd.DataFrame({
                'Feature': feature_names,
                'Importance': np.mean(fold_importances, axis=0),
                'Std': np.std(fold_importances, axis=0),
            }).sort_values('Importance', ascending=False)
        fold_cols = [f'Fold_{i+1}' for i in range(len(fold_importances))]
        df = pd.DataFrame({'Feature': feature_names, **{c: fold_importances[i] for i, c in enumerate(fold_cols)}})
        df['Mean'] = df[fold_cols].mean(axis=1)
        df['Std'] = df[fold_cols].std(axis=1)
        return df.sort_values('Mean', ascending=False)
    
    def _default_sklearn_scoring(self):
        """permutation_importance용 sklearn scoring 문자열. scoring_metric과 일치."""
        if self.task_type == 'regression':
            return {'rmse': 'neg_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}.get(self.scoring_metric, 'neg_mean_squared_error')
        return {'auc': 'roc_auc', 'logloss': 'neg_log_loss', 'accuracy': 'accuracy'}.get(self.scoring_metric, 'roc_auc')

    def get_permutation_importance(self, model_type: str, X: pd.DataFrame, y: Union[pd.Series, np.ndarray],
                                   feature_names: Optional[List[str]] = None,
                                   n_repeats: int = 10, random_state: Optional[int] = None,
                                   scoring: Optional[str] = None, n_jobs: int = -1) -> pd.DataFrame:
        """Fold별 permutation importance 계산 후 평균 반환. scoring 미지정 시 scoring_metric과 동일하게 선택."""
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found. Train the model first.")
        feature_names = feature_names or X.columns.tolist()
        models = self.models[model_type]
        scoring = scoring or self._default_sklearn_scoring()
        rs = random_state if random_state is not None else self.random_state
        fold_importances = []

        print(f"\n📊 {model_type.upper()} Permutation Importance (n_repeats={n_repeats}, folds={len(models)})")
        for fold_idx, model in enumerate(models, 1):
            print(f"   Fold {fold_idx}/{len(models)} 처리 중...", end='\r')
            try:
                if not hasattr(model, 'predict'):
                    continue
                perm = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=rs, scoring=scoring, n_jobs=n_jobs)
                imp = perm.importances_mean
                if scoring.startswith('neg_'):
                    imp = -imp
                fold_importances.append({'importances': imp, 'stds': perm.importances_std})
            except Exception as e:
                print(f"   ⚠️ Fold {fold_idx} 실패: {e}")
        print()
        if not fold_importances:
            return pd.DataFrame()
        all_imp = np.array([f['importances'] for f in fold_importances])
        mean_imp = np.mean(all_imp, axis=0)
        std_imp = np.std(all_imp, axis=0)
        if len(feature_names) != len(mean_imp):
            feature_names = feature_names[:len(mean_imp)] if len(feature_names) > len(mean_imp) else feature_names + [f'Feature_{i}' for i in range(len(feature_names), len(mean_imp))]
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': mean_imp, 'Std': std_imp}).sort_values('Importance', ascending=False)
        print("   상위 10개:", importance_df.head(10).to_string(index=False))
        return importance_df


class EnsembleModel:
    """OOF 예측에 대한 가중 평균/단순 평균/스태킹(가중치 최적화) 앙상블. scoring_metric에 따라 평가 지표 통일."""

    def __init__(self, task_type: str = 'regression', scoring_metric: Optional[str] = None):
        self.task_type = task_type
        self.scoring_metric = (scoring_metric or 'auto').lower()
        self.weights = None
        self.ensemble_score = None

    def _ensemble_score(self, y_true: np.ndarray, ensemble_pred: np.ndarray) -> float:
        """설정된 지표로 앙상블 점수 계산 (최소화 방향: RMSE/LogLoss는 그대로, AUC/Accuracy/R2는 음수)."""
        s, _ = _compute_score(y_true, ensemble_pred, self.task_type, self.scoring_metric)
        if self.scoring_metric in ('r2', 'auc', 'accuracy'):
            return -s
        return s

    def fit(self, predictions_dict: Dict[str, np.ndarray], y_true: np.ndarray,
            method: str = 'weighted_average', optimize: bool = True):
        self.method = method
        
        if method == 'simple_average':
            self.weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict.keys()}
        
        elif method == 'weighted_average':
            if optimize:
                self.weights = self._optimize_weights(predictions_dict, y_true)
            else:
                self.weights = {name: 1.0 / len(predictions_dict) for name in predictions_dict.keys()}
        
        elif method == 'stacking':
            self.weights = self._optimize_weights(predictions_dict, y_true)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        for name, pred in predictions_dict.items():
            if self.weights and name in self.weights:
                ensemble_pred += self.weights[name] * pred
        
        self.ensemble_score, _ = _compute_score(y_true, ensemble_pred, self.task_type, self.scoring_metric)
    
    def _optimize_weights(self, predictions_dict: Dict[str, np.ndarray], 
                         y_true: np.ndarray) -> Dict[str, float]:
        """가중치 최적화 (scipy.optimize 사용). 목적은 항상 최소화."""
        from scipy.optimize import minimize
        
        model_names = list(predictions_dict.keys())
        predictions_list = [predictions_dict[name] for name in model_names]
        
        def objective(weights):
            weighted_pred = np.zeros_like(predictions_list[0])
            for pred, weight in zip(predictions_list, weights):
                weighted_pred += weight * pred
            return self._ensemble_score(y_true, weighted_pred)
        
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
        """저장된 가중치로 앙상블 예측 반환."""
        if self.weights is None:
            raise ValueError("Ensemble model must be fitted first")
        
        ensemble_pred = np.zeros_like(list(predictions_dict.values())[0])
        
        for name, pred in predictions_dict.items():
            ensemble_pred += self.weights[name] * pred
        
        return ensemble_pred


def evaluate_model(y_true, y_pred, task_type: str = 'regression'):
    """task_type에 따라 RMSE/MAE/R2 또는 AUC/Accuracy/LogLoss 반환."""
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

