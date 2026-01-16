"""
모델 학습 및 앙상블 실행 스크립트
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.metrics import mean_squared_error, roc_auc_score
from preprocess.encoder import fit_encoder, transform_with_encoder, one_hot_encode, ordinal_encode
from preprocess.feature_engineering import clip_outliers, create_interaction_features, create_ratio_features, create_categorical_interactions
from eda import analyze_feature_importance, analyze_permutation_importance, analyze_high_error_samples
from eda.dataload import load_data
from modeling.model import ModelTrainer, EnsembleModel, evaluate_model
from modeling.hyperparameter import (
    HyperparameterOptimizer, 
    save_hyperparameters, 
    load_hyperparameters,
    optimize_hyperparameters
)
# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# eda 모듈 import (Feature Importance 분석)
def prepare_data(df_train, df_test, target_col='exam_score', 
                 use_feature_engineering=True, encoding_config=None):
    # 데이터 복사
    train = df_train.copy()
    test = df_test.copy()
    
    # ID 컬럼 제거
    if 'id' in train.columns:
        train = train.drop(columns=['id'])
    if 'id' in test.columns:
        test = test.drop(columns=['id'])
    
    # 타겟 분리
    y_train = train[target_col]
    X_train = train.drop(columns=[target_col])
    X_test = test.copy()
    
    # 컬럼 타입 분류 (Feature Engineering 전 원본 컬럼 정보 저장)
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    original_categorical_cols_before_fe = categorical_cols.copy()  # Feature Engineering 전 범주형 컬럼 저장
    
    print(f"\n📊 데이터 정보:")
    print(f"  학습 데이터 크기: {X_train.shape}")
    print(f"  테스트 데이터 크기: {X_test.shape}")
    print(f"  수치형 컬럼: {len(numeric_cols)}개")
    print(f"  범주형 컬럼: {len(categorical_cols)}개")
    
    # Feature Engineering 적용 (2단계로 분리)
    if use_feature_engineering:
        print("\n🔧 Feature Engineering 적용 중...")
        # 간단한 설정 예제 (필요에 따라 수정)
        config = {
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
                'operations': [
                    'multiply',
                    'add',
                ]
            },
            
            'create_interactions_after_encoding': {
                'flag': True,
                'feature_pairs': [
                    ('class_attendance', 'sleep_quality_encoded'),
                    ('study_hours', 'sleep_quality_encoded'),
                    ('study_hours', 'facility_rating_encoded'),
                    ('study_hours', 'exam_difficulty_encoded'),
                ],
                'operations': [
                    'multiply',
                    'multiply',
                    'multiply',
                    'multiply',
                ]
            },
        
            'create_frequency': {
                'flag': False
            },
            
            'create_ratios': {
                'flag': True,
                'numerator_cols': [
                    'study_hours',
                    'class_attendance',
                ],
                'denominator_cols': [
                    'study_hours_add_sleep_hours',
                    'study_hours',
                ],
                'ratio_feature_names': "_ratio"
            },
            'create_categorical_interactions': {
                'flag': True,
                'categorical_pairs': [
                    ('facility_rating', 'exam_difficulty'),
                    ('sleep_quality', 'exam_difficulty'),
                ],
                'separator': '_'
            }
        }
        
        # ========== Feature Engineering 1단계: 인코딩 전 ==========
        print("  📌 1단계: 인코딩 전 Feature Engineering")
        
        # 이상치 클리핑
        clip_cfg = config.get('clip_outliers', {})
        if clip_cfg.get('flag', False):
            clip_rules = clip_cfg.get('clip_rules', None)
            X_train = clip_outliers(X_train, numeric_cols, clip_rules)
            X_test = clip_outliers(X_test, numeric_cols, clip_rules)
        
        # 범주형 조합 특성 생성 (인코딩 전에만 가능)
        # cat_interaction_cfg = config.get('create_categorical_interactions', {})
        # if cat_interaction_cfg.get('flag', False):
        #     categorical_pairs = cat_interaction_cfg.get('categorical_pairs', [])
        #     separator = cat_interaction_cfg.get('separator', '_')
        #     X_train = create_categorical_interactions(X_train, categorical_pairs, separator)
        #     X_test = create_categorical_interactions(X_test, categorical_pairs, separator)
        
        # 인코딩 전 수치형 조합 (인코딩된 컬럼을 사용하지 않는 것들)
        interaction_before_cfg = config.get('create_interactions_before_encoding', {})
        if interaction_before_cfg.get('flag', False):
            feature_pairs = interaction_before_cfg.get('feature_pairs', [])
            operations = interaction_before_cfg.get('operations', [])
            X_train = create_interaction_features(X_train, feature_pairs, operations)
            X_test = create_interaction_features(X_test, feature_pairs, operations)

        # 업데이트된 컬럼 리스트
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 범주형 인코딩 (LightGBM, XGBoost용)
    encoder = None

    original_categorical_cols = original_categorical_cols_before_fe.copy()  # CatBoost용 원본 범주형 컬럼 저장 (FE 전)
    encoded_cols_tag = '_encoded'  # 인코딩된 컬럼 태그
    

    if len(original_categorical_cols_before_fe) > 0:
        categorical_cols_for_encoding = [col for col in original_categorical_cols_before_fe if col in X_train.columns]
    else:

        categorical_cols_for_encoding = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(categorical_cols_for_encoding) > 0:
        print("\n🔤 범주형 인코딩 적용 중...")
        print(f"  원본 범주형 컬럼: {len(categorical_cols_for_encoding)}개")
        print(f"  CatBoost는 원본 범주형 컬럼 사용, LightGBM/XGBoost는 인코딩된 컬럼 사용")
        
        if encoding_config is not None:
            # encoding_config가 있으면 onehot_cols와 ordinal_cols로 분리하여 인코딩
            onehot_cols = encoding_config.get('onehot_cols', [])
            ordinal_cols = encoding_config.get('ordinal_cols', [])
            onehot_params = encoding_config.get('onehot_params', {})
            ordinal_params = encoding_config.get('ordinal_params', {})
            
            # 실제로 존재하는 컬럼만 필터링
            onehot_cols = [col for col in onehot_cols if col in X_train.columns]
            ordinal_cols = [col for col in ordinal_cols if col in X_train.columns]
            
            print(f"  OneHot 인코딩: {onehot_cols} (개수: {len(onehot_cols)})")
            print(f"  Ordinal 인코딩: {ordinal_cols} (개수: {len(ordinal_cols)})")
            
            onehot_feature_count = 0
            # OneHot 인코딩: 원본 유지하고 인코딩된 컬럼 추가
            if len(onehot_cols) > 0:
                onehot_params_clean = {k: v for k, v in onehot_params.items() if k not in ['drop_original']}
                onehot_params_clean.setdefault('sparse', False)
                
                # OneHot 인코더 학습
                onehot_encoder = fit_encoder(
                    X_train,
                    categorical_cols=onehot_cols,
                    encoding_type='onehot',
                    **{k: v for k, v in onehot_params_clean.items() if k != 'sparse'}
                )
                
                # 학습 데이터 변환
                train_onehot_array = onehot_encoder.transform(X_train[onehot_cols])
                if hasattr(train_onehot_array, 'toarray'):
                    train_onehot_array = train_onehot_array.toarray()
                onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_cols)
                onehot_feature_count = len(onehot_feature_names)
                
                # 테스트 데이터 변환
                test_onehot_array = onehot_encoder.transform(X_test[onehot_cols])
                if hasattr(test_onehot_array, 'toarray'):
                    test_onehot_array = test_onehot_array.toarray()
                
                # 인코딩된 컬럼 추가
                for idx, feature_name in enumerate(onehot_feature_names):
                    encoded_col_name = f"{feature_name}{encoded_cols_tag}"
                    X_train[encoded_col_name] = train_onehot_array[:, idx]
                    X_test[encoded_col_name] = test_onehot_array[:, idx]
            
            # Ordinal 인코딩: 원본 유지하고 인코딩된 컬럼 추가
            if len(ordinal_cols) > 0:
                ordinal_params_clean = {k: v for k, v in ordinal_params.items() if k != 'drop_original'}
                
                # Ordinal 인코더 학습
                ordinal_encoder = fit_encoder(
                    X_train,
                    categorical_cols=ordinal_cols,
                    category_orders={
                        'sleep_quality': ['poor', 'average', 'good'],
                        'facility_rating': ['low', 'medium', 'high'],
                        'exam_difficulty': ['easy', 'moderate', 'hard']
                    },
                    encoding_type='ordinal',
                    **ordinal_params_clean
                )
                
                # 학습 및 테스트 데이터 변환
                train_ordinal_array = ordinal_encoder.transform(X_train[ordinal_cols])
                test_ordinal_array = ordinal_encoder.transform(X_test[ordinal_cols])
                
                # 인코딩된 컬럼 추가
                for idx, col in enumerate(ordinal_cols):
                    encoded_col_name = f"{col}{encoded_cols_tag}"
                    X_train[encoded_col_name] = train_ordinal_array[:, idx]
                    X_test[encoded_col_name] = test_ordinal_array[:, idx]
            
            total_encoded = onehot_feature_count + len(ordinal_cols)
            
            print(f"  인코딩된 컬럼 추가: OneHot {onehot_feature_count}개 + Ordinal {len(ordinal_cols)}개 = 총 {total_encoded}개")
        
        
        print(f"  원본 범주형 컬럼 유지: {len(categorical_cols_for_encoding)}개 (CatBoost용)")
    
    # ========== Feature Engineering 2단계: 인코딩 후 ==========
    if use_feature_engineering:
        print("\n  📌 2단계: 인코딩 후 Feature Engineering")
        
        # 인코딩 후 수치형 조합 (인코딩된 컬럼을 사용할 수 있는 것들)
        interaction_after_cfg = config.get('create_interactions_after_encoding', {})
        if interaction_after_cfg.get('flag', False):
            feature_pairs = interaction_after_cfg.get('feature_pairs', [])
            operations = interaction_after_cfg.get('operations', [])
       

            X_train = create_interaction_features(X_train, feature_pairs, operations)
            X_test = create_interaction_features(X_test, feature_pairs, operations)
        
        # 비율 특성 생성 (인코딩 후에도 가능)
        ratio_cfg = config.get('create_ratios', {})

        if ratio_cfg.get('flag', False):
            numerator_cols = ratio_cfg.get('numerator_cols', [])
            denominator_cols = ratio_cfg.get('denominator_cols', [])
            feature_names = ratio_cfg.get('ratio_feature_names', None)
       
            X_train = create_ratio_features(X_train, numerator_cols, denominator_cols, feature_names)
            X_test = create_ratio_features(X_test, numerator_cols, denominator_cols, feature_names)
          
        
        # 최종 컬럼 리스트 업데이트
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 결측치 처리
    for col in numeric_cols:
        if col in X_train.columns and X_train[col].isnull().sum() > 0:
            mean_val = X_train[col].mean()
            X_train[col].fillna(mean_val, inplace=True)
            if col in X_test.columns:
                X_test[col].fillna(mean_val, inplace=True)
    
    print("✅ 데이터 준비 완료!")

    return X_train, y_train, X_test, original_categorical_cols, numeric_cols, encoder, encoded_cols_tag


def train_models(X_train, y_train, X_test, categorical_cols, task_type='regression', 
                 n_folds=5, random_state=42, use_optuna=False, n_trials=50,
                 use_saved_params=None, params_filepath=None, encoded_cols_tag='_encoded',
                 use_gpu=False, optuna_sample_size=None):

    trainer = ModelTrainer(task_type=task_type, random_state=random_state, use_gpu=use_gpu)
    
    # 저장된 파라미터 자동 감지 (use_saved_params가 None인 경우)
    # 단, use_optuna=True인 경우는 Optuna 최적화를 우선하므로 use_saved_params=False로 설정
    if use_optuna:
        # Optuna 최적화를 실행하는 경우, 저장된 파라미터 사용하지 않음
        use_saved_params = False
        print(f"\n🔍 Optuna 최적화 모드: 저장된 파라미터 무시하고 최적화 실행")
    elif use_saved_params is None and params_filepath and os.path.exists(params_filepath):
        use_saved_params = True
        print(f"\n✅ 저장된 하이퍼파라미터 파일 자동 감지: {params_filepath}")
    elif use_saved_params is None:
        use_saved_params = False
    
    # 모델별로 사용할 컬럼 준비
    # CatBoost: 원본 범주형 컬럼 사용 (인코딩된 컬럼 제외)
    # LightGBM/XGBoost: 인코딩된 컬럼 사용 (원본 범주형 컬럼 제외)
    print('X_train.columns:', X_train.columns)
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    
    def get_features_for_model(model_type):
        """모델 타입에 따라 사용할 컬럼 반환"""
        if model_type == 'catboost':
            # CatBoost: 원본 범주형 컬럼 + 조합 범주형 컬럼 + 수치형 컬럼 (인코딩된 컬럼 제외)
            exclude_cols = encoded_cols
            return [col for col in X_train.columns if col not in exclude_cols]
        else:
            # LightGBM/XGBoost: 인코딩된 컬럼 + 수치형 컬럼만 사용
            # 모든 범주형 컬럼 제외 (원본 + 조합 특성 모두)
            # 실제 데이터프레임에서 모든 범주형 컬럼을 동적으로 찾아서 제외
            all_categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            exclude_cols = all_categorical_cols
            return [col for col in X_train.columns if col not in exclude_cols]
       
    
    # 하이퍼파라미터 설정
    if use_optuna or use_saved_params:
        # Optuna 최적화 또는 저장된 파라미터 사용
        best_params = optimize_hyperparameters(
            X_train, y_train, categorical_cols,
            task_type=task_type,
            n_trials=n_trials,
            random_state=random_state,
            use_saved_params=use_saved_params,
            params_filepath=params_filepath,
            encoded_cols_tag=encoded_cols_tag,
            use_gpu=use_gpu,
            sample_size=optuna_sample_size
        )
        
        # 모델별 파라미터 변환
        model_configs = {}
        
        # CatBoost 파라미터 변환
        if 'catboost' in best_params:
            cb_params = best_params['catboost'].copy()
            cb_params['iterations'] = cb_params.get('iterations', 1000)
            cb_params.setdefault('random_state', random_state)
            cb_params.setdefault('verbose', False)
            cb_params.setdefault('allow_writing_files', False)
            model_configs['catboost'] = cb_params
        
        # LightGBM 파라미터 변환
        if 'lightgbm' in best_params:
            lgb_params = best_params['lightgbm'].copy()
            lgb_params.setdefault('random_state', random_state)
            lgb_params.setdefault('verbosity', -1)
            model_configs['lightgbm'] = lgb_params
        
        # XGBoost 파라미터 변환
        if 'xgboost' in best_params:
            xgb_params = best_params['xgboost'].copy()
            xgb_params.setdefault('random_state', random_state)
            xgb_params.setdefault('verbosity', 0)
            model_configs['xgboost'] = xgb_params
        
        trainer.best_params = best_params
    
    # 각 모델 학습
    test_predictions = {}
    model_features = {}  # 각 모델이 사용한 컬럼 리스트 저장
    
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        print(f"\n{'='*60}")
        print(f"🚀 {model_type.upper()} 모델 학습 시작")
        print(f"{'='*60}")
        
        # 모델별로 사용할 컬럼 선택
        feature_cols = get_features_for_model(model_type)
        model_features[model_type] = feature_cols  # 사용한 컬럼 저장
        X_train_model = X_train[feature_cols].copy()
        X_test_model = X_test[feature_cols].copy()
        
        # CatBoost는 범주형 특성을 자동 처리하므로 cat_features 전달
        # LightGBM/XGBoost는 인코딩된 컬럼만 사용하므로 cat_features는 None
        if model_type == 'catboost':
            # 실제 데이터프레임에 있는 모든 범주형 컬럼 필터링 (조합 특성 포함)
            # 이렇게 하면 범주형 조합 특성도 자동으로 포함됨
            actual_categorical_cols = X_train_model.select_dtypes(include=['object', 'category']).columns.tolist()
            actual_numeric_cols = X_train_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_features = actual_categorical_cols
            print(f"  사용 컬럼: 범주형 {len(actual_categorical_cols)}개 (원본 + 조합 특성 포함) + 수치형 {actual_numeric_cols}")
            print(f"    총 피처 수: {len(feature_cols)}개")
            print(f"    범주형 컬럼: {actual_categorical_cols}")
        else:
            cat_features = None
            print(f"  사용 컬럼: 범주형 {len(actual_categorical_cols)}개 (원본 + 조합 특성 포함) + 수치형 {actual_numeric_cols}")
            print(f"    총 피처 수: {len(feature_cols)}개")
            print(f"  사용 컬럼: 인코딩된 컬럼 {len(encoded_cols)}개 + 수치형 (원본 범주형 컬럼 제외)")
        
        print('model_configs[model_type]:', model_configs[model_type])
        # K-Fold 교차 검증으로 학습
        result = trainer.train_with_cv(
            X_train_model, y_train,
            model_type=model_type,
            n_folds=n_folds,
            cat_features=cat_features,
            **model_configs[model_type]
        )
        
        # 테스트 데이터 예측
        test_pred = trainer.predict_test(X_test_model, model_type)
        test_predictions[model_type] = test_pred
        
        print(f"✅ {model_type.upper()} 학습 완료!")
    
    return {
        'trainer': trainer,
        'test_predictions': test_predictions,
        'oof_predictions': trainer.oof_predictions,
        'best_params': trainer.best_params,
        'model_features': model_features
    }


def create_ensemble(oof_predictions, y_train, test_predictions, task_type='regression'):
 
    print(f"\n{'='*60}")
    print("🎯 앙상블 모델 생성")
    print(f"{'='*60}")
    
    # OOF 예측으로 가중치 최적화
    ensemble = EnsembleModel(task_type=task_type)
    ensemble.fit(
        oof_predictions,
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        method='weighted_average',
        optimize=True
    )
    
    # 테스트 데이터 앙상블 예측
    ensemble_pred = ensemble.predict(test_predictions)
    
    print("✅ 앙상블 예측 완료!")
    
    return ensemble_pred, ensemble


def main(use_optuna=False, n_trials=50, use_saved_params=None, 
         params_filepath=None, use_gpu=False,
         optuna_sample_size=None, encoding_config=None,
         use_permutation_importance=False):

    print("="*60)
    print("🚀 모델 학습 및 앙상블 시작")
    print("="*60)
    
    # params_filepath 기본값 설정
    if params_filepath is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
        params_filepath = os.path.join(base_dir, 'best_hyperparameters.json')
    
    # 저장된 파라미터 파일이 있고 use_saved_params가 명시되지 않았으면 자동으로 사용
    if use_saved_params is False and params_filepath and os.path.exists(params_filepath):
        print(f"\n📂 저장된 하이퍼파라미터 파일 발견: {params_filepath}")
    elif use_saved_params is True and params_filepath and os.path.exists(params_filepath):
        print(f"\n✅ 저장된 하이퍼파라미터 사용: {params_filepath}")
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df_train, df_test, df_sub = load_data()
    
    # 데이터 준비
    X_train, y_train, X_test, categorical_cols, numeric_cols, encoder, encoded_cols_tag = prepare_data(
        df_train, df_test,
        target_col='exam_score',
        use_feature_engineering=True,
        encoding_config=encoding_config
    )
    
    # 모델 학습
    results = train_models(
        X_train, y_train, X_test, categorical_cols,  # CatBoost용 원본 범주형 컬럼
        task_type='regression',
        n_folds=5,
        random_state=42,
        use_optuna=use_optuna,
        n_trials=n_trials,
        use_saved_params=use_saved_params,
        params_filepath=params_filepath,
        encoded_cols_tag=encoded_cols_tag,
        use_gpu=use_gpu,
        optuna_sample_size=optuna_sample_size
    )
    
    # Feature Importance 분석
    if analyze_feature_importance is not None:
        print(f"\n{'='*60}")
        print("📊 Feature Importance 분석 시작")
        print(f"{'='*60}")
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        feature_importance_dir = os.path.join(base_dir, 'feature_importance_results')
        
       
        feature_importances = analyze_feature_importance(
            trainer=results['trainer'],
            X_train=X_train,
            categorical_cols=categorical_cols,
            encoded_cols_tag=encoded_cols_tag,
            top_n=30,
            save_dir=feature_importance_dir
        )
        results['feature_importances'] = feature_importances
     
    else:
        results['feature_importances'] = None
    
    # Permutation Importance 분석
    if use_permutation_importance and analyze_permutation_importance is not None:
        print(f"\n{'='*60}")
        print("📊 Permutation Importance 분석 시작")
        print(f"{'='*60}")
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        feature_importance_dir = os.path.join(base_dir, 'feature_importance_results')
        
        permutation_importances = analyze_permutation_importance(
            trainer=results['trainer'],
            X_train=X_train,
            y_train=y_train,
            categorical_cols=categorical_cols,
            encoded_cols_tag=encoded_cols_tag,
            top_n=30,
            n_repeats=10,
            save_dir=feature_importance_dir
        )
        results['permutation_importances'] = permutation_importances
    else:
        if use_permutation_importance:
            print(f"\n⚠️ Permutation Importance 분석을 요청했지만 analyze_permutation_importance를 사용할 수 없습니다.")
        results['permutation_importances'] = None
    
    # 앙상블 생성
    ensemble_pred, ensemble = create_ensemble(
        results['oof_predictions'],
        y_train,
        results['test_predictions'],
        task_type='regression'
    )
    
    # 앙상블 OOF 예측 계산 (오차 분석용)
    ensemble_oof_pred = np.zeros_like(list(results['oof_predictions'].values())[0])
    for name, pred in results['oof_predictions'].items():
        if hasattr(ensemble, 'weights') and ensemble.weights and name in ensemble.weights:
            ensemble_oof_pred += ensemble.weights[name] * pred
    
    # 오차가 큰 샘플 분석
    if analyze_high_error_samples is not None:
        print(f"\n{'='*60}")
        print("📊 오차가 큰 샘플 분석 시작")
        print(f"{'='*60}")
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        error_analysis_dir = os.path.join(base_dir, 'error_analysis_results')
        
        # 원본 학습 데이터 로드 (오차 분석용)
        df_train_original, _, _ = load_data()
        
        # 오차 분석 수행
        error_samples = analyze_high_error_samples(
            trainer=results['trainer'],
            X_train=df_train_original.drop(columns=['exam_score'] if 'exam_score' in df_train_original.columns else []),
            y_train=y_train,
            ensemble_pred=ensemble_oof_pred,
            top_n=100,
            error_threshold=None,  # 자동 설정
            save_dir=error_analysis_dir
        )
        results['error_samples'] = error_samples
    else:
        results['error_samples'] = None
    
    # 제출 파일 생성
    print("\n📝 제출 파일 생성 중...")
    submission = pd.DataFrame({
        'id': df_sub['id'],
        'exam_score': ensemble_pred
    })
    
    # 프로젝트 루트에 저장
    base_dir = os.path.dirname(os.path.dirname(__file__))
    submission_path = os.path.join(base_dir, 'submission.csv')
    
    submission.to_csv(submission_path, index=False)
    print(f"✅ 제출 파일 저장 완료: {submission_path}")
    print(f"   파일 크기: {os.path.getsize(submission_path) / 1024 / 1024:.2f} MB")
    print(f"   예측 개수: {len(submission)}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 학습 결과 요약")
    print(f"{'='*60}")
    
    trainer = results['trainer']
    summary_lines = []
    summary_lines.append("="*60)
    summary_lines.append("📊 학습 결과 요약")
    summary_lines.append("="*60)
    summary_lines.append(f"\n실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append(f"GPU 사용: {use_gpu}")
    summary_lines.append(f"\n모델별 CV Score:")
    
    for model_type, scores in trainer.cv_scores.items():
        model_summary = f"\n{model_type.upper()}:"
        model_summary += f"\n  CV Score: {scores['mean']:.4f} (std: {scores['std']:.4f})"
        if 'fold_scores' in scores:
            model_summary += f"\n  Fold Scores: {[f'{s:.4f}' for s in scores['fold_scores']]}"
        # 사용한 컬럼 리스트 추가
        if 'model_features' in results and model_type in results['model_features']:
            features = results['model_features'][model_type]
            model_summary += f"\n  사용 컬럼 수: {len(features)}개"
            model_summary += f"\n  사용 컬럼 리스트: {features}"
        print(model_summary)
        summary_lines.append(model_summary)
    
    # 앙상블 정보 추가
    summary_lines.append(f"\n앙상블 가중치:")
    if hasattr(ensemble, 'weights') and ensemble.weights:
        for name, weight in ensemble.weights.items():
            weight_info = f"  {name}: {weight:.4f}"
            print(weight_info)
            summary_lines.append(weight_info)
    else:
        summary_lines.append("  (가중치 정보 없음)")
    
    # 앙상블 최종 점수 추가
    if hasattr(ensemble, 'ensemble_score') and ensemble.ensemble_score is not None:
        if ensemble.task_type == 'regression':
            ensemble_info = f"\n앙상블 최종 RMSE: {ensemble.ensemble_score:.4f}"
        else:
            ensemble_info = f"\n앙상블 최종 AUC: {ensemble.ensemble_score:.4f}"
        print(ensemble_info)
        summary_lines.append(ensemble_info)
    else:
        # ensemble_score가 없는 경우 직접 계산
        ensemble_oof_pred = np.zeros_like(list(results['oof_predictions'].values())[0])
        for name, pred in results['oof_predictions'].items():
            if hasattr(ensemble, 'weights') and ensemble.weights and name in ensemble.weights:
                ensemble_oof_pred += ensemble.weights[name] * pred
        
        if ensemble.task_type == 'regression':
            ensemble_score = np.sqrt(mean_squared_error(y_train.values if isinstance(y_train, pd.Series) else y_train, ensemble_oof_pred))
            ensemble_info = f"\n앙상블 최종 RMSE: {ensemble_score:.4f}"
        else:
            ensemble_score = roc_auc_score(y_train.values if isinstance(y_train, pd.Series) else y_train, ensemble_oof_pred)
            ensemble_info = f"\n앙상블 최종 AUC: {ensemble_score:.4f}"
        print(ensemble_info)
        summary_lines.append(ensemble_info)
    
    # 인코딩 설정 정보 추가
    if encoding_config:
        summary_lines.append(f"\n인코딩 설정:")
        summary_lines.append(f"  One-Hot 컬럼: {encoding_config.get('onehot_cols', [])}")
        summary_lines.append(f"  Ordinal 컬럼: {encoding_config.get('ordinal_cols', [])}")
    
    # 제출 파일 정보
    summary_lines.append(f"\n제출 파일 정보:")
    summary_lines.append(f"  경로: {submission_path}")
    summary_lines.append(f"  파일 크기: {os.path.getsize(submission_path) / 1024 / 1024:.2f} MB")
    summary_lines.append(f"  예측 개수: {len(submission)}")
    summary_lines.append(f"  예측값 범위: [{submission['exam_score'].min():.2f}, {submission['exam_score'].max():.2f}]")
    summary_lines.append(f"  예측값 평균: {submission['exam_score'].mean():.2f}")
    summary_lines.append(f"  예측값 표준편차: {submission['exam_score'].std():.2f}")
    
    summary_lines.append(f"\n{'='*60}")
    summary_lines.append("✅ 모든 작업 완료!")
    summary_lines.append("="*60)
    
    # 결과 요약 파일 저장 (누적 형태)
    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(base_dir, 'training_summary.txt')
    
    # 기존 파일이 있으면 읽어서 내용 유지
    existing_content = ""
    execution_count = 0
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            # 기존 실행 기록 개수 계산 (구분선 기준)
            execution_count = existing_content.count("="*60 + "\n📊 학습 결과 요약")
        except Exception as e:
            print(f"⚠️ 기존 요약 파일 읽기 실패: {e}")
    
    # 새로운 내용을 기존 내용 뒤에 추가
    with open(summary_path, 'a', encoding='utf-8') as f:
        if existing_content:
            # 기존 내용이 있으면 구분선 추가
            f.write("\n\n" + "\n" + "="*80 + "\n")
            f.write("="*80 + "\n")
            f.write(f"새로운 실행 기록 #{execution_count + 1}\n")
            f.write(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
            f.write("="*80 + "\n\n")
        f.write(summary_text)
        f.write("\n")
    
    print(f"\n📝 학습 결과 요약이 {summary_path}에 추가되었습니다.")
    if existing_content:
        print(f"   (총 {execution_count + 1}개의 실행 기록이 저장되어 있습니다)")
    
    # 하이퍼파라미터 저장 (Optuna 최적화된 경우 또는 기본 파라미터)
    if 'best_params' in results and results['best_params']:
        best_params = results['best_params']
        # 빈 딕셔너리가 아닌 경우에만 저장
        if best_params and len(best_params) > 0:
            # Optuna로 최적화된 경우와 기본 파라미터를 구분
            is_optimized = use_optuna and not use_saved_params
            
            print(f"\n💾 하이퍼파라미터 저장 중...")
            if is_optimized:
                print("   (Optuna 최적화된 파라미터)")
            else:
                print("   (기본 파라미터)")
            
            save_hyperparameters(
                best_params,
                params_filepath,
                task_type='regression',
                additional_info={
                    'cv_scores': {k: {'mean': v['mean'], 'std': v['std']} 
                                for k, v in trainer.cv_scores.items()},
                    'is_optimized': is_optimized,
                    'note': 'Optuna 최적화된 파라미터' if is_optimized else '기본 파라미터'
                }
            )
        else:
            print(f"\n⚠️ 저장할 하이퍼파라미터가 없습니다.")
    
    print(f"\n✅ 모든 작업 완료!")
    
    return results, ensemble_pred, submission


if __name__ == "__main__":
    # Optuna 최적화 사용 여부 설정
    USE_OPTUNA = True  # True로 설정하면 Optuna 최적화 실행
    N_TRIALS = 50  # Optuna 시도 횟수
    USE_SAVED_PARAMS = True  # 저장된 파라미터 사용 여부
    PARAMS_FILEPATH = os.path.join(
        os.path.dirname(__file__), 
        'best_hyperparameters.json'
    )
    
    results, ensemble_pred, submission = main(
        use_optuna=USE_OPTUNA,
        n_trials=N_TRIALS,
        use_saved_params=USE_SAVED_PARAMS,
        params_filepath=PARAMS_FILEPATH
    )

