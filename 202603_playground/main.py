"""
메인 실행 파일
프로젝트별 설정을 config.py에서 불러와서 사용
"""
import sys
import os

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, '..'))

# 프로젝트별 설정 불러오기
from config import (
    ID_COL, TARGET_COL, ENCODING_CONFIG, FEATURE_ENGINEERING_CONFIG,
    TASK_TYPE, SCORING_METRIC, N_FOLDS, RANDOM_STATE, USE_GPU,
    USE_OPTUNA, USE_SAVED_PARAMS, N_TRIALS, PARAMS_FILEPATH, OPTUNA_SAMPLE_SIZE,
    SUBMISSION_FILEPATH,
)

# 공통 모듈 import
from common.data.loader import load_data
from common.preprocess.encoder import fit_encoder, transform_with_encoder
from common.preprocess.feature_engineering import (
    clip_outliers, create_interaction_features, create_ratio_features,
    create_categorical_interactions, create_categorical_encoded_interaction
)
from common.modeling.model import ModelTrainer
from common.modeling.model import EnsembleModel
from common.modeling.hyperparameter import optimize_hyperparameters, save_hyperparameters
from common.eda.feature_importance import analyze_feature_importance
from common.eda.error_analysis import analyze_high_error_samples
from visualization import run_eda_visualization
import pandas as pd
import numpy as np
from datetime import datetime


def prepare_data(df_train, df_test, target_col='churn', 
                 use_feature_engineering=True, encoding_config=None):
    """
    데이터 전처리 함수
    """
    # 데이터 복사
    train = df_train.copy()
    test = df_test.copy()
    
    # ID 컬럼 제거
    if ID_COL in train.columns:
        train = train.drop(columns=[ID_COL])
    if ID_COL in test.columns:
        test = test.drop(columns=[ID_COL])
    
    # 타겟 분리
    y_train = train[target_col]
    X_train = train.drop(columns=[target_col])
    X_test = test.copy()
    
    # 컬럼 타입 분류
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    original_categorical_cols = categorical_cols.copy()
    
    print(f"\n📊 데이터 정보:")
    print(f"  학습 데이터 크기: {X_train.shape}")
    print(f"  테스트 데이터 크기: {X_test.shape}")
    print(f"  데이터 정보: {X_train.info()}")
    print(f"  수치형 컬럼: {len(numeric_cols)}개")
    print(f"  범주형 컬럼: {len(categorical_cols)}개")
    
    # Feature Engineering 적용
    if use_feature_engineering:
        print("\n🔧 Feature Engineering 적용 중...")
        config = FEATURE_ENGINEERING_CONFIG

        # 1단계: 인코딩 전 Feature Engineering
        print("  📌 1단계: 인코딩 전 Feature Engineering")
        
        # 이상치 클리핑
        clip_cfg = config.get('clip_outliers', {})
        if clip_cfg.get('flag', False):
            clip_rules = clip_cfg.get('clip_rules', None)
            X_train = clip_outliers(X_train, numeric_cols, clip_rules)
            X_test = clip_outliers(X_test, numeric_cols, clip_rules)
        
        # 인코딩 전 상호작용 특성
        interaction_before_cfg = config.get('create_interactions_before_encoding', {})
        if interaction_before_cfg.get('flag', False):
            feature_pairs = interaction_before_cfg.get('feature_pairs', [])
            operations = interaction_before_cfg.get('operations', [])
            X_train = create_interaction_features(X_train, feature_pairs, operations)
            X_test = create_interaction_features(X_test, feature_pairs, operations)
        
        # 컬럼 리스트 업데이트
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 범주형 인코딩
    encoded_cols_tag = '_encoded'
    
    if len(original_categorical_cols) > 0 and encoding_config is not None:
        print("\n🔤 범주형 인코딩 적용 중...")
        
        onehot_cols = encoding_config.get('onehot_cols', [])
        ordinal_cols = encoding_config.get('ordinal_cols', [])
        onehot_params = encoding_config.get('onehot_params', {})
        ordinal_params = encoding_config.get('ordinal_params', {})
        
        # 실제로 존재하는 컬럼만 필터링
        onehot_cols = [col for col in onehot_cols if col in X_train.columns]
        ordinal_cols = [col for col in ordinal_cols if col in X_train.columns]
        
        print(f"  OneHot 인코딩: {onehot_cols} (개수: {len(onehot_cols)})")
        print(f"  Ordinal 인코딩: {ordinal_cols} (개수: {len(ordinal_cols)})")
        
        # OneHot 인코딩
        if len(onehot_cols) > 0:
            onehot_params_clean = {k: v for k, v in onehot_params.items() if k not in ['drop_original']}
            onehot_params_clean.setdefault('sparse', False)
            
            onehot_encoder = fit_encoder(
                X_train,
                categorical_cols=onehot_cols,
                encoding_type='onehot',
                **{k: v for k, v in onehot_params_clean.items() if k != 'sparse'}
            )
            
            train_onehot_array = onehot_encoder.transform(X_train[onehot_cols])
            if hasattr(train_onehot_array, 'toarray'):
                train_onehot_array = train_onehot_array.toarray()
            onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_cols)
            
            test_onehot_array = onehot_encoder.transform(X_test[onehot_cols])
            if hasattr(test_onehot_array, 'toarray'):
                test_onehot_array = test_onehot_array.toarray()
            
            for idx, feature_name in enumerate(onehot_feature_names):
                encoded_col_name = f"{feature_name}{encoded_cols_tag}"
                X_train[encoded_col_name] = train_onehot_array[:, idx]
                X_test[encoded_col_name] = test_onehot_array[:, idx]
        
        # Ordinal 인코딩
        if len(ordinal_cols) > 0:
            ordinal_params_clean = {k: v for k, v in ordinal_params.items() if k != 'drop_original'}
            
            ordinal_encoder = fit_encoder(
                X_train,
                categorical_cols=ordinal_cols,
                encoding_type='ordinal',
                **ordinal_params_clean
            )
            
            train_ordinal_array = ordinal_encoder.transform(X_train[ordinal_cols])
            test_ordinal_array = ordinal_encoder.transform(X_test[ordinal_cols])
            
            for idx, col in enumerate(ordinal_cols):
                encoded_col_name = f"{col}{encoded_cols_tag}"
                X_train[encoded_col_name] = train_ordinal_array[:, idx]
                X_test[encoded_col_name] = test_ordinal_array[:, idx]
        
        print(f"  원본 범주형 컬럼 유지: {len(original_categorical_cols)}개 (CatBoost용)")
    
    # 2단계: 인코딩 후 Feature Engineering
    if use_feature_engineering:
        print("\n  📌 2단계: 인코딩 후 Feature Engineering")
        
        interaction_after_cfg = config.get('create_interactions_after_encoding', {})
        if interaction_after_cfg.get('flag', False):
            feature_pairs = interaction_after_cfg.get('feature_pairs', [])
            operations = interaction_after_cfg.get('operations', [])
            X_train = create_interaction_features(X_train, feature_pairs, operations)
            X_test = create_interaction_features(X_test, feature_pairs, operations)
        
        ratio_cfg = config.get('create_ratios', {})
        if ratio_cfg.get('flag', False):
            numerator_cols = ratio_cfg.get('numerator_cols', [])
            denominator_cols = ratio_cfg.get('denominator_cols', [])
            feature_names = ratio_cfg.get('ratio_feature_names', None)
            X_train = create_ratio_features(X_train, numerator_cols, denominator_cols, feature_names)
            X_test = create_ratio_features(X_test, numerator_cols, denominator_cols, feature_names)
        
        # 범주형 인코딩 컬럼과 수치형 컬럼 상호작용
        cat_interaction_cfg = config.get('create_categorical_interactions', {})
        if cat_interaction_cfg.get('flag', False):
            categorical_pairs = cat_interaction_cfg.get('categorical_pairs', [])
            separator = cat_interaction_cfg.get('separator', '_')
            X_train = create_categorical_interactions(X_train, categorical_pairs, separator)
            X_test = create_categorical_interactions(X_test, categorical_pairs, separator)
    
    # 결측치 처리
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col in X_train.columns and X_train[col].isnull().sum() > 0:
            mean_val = X_train[col].mean()
            X_train[col].fillna(mean_val, inplace=True)
            if col in X_test.columns:
                X_test[col].fillna(mean_val, inplace=True)
    
    print("✅ 데이터 준비 완료!")
    
    return X_train, y_train, X_test, original_categorical_cols, numeric_cols, encoded_cols_tag


def main():
    """메인 실행 함수"""
    print("="*60)
    print("🎯 Kaggle Competition - 모델 학습 및 앙상블")
    print("="*60)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df_train, df_test, df_sub = load_data(data_dir=project_root)
    
    # 데이터 준비
    X_train, y_train, X_test, categorical_cols, numeric_cols, encoded_cols_tag = prepare_data(
        df_train, df_test,
        target_col=TARGET_COL,
        use_feature_engineering=True,
        encoding_config=ENCODING_CONFIG
    )

    # 분류 시 타겟이 문자열(Yes/No 등)이면 0/1로 변환 (LightGBM 등 모델·검증 지표 요구)
    if TASK_TYPE == 'classification' and (pd.api.types.is_string_dtype(y_train) or y_train.dtype == object):
        if set(y_train.dropna().unique()) <= {'Yes', 'No'}:
            mapping = {'No': 0, 'Yes': 1}
        else:
            unique_vals = sorted(y_train.dropna().unique().tolist())
            mapping = {v: i for i, v in enumerate(unique_vals)}
        y_train = y_train.map(mapping).astype(int)
        print(f"  타겟 인코딩: {mapping}")
    
    print("\n📊 EDA 시작...")
    print("="*60)
    df_for_vis = X_train.copy()
    df_for_vis[TARGET_COL] = y_train
    # run_eda_visualization(df_for_vis, target_col=TARGET_COL)
    
    # 모델 학습
    print("\n🚀 모델 학습 시작...")
    trainer = ModelTrainer(task_type=TASK_TYPE, scoring_metric=SCORING_METRIC, random_state=RANDOM_STATE, use_gpu=USE_GPU)
    
    # 하이퍼파라미터 최적화 또는 저장된 파라미터 사용
    if USE_OPTUNA or USE_SAVED_PARAMS:
        best_params = optimize_hyperparameters(
            X_train, y_train, categorical_cols,
            task_type=TASK_TYPE,
            n_trials=N_TRIALS,
            random_state=RANDOM_STATE,
            use_saved_params=USE_SAVED_PARAMS,
            params_filepath=PARAMS_FILEPATH,
            encoded_cols_tag=encoded_cols_tag,
            use_gpu=USE_GPU,
            sample_size=OPTUNA_SAMPLE_SIZE,
            additional_save_info={
                'n_folds': N_FOLDS,
                'use_gpu': USE_GPU,
                'target_col': TARGET_COL,
                'params_filepath': PARAMS_FILEPATH,
                'optuna_sample_size': OPTUNA_SAMPLE_SIZE,
            }
        )
    else:
        best_params = {}
    
    # 각 모델 학습
    test_predictions = {}
    model_features = {}
    
    for model_type in ['lightgbm', 'xgboost']:
        print(f"\n{'='*60}")
        print(f"🚀 {model_type.upper()} 모델 학습 시작")
        print(f"{'='*60}")
        
        # 모델별로 사용할 컬럼 선택
        encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
        
        if model_type == 'catboost':
            feature_cols = [col for col in X_train.columns if col not in encoded_cols]
            actual_categorical_cols = X_train[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
            cat_features = actual_categorical_cols
        else:
            all_categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            feature_cols = [col for col in X_train.columns if col not in all_categorical_cols]
            cat_features = None
        
        model_features[model_type] = feature_cols
        X_train_model = X_train[feature_cols].copy()
        X_test_model = X_test[feature_cols].copy()
        
        # 모델 파라미터 설정
        if model_type in best_params:
            model_params = best_params[model_type].copy()
        else:
            # 기본 파라미터 사용
            model_params = {}
        
        # K-Fold 교차 검증으로 학습
        result = trainer.train_with_cv(
            X_train_model, y_train,
            model_type=model_type,
            n_folds=N_FOLDS,
            cat_features=cat_features,
            **model_params
        )
        
        # 테스트 데이터 예측
        test_pred = trainer.predict_test(X_test_model, model_type)
        test_predictions[model_type] = test_pred
        
        print(f"✅ {model_type.upper()} 학습 완료!")
    
    # 앙상블 생성
    print(f"\n{'='*60}")
    print("🎯 앙상블 모델 생성")
    print(f"{'='*60}")
    
    ensemble = EnsembleModel(task_type=TASK_TYPE, scoring_metric=SCORING_METRIC)
    ensemble.fit(
        trainer.oof_predictions,
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        method='weighted_average',
        optimize=True
    )
    
    ensemble_pred = ensemble.predict(test_predictions)
    
    # 제출 파일 생성
    print("\n📝 제출 파일 생성 중...")
    submission = pd.DataFrame({
        ID_COL: df_sub[ID_COL],
        TARGET_COL: ensemble_pred
    })
    
    submission.to_csv(SUBMISSION_FILEPATH, index=False)
    print(f"✅ 제출 파일 저장 완료: {SUBMISSION_FILEPATH}")
    
    print("\n✅ 모든 작업 완료!")
    
    return trainer, ensemble, submission


if __name__ == "__main__":
    trainer, ensemble, submission = main()
