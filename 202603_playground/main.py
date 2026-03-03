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
    ENSEMBLE_METHOD, RIDGE_ALPHA, OPTUNA_RIDGE_ALPHA, RIDGE_ALPHA_N_TRIALS,
    USE_OPTUNA, USE_SAVED_PARAMS, N_TRIALS, PARAMS_FILEPATH, OPTUNA_SAMPLE_SIZE,
    SUBMISSION_FILEPATH, PROJECT_ROOT, MODEL_TYPES, USE_FEATURE_ENGINEERING_FOR_MODELS,
    USE_PERMUTATION_IMPORTANCE,
)

# 공통 모듈 import
from common.data.loader import load_data
from common.preprocess.encoder import fit_encoder, transform_with_encoder
from common.preprocess.feature_engineering import (
    clip_outliers, create_interaction_features, create_ratio_features,
    create_categorical_interactions, create_categorical_encoded_interaction, create_statistical_features,
    convert_ordered_categorical_to_numeric, transform_numeric_features
)
from common.modeling.model import ModelTrainer
from common.modeling.model import EnsembleModel
from common.modeling.hyperparameter import optimize_hyperparameters, save_hyperparameters, optimize_ridge_alpha
from common.eda.feature_importance import (
    analyze_feature_importance,
    analyze_permutation_importance,
    analyze_ensemble_feature_importance,
    compare_model_and_ensemble_importance
)
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
        
        # 순서가 있는 범주형 변수를 숫자로 변환 (Contract 등)
        convert_cat_cfg = config.get('convert_ordered_categorical_to_numeric', {})
        if convert_cat_cfg.get('flag', False):
            mappings = convert_cat_cfg.get('mappings', {})
            for cat_col, mapping in mappings.items():
                if cat_col in X_train.columns:
                    X_train = convert_ordered_categorical_to_numeric(X_train, cat_col, mapping)
                    X_test = convert_ordered_categorical_to_numeric(X_test, cat_col, mapping)
                    print(f"  ✅ {cat_col} → {cat_col}_numeric 변환 완료 (매핑: {mapping})")
        
        # 컬럼 리스트 업데이트 (숫자로 변환된 컬럼 포함)
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # 인코딩 전 상호작용 특성
        interaction_before_cfg = config.get('create_interactions_before_encoding', {})
        if interaction_before_cfg.get('flag', False):
            feature_pairs = interaction_before_cfg.get('feature_pairs', [])
            operations = interaction_before_cfg.get('operations', [])
            X_train = create_interaction_features(X_train, feature_pairs, operations)
            X_test = create_interaction_features(X_test, feature_pairs, operations)
        
        # 인코딩 전 통계/집계 특성
        create_statistical_cfg = config.get('create_statistical_features', {})
        if create_statistical_cfg.get('flag', False):
            feature_groups = create_statistical_cfg.get('feature_groups', [])
            statistics = create_statistical_cfg.get('statistics', ['mean', 'std'])
            X_train = create_statistical_features(X_train, feature_groups, statistics)
            X_test = create_statistical_features(X_test, feature_groups, statistics)
        
        # 인코딩 전 추가 Feature Engineering (수기 정의)
        
        # 2) 인터넷 사용 여부 플래그
        if 'InternetService' in X_train.columns:
            X_train['has_internet'] = (X_train['InternetService'] != 'No').astype(int)
        if 'InternetService' in X_test.columns:
            X_test['has_internet'] = (X_test['InternetService'] != 'No').astype(int)
        
        # 3) 고위험 조합: 월단위 계약 + 전자 청구서 결제
        if 'Contract' in X_train.columns and 'PaymentMethod' in X_train.columns:
            high_risk_train = (
                (X_train['Contract'] == 'Month-to-month') &
                (X_train['PaymentMethod'] == 'Electronic check')
            )
            X_train['high_risk_combo'] = high_risk_train.astype(int)
        if 'Contract' in X_test.columns and 'PaymentMethod' in X_test.columns:
            high_risk_test = (
                (X_test['Contract'] == 'Month-to-month') &
                (X_test['PaymentMethod'] == 'Electronic check')
            )
            X_test['high_risk_combo'] = high_risk_test.astype(int)
        
        # 4) Senior × tenure 상호작용 (장기 이용 고령 고객 구분)
        if 'SeniorCitizen' in X_train.columns and 'tenure' in X_train.columns:
            X_train['SeniorCitizen_tenure_FE'] = X_train['SeniorCitizen'] * X_train['tenure']
        if 'SeniorCitizen' in X_test.columns and 'tenure' in X_test.columns:
            X_test['SeniorCitizen_tenure_FE'] = X_test['SeniorCitizen'] * X_test['tenure']

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
        drop_original = encoding_config.get('drop_original', False)
        
        # 실제로 존재하는 컬럼만 필터링
        onehot_cols = [col for col in onehot_cols if col in X_train.columns]
        ordinal_cols = [col for col in ordinal_cols if col in X_train.columns]
        
        print(f"  OneHot 인코딩: {onehot_cols} (개수: {len(onehot_cols)})")
        print(f"  Ordinal 인코딩: {ordinal_cols} (개수: {len(ordinal_cols)})")
        print(f"  drop_original: {drop_original}")
        
        # OneHot 인코딩
        if len(onehot_cols) > 0:
            onehot_params_clean = onehot_params.copy()
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
        
        # Ordinal 인코딩: 원본 컬럼 삭제 (CatBoost용으로는 별도 보관)
        catboost_original_cols_train = {}
        catboost_original_cols_test = {}
        if len(ordinal_cols) > 0:
            ordinal_params_clean = ordinal_params.copy()
            
            # CatBoost용으로 원본 범주형 컬럼 보관 (ordinal_cols만)
            for col in ordinal_cols:
                if col in X_train.columns:
                    catboost_original_cols_train[col] = X_train[col].copy()
                    catboost_original_cols_test[col] = X_test[col].copy()
            
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
            
            # Ordinal 인코딩된 컬럼의 원본 삭제
                      # Ordinal 인코딩된 컬럼의 원본 삭제 (drop_original 설정에 따라)
            if drop_original:
                X_train = X_train.drop(columns=ordinal_cols)
                X_test = X_test.drop(columns=ordinal_cols)
                print(f"  ✅ Ordinal 인코딩 완료: {len(ordinal_cols)}개 컬럼 → 원본 삭제, 인코딩된 컬럼 추가")
            else:
                print(f"  ✅ Ordinal 인코딩 완료: {len(ordinal_cols)}개 컬럼 → 원본 유지, 인코딩된 컬럼 추가")
        
        if not drop_original:
            print(f"  원본 범주형 컬럼 유지: {len([c for c in original_categorical_cols if c in X_train.columns])}개 (CatBoost용)")
    
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
    
        transform_cfg = config.get('transform_numeric_features', {})
        if transform_cfg.get('flag', False):
            numeric_cols_to_transform = transform_cfg.get('columns', ['tenure'])   # 대상 컬럼
            transformations = transform_cfg.get('transformations', ['log', 'square'])
            X_train = transform_numeric_features(X_train, numeric_cols_to_transform, transformations)
            X_test = transform_numeric_features(X_test, numeric_cols_to_transform, transformations)

    # 모든 Feature Engineering이 끝난 후 원본 컬럼 삭제
    if len(original_categorical_cols) > 0 and encoding_config is not None:
        drop_original = encoding_config.get('drop_original', True)
        if drop_original:
            cols_to_drop = [col for col in original_categorical_cols if col in X_train.columns]
            if cols_to_drop:
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=cols_to_drop)
                print(f"\n✅ 원본 범주형 컬럼 삭제 완료: {len(cols_to_drop)}개")
    
    # 결측치 처리
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col in X_train.columns and X_train[col].isnull().sum() > 0:
            mean_val = X_train[col].mean()
            X_train[col].fillna(mean_val, inplace=True)
            if col in X_test.columns:
                X_test[col].fillna(mean_val, inplace=True)
    
    print("✅ 데이터 준비 완료!")
    
    return X_train, y_train, X_test, original_categorical_cols, numeric_cols, encoded_cols_tag, catboost_original_cols_train, catboost_original_cols_test


def main():
    """메인 실행 함수"""
    print("="*60)
    print("🎯 Kaggle Competition - 모델 학습 및 앙상블")
    print("="*60)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df_train, df_test, df_sub = load_data(data_dir=project_root)
    
    # 데이터 준비
    X_train, y_train, X_test, categorical_cols, numeric_cols, encoded_cols_tag, catboost_original_cols_train, catboost_original_cols_test = prepare_data(
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
    # EDA는 전처리된 데이터로 수행
    df_for_vis = X_train.copy()
    if isinstance(y_train, pd.Series):
        df_for_vis[TARGET_COL] = y_train.values
    else:
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
            model_types=MODEL_TYPES,
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
    
    # AUC 히스토리 저장 파일 경로
    auc_history_path = os.path.join(PROJECT_ROOT, 'auc_history.csv')
    
    # 각 모델 학습
    test_predictions = {}
    model_features = {}
    
    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"🚀 {model_type.upper()} 모델 학습 시작")
        print(f"{'='*60}")
        
        # 모델별로 사용할 컬럼 선택
        encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
        
        if model_type == 'catboost':
            # CatBoost: 원본 범주형 컬럼 + 인코딩된 컬럼 모두 사용 + ordinal 원본 컬럼 복원
            # 인코딩된 컬럼도 포함하여 사용 (원본과 인코딩 모두 활용)
            feature_cols = [col for col in X_train.columns]  # 모든 컬럼 포함
            X_train_model = X_train[feature_cols].copy()
            X_test_model = X_test[feature_cols].copy()
            
            # Ordinal 인코딩된 컬럼의 원본 복원 (원본 범주형 컬럼이 이미 있으면 덮어쓰기)
            for col, original_data in catboost_original_cols_train.items():
                X_train_model[col] = original_data
            for col, original_data in catboost_original_cols_test.items():
                X_test_model[col] = original_data
            
            # Feature Engineering 특성 제거 (설정에 따라)
            use_fe_for_model = USE_FEATURE_ENGINEERING_FOR_MODELS.get(model_type, True)
            if not use_fe_for_model:
                # 인코딩된 컬럼만 제거 (수치형 변수는 모두 유지)
                encoded_cols_to_remove = [col for col in X_train_model.columns 
                                        if col.endswith(encoded_cols_tag)]
                if encoded_cols_to_remove:
                    X_train_model = X_train_model.drop(columns=encoded_cols_to_remove)
                    X_test_model = X_test_model.drop(columns=[col for col in encoded_cols_to_remove if col in X_test_model.columns])
                    print(f"  🗑️ CatBoost: 인코딩된 컬럼 제거 ({len(encoded_cols_to_remove)}개)")
            else:
                # 기존 로직: 통계 특성 중 일부만 제거 (mean만 유지, std/min/max 제거)
                stats_cols_to_remove = [col for col in X_train_model.columns 
                                       if any(stat in col for stat in ['_std', '_min', '_max'])]
                if stats_cols_to_remove:
                    X_train_model = X_train_model.drop(columns=stats_cols_to_remove)
                    X_test_model = X_test_model.drop(columns=[col for col in stats_cols_to_remove if col in X_test_model.columns])
                    print(f"  🗑️ CatBoost: 통계 특성 제거 ({len(stats_cols_to_remove)}개: std/min/max)")
            
            # 범주형 컬럼은 원본만 사용 (인코딩된 컬럼은 수치형으로 처리)
            actual_categorical_cols = X_train_model.select_dtypes(include=['object', 'category']).columns.tolist()
            cat_features = actual_categorical_cols
            feature_cols = X_train_model.columns.tolist()
            
            if not use_fe_for_model:
                numeric_cols_count = len([c for c in feature_cols if c not in actual_categorical_cols])
                print(f"  ✅ CatBoost: 원본 범주형 {len(actual_categorical_cols)}개 + 수치형 {numeric_cols_count}개 사용 (인코딩 컬럼 제외)")
            else:
                print(f"  ✅ CatBoost: 원본 범주형 {len(actual_categorical_cols)}개 + 인코딩된 컬럼 {len([c for c in feature_cols if c.endswith(encoded_cols_tag)])}개 사용")
        else:
            # LightGBM/XGBoost: 모든 범주형 컬럼 제외 (인코딩된 컬럼만 사용)
            all_categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            feature_cols = [col for col in X_train.columns if col not in all_categorical_cols]
            X_train_model = X_train[feature_cols].copy()
            X_test_model = X_test[feature_cols].copy()
            cat_features = None
        
        model_features[model_type] = feature_cols
        
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
        
        # AUC 값 저장
        cv_score = trainer.cv_scores[model_type]['mean']
        cv_std = trainer.cv_scores[model_type]['std']
        fold_scores = trainer.cv_scores[model_type].get('fold_scores', [])
        
        auc_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'cv_score': cv_score,
            'std': cv_std,
            'fold_scores': ','.join([f'{s:.4f}' for s in fold_scores]) if fold_scores else '',
            'n_folds': N_FOLDS,
            'scoring_metric': SCORING_METRIC
        }
        
        # CSV에 추가 (파일이 없으면 생성, 있으면 append)
        auc_df = pd.DataFrame([auc_record])
        if os.path.exists(auc_history_path):
            auc_df.to_csv(auc_history_path, mode='a', header=False, index=False)
        else:
            auc_df.to_csv(auc_history_path, mode='w', header=True, index=False)
        
        print(f"✅ {model_type.upper()} 학습 완료! (CV Score: {cv_score:.4f}, 저장 완료)")
    
    # 모델별 Feature Importance 분석
    print(f"\n{'='*60}")
    print("📊 모델별 Feature Importance 분석")
    print(f"{'='*60}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_importance_dir = os.path.join(base_dir, 'feature_importance_results')
    
    model_importances = analyze_feature_importance(
        trainer=trainer,
        X_train=X_train,
        categorical_cols=categorical_cols,
        encoded_cols_tag=encoded_cols_tag,
        top_n=30,
        save_dir=feature_importance_dir
    )
    
    # Permutation Importance 분석 (config에서 활성화 시)
    if USE_PERMUTATION_IMPORTANCE:
        print(f"\n{'='*60}")
        print("📊 Permutation Importance 분석")
        print(f"{'='*60}")
        analyze_permutation_importance(
            trainer=trainer,
            X_train=X_train,
            y_train=y_train,
            categorical_cols=categorical_cols,
            encoded_cols_tag=encoded_cols_tag,
            top_n=30,
            n_repeats=10,
            save_dir=feature_importance_dir
        )
    
    # 앙상블 생성
    print(f"\n{'='*60}")
    print("🎯 앙상블 모델 생성")
    print(f"{'='*60}")
    
    ensemble = EnsembleModel(task_type=TASK_TYPE, scoring_metric=SCORING_METRIC)
    ridge_alpha = RIDGE_ALPHA
    if ENSEMBLE_METHOD == 'ridge_meta' and OPTUNA_RIDGE_ALPHA:
        print("\n🔍 Ridge alpha Optuna 최적화 중...")
        ridge_alpha = optimize_ridge_alpha(
            trainer.oof_predictions,
            y_train.values if isinstance(y_train, pd.Series) else y_train,
            task_type=TASK_TYPE,
            scoring_metric=SCORING_METRIC,
            n_trials=RIDGE_ALPHA_N_TRIALS,
            random_state=RANDOM_STATE
        )
        print(f"  ✅ 최적 Ridge alpha: {ridge_alpha:.6f}")
    ensemble.fit(
        trainer.oof_predictions,
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        method=ENSEMBLE_METHOD,
        optimize=(ENSEMBLE_METHOD == 'weighted_average'),
        ridge_alpha=ridge_alpha
    )
    
    ensemble_pred = ensemble.predict(test_predictions)
    
    # 앙상블 AUC 값 저장
    if hasattr(ensemble, 'ensemble_score') and ensemble.ensemble_score is not None:
        ensemble_auc = ensemble.ensemble_score
        ensemble_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'ensemble',
            'cv_score': ensemble_auc,
            'std': '',
            'fold_scores': '',
            'n_folds': N_FOLDS,
            'scoring_metric': SCORING_METRIC
        }
        
        ensemble_df = pd.DataFrame([ensemble_record])
        if os.path.exists(auc_history_path):
            ensemble_df.to_csv(auc_history_path, mode='a', header=False, index=False)
        else:
            ensemble_df.to_csv(auc_history_path, mode='w', header=True, index=False)
        
        print(f"✅ 앙상블 AUC 저장 완료: {ensemble_auc:.4f}")
    
    # 앙상블 Feature Importance 분석
    print(f"\n{'='*60}")
    print("📊 앙상블 Feature Importance 분석")
    print(f"{'='*60}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    feature_importance_dir = os.path.join(base_dir, 'feature_importance_results')
    
    ensemble_importance = analyze_ensemble_feature_importance(
        trainer=trainer,
        ensemble=ensemble,
        X_train=X_train,
        categorical_cols=categorical_cols,
        encoded_cols_tag=encoded_cols_tag,
        top_n=30,
        save_dir=feature_importance_dir
    )
    
    # 모델별 vs 앙상블 Feature Importance 비교
    if model_importances is not None and ensemble_importance is not None:
        print(f"\n{'='*60}")
        print("📊 모델별 vs 앙상블 Feature Importance 비교")
        print(f"{'='*60}")
        
        comparison_df = compare_model_and_ensemble_importance(
            model_importances=model_importances,
            ensemble_importance=ensemble_importance,
            top_n=30,
            save_dir=feature_importance_dir
        )
    
    # 제출 파일 생성 (분류 시 확률이 (n,2)이면 양성 클래스 확률만 사용)
    print("\n📝 제출 파일 생성 중...")
    pred_submit = ensemble_pred[:, 1] if getattr(ensemble_pred, 'ndim', 1) == 2 else ensemble_pred
    submission = pd.DataFrame({
        ID_COL: df_sub[ID_COL],
        TARGET_COL: pred_submit
    })
    
    submission.to_csv(SUBMISSION_FILEPATH, index=False)
    print(f"✅ 제출 파일 저장 완료: {SUBMISSION_FILEPATH}")
    
    print("\n✅ 모든 작업 완료!")
    
    return trainer, ensemble, submission


if __name__ == "__main__":
    trainer, ensemble, submission = main()
