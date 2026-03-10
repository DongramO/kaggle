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
    TASK_TYPE, SCORING_METRIC, N_FOLDS, RANDOM_STATE, USE_GPU, EARLY_STOPPING_ROUNDS,
    ENSEMBLE_METHOD, RIDGE_ALPHA, OPTUNA_RIDGE_ALPHA, RIDGE_ALPHA_N_TRIALS,
    USE_OPTUNA, USE_SAVED_PARAMS, N_TRIALS, OPTUNA_N_FOLDS, PARAMS_FILEPATH, OPTUNA_SAMPLE_SIZE,
    SUBMISSION_FILEPATH, PROJECT_ROOT, MODEL_TYPES, USE_FEATURE_ENGINEERING_FOR_MODELS,
    USE_PERMUTATION_IMPORTANCE,
    USE_MLP_FOCUS_ANALYSIS,
    USE_MLP_FOCUS_ONLY,
)

# 공통 모듈 import
from common.data.loader import load_data
from common.preprocess.encoder import fit_encoder, transform_with_encoder
from common.preprocess.feature_engineering import (
    clip_outliers, create_interaction_features, create_ratio_features,
    create_categorical_interactions, create_categorical_encoded_interaction, create_statistical_features,
    convert_ordered_categorical_to_numeric, transform_numeric_features, apply_target_encoding
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
from common.eda.mlp_focus_analysis import analyze_mlp_focus
from common.eda.error_analysis import analyze_high_error_samples
from common.eda.correlation import (
    calculate_correlation_matrix,
    plot_correlation_heatmap,
    CorrelationConfig as CorrConfig,
)
from visualization import run_eda_visualization
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _plot_train_vs_val_scores(trainer, model_types, scoring_metric, n_folds, output_dir=None):
    """Fold별 Train vs Validation 점수를 막대 그래프로 저장."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_importance_results')
    os.makedirs(output_dir, exist_ok=True)
    metric_label = scoring_metric.upper()
    n_models = len(model_types)
    if n_models == 0:
        return
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    folds = list(range(1, n_folds + 1))
    x = np.arange(n_folds)
    width = 0.35
    for i, model_type in enumerate(model_types):
        ax = axes[i]
        info = trainer.cv_scores.get(model_type, {})
        train_scores = info.get('train_fold_scores', [])
        val_scores = info.get('fold_scores', [])
        if not train_scores or not val_scores:
            ax.text(0.5, 0.5, f'{model_type}\nNo data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper())
            continue
        bars_train = ax.bar(x - width / 2, train_scores, width, label='Train', color='steelblue', alpha=0.9)
        bars_val = ax.bar(x + width / 2, val_scores, width, label='Validation', color='coral', alpha=0.9)
        ax.bar_label(bars_train, labels=[f'{s:.3f}' for s in train_scores], fontsize=8, padding=2)
        ax.bar_label(bars_val, labels=[f'{s:.3f}' for s in val_scores], fontsize=8, padding=2)
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric_label)
        ax.set_title(f'{model_type.upper()} — Train vs Val')
        ax.set_xticks(x)
        ax.set_xticklabels(folds)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(output_dir, 'train_vs_val_scores.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  📊 Train vs Validation 그래프 저장: {out_path}")


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
        
        # 타겟 인코딩 (인코딩/숫자 변환 전에 원본 범주형 기준으로 생성)
        te_cfg = config.get('target_encoding', {})
        if te_cfg.get('flag', False):
            te_cols = te_cfg.get('cols', [])
            # 실제 존재하는 컬럼만 사용
            te_cols = [c for c in te_cols if c in X_train.columns]
            if len(te_cols) > 0:
                n_splits_te = te_cfg.get('n_splits', N_FOLDS)
                smoothing_te = te_cfg.get('smoothing', 10.0)
                X_train, X_test = apply_target_encoding(
                    X_train,
                    y_train,
                    X_test,
                    cols=te_cols,
                    n_splits=n_splits_te,
                    smoothing=smoothing_te,
                    random_state=RANDOM_STATE,
                )
                print(f"  ✅ Target Encoding 적용: {te_cols} (n_splits={n_splits_te}, smoothing={smoothing_te})")
        
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
        
         # 2) 인터넷 사용 여부 플래그
        if 'InternetService' in X_train.columns:
            X_train['has_internet'] = (X_train['InternetService'] != 'No').astype(int)
        if 'InternetService' in X_test.columns:
            X_test['has_internet'] = (X_test['InternetService'] != 'No').astype(int)

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
        
        transform_cfg = config.get('transform_numeric_features', {})
        if transform_cfg.get('flag', False):
            numeric_cols_to_transform = transform_cfg.get('columns', ['tenure'])   # 대상 컬럼
            transformations = transform_cfg.get('transformations', ['log', 'square'])
            X_train = transform_numeric_features(X_train, numeric_cols_to_transform, transformations)
            X_test = transform_numeric_features(X_test, numeric_cols_to_transform, transformations)
            
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

        # tenure, MonthlyCharges, TotalCharges 사칙연산 신규 특성 (add/mul/sub/div, 쌍별 양방향)
        arithmetic_cfg = config.get('create_tenure_monthly_total_arithmetic', {})
        if arithmetic_cfg.get('flag', False):
            feature_pairs = arithmetic_cfg.get('feature_pairs', [])
            operations = arithmetic_cfg.get('operations', [])
            X_train = create_interaction_features(X_train, feature_pairs, operations)
            X_test = create_interaction_features(X_test, feature_pairs, operations)
            print(f"  ✅ tenure/MonthlyCharges/TotalCharges 사칙연산 특성 추가 ({len(feature_pairs)}개)")

        # tenure_years(가입 년수), tenure_segment(0,12,24,48,72 구간)
        tenure_derived_cfg = config.get('create_tenure_derived', {})
        if tenure_derived_cfg.get('flag', False) and 'tenure' in X_train.columns:
            bins = tenure_derived_cfg.get('tenure_segment_bins', [0, 12, 24, 48, 72])
            # 구간 4개: (0,12], (12,24], (24,48], (48,72]. 72 초과도 마지막 구간(3)으로 처리
            bins_ext = bins + [float('inf')]
            seg_labels = [0, 1, 2, 3, 3]  # 5개 구간 중 (48,72]와 (72,inf) 모두 3
            for df in [X_train, X_test]:
                df['tenure_years'] = (df['tenure'] / 12).apply(np.floor).astype(int)
                seg = pd.cut(df['tenure'], bins=bins_ext, labels=seg_labels, include_lowest=True, ordered=False)
                df['tenure_segment'] = seg.astype(float).fillna(3).astype(int)
            print(f"  ✅ tenure_years, tenure_segment(0,12,24,48,72) 특성 추가")
        
        # 이용하는 인터넷 서비스 개수 (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies 중 Yes=1 개수)
        internet_service_cols = [
            'OnlineSecurity_numeric', 'OnlineBackup_numeric', 'DeviceProtection_numeric',
            'TechSupport_numeric', 'StreamingTV_numeric', 'StreamingMovies_numeric',
        ]
        existing_is_cols = [c for c in internet_service_cols if c in X_train.columns]
        if len(existing_is_cols) == len(internet_service_cols):
            X_train['num_internet_services'] = X_train[existing_is_cols].sum(axis=1)
            X_test['num_internet_services'] = X_test[existing_is_cols].sum(axis=1)
            print(f"  ✅ 이용 인터넷 서비스 개수 특성 추가: num_internet_services (0~6)")
        
        # Contract_numeric 활용: 역수(계약 짧을수록 큼), 월단위 여부
        _eps = 1e-6
        if 'Contract_numeric' in X_train.columns:
            X_train['Contract_numeric_inv'] = 1 / (X_train['Contract_numeric'] + _eps)
            X_test['Contract_numeric_inv'] = 1 / (X_test['Contract_numeric'] + _eps)
            X_train['is_month_to_month_x_payment_method_electronic_check'] = (X_train['Contract_numeric'] == 1).astype(int) * X_train['PaymentMethod_Electronic check_encoded']
            X_test['is_month_to_month_x_payment_method_electronic_check'] = (X_test['Contract_numeric'] == 1).astype(int) * X_test['PaymentMethod_Electronic check_encoded']
        
        # 범주형 인코딩 컬럼과 수치형 컬럼 상호작용
        cat_interaction_cfg = config.get('create_categorical_interactions', {})
        if cat_interaction_cfg.get('flag', False):
            categorical_pairs = cat_interaction_cfg.get('categorical_pairs', [])
            separator = cat_interaction_cfg.get('separator', '_')
            X_train = create_categorical_interactions(X_train, categorical_pairs, separator)
            X_test = create_categorical_interactions(X_test, categorical_pairs, separator)
    
    
        if 'SeniorCitizen' in X_train.columns and 'tenure' in X_train.columns:
            X_train['SeniorCitizen_MonthlyCharges_FE'] = X_train['SeniorCitizen'] * X_train['MonthlyCharges']
            X_train['SeniorCitizen_Dependents_Yes_FE'] = X_train['SeniorCitizen'] * (1- X_train['Dependents_Yes_encoded'] )
            X_train['SeniorCitizen_Partner_Yes_FE'] = X_train['SeniorCitizen'] * (1- X_train['Partner_Yes_encoded'] )
        if 'SeniorCitizen' in X_test.columns and 'tenure' in X_test.columns:
            X_test['SeniorCitizen_MonthlyCharges_FE'] = X_test['SeniorCitizen'] * X_test['MonthlyCharges']
            X_test['SeniorCitizen_Dependents_Yes_FE'] = X_test['SeniorCitizen'] * (1- X_test['Dependents_Yes_encoded'] )
            X_test['SeniorCitizen_Partner_Yes_FE'] = X_test['SeniorCitizen'] * (1- X_test['Partner_Yes_encoded'] )


        # 3) 고위험 조합: 월단위 계약 + 전자 청구서 결제
        # if 'Contract' in X_train.columns and 'PaymentMethod' in X_train.columns:
        #     high_risk_train = (
        #         (X_train['Contract'] == 'Month-to-month') &
        #         (X_train['PaymentMethod'] == 'Electronic check')
        #     )
        #     X_train['high_risk_combo'] = high_risk_train.astype(int)
        # if 'Contract' in X_test.columns and 'PaymentMethod' in X_test.columns:
        #     high_risk_test = (
        #         (X_test['Contract'] == 'Month-to-month') &
        #         (X_test['PaymentMethod'] == 'Electronic check')
        #     )
        #     X_test['high_risk_combo'] = high_risk_test.astype(int)

    # 모든 Feature Engineering이 끝난 후 원본 컬럼 삭제
    if len(original_categorical_cols) > 0 and encoding_config is not None:
        drop_original = encoding_config.get('drop_original', True)
        if drop_original:
            cols_to_drop = [col for col in original_categorical_cols if col in X_train.columns]
            if cols_to_drop:
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=cols_to_drop)
                print(f"\n✅ 원본 범주형 컬럼 삭제 완료: {len(cols_to_drop)}개")
    
    # X_train = X_train.drop(columns=['tenure'])
    # X_test = X_test.drop(columns=['tenure'])

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

    print('df.columns: ', X_train.columns)
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

    # 상관계수 히트맵: 학습에 쓰이는 특성(수치형) 간 상관 여부 확인 → 노이즈/중복 특성 판단용
    eda_dir = os.path.join(PROJECT_ROOT, 'eda_results')
    os.makedirs(eda_dir, exist_ok=True)
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) >= 2:
        corr_matrix = calculate_correlation_matrix(
            X_train[num_cols], method='pearson', numeric_only=True
        )
        corr_config = CorrConfig(figsize=(14, 12), threshold=0)
        fig = plot_correlation_heatmap(
            corr_matrix, config=corr_config,
            title='상관계수 히트맵 (학습 특성, FE·인코딩 후)'
        )
        heatmap_path = os.path.join(eda_dir, 'correlation_heatmap_after_fe.png')
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        import matplotlib.pyplot as plt
        plt.close(fig)

    # MLP 분석만 실행 (모델 학습·앙상블·제출 건너뜀)
    if USE_MLP_FOCUS_ONLY:
        print("\n📌 USE_MLP_FOCUS_ONLY=True: 모델 학습 없이 MLP 분석만 진행합니다.")
        feature_importance_dir = os.path.join(PROJECT_ROOT, 'feature_importance_results')
        all_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols_for_mlp = [c for c in X_train.columns if c not in all_cat_cols]
        X_train_numeric = X_train[numeric_cols_for_mlp]
        analyze_mlp_focus(
            X_train=X_train_numeric,
            y_train=y_train,
            task_type=TASK_TYPE,
            random_state=RANDOM_STATE,
            save_dir=feature_importance_dir,
            tree_importance_df=None,
            top_n=30,
            top_interactions=50,
        )
        print(f"✅ MLP 분석 완료. 결과: {feature_importance_dir}")
        return None, None, None

    # 모델 학습
    print("\n🚀 모델 학습 시작...")
    trainer = ModelTrainer(task_type=TASK_TYPE, scoring_metric=SCORING_METRIC, random_state=RANDOM_STATE, use_gpu=USE_GPU, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    
    # 하이퍼파라미터 최적화 또는 저장된 파라미터 사용
    if USE_OPTUNA or USE_SAVED_PARAMS:
        best_params = optimize_hyperparameters(
            X_train, y_train, categorical_cols,
            task_type=TASK_TYPE,
            n_trials=N_TRIALS,
            n_folds=OPTUNA_N_FOLDS,
            random_state=RANDOM_STATE,
            use_saved_params=USE_SAVED_PARAMS,
            params_filepath=PARAMS_FILEPATH,
            encoded_cols_tag=encoded_cols_tag,
            use_gpu=USE_GPU,
            sample_size=OPTUNA_SAMPLE_SIZE,
            # 특정 모델만 Optuna로 튜닝하고 싶다면 여기서 지정 (예: ['lightgbm'])
            model_types=[],
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
            # LightGBM/XGBoost: 수치형 컬럼만 사용 (object/category/string 등 비수치 제외 → "to numeric" 에러 방지)
            feature_cols = [col for col in X_train.columns if pd.api.types.is_numeric_dtype(X_train[col])]
            X_train_model = X_train[feature_cols].copy()
            X_test_model = X_test[[c for c in feature_cols if c in X_test.columns]].copy()
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
        
        # AUC 값 저장 및 터미널 출력
        cv_score = trainer.cv_scores[model_type]['mean']
        cv_std = trainer.cv_scores[model_type]['std']
        fold_scores = trainer.cv_scores[model_type].get('fold_scores', [])
        print(f"  ✅ {model_type.upper()} 학습 완료 | CV {SCORING_METRIC.upper()}: {cv_score:.4f} (±{cv_std:.4f})")
        
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
        
    # Train vs Validation 점수 그래프 저장
    _plot_train_vs_val_scores(trainer, list(trainer.cv_scores.keys()), SCORING_METRIC, N_FOLDS)
    
    # 모델별 Feature Importance 분석
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
    
    # MLP 기반 '분석 초점' 수치화 (예측용 아님): 어디에 초점 맞출지 특성 중요도·상호작용·트리 비교
    if USE_MLP_FOCUS_ANALYSIS:
        all_cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols_for_mlp = [c for c in X_train.columns if c not in all_cat_cols]
        X_train_numeric = X_train[numeric_cols_for_mlp]
        # 모델별 importance 유지: CatBoost, LightGBM, XGBoost 각각 컬럼으로 저장
        analyze_mlp_focus(
            X_train=X_train_numeric,
            y_train=y_train,
            task_type=TASK_TYPE,
            random_state=RANDOM_STATE,
            save_dir=feature_importance_dir,
            model_importances=model_importances,
            top_n=30,
            top_interactions=50,
        )
    
    # Permutation Importance 분석 (config에서 활성화 시)
    if USE_PERMUTATION_IMPORTANCE:
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
    ensemble = EnsembleModel(task_type=TASK_TYPE, scoring_metric=SCORING_METRIC)
    ridge_alpha = RIDGE_ALPHA
    if ENSEMBLE_METHOD == 'ridge_meta' and OPTUNA_RIDGE_ALPHA:
        ridge_alpha = optimize_ridge_alpha(
            trainer.oof_predictions,
            y_train.values if isinstance(y_train, pd.Series) else y_train,
            task_type=TASK_TYPE,
            scoring_metric=SCORING_METRIC,
            n_trials=RIDGE_ALPHA_N_TRIALS,
            random_state=RANDOM_STATE
        )
    ensemble.fit(
        trainer.oof_predictions,
        y_train.values if isinstance(y_train, pd.Series) else y_train,
        method=ENSEMBLE_METHOD,
        optimize=(ENSEMBLE_METHOD == 'weighted_average'),
        ridge_alpha=ridge_alpha
    )
    
    # 앙상블 가중치(모델별 비율) 로그 출력
    if hasattr(ensemble, 'weights') and ensemble.weights:
        w_sum = sum(ensemble.weights.values())
        print(f"\n📊 앙상블 가중치 ({ENSEMBLE_METHOD}, 모델별 비율):")
        for name in sorted(ensemble.weights.keys()):
            w = ensemble.weights[name]
            pct = (w / w_sum * 100) if w_sum and abs(w_sum) > 1e-9 else 0.0
            print(f"  {name:12s}: {w:+.6f}  ({pct:5.1f}%)")
        if w_sum and abs(w_sum - 1.0) > 1e-6:
            print(f"  (가중치 합: {w_sum:.6f})")
    
    ensemble_pred = ensemble.predict(test_predictions)
    
    # 앙상블 AUC 값 저장 및 터미널 출력
    if hasattr(ensemble, 'ensemble_score') and ensemble.ensemble_score is not None:
        ensemble_auc = ensemble.ensemble_score
        print(f"  ✅ 앙상블 완료 | CV {SCORING_METRIC.upper()}: {ensemble_auc:.4f}")
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
        
    
    # 앙상블 Feature Importance 분석
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
        comparison_df = compare_model_and_ensemble_importance(
            model_importances=model_importances,
            ensemble_importance=ensemble_importance,
            top_n=30,
            save_dir=feature_importance_dir
        )
    
    # 제출 파일 생성 (분류 시 확률이 (n,2)이면 양성 클래스 확률만 사용)
    pred_submit = ensemble_pred[:, 1] if getattr(ensemble_pred, 'ndim', 1) == 2 else ensemble_pred
    submission = pd.DataFrame({
        ID_COL: df_sub[ID_COL],
        TARGET_COL: pred_submit
    })
    submission.to_csv(SUBMISSION_FILEPATH, index=False)
    
    return trainer, ensemble, submission


if __name__ == "__main__":
    trainer, ensemble, submission = main()
