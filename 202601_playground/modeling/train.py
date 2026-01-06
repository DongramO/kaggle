"""
모델 학습 및 앙상블 실행 스크립트
"""
import sys
import os
import pandas as pd
import numpy as np

# 상위 디렉토리 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 데이터 로드 함수 (경로에 맞게 수정)
def load_data():
    """데이터 로드 함수"""
    import os
    # modeling/train.py -> 202601_playground
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, 'data')
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    
    return df_train, df_test, df_sub
from preprocess.encoder import fit_encoder, transform_with_encoder
from preprocess.feature_engineering import apply_feature_engineering_pipeline
from modeling.model import (
    ModelTrainer, EnsembleModel, evaluate_model,
    HyperparameterOptimizer, save_hyperparameters, load_hyperparameters
)


def prepare_data(df_train, df_test, target_col='exam_score', 
                 use_feature_engineering=True, encoding_config=None):
    """
    데이터 전처리 및 준비
    
    Parameters:
    -----------
    df_train : pd.DataFrame
        학습 데이터
    df_test : pd.DataFrame
        테스트 데이터
    target_col : str
        타겟 컬럼 이름
    use_feature_engineering : bool
        Feature Engineering 적용 여부
    encoding_config : dict, optional
        인코딩 설정
        
    Returns:
    --------
    tuple
        (X_train, y_train, X_test, categorical_cols, numeric_cols, encoder)
    """
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
    
    # 컬럼 타입 분류
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"\n📊 데이터 정보:")
    print(f"  학습 데이터 크기: {X_train.shape}")
    print(f"  테스트 데이터 크기: {X_test.shape}")
    print(f"  수치형 컬럼: {len(numeric_cols)}개")
    print(f"  범주형 컬럼: {len(categorical_cols)}개")
    
    # Feature Engineering 적용
    if use_feature_engineering:
        print("\n🔧 Feature Engineering 적용 중...")
        # 간단한 설정 예제 (필요에 따라 수정)
        fe_config = {
            'clip_outliers': True,
            'create_frequency': True,  # 빈도 인코딩
        }
        
        X_train = apply_feature_engineering_pipeline(
            X_train,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=target_col,
            config=fe_config
        )
        
        X_test = apply_feature_engineering_pipeline(
            X_test,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=None,  # 테스트 데이터는 타겟 없음
            config=fe_config
        )
        
        # 업데이트된 컬럼 리스트
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 범주형 인코딩 (LightGBM, XGBoost용)
    # CatBoost는 범주형을 자동 처리하므로 원본 컬럼을 유지하고,
    # 인코딩된 컬럼은 별도로 태그를 달아서 구분
    encoder = None
    original_categorical_cols = categorical_cols.copy()  # CatBoost용 원본 범주형 컬럼 저장
    encoded_cols_tag = '_encoded'  # 인코딩된 컬럼 태그
    
    if len(categorical_cols) > 0:
        print("\n🔤 범주형 인코딩 적용 중...")
        print(f"  원본 범주형 컬럼: {len(categorical_cols)}개")
        print(f"  CatBoost는 원본 범주형 컬럼 사용, LightGBM/XGBoost는 인코딩된 컬럼 사용")
        
        if encoding_config is None:
            # 학습 데이터로 인코더 학습
            encoder = fit_encoder(
                X_train,
                categorical_cols=categorical_cols,
                encoding_type='ordinal',
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            
            # 인코더를 직접 사용하여 인코딩 (원본 컬럼은 유지)
            # OrdinalEncoder는 같은 이름의 컬럼을 덮어쓰므로, 새로운 이름으로 추가
            train_encoded_array = encoder.transform(X_train[categorical_cols])
            test_encoded_array = encoder.transform(X_test[categorical_cols])
            
            # 인코딩된 컬럼을 새로운 이름으로 추가
            for idx, col in enumerate(categorical_cols):
                encoded_col_name = f"{col}{encoded_cols_tag}"
                X_train[encoded_col_name] = train_encoded_array[:, idx]
                X_test[encoded_col_name] = test_encoded_array[:, idx]
            
            print(f"  인코딩된 컬럼 추가: {len(categorical_cols)}개 (태그: '{encoded_cols_tag}')")
            print(f"  원본 범주형 컬럼 유지: {len(categorical_cols)}개 (CatBoost용)")
    
    # 결측치 처리 (간단하게 평균/최빈값으로 대체)
    print("\n🔍 결측치 처리 중...")
    for col in numeric_cols:
        if col in X_train.columns and X_train[col].isnull().sum() > 0:
            mean_val = X_train[col].mean()
            X_train[col].fillna(mean_val, inplace=True)
            if col in X_test.columns:
                X_test[col].fillna(mean_val, inplace=True)
    
    print("✅ 데이터 준비 완료!")
    
    # 반환값:
    # - categorical_cols: CatBoost용 원본 범주형 컬럼 (인코딩되지 않음)
    # - encoded_cols_tag: 인코딩된 컬럼 태그 (LightGBM/XGBoost에서 사용)
    return X_train, y_train, X_test, original_categorical_cols, numeric_cols, encoder, encoded_cols_tag


def optimize_hyperparameters(X_train, y_train, categorical_cols, task_type='regression',
                              n_trials=50, random_state=42, use_saved_params=False, 
                              params_filepath=None, encoded_cols_tag='_encoded'):
    """
    Optuna를 사용한 하이퍼파라미터 최적화
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        학습 데이터
    y_train : pd.Series
        타겟 데이터
    categorical_cols : list
        CatBoost용 원본 범주형 컬럼 리스트
    task_type : str
        작업 타입 ('regression' or 'classification')
    n_trials : int
        Optuna 시도 횟수
    random_state : int
        랜덤 시드
    use_saved_params : bool
        저장된 파라미터 사용 여부
    params_filepath : str, optional
        파라미터 파일 경로
    encoded_cols_tag : str
        인코딩된 컬럼 태그
        
    Returns:
    --------
    dict
        최적화된 하이퍼파라미터
    """
    if use_saved_params and params_filepath and os.path.exists(params_filepath):
        print(f"\n📂 저장된 하이퍼파라미터 불러오기: {params_filepath}")
        best_params = load_hyperparameters(params_filepath)
        return best_params
    
    print(f"\n{'='*60}")
    print("🔍 Optuna를 사용한 하이퍼파라미터 최적화 시작")
    print(f"{'='*60}")
    
    optimizer = HyperparameterOptimizer(task_type=task_type, random_state=random_state)
    best_params = {}
    
    # 모델별로 사용할 컬럼 준비
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    
    # 각 모델별로 최적화
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        print(f"\n{'='*60}")
        print(f"🔍 {model_type.upper()} 하이퍼파라미터 최적화 (n_trials={n_trials})")
        print(f"{'='*60}")
        
        # 모델별로 사용할 컬럼 선택
        if model_type == 'catboost':
            # CatBoost: 원본 범주형 컬럼 사용
            feature_cols = [col for col in X_train.columns if col not in encoded_cols]
            X_train_model = X_train[feature_cols].copy()
            print(f"  CatBoost: 원본 범주형 컬럼 {len(categorical_cols)}개 사용")
            
            params = optimizer.optimize_catboost(
                X_train_model, y_train,
                n_trials=n_trials,
                cat_features=categorical_cols
            )
        elif model_type == 'lightgbm':
            # LightGBM: 인코딩된 컬럼 사용
            feature_cols = [col for col in X_train.columns if col not in categorical_cols]
            X_train_model = X_train[feature_cols].copy()
            print(f"  LightGBM: 인코딩된 컬럼 {len(encoded_cols)}개 사용")
            
            params = optimizer.optimize_lightgbm(
                X_train_model, y_train,
                n_trials=n_trials
            )
        elif model_type == 'xgboost':
            # XGBoost: 인코딩된 컬럼 사용
            feature_cols = [col for col in X_train.columns if col not in categorical_cols]
            X_train_model = X_train[feature_cols].copy()
            print(f"  XGBoost: 인코딩된 컬럼 {len(encoded_cols)}개 사용")
            
            params = optimizer.optimize_xgboost(
                X_train_model, y_train,
                n_trials=n_trials
            )
        
        best_params[model_type] = params
        
        print(f"\n✅ {model_type.upper()} 최적화 완료!")
        print(f"최적 파라미터:")
        for key, value in sorted(params.items()):
            print(f"  {key:25s}: {value}")
    
    # 최적화된 파라미터 저장
    if params_filepath:
        save_hyperparameters(
            best_params,
            params_filepath,
            task_type=task_type
        )
    
    return best_params


def train_models(X_train, y_train, X_test, categorical_cols, task_type='regression', 
                 n_folds=5, random_state=42, use_optuna=False, n_trials=50,
                 use_saved_params=False, params_filepath=None, encoded_cols_tag='_encoded',
                 use_gpu=False):
    """
    여러 모델 학습 및 예측
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        학습 데이터
    y_train : pd.Series
        타겟 데이터
    X_test : pd.DataFrame
        테스트 데이터
    categorical_cols : list
        CatBoost용 원본 범주형 컬럼 리스트
    task_type : str
        작업 타입 ('regression' or 'classification')
    n_folds : int
        K-Fold 개수
    random_state : int
        랜덤 시드
    use_optuna : bool
        Optuna를 사용한 하이퍼파라미터 최적화 여부
    n_trials : int
        Optuna 시도 횟수 (use_optuna=True일 때)
    use_saved_params : bool
        저장된 파라미터 사용 여부
    params_filepath : str, optional
        파라미터 파일 경로
    encoded_cols_tag : str
        인코딩된 컬럼 태그 (LightGBM/XGBoost에서 인코딩된 컬럼 식별용)
    use_gpu : bool
        GPU 사용 여부 (기본값: False)
        
    Returns:
    --------
    dict
        모델 학습 결과 및 예측
    """
    trainer = ModelTrainer(task_type=task_type, random_state=random_state, use_gpu=use_gpu)
    
    # 모델별로 사용할 컬럼 준비
    # CatBoost: 원본 범주형 컬럼 사용 (인코딩된 컬럼 제외)
    # LightGBM/XGBoost: 인코딩된 컬럼 사용 (원본 범주형 컬럼 제외)
    encoded_cols = [col for col in X_train.columns if col.endswith(encoded_cols_tag)]
    
    def get_features_for_model(model_type):
        """모델 타입에 따라 사용할 컬럼 반환"""
        if model_type == 'catboost':
            # CatBoost: 원본 범주형 컬럼 + 수치형 컬럼 (인코딩된 컬럼 제외)
            exclude_cols = encoded_cols
            return [col for col in X_train.columns if col not in exclude_cols]
        else:
            # LightGBM/XGBoost: 인코딩된 컬럼 + 수치형 컬럼 (원본 범주형 컬럼 제외)
            exclude_cols = categorical_cols
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
            encoded_cols_tag=encoded_cols_tag
        )
        
        # 모델별 파라미터 변환
        model_configs = {}
        
        # CatBoost 파라미터 변환
        if 'catboost' in best_params:
            cb_params = best_params['catboost'].copy()
            cb_params['iterations'] = cb_params.get('iterations', 1000)
            # CatBoost에 필요한 기본 파라미터 추가
            cb_params.setdefault('random_state', random_state)
            cb_params.setdefault('verbose', False)
            cb_params.setdefault('allow_writing_files', False)
            model_configs['catboost'] = cb_params
        
        # LightGBM 파라미터 변환
        if 'lightgbm' in best_params:
            lgb_params = best_params['lightgbm'].copy()
            # LightGBM은 num_boost_round를 fit에서 전달하므로 여기서는 제거
            # 필요한 기본 파라미터 추가
            lgb_params.setdefault('random_state', random_state)
            lgb_params.setdefault('verbosity', -1)
            model_configs['lightgbm'] = lgb_params
        
        # XGBoost 파라미터 변환
        if 'xgboost' in best_params:
            xgb_params = best_params['xgboost'].copy()
            # XGBoost는 num_boost_round를 fit에서 전달하므로 여기서는 제거
            # 필요한 기본 파라미터 추가
            xgb_params.setdefault('random_state', random_state)
            xgb_params.setdefault('verbosity', 0)
            model_configs['xgboost'] = xgb_params
        
        trainer.best_params = best_params
    else:
        # 기본 하이퍼파라미터 설정
        model_configs = {
            'catboost': {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'random_strength': 1,
                'bagging_temperature': 1,
                'od_type': 'Iter',
                'od_wait': 50,
            },
            'lightgbm': {
                'num_boost_round': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': 6,
                'min_data_in_leaf': 20,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
            },
            'xgboost': {
                'num_boost_round': 1000,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
            }
        }
    
    # 각 모델 학습
    test_predictions = {}
    
    for model_type in ['catboost', 'lightgbm', 'xgboost']:
        print(f"\n{'='*60}")
        print(f"🚀 {model_type.upper()} 모델 학습 시작")
        print(f"{'='*60}")
        
        # 모델별로 사용할 컬럼 선택
        feature_cols = get_features_for_model(model_type)
        X_train_model = X_train[feature_cols].copy()
        X_test_model = X_test[feature_cols].copy()
        
        # CatBoost는 범주형 특성을 자동 처리하므로 cat_features 전달
        # LightGBM/XGBoost는 인코딩된 컬럼만 사용하므로 cat_features는 None
        cat_features = categorical_cols if model_type == 'catboost' else None
        
        if model_type == 'catboost':
            print(f"  사용 컬럼: 원본 범주형 {len(categorical_cols)}개 + 수치형 (인코딩된 컬럼 제외)")
        else:
            print(f"  사용 컬럼: 인코딩된 컬럼 {len(encoded_cols)}개 + 수치형 (원본 범주형 컬럼 제외)")
        
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
        'best_params': trainer.best_params
    }


def create_ensemble(oof_predictions, y_train, test_predictions, task_type='regression'):
    """
    앙상블 모델 생성 및 예측
    
    Parameters:
    -----------
    oof_predictions : dict
        OOF 예측 딕셔너리
    y_train : pd.Series
        실제 타겟 값
    test_predictions : dict
        테스트 예측 딕셔너리
    task_type : str
        작업 타입
        
    Returns:
    --------
    np.ndarray
        앙상블 예측 결과
    """
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


def main(use_optuna=False, n_trials=50, use_saved_params=False, 
         params_filepath='../data/modeling/best_hyperparameters.json', use_gpu=False):
    """
    메인 실행 함수
    
    Parameters:
    -----------
    use_optuna : bool
        Optuna를 사용한 하이퍼파라미터 최적화 여부
    n_trials : int
        Optuna 시도 횟수
    use_saved_params : bool
        저장된 파라미터 사용 여부
    params_filepath : str
        파라미터 파일 경로
    use_gpu : bool
        GPU 사용 여부 (기본값: False)
    """
    print("="*60)
    print("🚀 모델 학습 및 앙상블 시작")
    print("="*60)
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    df_train, df_test, df_sub = load_data()
    
    # 데이터 준비
    X_train, y_train, X_test, categorical_cols, numeric_cols, encoder, encoded_cols_tag = prepare_data(
        df_train, df_test,
        target_col='exam_score',
        use_feature_engineering=True
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
        use_gpu=use_gpu
    )
    
    # 앙상블 생성
    ensemble_pred, ensemble = create_ensemble(
        results['oof_predictions'],
        y_train,
        results['test_predictions'],
        task_type='regression'
    )
    
    # 제출 파일 생성
    print("\n📝 제출 파일 생성 중...")
    submission = pd.DataFrame({
        'id': df_sub['id'],
        'exam_score': ensemble_pred
    })
    
    submission_path = '../submission_ensemble.csv'
    submission.to_csv(submission_path, index=False)
    print(f"✅ 제출 파일 저장 완료: {submission_path}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 학습 결과 요약")
    print(f"{'='*60}")
    
    trainer = results['trainer']
    for model_type, scores in trainer.cv_scores.items():
        print(f"\n{model_type.upper()}:")
        print(f"  CV Score: {scores['mean']:.4f} (std: {scores['std']:.4f})")
    
    # 최적화된 하이퍼파라미터가 있으면 저장
    if 'best_params' in results and results['best_params']:
        print(f"\n💾 최적화된 하이퍼파라미터 저장 중...")
        save_hyperparameters(
            results['best_params'],
            params_filepath,
            task_type='regression',
            additional_info={
                'cv_scores': {k: {'mean': v['mean'], 'std': v['std']} 
                            for k, v in trainer.cv_scores.items()}
            }
        )
    
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

