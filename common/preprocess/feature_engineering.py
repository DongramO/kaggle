import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any


def _get_config_value(config: dict, key: str, default_flag: bool = False) -> tuple[bool, dict]:
    """config에서 설정값 추출 (헬퍼 함수)"""
    config_value = config.get(key, default_flag if default_flag else {})
    
    # dict 구조인 경우
    if isinstance(config_value, dict):
        flag = config_value.get('flag', default_flag if default_flag else False)
        return flag, config_value
    
    # bool 값인 경우 (기존 flat 구조 호환성)
    if isinstance(config_value, bool):
        if config_value:
            # flat 구조에서 설정값들을 추출
            flat_config = {}
            # 각 기능별로 flat 구조의 키들을 매핑
            flat_key_mapping = {
                'clip_outliers': ['clip_rules'],
                'transform_numeric': ['transformations'],
                'create_interactions': ['feature_pairs', 'operations'],
                'create_ratios': ['numerator_cols', 'denominator_cols', 'ratio_feature_names'],
                'create_statistical': ['feature_groups', 'statistics'],
                'create_categorical_interactions': ['categorical_pairs', 'separator'],
                'create_frequency': [],
                'create_binning': ['n_bins', 'binning_strategy'],
            }
            
            for flat_key in flat_key_mapping.get(key, []):
                if flat_key in config:
                    flat_config[flat_key] = config[flat_key]
            
            return True, flat_config
        else:
            return False, {}
    
    # 기본값
    return default_flag, {}


def transform_numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    transformations: Optional[List[str]] = None
) -> pd.DataFrame:
    """수치형 특성에 변환 적용 (log, sqrt, square, reciprocal 등)"""
    if transformations is None:
        transformations = ['log', 'sqrt']
    
    df = df.copy()
    eps = 1e-6
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
            
        for transform in transformations:
            if transform == 'log':
                df[f'{col}_log'] = np.log1p(df[col] + eps)
            elif transform == 'sqrt':
                df[f'{col}_sqrt'] = np.sqrt(df[col] + eps)
            elif transform == 'square':
                df[f'{col}_square'] = df[col] ** 2
            elif transform == 'reciprocal':
                df[f'{col}_reciprocal'] = 1 / (df[col] + eps)
    
    return df


def create_interaction_features(
    df: pd.DataFrame,
    feature_pairs: List[tuple],
    operations: Optional[List[str]] = None
) -> pd.DataFrame:
    """특성 간 상호작용 특성 생성 (multiply, divide, add, subtract)"""
    df = df.copy()
    eps = 1e-6
    
    if operations is None:
        # operations가 지정되지 않으면 feature_pairs 개수만큼 'multiply' 반복
        operations = ['multiply'] * len(feature_pairs)
    
    # 1:1 매핑: feature_pairs와 operations 개수가 일치해야 함
    if len(feature_pairs) != len(operations):
        raise ValueError(
            f"feature_pairs({len(feature_pairs)}개)와 operations({len(operations)}개)의 개수가 일치하지 않습니다. "
            f"1:1 매핑을 위해서는 개수가 같아야 합니다."
        )
    
    # 각 feature_pair에 대해 해당하는 operation만 적용
    for (col1, col2), op in zip(feature_pairs, operations):
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        if op == 'multiply':
            df[f'{col1}_x_{col2}_FE'] = df[col1] * df[col2]
        elif op == 'divide':
            df[f'{col1}_div_{col2}_FE'] = df[col1] / (df[col2] + eps)
        elif op == 'add':
            df[f'{col1}_add_{col2}_FE'] = df[col1] + df[col2]
        elif op == 'subtract':
            df[f'{col1}_subtract_{col2}_FE'] = df[col1] - df[col2]
        else:
            raise ValueError(f"지원하지 않는 연산: {op}. 'multiply', 'divide', 'add', 'subtract' 중 하나를 사용하세요.")
    
    return df


def create_ratio_features(
    df: pd.DataFrame,
    numerator_cols: List[str],
    denominator_cols: List[str],
    feature_names: Optional[Union[List[str], str]] = None
) -> pd.DataFrame:
    """비율 특성 생성"""
    df = df.copy()
    eps = 1e-6
    
    for i, (num_col, denom_col) in enumerate(zip(numerator_cols, denominator_cols)):
        if num_col not in df.columns or denom_col not in df.columns:
            continue
        
        if feature_names is None:
            # 기본 형식: {num_col}_{denom_col}_ratio
            feature_name = f'{num_col}_{denom_col}_ratio'
        elif isinstance(feature_names, str):
            # 문자열인 경우: suffix로 사용 {num_col}_{denom_col}{suffix}
            feature_name = f'{num_col}_{denom_col}{feature_names}'
        elif isinstance(feature_names, list) and i < len(feature_names):
            # 리스트인 경우: 지정된 이름 사용
            feature_name = feature_names[i]
        else:
            # 기본 형식 (fallback)
            feature_name = f'{num_col}_{denom_col}_ratio'
        
        df[feature_name] = df[num_col] / (df[denom_col] + eps)
    
    return df


def create_statistical_features(
    df: pd.DataFrame,
    feature_groups: List[List[str]],
    statistics: Optional[List[str]] = None
) -> pd.DataFrame:
    """여러 특성의 통계적 특성 생성 (mean, std, max, min, median, sum 등)"""
    if statistics is None:
        statistics = ['mean', 'std', 'max', 'min']
    
    df = df.copy()
    
    for group in feature_groups:
        valid_cols = [col for col in group if col in df.columns]
        if len(valid_cols) == 0:
            continue
        
        group_name = '_'.join(valid_cols[:2])  # 그룹 이름 생성
        
        for stat in statistics:
            if stat == 'mean':
                df[f'{group_name}_mean'] = df[valid_cols].mean(axis=1)
            elif stat == 'std':
                df[f'{group_name}_std'] = df[valid_cols].std(axis=1)
            elif stat == 'max':
                df[f'{group_name}_max'] = df[valid_cols].max(axis=1)
            elif stat == 'min':
                df[f'{group_name}_min'] = df[valid_cols].min(axis=1)
            elif stat == 'median':
                df[f'{group_name}_median'] = df[valid_cols].median(axis=1)
            elif stat == 'sum':
                df[f'{group_name}_sum'] = df[valid_cols].sum(axis=1)
    
    return df


def create_categorical_encoded_interaction(
    df: pd.DataFrame,
    categorical_col: str,
    numeric_col: str,
    encoded_cols_tag: str = '_encoded',
    feature_name: Optional[str] = None
) -> pd.DataFrame:
    """범주형 one-hot 인코딩 컬럼과 수치형 컬럼의 상호작용 특성 생성"""
    df = df.copy()
    
    # 컬럼 존재 확인
    if categorical_col not in df.columns:
        return df
    if numeric_col not in df.columns:
        return df
    
    # 해당 범주형 컬럼의 모든 encoded 컬럼 찾기
    encoded_cols = [col for col in df.columns 
                  if categorical_col in col and col.endswith(encoded_cols_tag)]
    
    if len(encoded_cols) == 0:
        return df
    
    # 매핑 딕셔너리: 범주형 값 -> encoded 컬럼 이름
    # 예: 'coaching' -> 'study_method_coaching_encoded'
    value_to_col = {}
    for col in encoded_cols:
        # 'study_method_coaching_encoded' -> 'coaching'
        value = col.replace(f'{categorical_col}_', '').replace(encoded_cols_tag, '')
        value_to_col[value] = col
    
    # 특성 이름 생성
    if feature_name is None:
        feature_name = f'{categorical_col}_encoded_x_{numeric_col}_FE'
    
    # 벡터화된 연산: 각 범주형 값에 해당하는 encoded 컬럼만 선택하여 곱하기
    result = pd.Series(0.0, index=df.index, dtype=float)
    for value, col_name in value_to_col.items():
        if col_name in df.columns:
            # 해당 범주형 값을 가진 row들만 선택
            mask = df[categorical_col] == value
            # encoded 값(0 또는 1) * 수치형 값
            result[mask] = df.loc[mask, col_name] * df.loc[mask, numeric_col]
    
    df[feature_name] = result
    
    return df


def create_categorical_interactions(
    df: pd.DataFrame,
    categorical_pairs: List[tuple],
    separator: str = '_'
) -> pd.DataFrame:
    """범주형 특성 간 조합 특성 생성"""
    df = df.copy()
    
    for col1, col2 in categorical_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        feature_name = f'{col1}{separator}{col2}'
        df[feature_name] = df[col1].astype(str) + separator + df[col2].astype(str)
    
    return df


def convert_ordered_categorical_to_numeric(
    df: pd.DataFrame,
    categorical_col: str,
    mapping: Dict[str, Union[int, float]]
) -> pd.DataFrame:
    """순서가 있는 범주형 변수를 숫자로 변환"""
    df = df.copy()
    
    if categorical_col not in df.columns:
        return df
    
    numeric_col_name = f'{categorical_col}_numeric'
    df[numeric_col_name] = df[categorical_col].map(mapping)
    
    # 매핑되지 않은 값은 NaN이 될 수 있으므로 처리
    if df[numeric_col_name].isna().any():
        unmapped = df[df[numeric_col_name].isna()][categorical_col].unique()
        print(f"   ⚠️ {categorical_col}에서 매핑되지 않은 값: {unmapped}")
        # 기본값으로 가장 작은 값 사용
        default_value = min(mapping.values()) if mapping else 0
        df[numeric_col_name] = df[numeric_col_name].fillna(default_value)
    
    return df


def clip_outliers(
    df: pd.DataFrame,
    numeric_cols: List[str],
    clip_rules: Optional[dict] = None,
    default_quantile: float = 0.995
) -> pd.DataFrame:
    """이상치 클리핑"""
    df = df.copy()
    
    if clip_rules is None:
        clip_rules = {col: default_quantile for col in numeric_cols}
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        quantile = clip_rules.get(col, default_quantile)
        upper = df[col].quantile(quantile)
        lower = df[col].quantile(1 - quantile) if quantile > 0.5 else df[col].min()
        
        df[col] = df[col].clip(lower=lower, upper=upper)
    
    return df


def create_frequency_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> pd.DataFrame:
    """범주형 특성의 빈도 인코딩 특성 생성"""
    df = df.copy()
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        frequency_map = df[col].value_counts().to_dict()
        df[f'{col}_freq'] = df[col].map(frequency_map)
    
    return df


def create_binning_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    n_bins: int = 5,
    strategy: str = 'quantile'
) -> pd.DataFrame:
    """수치형 특성을 구간화(binning)하여 범주형 특성으로 변환"""
    from sklearn.preprocessing import KBinsDiscretizer
    
    df = df.copy()
    
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        if strategy == 'quantile':
            encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        elif strategy == 'uniform':
            encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        elif strategy == 'kmeans':
            encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
        else:
            continue
        
        df[f'{col}_binned'] = encoder.fit_transform(df[[col]]).astype(int).flatten()
    
    return df


def create_polynomial_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    degree: int = 2,
    interaction_only: bool = False
) -> pd.DataFrame:
    """다항식 특성 생성"""
    from sklearn.preprocessing import PolynomialFeatures
    
    df = df.copy()
    valid_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(valid_cols) == 0:
        return df
    
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_features = poly.fit_transform(df[valid_cols])
    
    feature_names = poly.get_feature_names_out(valid_cols)
    
    for i, feature_name in enumerate(feature_names):
        if feature_name not in df.columns:  # 원본 특성 제외
            df[feature_name] = poly_features[:, i]
    
    return df


def apply_feature_engineering_pipeline(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    target_col: Optional[str] = None,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """전체 feature engineering 파이프라인 적용"""
    if config is None:
        config = {}
    
    df = df.copy()
    
    # 이상치 클리핑
    is_enabled, clip_cfg = _get_config_value(config, 'clip_outliers', default_flag=True)
    if is_enabled:
        df = clip_outliers(df, numeric_cols, clip_cfg.get('clip_rules', None))
    
    # 수치형 특성 변환
    is_enabled, transform_cfg = _get_config_value(config, 'transform_numeric', default_flag=False)
    if is_enabled:
        transformations = transform_cfg.get('transformations', ['log', 'sqrt'])
        df = transform_numeric_features(df, numeric_cols, transformations)
    
    # 상호작용 특성 생성
    is_enabled, interaction_cfg = _get_config_value(config, 'create_interactions', default_flag=False)
    if is_enabled:
        feature_pairs = interaction_cfg.get('feature_pairs', [])
        operations = interaction_cfg.get('operations', ['multiply', 'divide'])
        df = create_interaction_features(df, feature_pairs, operations)
    
    # 비율 특성 생성
    is_enabled, ratio_cfg = _get_config_value(config, 'create_ratios', default_flag=False)
    if is_enabled:
        numerator_cols = ratio_cfg.get('numerator_cols', [])
        denominator_cols = ratio_cfg.get('denominator_cols', [])
        feature_names = ratio_cfg.get('ratio_feature_names', None)
        df = create_ratio_features(df, numerator_cols, denominator_cols, feature_names)
    
    # 통계적 특성 생성
    is_enabled, statistical_cfg = _get_config_value(config, 'create_statistical', default_flag=False)
    if is_enabled:
        feature_groups = statistical_cfg.get('feature_groups', [])
        statistics = statistical_cfg.get('statistics', ['mean', 'std'])
        df = create_statistical_features(df, feature_groups, statistics)
    
    # 범주형 조합 특성 생성
    is_enabled, cat_interaction_cfg = _get_config_value(config, 'create_categorical_interactions', default_flag=False)
    if is_enabled:
        categorical_pairs = cat_interaction_cfg.get('categorical_pairs', [])
        separator = cat_interaction_cfg.get('separator', '_')
        df = create_categorical_interactions(df, categorical_pairs, separator)
    
    # 빈도 인코딩
    is_enabled, _ = _get_config_value(config, 'create_frequency', default_flag=False)
    if is_enabled:
        df = create_frequency_features(df, categorical_cols)
    
    # 구간화
    is_enabled, binning_cfg = _get_config_value(config, 'create_binning', default_flag=False)
    if is_enabled:
        n_bins = binning_cfg.get('n_bins', 5)
        strategy = binning_cfg.get('binning_strategy', 'quantile')
        df = create_binning_features(df, numeric_cols, n_bins, strategy)
    
    return df


# ============================================================================
# Config 예제
# ============================================================================

# example_config_full = {
#     # 이상치 클리핑
#     'clip_outliers': True,
#     'clip_rules': {
#         'study_hours': 0.995,
#         'class_attendance': 0.99,
#         'sleep_hours': 0.995,
#         'age': 0.99,
#     },
    
#     # 수치형 특성 변환
#     'transform_numeric': True,
#     'transformations': ['log', 'sqrt'],
    
#     # 상호작용 특성 생성
#     'create_interactions': True,
#     'feature_pairs': [
#         ('study_hours', 'class_attendance'),
#         ('study_hours', 'sleep_hours'),
#         ('class_attendance', 'sleep_hours'),
#         ('age', 'study_hours'),
#     ],
#     'operations': ['multiply', 'divide'],
    
#     # 비율 특성 생성
#     'create_ratios': True,
#     'numerator_cols': ['study_hours', 'class_attendance'],
#     'denominator_cols': ['sleep_hours', 'age'],
#     'ratio_feature_names': ['study_efficiency', 'attendance_rate'],
    
#     # 통계적 특성 생성
#     'create_statistical': True,
#     'feature_groups': [
#         ['study_hours', 'class_attendance', 'sleep_hours'],
#         ['age', 'study_hours'],
#     ],
#     'statistics': ['mean', 'std', 'max', 'min'],
    
#     # 범주형 조합 특성 생성
#     'create_categorical_interactions': True,
#     'categorical_pairs': [
#         ('gender', 'course'),
#         ('study_method', 'facility_rating'),
#         ('sleep_quality', 'exam_difficulty'),
#     ],
#     'separator': '_',
    
#     # 빈도 인코딩
#     'create_frequency': True,
    
#     # 구간화
#     'create_binning': True,
#     'n_bins': 5,
#     'binning_strategy': 'quantile',
# }

# # 예제 4: 고급 설정 (다양한 변환 포함)
# example_config_advanced = {
#     'clip_outliers': True,
#     'clip_rules': {
#         'study_hours': 0.99,
#         'class_attendance': 0.995,
#         'sleep_hours': 0.99,
#     },
    
#     'transform_numeric': True,
#     'transformations': ['log', 'sqrt', 'square'],
    
#     'create_interactions': True,
#     'feature_pairs': [
#         ('study_hours', 'class_attendance'),
#         ('study_hours', 'sleep_hours'),
#         ('class_attendance', 'sleep_hours'),
#         ('age', 'study_hours'),
#         ('age', 'sleep_hours'),
#     ],
#     'operations': ['multiply', 'divide', 'add'],
    
#     'create_ratios': True,
#     'numerator_cols': [
#         'study_hours',
#         'class_attendance',
#         'study_hours',
#     ],
#     'denominator_cols': [
#         'sleep_hours',
#         'age',
#         'age',
#     ],
#     'ratio_feature_names': [
#         'study_to_sleep_ratio',
#         'attendance_to_age_ratio',
#         'study_to_age_ratio',
#     ],
    
#     'create_statistical': True,
#     'feature_groups': [
#         ['study_hours', 'class_attendance', 'sleep_hours'],
#         ['age', 'study_hours', 'class_attendance'],
#     ],
#     'statistics': ['mean', 'std', 'max', 'min', 'median'],
    
#     'create_categorical_interactions': True,
#     'categorical_pairs': [
#         ('gender', 'course'),
#         ('study_method', 'facility_rating'),
#         ('sleep_quality', 'exam_difficulty'),
#         ('gender', 'study_method'),
#     ],
#     'separator': '_',
    
#     'create_frequency': True,
    
#     'create_binning': True,
#     'n_bins': 10,
#     'binning_strategy': 'quantile',
# }

# 사용 예제:
# from preprocess.feature_engineering import apply_feature_engineering_pipeline, example_config_full
# 
# numeric_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
# categorical_cols = ['gender', 'course', 'internet_access', 'sleep_quality', 
#                     'study_method', 'facility_rating', 'exam_difficulty']
# 
# df_processed = apply_feature_engineering_pipeline(
#     df=df_train,
#     numeric_cols=numeric_cols,
#     categorical_cols=categorical_cols,
#     target_col='exam_score',
#     config=example_config_full
# )

