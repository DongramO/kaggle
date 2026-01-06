import numpy as np
import pandas as pd
from typing import List, Optional, Union


def transform_numeric_features(
    df: pd.DataFrame,
    numeric_cols: List[str],
    transformations: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    수치형 특성에 다양한 변환을 적용합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    numeric_cols : List[str]
        변환할 수치형 컬럼 리스트
    transformations : List[str], optional
        적용할 변환 방법 리스트 ('log', 'sqrt', 'square', 'reciprocal' 등)
        기본값: ['log', 'sqrt']
    
    Returns:
    --------
    pd.DataFrame
        변환이 적용된 데이터프레임
    """
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
    """
    두 특성 간의 상호작용 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    feature_pairs : List[tuple]
        (col1, col2) 형태의 튜플 리스트
    operations : List[str], optional
        적용할 연산 리스트 ('multiply', 'divide', 'add', 'subtract' 등)
        기본값: ['multiply', 'divide']
    
    Returns:
    --------
    pd.DataFrame
        상호작용 특성이 추가된 데이터프레임
    """
    if operations is None:
        operations = ['multiply', 'divide', 'add', 'subtract']
    
    df = df.copy()
    eps = 1e-6
    
    for col1, col2 in feature_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        for op in operations:
            if op == 'multiply':
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
            elif op == 'divide':
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + eps)
            elif op == 'add':
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            elif op == 'subtract':
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
    
    return df


def create_ratio_features(
    df: pd.DataFrame,
    numerator_cols: List[str],
    denominator_cols: List[str],
    feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    비율 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    numerator_cols : List[str]
        분자 컬럼 리스트
    denominator_cols : List[str]
        분모 컬럼 리스트
    feature_names : List[str], optional
        생성될 특성 이름 리스트 (기본값: 자동 생성)
    
    Returns:
    --------
    pd.DataFrame
        비율 특성이 추가된 데이터프레임
    """
    df = df.copy()
    eps = 1e-6
    
    for i, (num_col, denom_col) in enumerate(zip(numerator_cols, denominator_cols)):
        if num_col not in df.columns or denom_col not in df.columns:
            continue
        
        if feature_names and i < len(feature_names):
            feature_name = feature_names[i]
        else:
            feature_name = f'{num_col}_div_{denom_col}'
        
        df[feature_name] = df[num_col] / (df[denom_col] + eps)
    
    return df


def create_statistical_features(
    df: pd.DataFrame,
    feature_groups: List[List[str]],
    statistics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    여러 특성의 통계적 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    feature_groups : List[List[str]]
        통계를 계산할 특성 그룹 리스트
    statistics : List[str], optional
        계산할 통계량 리스트 ('mean', 'std', 'max', 'min', 'median' 등)
        기본값: ['mean', 'std', 'max', 'min']
    
    Returns:
    --------
    pd.DataFrame
        통계적 특성이 추가된 데이터프레임
    """
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


def create_categorical_interactions(
    df: pd.DataFrame,
    categorical_pairs: List[tuple],
    separator: str = '_'
) -> pd.DataFrame:
    """
    범주형 특성 간의 조합 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    categorical_pairs : List[tuple]
        (col1, col2) 형태의 범주형 컬럼 튜플 리스트
    separator : str
        조합된 특성 이름의 구분자 (기본값: '_')
    
    Returns:
    --------
    pd.DataFrame
        범주형 조합 특성이 추가된 데이터프레임
    """
    df = df.copy()
    
    for col1, col2 in categorical_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        feature_name = f'{col1}{separator}{col2}'
        df[feature_name] = df[col1].astype(str) + separator + df[col2].astype(str)
    
    return df


def clip_outliers(
    df: pd.DataFrame,
    numeric_cols: List[str],
    clip_rules: Optional[dict] = None,
    default_quantile: float = 0.995
) -> pd.DataFrame:
    """
    이상치를 클리핑합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    numeric_cols : List[str]
        클리핑할 수치형 컬럼 리스트
    clip_rules : dict, optional
        컬럼별 클리핑 규칙 {col_name: quantile_value}
        기본값: 모든 컬럼에 default_quantile 적용
    default_quantile : float
        기본 분위수 값 (기본값: 0.995)
    
    Returns:
    --------
    pd.DataFrame
        이상치가 클리핑된 데이터프레임
    """
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
    """
    범주형 특성의 빈도 인코딩 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    categorical_cols : List[str]
        빈도 인코딩할 범주형 컬럼 리스트
    
    Returns:
    --------
    pd.DataFrame
        빈도 인코딩 특성이 추가된 데이터프레임
    """
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
    """
    수치형 특성을 구간화(binning)하여 범주형 특성으로 변환합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    numeric_cols : List[str]
        구간화할 수치형 컬럼 리스트
    n_bins : int
        구간 개수 (기본값: 5)
    strategy : str
        구간화 전략 ('quantile', 'uniform', 'kmeans')
        기본값: 'quantile'
    
    Returns:
    --------
    pd.DataFrame
        구간화된 특성이 추가된 데이터프레임
    """
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
    """
    다항식 특성을 생성합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    numeric_cols : List[str]
        다항식 특성을 생성할 수치형 컬럼 리스트
    degree : int
        다항식 차수 (기본값: 2)
    interaction_only : bool
        상호작용 항만 생성할지 여부 (기본값: False)
    
    Returns:
    --------
    pd.DataFrame
        다항식 특성이 추가된 데이터프레임
    """
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
    """
    전체 feature engineering 파이프라인을 적용합니다.
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    numeric_cols : List[str]
        수치형 컬럼 리스트
    categorical_cols : List[str]
        범주형 컬럼 리스트
    target_col : str, optional
        타겟 컬럼 이름 (타겟 인코딩에 사용)
    config : dict, optional
        파이프라인 설정 딕셔너리
    
    Returns:
    --------
    pd.DataFrame
        feature engineering이 적용된 데이터프레임
    """
    if config is None:
        config = {}
    
    df = df.copy()
    
    # 이상치 클리핑
    if config.get('clip_outliers', True):
        clip_rules = config.get('clip_rules', None)
        df = clip_outliers(df, numeric_cols, clip_rules)
    
    # 수치형 특성 변환
    if config.get('transform_numeric', False):
        transformations = config.get('transformations', ['log', 'sqrt'])
        df = transform_numeric_features(df, numeric_cols, transformations)
    
    # 상호작용 특성 생성
    if config.get('create_interactions', False):
        feature_pairs = config.get('feature_pairs', [])
        operations = config.get('operations', ['multiply', 'divide'])
        df = create_interaction_features(df, feature_pairs, operations)
    
    # 비율 특성 생성
    if config.get('create_ratios', False):
        numerator_cols = config.get('numerator_cols', [])
        denominator_cols = config.get('denominator_cols', [])
        feature_names = config.get('ratio_feature_names', None)
        df = create_ratio_features(df, numerator_cols, denominator_cols, feature_names)
    
    # 통계적 특성 생성
    if config.get('create_statistical', False):
        feature_groups = config.get('feature_groups', [])
        statistics = config.get('statistics', ['mean', 'std'])
        df = create_statistical_features(df, feature_groups, statistics)
    
    # 범주형 조합 특성 생성
    if config.get('create_categorical_interactions', False):
        categorical_pairs = config.get('categorical_pairs', [])
        separator = config.get('separator', '_')
        df = create_categorical_interactions(df, categorical_pairs, separator)
    
    # 빈도 인코딩
    if config.get('create_frequency', False):
        df = create_frequency_features(df, categorical_cols)
    
    # 구간화
    if config.get('create_binning', False):
        n_bins = config.get('n_bins', 5)
        strategy = config.get('binning_strategy', 'quantile')
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

