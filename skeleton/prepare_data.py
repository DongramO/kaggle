"""
데이터 준비(전처리) 모듈 (독립 스켈레톤)
- 결측 처리, 타입 변환, 기본 전처리
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


def prepare_data(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    기본 전처리 적용.
    - 결측치 처리
    - 필요 시 타입 변환
    """
    df_cp = df.copy()

    numeric_cols = df_cp.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_cp.select_dtypes(include=['object', 'category']).columns.tolist()

    print('numeric_cols: ', numeric_cols)
    print('categorical_cols: ', categorical_cols)
    print('--------------------------------')
    
    for col in numeric_cols:

        # numeric_col에 대한 결측치 처리
        df_cp[col] = df_cp[col].fillna(df_cp[col].median())
        
        # numeric_col에 대한 clipping 진행
        upper = df_cp[col].quantile(0.99)
        lower = df_cp[col].quantile(0.01)
        df_cp[col] = df_cp[col].clip(lower=lower, upper=upper)


    for col in categorical_cols:

        # categorical_col에 대한 결측치 처리 (나중에 다시 확인하기)
        df_cp[col] = df_cp[col].fillna(df_cp[col].mode().iloc[0] if len(df_cp[col].mode()) else "")


    # encoding 관련 작업 진행 (target_col 제외한 범주형 컬럼)
    encode_cols = [c for c in categorical_cols if c != target_col] if target_col else categorical_cols

    df_cp = one_hot_encoding(df_cp, target_cols=[])
    df_cp = ordinal_encoding(df_cp, target_cols=[])
    df_cp = label_encoding(df_cp, target_cols=encode_cols)
    df_cp = target_encoding(df_cp, target_cols=[], target_col=target_col)

    
    # feature_engineeering 작업 진행

    df_cp = feature_engineering(df_cp)


    return df_cp

def get_feature_columns(df: pd.DataFrame, target_col: str | None = None, id_col: str | None = None) -> list[str]:
    """특성 컬럼 목록 반환 (target, id 제외)."""
    exclude = {c for c in (target_col, id_col) if c is not None}
    return [c for c in df.columns if c not in exclude]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """신규 컬럼 생성을 위해 특성 공학 진행 """
    eps = 1e-6

    X = df.copy()


    return X


def one_hot_encoding(df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
    """범주형 컬럼을 원-핫 인코딩."""
    X = df.copy()

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded = encoder.fit_transform(X[target_cols].astype(str).fillna("__NA__"))
    feature_names = encoder.get_feature_names_out(target_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
    X = X.drop(columns=target_cols)
    X = pd.concat([X, encoded_df], axis=1)
    return X


def ordinal_encoding(df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
    """범주형 컬럼을 순서형 인코딩."""
    X = df.copy()

    categories = [sorted(X[col].astype(str).fillna("__NA__").unique().tolist()) for col in target_cols]
    encoder = OrdinalEncoder(categories=categories, handle_unknown="use_encoded_value", unknown_value=-1)
    X[target_cols] = encoder.fit_transform(X[target_cols].astype(str).fillna("__NA__"))
    return X


def label_encoding(df: pd.DataFrame, target_cols: list[str] | None = None) -> pd.DataFrame:
    """범주형 컬럼을 수치로 인코딩."""
    X = df.copy()

    for col in target_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna("__NA__"))
    
    return X


def target_encoding(df: pd.DataFrame, target_cols: list[str] | None = None, target_col: str | None = None) -> pd.DataFrame:
    """범주형 컬럼을 타겟 평균으로 인코딩."""
    X = df.copy()

    y = X[target_col]
    global_mean = y.mean()
    smoothing = 10.0

    for col in target_cols:
        stats = X.groupby(col)[target_col].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        X[col] = X[col].map(smooth).fillna(global_mean)
    
    return X