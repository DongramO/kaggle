"""
데이터 준비(전처리) 모듈 (독립 스켈레톤)
- 결측 처리, 타입 변환, 기본 전처리
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from eda import correlation_analysis

# 순서 없는 범주형 → OneHot
ONEHOT_COLS = ['gender', 'PaymentMethod', 'InternetService']

# 순서 있는 범주형 → Ordinal (순서 명시)
ORDINAL_COLS = {
    'Contract': ['Month-to-month', 'One year', 'Two year'],  # 0, 1, 2
}

# 의미 기반 직접 매핑 (No/서비스없음=0, Yes=1)
DIRECT_MAPPING_COLS = {
    'PhoneService':      {'No': 0, 'Yes': 1},
    'Partner':           {'No': 0, 'Yes': 1},
    'Dependents':        {'No': 0, 'Yes': 1},
    'PaperlessBilling':  {'No': 0, 'Yes': 1},
    'MultipleLines':     {'No': 0, 'Yes': 1, 'No phone service': 0},
    'OnlineSecurity':    {'No': 0, 'Yes': 1, 'No internet service': 0},
    'OnlineBackup':      {'No': 0, 'Yes': 1, 'No internet service': 0},
    'DeviceProtection':  {'No': 0, 'Yes': 1, 'No internet service': 0},
    'TechSupport':       {'No': 0, 'Yes': 1, 'No internet service': 0},
    'StreamingTV':       {'No': 0, 'Yes': 1, 'No internet service': 0},
    'StreamingMovies':   {'No': 0, 'Yes': 1, 'No internet service': 0},
}


def prepare_data(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    기본 전처리 적용.
    - 결측치 처리
    - 타입 변환 및 인코딩
    - feature engineering
    """
    df_cp = df.copy()

    # ID 컬럼 제거
    drop_cols = ['id']
    for col in drop_cols:
        if col in df_cp.columns:
            df_cp = df_cp.drop(columns=[col])

    numeric_cols = df_cp.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_cp.select_dtypes(include=['object', 'category']).columns.tolist()

    print('numeric_cols: ', numeric_cols)
    print('categorical_cols: ', categorical_cols)
    print('--------------------------------')

    # 수치형 결측치 처리 + 클리핑
    for col in numeric_cols:
        df_cp[col] = df_cp[col].fillna(df_cp[col].median())
        upper = df_cp[col].quantile(0.99)
        lower = df_cp[col].quantile(0.01)
        df_cp[col] = df_cp[col].clip(lower=lower, upper=upper)

    # 범주형 결측치 처리
    for col in categorical_cols:
        df_cp[col] = df_cp[col].fillna(df_cp[col].mode().iloc[0] if len(df_cp[col].mode()) else "")

    # 1단계: 의미 기반 직접 매핑 (No/서비스없음=0, Yes=1)
    df_cp = direct_mapping(df_cp)

    # 2단계: feature engineering (직접 매핑 후 → 서비스 컬럼이 0/1이므로 카운트 정확)
    df_cp = feature_engineering(df_cp)

    # 3단계: OneHot 인코딩 (순서 없는 범주형)
    onehot_cols = [c for c in ONEHOT_COLS if c in df_cp.columns]
    if onehot_cols:
        df_cp = one_hot_encoding(df_cp, target_cols=onehot_cols)

    # 4단계: Ordinal 인코딩 (순서 있는 범주형)
    ordinal_cols = [c for c in ORDINAL_COLS if c in df_cp.columns]
    if ordinal_cols:
        df_cp = ordinal_encoding(df_cp, target_cols=ordinal_cols)

    # 5단계: Target 인코딩 (범주형 컬럼 → 타겟 평균)
    encode_cols = [c for c in categorical_cols if c != target_col] if target_col else categorical_cols
    df_cp = target_encoding(df_cp, target_cols=[], target_col=target_col)

    # 6단계: 타겟 컬럼 LabelEncoder (target_col만)
    if target_col and target_col in df_cp.columns:
        df_cp = label_encoding(df_cp, target_cols=[target_col])

    # 7단계: 상관 분석 기반 특성 필터링
    df_cp = filter_correlated_features(df_cp, target_col=target_col)

    return df_cp


def get_feature_columns(df: pd.DataFrame, target_col: str | None = None, id_col: str | None = None) -> list[str]:
    """특성 컬럼 목록 반환 (target, id 제외)."""
    exclude = {c for c in (target_col, id_col) if c is not None}
    return [c for c in df.columns if c not in exclude]


def direct_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """의미 기반 직접 매핑으로 이진/삼진 범주형 컬럼을 0/1 수치로 변환."""
    X = df.copy()
    for col, mapping in DIRECT_MAPPING_COLS.items():
        if col in X.columns:
            X[col] = X[col].map(mapping).fillna(0).astype(int)
    return X


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """신규 컬럼 생성을 위해 특성 공학 진행"""
    eps = 1e-6
    X = df.copy()

    # 실제 평균 월 요금 (전체 납부액 / 가입 기간)
    X['avg_monthly_charge'] = X['TotalCharges'] / (X['tenure'] + eps)

    # 요금 인상 신호 (현재 월 요금 - 과거 평균 월 요금, 양수면 최근 요금 인상)
    X['charge_increase'] = X['MonthlyCharges'] - X['avg_monthly_charge']

    # 스트리밍 서비스 수 (StreamingTV + StreamingMovies)
    X['streaming_count'] = X['StreamingTV'] + X['StreamingMovies']

    # 보안/지원 서비스 수 (OnlineSecurity + OnlineBackup + DeviceProtection + TechSupport)
    X['security_count'] = X['OnlineSecurity'] + X['OnlineBackup'] + X['DeviceProtection'] + X['TechSupport']

    # 전체 부가 서비스 수
    X['total_services'] = X['security_count'] + X['streaming_count']

    # 신규 고객 플래그 (가입 6개월 이내)
    X['is_new_customer'] = (X['tenure'] <= 6).astype(int)

    # 서비스당 월 요금 (요금 대비 구독 서비스 수 — 가성비 인식)
    X['monthly_charge_per_service'] = X['MonthlyCharges'] / (X['total_services'] + 1)

    # 요금 일관성 (실제 납부액 / 기대 납부액 — 1보다 낮으면 중단/할인 이력)
    X['charge_consistency'] = X['TotalCharges'] / (X['MonthlyCharges'] * X['tenure'] + eps)

    return X


def one_hot_encoding(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """순서 없는 범주형 컬럼을 원-핫 인코딩."""
    X = df.copy()
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    encoded = encoder.fit_transform(X[target_cols].astype(str).fillna('__NA__'))
    feature_names = encoder.get_feature_names_out(target_cols)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X.index)
    X = X.drop(columns=target_cols)
    X = pd.concat([X, encoded_df], axis=1)
    return X


def ordinal_encoding(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """순서 있는 범주형 컬럼을 순서형 인코딩."""
    X = df.copy()
    categories = [ORDINAL_COLS[col] for col in target_cols]
    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    X[target_cols] = encoder.fit_transform(X[target_cols].astype(str).fillna('__NA__'))
    return X


def label_encoding(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """타겟 컬럼을 수치로 인코딩."""
    X = df.copy()
    for col in target_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).fillna('__NA__'))
    return X


def filter_correlated_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    high_corr_threshold: float = 0.9,
    low_target_corr_threshold: float = 0.01,
    save_dir: str = 'eda_results',
) -> pd.DataFrame:
    result = correlation_analysis(
        df,
        target_col=target_col,
        high_corr_threshold=high_corr_threshold,
        low_target_corr_threshold=low_target_corr_threshold,
        save_dir=save_dir,
    )
    drop_cols = [c for c in result['drop_candidates'] if c in df.columns]
    return df.drop(columns=drop_cols)


def target_encoding(df: pd.DataFrame, target_cols: list[str] | None = None, target_col: str | None = None) -> pd.DataFrame:
    """범주형 컬럼을 타겟 평균으로 인코딩. target_col이 없으면(테스트 등) 그대로 반환."""
    X = df.copy()

    if target_col and target_col not in X.columns:
        return X

    if target_col and target_col in X.columns:
        if X[target_col].dtype in ("object", "category") or (hasattr(X[target_col].dtype, "name") and X[target_col].dtype.name == "category"):
            s = X[target_col].astype(str).str.strip().str.lower()
            X[target_col] = s.map({"yes": 1, "no": 0})

    y = X[target_col]
    global_mean = y.mean()
    smoothing = 10.0

    for col in target_cols:
        stats = X.groupby(col)[target_col].agg(["count", "mean"])
        smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
        X[col] = X[col].map(smooth).fillna(global_mean)

    return X
