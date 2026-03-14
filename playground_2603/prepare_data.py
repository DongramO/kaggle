import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from eda import correlation_analysis

ONEHOT_COLS = ['PaymentMethod', 'InternetService']
DROP_COLS   = ['gender']   # permutation importance 낮음

ORDINAL_COLS = {
    'Contract': ['Month-to-month', 'One year', 'Two year'],
}


FE_FLAGS_PER_MODEL = {
    "catboost": {
        "charge_per_log_tenure":      True,
        "contract_x_avg_charge":      True,
        "contract_x_charge_log":      True,
        "fiber_x_echeck":             False,   # perm 낮음
        "contract_x_fiber":           False,   # perm 낮음
        "senior_x_monthly":           False,
        # 신규 후보
        "total_services":             False,
        "tenure_bin":                 False,
        "loyalty_score":              False,
        "charge_per_service":         False,   # perm 낮음
        "no_support":                 False,
        "partner_x_dependents":       False,
    },
    "lightgbm": {
        "avg_monthly_charge":         True,
        "charge_increase":            True,
        "charge_consistency":         True,
        "security_count":             True,
        "senior_x_charge_increase":   True,
        "charge_ratio":               False,
        "tenure_x_avg_charge":        False,
        # 신규 후보
        "total_services":             False,   # perm 낮음
        "tenure_bin":                 False,
        "loyalty_score":              False,
        "charge_per_service":         False,
        "no_support":                 False,
        "partner_x_dependents":       False,
    },
    "xgboost": {
        "high_risk_combo":               True,
        "contract_x_electronic_check":   True,
        "fiber_x_paperless":             False,  # perm 낮음
        "streaming_count":               True,
        "mtm_x_fiber":                   True,
        "mtm_x_echeck":                  True,
        "senior_x_fiber":                False,
        "no_security_x_fiber":           False,
        # 신규 후보
        "total_services":                False,  # perm 낮음
        "tenure_bin":                    False,
        "loyalty_score":                 False,
        "charge_per_service":            False,
        "no_support":                    False,
        "partner_x_dependents":          False,
    },
}

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


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    for df in [train, test]:
        drop = ['id'] + [c for c in DROP_COLS if c in df.columns]
        df.drop(columns=drop, inplace=True)

    y_train = train.pop(target_col)
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train.astype(str)), index=y_train.index, name=target_col)

    train, test = _basic_preprocessing(train, test)
    train, test = _direct_mapping(train, test)
    train, test = _fit_transform_onehot(train, test)
    train, test = _fit_transform_ordinal(train, test)

    return train, y_train, test


def get_feature_columns(df: pd.DataFrame, target_col: str | None = None, id_col: str | None = None) -> list[str]:
    exclude = {c for c in (target_col, id_col) if c is not None}
    return [c for c in df.columns if c not in exclude]


def feature_engineering(df: pd.DataFrame, flags: dict | None = None) -> pd.DataFrame:
    """
    FE_FLAGS의 각 키를 True로 설정해 피처를 하나씩 활성화.
    의존 관계가 있는 피처(charge_increase → avg_monthly_charge 등)는
    중간값을 내부에서 계산해 처리.
    """
    if flags is None:
        flags = {}

    eps = 1e-6
    X = df.copy()

    # 중간값 (플래그와 무관하게 필요 시 계산)
    avg_monthly_charge   = X['TotalCharges'] / (X['tenure'] + eps)
    security_count       = X['OnlineSecurity'] + X['OnlineBackup'] + X['DeviceProtection'] + X['TechSupport']
    charge_per_log_tenure = X['MonthlyCharges'] / np.log1p(X['tenure'])
    contract_map         = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_numeric     = X['Contract'].map(contract_map).fillna(0)

    if flags.get("avg_monthly_charge"):
        X['avg_monthly_charge'] = avg_monthly_charge

    if flags.get("charge_increase"):
        X['charge_increase'] = X['MonthlyCharges'] - avg_monthly_charge

    if flags.get("security_count"):
        X['security_count'] = security_count

    if flags.get("monthly_charge_per_service"):
        X['monthly_charge_per_service'] = X['MonthlyCharges'] / (security_count + 1)

    if flags.get("charge_consistency"):
        X['charge_consistency'] = X['TotalCharges'] / (X['MonthlyCharges'] * X['tenure'] + eps)

    if flags.get("charge_per_log_tenure"):
        X['charge_per_log_tenure'] = charge_per_log_tenure

    if flags.get("contract_x_charge_log"):
        X['contract_x_charge_log'] = contract_numeric * charge_per_log_tenure

    if flags.get("contract_x_avg_charge"):
        X['contract_x_avg_charge'] = contract_numeric * avg_monthly_charge

    return X


def apply_model_fe(X: pd.DataFrame, flags: dict | None = None) -> pd.DataFrame:
    """
    인코딩 완료된 데이터에 모델별 FE 피처를 추가.
    Contract는 ordinal 인코딩 후 0/1/2 숫자로 사용.
    FE_FLAGS_PER_MODEL의 모델별 플래그를 전달해 사용.
    """
    if not flags:
        return X

    eps = 1e-6
    df = X.copy()

    avg_monthly_charge    = df['TotalCharges'] / (df['tenure'] + eps)
    charge_per_log_tenure = df['MonthlyCharges'] / np.log1p(df['tenure'])
    security_count        = df['OnlineSecurity'] + df['OnlineBackup'] + df['DeviceProtection'] + df['TechSupport']
    contract_numeric      = df['Contract']  # ordinal 인코딩 후 0.0 / 1.0 / 2.0

    if flags.get("avg_monthly_charge"):
        df['avg_monthly_charge'] = avg_monthly_charge

    if flags.get("charge_increase"):
        df['charge_increase'] = df['MonthlyCharges'] - avg_monthly_charge

    if flags.get("security_count"):
        df['security_count'] = security_count

    if flags.get("monthly_charge_per_service"):
        df['monthly_charge_per_service'] = df['MonthlyCharges'] / (security_count + 1)

    if flags.get("charge_consistency"):
        df['charge_consistency'] = df['TotalCharges'] / (df['MonthlyCharges'] * df['tenure'] + eps)

    if flags.get("charge_per_log_tenure"):
        df['charge_per_log_tenure'] = charge_per_log_tenure

    if flags.get("contract_x_charge_log"):
        df['contract_x_charge_log'] = contract_numeric * charge_per_log_tenure

    if flags.get("contract_x_avg_charge"):
        df['contract_x_avg_charge'] = contract_numeric * avg_monthly_charge

    # --- XGBoost 전용: 범주형 조합 ---
    is_mtm    = (df['Contract'] == 0).astype(int)
    is_echeck = df.get('PaymentMethod_Electronic check', pd.Series(0, index=df.index))
    is_fiber  = df.get('InternetService_Fiber optic',    pd.Series(0, index=df.index))

    if flags.get("high_risk_combo"):
        df['high_risk_combo'] = is_mtm * is_echeck * is_fiber

    if flags.get("contract_x_electronic_check"):
        df['contract_x_electronic_check'] = contract_numeric * is_echeck

    if flags.get("fiber_x_paperless"):
        df['fiber_x_paperless'] = is_fiber * df['PaperlessBilling']

    if flags.get("streaming_count"):
        df['streaming_count'] = df['StreamingTV'] + df['StreamingMovies']

    # XGBoost 신규: 2-way 고위험 조합
    if flags.get("mtm_x_fiber"):
        df['mtm_x_fiber'] = is_mtm * is_fiber

    if flags.get("mtm_x_echeck"):
        df['mtm_x_echeck'] = is_mtm * is_echeck

    if flags.get("senior_x_fiber"):
        df['senior_x_fiber'] = df['SeniorCitizen'] * is_fiber

    if flags.get("no_security_x_fiber"):
        df['no_security_x_fiber'] = (1 - df['OnlineSecurity']) * is_fiber

    # --- CatBoost 신규 ---
    if flags.get("fiber_x_echeck"):
        df['fiber_x_echeck'] = is_fiber * is_echeck

    if flags.get("contract_x_fiber"):
        df['contract_x_fiber'] = contract_numeric * is_fiber

    if flags.get("senior_x_monthly"):
        df['senior_x_monthly'] = df['SeniorCitizen'] * df['MonthlyCharges']

    # --- LightGBM 신규 ---
    if flags.get("charge_ratio"):
        # 현재 월비용 / 평균 월비용 → 1보다 크면 비용 증가 추세
        df['charge_ratio'] = df['MonthlyCharges'] / (avg_monthly_charge + eps)

    if flags.get("senior_x_charge_increase"):
        df['senior_x_charge_increase'] = df['SeniorCitizen'] * (df['MonthlyCharges'] - avg_monthly_charge)

    if flags.get("tenure_x_avg_charge"):
        df['tenure_x_avg_charge'] = np.log1p(df['tenure']) * avg_monthly_charge

    # --- 공통 신규 후보 ---
    total_services = (
        df['MultipleLines'] + df['OnlineSecurity'] + df['OnlineBackup'] +
        df['DeviceProtection'] + df['TechSupport'] + df['StreamingTV'] + df['StreamingMovies']
    )

    if flags.get("total_services"):
        df['total_services'] = total_services

    if flags.get("tenure_bin"):
        # 0: 신규(≤12개월), 1: 중기(13~36개월), 2: 장기(>36개월)
        df['tenure_bin'] = pd.cut(
            df['tenure'], bins=[-1, 12, 36, float('inf')], labels=[0, 1, 2]
        ).astype(float)

    if flags.get("loyalty_score"):
        df['loyalty_score'] = df['tenure'] * contract_numeric

    if flags.get("charge_per_service"):
        df['charge_per_service'] = df['MonthlyCharges'] / (total_services + 1)

    if flags.get("no_support"):
        df['no_support'] = ((df['OnlineSecurity'] == 0) & (df['TechSupport'] == 0)).astype(int)

    if flags.get("partner_x_dependents"):
        df['partner_x_dependents'] = df['Partner'] * df['Dependents']

    return df


def filter_correlated_features(
    df: pd.DataFrame,
    target_col: str | None = None,
    high_corr_threshold: float = 0.9,
    low_target_corr_threshold: float = 0.01,
    save_dir: str = 'eda_results',
) -> list[str]:
    result = correlation_analysis(
        df,
        target_col=target_col,
        high_corr_threshold=high_corr_threshold,
        low_target_corr_threshold=low_target_corr_threshold,
        save_dir=save_dir,
    )
    return [c for c in result['drop_candidates'] if c in df.columns]


def _basic_preprocessing(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()

    num_fill = train[num_cols].median()
    cat_fill = {col: train[col].mode().iloc[0] for col in cat_cols if len(train[col].mode()) > 0}
    clip_bounds = {col: (train[col].quantile(0.01), train[col].quantile(0.99)) for col in num_cols}

    for df in [train, test]:
        df[num_cols] = df[num_cols].fillna(num_fill)
        for col, val in cat_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)
        for col, (lower, upper) in clip_bounds.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)

    return train, test


def _direct_mapping(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for df in [train, test]:
        for col, mapping in DIRECT_MAPPING_COLS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)
    return train, test


def _kfold_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    y: pd.Series,
    n_splits: int = 5,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_mean = y.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cols:
        if col not in train.columns:
            continue

        te_col = f'{col}_TE'
        train[te_col] = np.nan

        for tr_idx, val_idx in skf.split(train, y):
            fold_df = pd.DataFrame({'col': train.iloc[tr_idx][col].values, 'target': y.iloc[tr_idx].values})
            stats = fold_df.groupby('col')['target'].agg(['count', 'mean'])
            smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            train.loc[train.index[val_idx], te_col] = train.iloc[val_idx][col].map(smooth).values

        train[te_col] = train[te_col].fillna(global_mean)

        full_df = pd.DataFrame({'col': train[col].values, 'target': y.values})
        stats = full_df.groupby('col')['target'].agg(['count', 'mean'])
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        test[te_col] = test[col].map(smooth).fillna(global_mean)

    return train, test


def _fit_transform_onehot(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in ONEHOT_COLS if c in train.columns]
    if not cols:
        return train, test

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    encoder.fit(train[cols].astype(str))

    feature_names = encoder.get_feature_names_out(cols)

    for df in [train, test]:
        encoded = encoder.transform(df[cols].astype(str))
        encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        df.drop(columns=cols, inplace=True)
        for feat in feature_names:
            df[feat] = encoded_df[feat]

    return train, test


def _fit_transform_ordinal(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [c for c in ORDINAL_COLS if c in train.columns]
    if not cols:
        return train, test

    categories = [ORDINAL_COLS[col] for col in cols]
    encoder = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
    encoder.fit(train[cols].astype(str))

    for df in [train, test]:
        df[cols] = encoder.transform(df[cols].astype(str))

    return train, test
