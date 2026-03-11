import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from eda import correlation_analysis

ONEHOT_COLS = ['gender', 'PaymentMethod', 'InternetService']

ORDINAL_COLS = {
    'Contract': ['Month-to-month', 'One year', 'Two year'],
}

TARGET_ENCODING_COLS = ['PaymentMethod', 'InternetService']

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
        if 'id' in df.columns:
            df.drop(columns=['id'], inplace=True)

    y_train = train.pop(target_col)
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train.astype(str)), index=y_train.index, name=target_col)

    train, test = _basic_preprocessing(train, test)
    train, test = _direct_mapping(train, test)
    train = feature_engineering(train)
    test = feature_engineering(test)
    train, test = _kfold_target_encoding(train, test, TARGET_ENCODING_COLS, y_train)
    train, test = _fit_transform_onehot(train, test)
    train, test = _fit_transform_ordinal(train, test)

    return train, y_train, test


def get_feature_columns(df: pd.DataFrame, target_col: str | None = None, id_col: str | None = None) -> list[str]:
    exclude = {c for c in (target_col, id_col) if c is not None}
    return [c for c in df.columns if c not in exclude]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-6
    X = df.copy()

    X['avg_monthly_charge'] = X['TotalCharges'] / (X['tenure'] + eps)
    X['charge_increase'] = X['MonthlyCharges'] - X['avg_monthly_charge']
    X['streaming_count'] = X['StreamingTV'] + X['StreamingMovies']
    X['security_count'] = X['OnlineSecurity'] + X['OnlineBackup'] + X['DeviceProtection'] + X['TechSupport']
    X['total_services'] = X['security_count'] + X['streaming_count']
    X['is_new_customer'] = (X['tenure'] <= 6).astype(int)
    X['monthly_charge_per_service'] = X['MonthlyCharges'] / (X['total_services'] + 1)
    X['charge_consistency'] = X['TotalCharges'] / (X['MonthlyCharges'] * X['tenure'] + eps)

    return X


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
