"""
데이터 준비(전처리) 모듈 (독립 스켈레톤)
- 결측 처리, 인코딩, 피처 엔지니어링 뼈대
- 문제에 맞게 상수와 FE 로직을 수정해 사용
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from eda import correlation_analysis


# ========== 문제별 수정 영역 ==========
# 원-핫 인코딩 대상 컬럼
ONEHOT_COLS: list[str] = []

# 제거할 컬럼 (feature importance 낮음 등)
DROP_COLS: list[str] = []

# N-gram Target Encoding 대상 컬럼
NGRAM_TE_COLS: list[str] = []

# 순서형 인코딩 컬럼 및 카테고리 순서
ORDINAL_COLS: dict[str, list] = {}

# 직접 값 매핑 컬럼 (이진/레이블 등)
DIRECT_MAPPING_COLS: dict[str, dict] = {}

# 모델별 FE 활성화 플래그 (True로 설정된 것만 apply_model_fe에서 생성)
FE_FLAGS_PER_MODEL: dict[str, dict] = {
    "catboost":  {},
    "lightgbm":  {},
    "xgboost":   {},
}
# ======================================


def prepare_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    train = train_df.copy()
    test = test_df.copy()

    for df in [train, test]:
        drop = ['id'] + [c for c in DROP_COLS if c in df.columns]
        df.drop(columns=[c for c in drop if c in df.columns], inplace=True)

    y_train = train.pop(target_col)
    le = LabelEncoder()
    y_train = pd.Series(le.fit_transform(y_train.astype(str)), index=y_train.index, name=target_col)

    train, test = _basic_preprocessing(train, test)

    if NGRAM_TE_COLS:
        train, test = _ngram_target_encoding(train, test, NGRAM_TE_COLS, y_train)
    if DIRECT_MAPPING_COLS:
        train, test = _direct_mapping(train, test)
    if ONEHOT_COLS:
        train, test = _fit_transform_onehot(train, test)
    if ORDINAL_COLS:
        train, test = _fit_transform_ordinal(train, test)

    return train, y_train, test


def get_feature_columns(df: pd.DataFrame, target_col: str | None = None, id_col: str | None = None) -> list[str]:
    exclude = {c for c in (target_col, id_col) if c is not None}
    return [c for c in df.columns if c not in exclude]


def apply_model_fe(X: pd.DataFrame, flags: dict | None = None) -> pd.DataFrame:
    """
    인코딩 완료된 데이터에 모델별 FE 피처를 추가.
    FE_FLAGS_PER_MODEL의 모델별 플래그를 전달해 사용.
    flags가 비어있으면 그대로 반환.
    """
    if not flags:
        return X
    df = X.copy()

    # TODO: 문제별 FE 로직 구현
    # 예시 패턴:
    # if flags.get("some_feature"):
    #     df['some_feature'] = ...

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


def _ngram_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    y: pd.Series,
    ns: list[int] = [2, 3],
    smoothing: float = 10.0,
    n_splits: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    여러 카테고리 컬럼을 하나의 문장으로 결합 후 word n-gram Target Encoding.
    OHE/Ordinal 인코딩 전에 원본 문자열 값이 필요하므로 가장 먼저 호출.
    KFold TE로 train leakage 방지, test는 전체 train 통계 사용.
    """
    valid_cols = [c for c in cols if c in train.columns]
    if not valid_cols:
        return train, test

    global_mean = float(y.mean())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def combine(df: pd.DataFrame) -> pd.Series:
        return df[valid_cols].astype(str).apply(lambda row: " ".join(row), axis=1)

    def get_ngrams(text: str, n: int) -> list[str]:
        words = text.split()
        if len(words) < n:
            return words
        return [" ".join(words[i: i + n]) for i in range(len(words) - n + 1)]

    def build_ngram_stats(sentences: pd.Series, targets: pd.Series, n: int) -> dict:
        stats: dict[str, list] = {}
        for text, t in zip(sentences, targets):
            for ng in get_ngrams(text, n):
                if ng not in stats:
                    stats[ng] = [0.0, 0]
                stats[ng][0] += t
                stats[ng][1] += 1
        return stats

    def stats_to_te(stats: dict, smoothing: float, global_mean: float) -> dict:
        return {
            ng: (v[0] + smoothing * global_mean) / (v[1] + smoothing)
            for ng, v in stats.items()
        }

    train_combined = combine(train)
    test_combined  = combine(test)

    for n in ns:
        col_name = f"ngram_te_{n}"
        train[col_name] = global_mean

        for tr_idx, val_idx in skf.split(train, y):
            stats = build_ngram_stats(train_combined.iloc[tr_idx], y.iloc[tr_idx], n)
            te_map = stats_to_te(stats, smoothing, global_mean)
            val_sentences = train_combined.iloc[val_idx]
            te_vals = val_sentences.apply(
                lambda text: float(np.mean([te_map.get(ng, global_mean) for ng in get_ngrams(text, n)]))
            )
            train.loc[train.index[val_idx], col_name] = te_vals.values

        full_stats = build_ngram_stats(train_combined, y, n)
        full_te    = stats_to_te(full_stats, smoothing, global_mean)
        test[col_name] = test_combined.apply(
            lambda text: float(np.mean([full_te.get(ng, global_mean) for ng in get_ngrams(text, n)]))
        )

        bi_or_tri = "Bigram" if n == 2 else "Trigram"
        print(f"      [{bi_or_tri} TE] {col_name} 추가 — 고유 n-gram 수: {len(full_stats)}")

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
