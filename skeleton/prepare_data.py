"""
데이터 준비(전처리) 모듈 (독립 스켈레톤)
- 결측 처리, 타입 변환, 기본 전처리
"""

import pandas as pd


def prepare_data(df: pd.DataFrame, target_col: str | None = None) -> pd.DataFrame:
    """
    기본 전처리 적용.
    - 결측치 처리
    - 필요 시 타입 변환
    """
    out = df.copy()
    # 결측치: 수치형은 중앙값, 범주형은 최빈값 등으로 채우기
    for col in out.columns:
        if col == target_col:
            continue
        if out[col].dtype in ("int64", "float64"):
            out[col] = out[col].fillna(out[col].median())
        else:
            out[col] = out[col].fillna(out[col].mode().iloc[0] if len(out[col].mode()) else "")
    return out


def get_feature_columns(df: pd.DataFrame, target_col: str | None = None, id_col: str | None = None) -> list[str]:
    """특성 컬럼 목록 반환 (target, id 제외)."""
    exclude = {c for c in (target_col, id_col) if c is not None}
    return [c for c in df.columns if c not in exclude]
