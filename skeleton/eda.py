"""
EDA 모듈 (독립 스켈레톤)
- 요약 통계, 결측/분포 확인, 시각화 진입점
"""

import pandas as pd


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """기본 요약 통계."""
    return df.describe(include="all")


def missing_report(df: pd.DataFrame) -> pd.Series:
    """컬럼별 결측 개수."""
    return df.isna().sum()


def run_eda(df: pd.DataFrame, target_col: str | None = None) -> dict:
    """
    EDA 실행: 요약, 결측, (선택) 타겟 분포.
    반환값으로 후속 시각화/분석에 활용.
    """
    result = {
        "shape": df.shape,
        "describe": summary(df),
        "missing": missing_report(df),
        "dtypes": df.dtypes,
    }
    if target_col and target_col in df.columns:
        result["target_distribution"] = df[target_col].value_counts()
    return result
