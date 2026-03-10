"""
데이터 로드 모듈 (독립 스켈레톤)
- train / test / sample_submission 등 원시 데이터 로드
"""

import pandas as pd
from pathlib import Path


def load_train(data_dir: str | Path) -> pd.DataFrame:
    """학습 데이터 로드."""
    path = Path(data_dir) / "train.csv"
    return pd.read_csv(path)


def load_test(data_dir: str | Path) -> pd.DataFrame:
    """테스트 데이터 로드."""
    path = Path(data_dir) / "test.csv"
    return pd.read_csv(path)


def load_sample_submission(data_dir: str | Path) -> pd.DataFrame:
    """제출 양식 로드."""
    path = Path(data_dir) / "sample_submission.csv"
    return pd.read_csv(path)


def load_all(data_dir: str | Path):
    """train, test, sample_submission 한 번에 로드."""
    return (
        load_train(data_dir),
        load_test(data_dir),
        load_sample_submission(data_dir),
    )
