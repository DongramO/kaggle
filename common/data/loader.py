"""
데이터 로드 모듈
로컬 환경 데이터 로드
"""
import os
import pandas as pd
from typing import Tuple, Optional


def load_data(
    data_dir: Optional[str] = None,
    train_path: Optional[str] = None,
    test_path: Optional[str] = None,
    sub_path: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드
    
    Parameters:
    -----------
    data_dir : str, optional
        데이터 디렉토리 경로 (상대경로는 현재 작업 디렉토리 기준)
        지정 시 train.csv, test.csv, sample_submission.csv를 찾음
    train_path : str, optional
        train.csv 파일 경로 (상대경로는 현재 작업 디렉토리 기준, data_dir 미지정 시 필수)
    test_path : str, optional
        test.csv 파일 경로 (상대경로는 현재 작업 디렉토리 기준, data_dir 미지정 시 필수)
    sub_path : str, optional
        sample_submission.csv 파일 경로 (상대경로는 현재 작업 디렉토리 기준, data_dir 미지정 시 필수)
    
    Returns:
    --------
    tuple: (df_train, df_test, df_sub)
    """
    # 경로 결정 및 절대경로 변환
    if data_dir:
        # data_dir이 지정된 경우
        data_dir = os.path.abspath(data_dir)
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        sub_path = os.path.join(data_dir, 'sample_submission.csv')
    else:
        # 개별 경로가 모두 지정되어야 함
        if train_path is None or test_path is None or sub_path is None:
            raise ValueError(
                "data_dir을 지정하거나 train_path, test_path, sub_path를 모두 지정해주세요."
            )
        # 상대경로를 절대경로로 변환
        train_path = os.path.abspath(train_path)
        test_path = os.path.abspath(test_path)
        sub_path = os.path.abspath(sub_path)
    
    print(f"📂 데이터 경로:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    print(f"  Submission: {sub_path}")
    
    # 파일 존재 확인
    for path, name in [(train_path, 'Train'), (test_path, 'Test'), (sub_path, 'Submission')]:
        if not os.path.exists(path):
            print(f"  ⚠️ {name} 파일을 찾을 수 없습니다: {path}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sub = pd.read_csv(sub_path)
    
    print(f"✅ 데이터 로드 완료!")
    print(f"  Train: {df_train.shape}, Test: {df_test.shape}, Submission: {df_sub.shape}")
    
    return df_train, df_test, df_sub
