import numpy as np
import pandas as pd
import os

def get_data_paths():
    """
    데이터 파일 경로 반환
    
    Returns:
    --------
    tuple
        (train_path, test_path, sub_path)
    """
    # 현재 파일 기준으로 프로젝트 루트 찾기
    # eda/dataload.py -> 202601_playground/eda/dataload.py
    # 따라서 상위 디렉토리가 프로젝트 루트
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    sub_path = os.path.join(data_dir, 'sample_submission.csv')
    
    return train_path, test_path, sub_path

def load_data():
    """
    데이터 로드 함수
    
    Returns:
    --------
    tuple
        (df_train, df_test, df_sub)
    """
    train_path, test_path, sub_path = get_data_paths()
    
    print(f"📂 데이터 경로:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")
    print(f"  Submission: {sub_path}")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sub = pd.read_csv(sub_path)
    
    return df_train, df_test, df_sub


def data_summary(df_train, df_test, df_sub):
    
    print(f"df_train shape: {df_train.shape}")
    print(f"df_test shape: {df_test.shape}")
    print(f"df_sub shape: {df_sub.shape}")

    print(f"df_train.info(): {df_train.info()}")
    print(f"df_test.info(): {df_test.info()}")
    print(f"df_train.describe() \n: {df_train.describe()}")


    cols = df_train.columns
    
    for col in cols:
        # unique(): 고유 값 배열 반환 → 데이터 확인용
        # nunique(): 고유 값 개수 반환 → 통계/요약용
        print("culmuns name :", col)
        print("colums nunique  : ", df_train[col].nunique())
        

def data_classifier(df_train):
    
    # 단순히 데이터 카테고리를 분류하는 방법이 데이터 타입에 의해서 나뉘어져도 괜찮은걸까?
    numeric_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # id 컬럼은 시각화 불필요하므로 제외
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    df_exam_score = df_train['exam_score']
    
    return numeric_cols, categorical_cols, df_exam_score



