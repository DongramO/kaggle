import numpy as np
import pandas as pd

def load_data():
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    df_sub = pd.read_csv('../data/sample_submission.csv')
    
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
    if 'id' in categorical_cols:
        categorical_cols.remove('id')
    
    df_exam_score = df_train['exam_score']
    
    return numeric_cols, categorical_cols, df_exam_score



