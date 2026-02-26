import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def one_hot_encode(
    df: pd.DataFrame,
    categorical_cols: List[str],
    drop_original: bool = True,
    handle_unknown: str = 'ignore',
    sparse: bool = False
) -> pd.DataFrame:
    """
    범주형 변수에 대해 one-hot 인코딩 수행
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    categorical_cols : List[str]
        one-hot 인코딩을 적용할 범주형 컬럼 리스트
    drop_original : bool
        원본 컬럼 삭제 여부 (기본값: True)
    handle_unknown : str
        미지 카테고리 처리 방법 ('error' 또는 'ignore', 기본값: 'ignore')
    sparse : bool
        희소 행렬 반환 여부 (기본값: False)
    
    Returns:
    --------
    pd.DataFrame: one-hot 인코딩이 적용된 데이터프레임
    """
    df = df.copy()
    
    # 존재하는 컬럼만 필터링
    valid_cols = [col for col in categorical_cols if col in df.columns]
    if len(valid_cols) == 0:
        return df
    
    # OneHotEncoder 인스턴스 생성
    encoder = OneHotEncoder(
        handle_unknown=handle_unknown,
        sparse=sparse,
        drop=None  # 모든 카테고리 유지
    )
    
    # 인코딩 수행
    encoded_array = encoder.fit_transform(df[valid_cols])
    
    # 희소 행렬인 경우 밀집 행렬로 변환
    if sparse:
        encoded_array = encoded_array.toarray()
    
    # 컬럼 이름 생성
    feature_names = encoder.get_feature_names_out(valid_cols)
    
    # DataFrame으로 변환
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=df.index
    )
    
    # 원본 컬럼 삭제
    if drop_original:
        df = df.drop(columns=valid_cols)
    
    # 인코딩된 컬럼 추가
    df = pd.concat([df, encoded_df], axis=1)
    
    return df


def ordinal_encode(
    df: pd.DataFrame,
    categorical_cols: List[str],
    categories: Optional[Union[List[List], str]] = 'auto',
    category_orders: Optional[Dict[str, List]] = None,
    drop_original: bool = True,
    handle_unknown: str = 'error',
    unknown_value: Optional[Union[int, float]] = None
) -> pd.DataFrame:
    """
    범주형 변수에 대해 ordinal 인코딩 수행
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    categorical_cols : List[str]
        ordinal 인코딩을 적용할 범주형 컬럼 리스트
    categories : List[List] or str, optional
        각 컬럼의 카테고리 순서 리스트 또는 'auto' (기본값: 'auto')
    category_orders : Dict[str, List], optional
        각 컬럼별로 카테고리 순서를 딕셔너리로 지정 (categories보다 우선순위 높음)
    drop_original : bool
        원본 컬럼 삭제 여부 (기본값: True)
    handle_unknown : str
        미지 카테고리 처리 방법 ('error' 또는 'use_encoded_value', 기본값: 'error')
    unknown_value : int or float, optional
        handle_unknown='use_encoded_value'일 때 사용할 값
    
    Returns:
    --------
    pd.DataFrame: ordinal 인코딩이 적용된 데이터프레임
    """
    df = df.copy()
    
    # 존재하는 컬럼만 필터링
    valid_cols = [col for col in categorical_cols if col in df.columns]
    if len(valid_cols) == 0:
        return df
    
    # category_orders가 지정된 경우, categories 리스트로 변환
    if category_orders is not None:
        categories_list = []
        for col in valid_cols:
            if col in category_orders:
                categories_list.append(category_orders[col])
            else:
                # 지정되지 않은 컬럼은 자동으로 처리
                unique_values = sorted(df[col].dropna().unique().tolist())
                categories_list.append(unique_values)
        categories = categories_list
    elif categories == 'auto' or categories is None:
        # 'auto'인 경우, 각 컬럼의 고유값을 알파벳 순서로 정렬
        categories = [sorted(df[col].dropna().unique().tolist()) for col in valid_cols]
    
    # OrdinalEncoder 파라미터 설정
    encoder_params = {
        'handle_unknown': handle_unknown,
        'categories': categories
    }
    
    # unknown_value 설정 (handle_unknown='use_encoded_value'일 때)
    if handle_unknown == 'use_encoded_value':
        if unknown_value is None:
            raise ValueError(
                "unknown_value must be provided when handle_unknown='use_encoded_value'"
            )
        encoder_params['unknown_value'] = unknown_value
    
    # OrdinalEncoder 인스턴스 생성
    encoder = OrdinalEncoder(**encoder_params)
    
    # 인코딩 수행
    encoded_array = encoder.fit_transform(df[valid_cols])
    
    # 인코딩된 결과를 DataFrame으로 변환
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=valid_cols,
        index=df.index
    )
    
    # 원본 컬럼 삭제
    if drop_original:
        df = df.drop(columns=valid_cols)
    
    # 인코딩된 컬럼 추가 (또는 원본 컬럼 교체)
    for col in valid_cols:
        df[col] = encoded_df[col]
    
    return df


def fit_encoder(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoding_type: str = 'onehot',
    category_orders: Optional[Dict[str, List]] = None,
    **kwargs
) -> Union[OneHotEncoder, OrdinalEncoder]:
    """
    인코더 학습 (학습/테스트 데이터 분리 시 사용)
    
    Parameters:
    -----------
    df : pd.DataFrame
        학습 데이터프레임
    categorical_cols : List[str]
        인코딩할 범주형 컬럼 리스트
    encoding_type : str
        인코딩 타입 ('onehot' or 'ordinal', 기본값: 'onehot')
    category_orders : Dict[str, List], optional
        ordinal 인코딩 시 각 컬럼별 카테고리 순서 (encoding_type='ordinal'일 때만 사용)
    **kwargs
        인코더에 전달할 추가 파라미터
    
    Returns:
    --------
    OneHotEncoder or OrdinalEncoder: 학습된 인코더 객체
    """
    # 존재하는 컬럼만 필터링
    valid_cols = [col for col in categorical_cols if col in df.columns]
    if len(valid_cols) == 0:
        raise ValueError("No valid categorical columns found in the dataframe")
    
    # encoding_type에 따라 인코더 생성
    if encoding_type == 'onehot':
        # OneHotEncoder 기본 파라미터
        default_params = {
            'handle_unknown': 'ignore',
        }
        # kwargs로 기본값 덮어쓰기
        encoder_params = {**default_params, **kwargs}
        encoder = OneHotEncoder(**encoder_params)
        
    elif encoding_type == 'ordinal':
        # category_orders가 지정된 경우 categories 리스트로 변환
        if category_orders is not None:
            categories_list = []
            for col in valid_cols:
                if col in category_orders:
                    categories_list.append(category_orders[col])
                else:
                    # 지정되지 않은 컬럼은 자동으로 처리
                    unique_values = sorted(df[col].dropna().unique().tolist())
                    categories_list.append(unique_values)
            # kwargs에 categories가 이미 있으면 덮어쓰지 않음 (명시적 지정 우선)
            if 'categories' not in kwargs:
                kwargs['categories'] = categories_list
        
        # OrdinalEncoder 기본 파라미터
        default_params = {
            'handle_unknown': 'error',
            'categories': 'auto'
        }
        # kwargs로 기본값 덮어쓰기
        encoder_params = {**default_params, **kwargs}
        
        # categories가 'auto'인 경우, 각 컬럼의 고유값을 알파벳 순서로 정렬
        if encoder_params.get('categories') == 'auto':
            encoder_params['categories'] = [
                sorted(df[col].dropna().unique().tolist()) for col in valid_cols
            ]
        
        encoder = OrdinalEncoder(**encoder_params)
        
    else:
        raise ValueError(f"Unknown encoding_type: {encoding_type}. Must be 'onehot' or 'ordinal'")
    
    # 인코더 학습
    encoder.fit(df[valid_cols])
    
    return encoder


def transform_with_encoder(
    df: pd.DataFrame,
    encoder: Union[OneHotEncoder, OrdinalEncoder],
    categorical_cols: List[str],
    drop_original: bool = True
) -> pd.DataFrame:
    """
    학습된 인코더를 사용하여 데이터 변환
    
    Parameters:
    -----------
    df : pd.DataFrame
        변환할 데이터프레임
    encoder : OneHotEncoder or OrdinalEncoder
        학습된 인코더 객체
    categorical_cols : List[str]
        인코딩할 범주형 컬럼 리스트
    drop_original : bool
        원본 컬럼 삭제 여부 (기본값: True)
    
    Returns:
    --------
    pd.DataFrame: 인코딩이 적용된 데이터프레임
    """
    df = df.copy()
    
    # 존재하는 컬럼만 필터링
    valid_cols = [col for col in categorical_cols if col in df.columns]
    if len(valid_cols) == 0:
        return df
    
    # 학습된 인코더로 변환
    encoded_array = encoder.transform(df[valid_cols])
    
    # OneHotEncoder인 경우
    if isinstance(encoder, OneHotEncoder):
        # 희소 행렬인 경우 밀집 행렬로 변환
        if hasattr(encoded_array, 'toarray'):
            encoded_array = encoded_array.toarray()
        
        # 컬럼 이름 생성
        feature_names = encoder.get_feature_names_out(valid_cols)
        
        # DataFrame으로 변환
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=feature_names,
            index=df.index
        )
        
        # 원본 컬럼 삭제
        if drop_original:
            df = df.drop(columns=valid_cols)
        
        # 인코딩된 컬럼 추가
        df = pd.concat([df, encoded_df], axis=1)
        
    # OrdinalEncoder인 경우
    elif isinstance(encoder, OrdinalEncoder):
        # 인코딩된 결과를 DataFrame으로 변환
        encoded_df = pd.DataFrame(
            encoded_array,
            columns=valid_cols,
            index=df.index
        )
        
        # 원본 컬럼 삭제
        if drop_original:
            df = df.drop(columns=valid_cols)
        
        # 인코딩된 컬럼 추가 (또는 원본 컬럼 교체)
        for col in valid_cols:
            df[col] = encoded_df[col]
    
    return df


def apply_encoding_pipeline(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoding_config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    인코딩 파이프라인 적용
    
    Parameters:
    -----------
    df : pd.DataFrame
        입력 데이터프레임
    categorical_cols : List[str]
        인코딩할 범주형 컬럼 리스트
    encoding_config : dict, optional
        인코딩 설정 딕셔너리 ({'onehot_cols': [...], 'ordinal_cols': [...], ...})
    
    Returns:
    --------
    pd.DataFrame: 인코딩이 적용된 데이터프레임
    """
    if encoding_config is None:
        encoding_config = {}
    
    df = df.copy()
    
    # OneHot 인코딩 적용
    onehot_cols = encoding_config.get('onehot_cols', [])
    if onehot_cols:
        onehot_params = encoding_config.get('onehot_params', {})
        df = one_hot_encode(
            df,
            categorical_cols=onehot_cols,
            **onehot_params
        )
    
    # Ordinal 인코딩 적용
    ordinal_cols = encoding_config.get('ordinal_cols', [])
    if ordinal_cols:
        ordinal_params = encoding_config.get('ordinal_params', {})
        df = ordinal_encode(
            df,
            categorical_cols=ordinal_cols,
            **ordinal_params
        )
    
    return df


# 방법 3: 파이프라인 사용 (category_orders 포함)
# config = {
#     'onehot_cols': ['color', 'gender'],
#     'ordinal_cols': ['quality', 'size'],
#     'onehot_params': {'handle_unknown': 'ignore'},
#     'ordinal_params': {
#         'category_orders': {
#             'quality': ['poor', 'average', 'good', 'excellent'],
#             'size': ['small', 'medium', 'large']
#         },
#         'handle_unknown': 'use_encoded_value',
#         'unknown_value': -1
#     }
# }
# df_encoded = apply_encoding_pipeline(df, ['color', 'gender', 'quality', 'size'], config)


# ============================================================================
# Unknown Category 처리 방법 가이드
# ============================================================================

"""
검증/테스트 데이터에서 학습 데이터에 없는 새로운 카테고리 값이 나타날 때 처리 방법:

1. One-Hot Encoding의 경우:
   -------------------------
   handle_unknown='ignore' 사용 (권장)
   
   예시:
   - 학습 데이터: ['red', 'blue', 'green'] → 3개 컬럼 생성
   - 검증 데이터: ['red', 'yellow', 'blue'] → yellow는 학습 시 없었음
   - 결과: yellow는 [0, 0, 0]으로 인코딩됨
   
   장점:
   - 컬럼 구조가 일치함 (학습/검증 모두 동일한 컬럼 수)
   - 모델이 "알 수 없는 값"을 학습할 수 있음
   
   단점:
   - 모든 컬럼이 0이면 정보 손실이 있음
   - 드물게 나타나는 값과 구분이 안 됨

2. Ordinal Encoding의 경우:
   ------------------------
   handle_unknown='use_encoded_value', unknown_value=-1 사용
   
   예시:
   - 학습 데이터: ['low', 'medium', 'high'] → [0, 1, 2]로 인코딩
   - 검증 데이터: ['low', 'unknown', 'high'] → unknown은 학습 시 없었음
   - 결과: unknown은 -1로 인코딩됨
   
   장점:
   - 명확하게 "알 수 없는 값"을 표현
   - 모델이 특별한 값으로 인식 가능
   
   단점:
   - -1이 실제 의미를 가질 수 있음 (순서가 있는 경우)

3. 대안 방법들:
   -------------
   
   a) Rare Category 처리:
      - 학습 시 빈도가 낮은 카테고리들을 미리 'rare' 또는 'other'로 그룹화
      - 검증 데이터의 새로운 값도 'rare'로 매핑
   
   b) 빈도 기반 임계값:
      - 학습 데이터에서 빈도가 특정 임계값(예: 5회) 미만인 값들을 'rare'로 처리
      - 검증 데이터의 새로운 값도 자동으로 'rare'로 처리
   
   c) Target Encoding 사용:
      - 카테고리별 타겟 평균으로 인코딩
      - 새로운 값은 전체 평균 또는 별도 처리

4. Ordinal Encoding with category_orders 사용 예시:
   -------------------------------------------------
   
   # 방법 1: ordinal_encode 함수 직접 사용
   df_encoded = ordinal_encode(
       df,
       categorical_cols=['sleep_quality', 'facility_rating', 'exam_difficulty'],
       category_orders={
           'sleep_quality': ['poor', 'average', 'good'],
           'facility_rating': ['low', 'medium', 'high'],
           'exam_difficulty': ['easy', 'moderate', 'hard']
       },
       handle_unknown='use_encoded_value',
       unknown_value=-1
   )
   
   # 방법 2: fit_encoder와 transform_with_encoder 사용 (학습/테스트 분리)
   ordinal_encoder = fit_encoder(
       X_train,
       categorical_cols=['sleep_quality', 'facility_rating'],
       encoding_type='ordinal',
       category_orders={
           'sleep_quality': ['poor', 'average', 'good'],
           'facility_rating': ['low', 'medium', 'high']
       },
       handle_unknown='use_encoded_value',
       unknown_value=-1
   )
   
   # 학습 데이터 변환
   X_train_encoded = transform_with_encoder(
       X_train,
       ordinal_encoder,
       categorical_cols=['sleep_quality', 'facility_rating']
   )
   
   # 테스트 데이터 변환
   X_test_encoded = transform_with_encoder(
       X_test,
       ordinal_encoder,
       categorical_cols=['sleep_quality', 'facility_rating']
   )
   
   # 방법 3: 파이프라인 사용 (encoding_config 활용)
   encoding_config = {
       'onehot_cols': ['gender', 'course'],
       'ordinal_cols': ['sleep_quality', 'facility_rating', 'exam_difficulty'],
       'onehot_params': {'handle_unknown': 'ignore'},
       'ordinal_params': {
           'category_orders': {
               'sleep_quality': ['poor', 'average', 'good'],
               'facility_rating': ['low', 'medium', 'high'],
               'exam_difficulty': ['easy', 'moderate', 'hard']
           },
           'handle_unknown': 'use_encoded_value',
           'unknown_value': -1
       }
   }
   df_encoded = apply_encoding_pipeline(
       df,
       categorical_cols=['gender', 'course', 'sleep_quality', 'facility_rating', 'exam_difficulty'],
       encoding_config=encoding_config
   )
"""

