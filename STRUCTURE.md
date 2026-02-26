# Kaggle Workspace 구조

## 전체 구조

```
kaggle/
├── common/                          # 공통 모듈 (재사용 가능)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                # Kaggle/로컬 환경 자동 감지 및 데이터 로드
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── encoder.py               # OneHot, Ordinal 인코딩
│   │   └── feature_engineering.py   # Feature Engineering 함수들
│   ├── modeling/
│   │   ├── __init__.py
│   │   ├── model.py                 # CatBoost, LightGBM, XGBoost 모델 클래스
│   │   ├── train.py                 # K-Fold CV 학습 로직 (향후 trainer.py로 변경 예정)
│   │   └── hyperparameter.py        # Optuna 하이퍼파라미터 최적화
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── visualization.py         # 시각화 함수들
│   │   ├── feature_importance.py    # Feature Importance 분석
│   │   └── error_analysis.py        # 에러 분석
│   └── utils/
│       ├── __init__.py
│       └── helpers.py               # 유틸리티 함수들 (GPU 체크 등)
│
├── 202511_playground/               # 기존 프로젝트 (유지)
├── 202601_playground/               # 기존 프로젝트 (유지)
│
└── 202611_playground/               # 새 프로젝트 템플릿
    ├── main.py                      # 메인 실행 파일
    ├── config.py                    # 프로젝트별 설정
    ├── README.md                    # 프로젝트 설명
    └── data/                        # 프로젝트별 데이터
        ├── train.csv
        ├── test.csv
        └── sample_submission.csv
```

## 주요 변경 사항

### 1. 공통 모듈 (`common/`)
- **데이터 로드**: Kaggle/로컬 환경 자동 감지
- **전처리**: 인코딩, Feature Engineering
- **모델링**: CatBoost, LightGBM, XGBoost, 앙상블
- **EDA**: 시각화, Feature Importance, Error Analysis
- **유틸리티**: GPU 체크 등 공통 함수

### 2. 새 프로젝트 템플릿 (`202611_playground/`)
- `config.py`: 프로젝트별 설정 (타겟 컬럼, 인코딩, Feature Engineering 등)
- `main.py`: 공통 모듈을 사용하는 메인 실행 파일
- `data/`: 프로젝트별 데이터 저장

## 사용 방법

### 새 프로젝트 생성
1. `202611_playground/` 디렉토리를 복사하여 새 프로젝트 생성
2. `config.py`에서 프로젝트별 설정 수정
3. `data/` 디렉토리에 데이터 파일 넣기
4. `main.py` 실행

### 공통 모듈 사용 예시

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.data.loader import load_data
from common.preprocess.encoder import fit_encoder
from common.modeling.model import ModelTrainer
from common.modeling.model import EnsembleModel

# 데이터 로드 (프로젝트 루트 자동 감지)
df_train, df_test, df_sub = load_data(
    project_root=os.path.dirname(__file__)
)
```

## 향후 개선 사항

1. **모델링 모듈 정리**
   - `train.py` → `trainer.py`로 이름 변경
   - `EnsembleModel`을 별도 파일로 분리 (`ensemble.py`)
   - `model.py`에서 GPU 함수를 `common.utils.helpers`로 이동

2. **경로 참조 수정**
   - 기존 프로젝트들이 `common/` 모듈을 사용하도록 수정 (선택사항)

3. **문서화**
   - 각 모듈별 사용 예시 추가
   - API 문서 작성

4. **테스트**
   - 새 프로젝트 템플릿 테스트
   - 기존 프로젝트와의 호환성 확인
