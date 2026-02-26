# 202611_playground

새로운 Kaggle Playground 프로젝트 템플릿

## 구조

```
202611_playground/
├── main.py                      # 메인 실행 파일
├── config.py                    # 프로젝트별 설정
├── data/                        # 프로젝트별 데이터
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── best_hyperparameters.json    # 최적화된 하이퍼파라미터
├── submission.csv               # 제출 파일
└── training_summary.txt         # 학습 결과 요약
```

## 사용 방법

### 1. 데이터 준비
`data/` 디렉토리에 다음 파일들을 넣어주세요:
- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 2. 설정 수정
`config.py`에서 프로젝트별 설정을 수정하세요:
- 타겟 컬럼 이름
- 인코딩 설정
- Feature Engineering 설정
- 하이퍼파라미터 최적화 설정

### 3. 실행
```bash
python main.py
```

## 공통 모듈 사용

이 프로젝트는 `common/` 디렉토리의 공통 모듈을 사용합니다:

- `common.data.loader`: 데이터 로드 (Kaggle/로컬 환경 자동 감지)
- `common.preprocess`: 전처리 (인코딩, Feature Engineering)
- `common.modeling`: 모델링 (CatBoost, LightGBM, XGBoost, 앙상블)
- `common.eda`: EDA (시각화, Feature Importance, Error Analysis)
- `common.utils`: 유틸리티 함수들

## 예시

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from common.data.loader import load_data
from common.preprocess.encoder import fit_encoder
from common.modeling.trainer import ModelTrainer
from common.modeling.ensemble import EnsembleModel

# 데이터 로드
df_train, df_test, df_sub = load_data(
    project_root=os.path.dirname(__file__)
)
```
