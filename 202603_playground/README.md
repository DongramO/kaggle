## 202603_playground

Kaggle Playground 텔레콤 Churn 분류 프로젝트입니다.  
`common/` 공통 모듈을 사용하는 **이진 분류(Churn)** 파이프라인이며, 평가·제출은 **ROC AUC 기준 확률 예측**으로 통일되어 있습니다.  
CatBoost, LightGBM, XGBoost를 K-Fold CV로 학습하고, **앙상블(가중 평균 / Ridge 메타모델)**, **하이퍼파라미터 재사용**, **Feature Importance / EDA 시각화**를 지원합니다.

---

## 프로세스 개요

```text
1. 데이터 로드 (로컬 data/ 디렉토리)
   ↓
2. 전처리 (prepare_data)
   - ID·타겟 분리, 컬럼 타입 분류
   - Feature Engineering 1단계 (인코딩 전: 이상치 클리핑, 수작업 상호작용, 통계 특성)
   - 범주형 인코딩 (OneHot, Ordinal)
   - Feature Engineering 2단계 (인코딩 후 상호작용, 비율 특성 등)
   - 결측치 처리
   ↓
3. (옵션) Optuna 하이퍼파라미터 최적화 또는 저장된 파라미터 로드
   ↓
4. 모델 학습 (K-Fold CV)
   - CatBoost, LightGBM, XGBoost
   - OOF 예측 및 CV 점수 기록
   ↓
5. 앙상블 (가중 평균 / Ridge 메타모델)
   ↓
6. Feature Importance / Permutation Importance / 앙상블 중요도 분석
   ↓
7. 제출 파일 생성 (submission.csv: id, Churn 확률)
   ↓
8. EDA 시각화 (boxplot, histogram, 그룹별 분포, 상관계수·타겟 상관계수)
```

---

## 프로젝트 구조

```text
202603_playground/
├── main.py                      # 메인 실행 파일 (전처리 + 모델 학습 + 앙상블 + 제출 + EDA)
├── config.py                    # 프로젝트별 설정 (타겟, 인코딩, FE, Optuna, 앙상블 등)
├── README.md                    # 프로젝트 설명
├── data/                        # 로컬 데이터 디렉토리
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── best_hyperparameters.json    # 베스트 하이퍼파라미터 세트 (USE_SAVED_PARAMS=True일 때 사용)
├── auc_history.csv              # 각 모델/앙상블의 CV 점수·히스토리
├── submission.csv               # 최신 실행의 제출 파일
├── feature_importance_results/  # 모델·앙상블 Feature Importance 및 비교 결과
├── eda_results/                 # EDA 시각화 결과 (PNG 등)
└── visualization.py             # EDA 전용 실행 스크립트
```

---

## 설정 요약 (config.py 기준)

### 데이터 / 태스크

- **타겟 컬럼**: `Churn` (이탈 여부, 이진 분류)
- **ID 컬럼**: `id`
- **태스크 타입**: `TASK_TYPE = 'classification'`
- **평가 지표**: `SCORING_METRIC = 'auc'`  
  - 내부 CV, 앙상블, 제출 모두 **양성 클래스(Churn=1) 확률** 기반 AUC를 사용

### 모델 / CV / GPU

- **사용 모델**: `MODEL_TYPES = ['catboost', 'lightgbm', 'xgboost']`
- **교차검증**: `N_FOLDS = 5` (Classification → StratifiedKFold)
- **랜덤 시드**: `RANDOM_STATE = 42`
- **GPU 사용**: `USE_GPU = True`  
  - 가능 시 `common.utils.helpers.setup_gpu_params`를 통해 각 모델별 GPU 설정 자동 적용

### 인코딩 설정 (ENCODING_CONFIG)

- **OneHot 인코딩 컬럼** (`onehot_cols`)
  - `['gender', 'InternetService', 'PaymentMethod']`
- **Ordinal 인코딩 컬럼** (`ordinal_cols`)
  - `['PhoneService', 'PaperlessBilling', 'Contract']`
- **OneHot 옵션**
  - `handle_unknown='ignore'`, `drop='first'`
- **Ordinal 옵션**
  - `handle_unknown='use_encoded_value'`, `unknown_value=-1`
  - `category_orders`는 Churn 데이터에 맞는 순서로 설정
- **원본 범주형 유지 여부**
  - `drop_original = False`  
  - 인코딩된 컬럼을 추가하고 원본 범주형 컬럼도 유지  
  - CatBoost는 원본 범주형 + 인코딩 컬럼 모두 사용, LightGBM/XGBoost는 인코딩 컬럼만 사용

### Feature Engineering 설정 (FEATURE_ENGINEERING_CONFIG)

- **이상치 클리핑 (`clip_outliers`)**
  - `flag = True`
  - 0.99 분위 기반 클리핑:
    - `tenure`, `MonthlyCharges`, `TotalCharges`
- **숫자 변환 / 이진 매핑 (`change_to_numeric`, `convert_ordered_categorical_to_numeric`)**
  - 대상 컬럼:
    - `Partner`, `Dependents`, `PhoneService`, `MultipleLines`,
      `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`,
      `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`
  - 예시 매핑:
    - `Partner/Dependents`: No→0, Yes→1
    - `Contract`: Month-to-month→1, One year→12, Two year→24
    - `No internet service` 등은 0으로 통합
- **인코딩 전 상호작용 (`create_interactions_before_encoding`)**
  - `flag = True`
  - 주요 조합:
    - `MonthlyCharges` vs `MonthlyCharges_mean`
    - `TotalCharges` vs `TotalCharges_mean`
    - `InternetService` × (`OnlineSecurity_numeric`, `OnlineBackup_numeric`, `DeviceProtection_numeric`, `TechSupport_numeric`, `StreamingTV_numeric`, `StreamingMovies_numeric`)
    - `SeniorCitizen` × `MonthlyCharges`
    - `Dependents_numeric` × `MonthlyCharges`
- **인코딩 후 상호작용 (`create_interactions_after_encoding`)**
  - `flag = True`
  - 예:
    - `InternetService_Fiber optic_encoded` × `MonthlyCharges`
    - `InternetService_No_encoded` × `MonthlyCharges`
- **비율 특성 (`create_ratios`)**
  - `flag = True`
  - `numerator_cols = ['TotalCharges', 'tenure', 'MonthlyCharges']`
  - `denominator_cols = ['tenure', 'Contract_numeric', 'TotalCharges']`
  - 이름 규칙: `ratio_feature_names = '_ratio'`
- **통계 특성 (`create_statistical_features`)**
  - `flag = True`
  - 그룹: `['tenure', 'MonthlyCharges', 'TotalCharges']`
  - 통계량: `['mean']`
- **수치 변환 (`transform_numeric_features`)**
  - 현재 `flag = False` (필요 시 log 등의 변환 추가 가능)

### 앙상블 / 하이퍼파라미터

- **앙상블 방법 (`ENSEMBLE_METHOD`)**
  - `'weighted_average'` (기본)
    - `EnsembleModel._optimize_weights`로 가중치 최적화 (각 모델 최소 가중치 0.05 이상)
  - `'simple_average'`
    - 모든 모델 동일 가중치
  - `'ridge_meta'`
    - OOF 예측을 입력으로 Ridge 회귀 메타모델 학습, 계수를 가중치처럼 사용
- **Ridge 메타모델**
  - `RIDGE_ALPHA`: Ridge 정규화 강도 (기본 1.0)
  - `OPTUNA_RIDGE_ALPHA = True`이면 `optimize_ridge_alpha`로 알파를 Optuna로 최적화
- **하이퍼파라미터 재사용 / 최적화**
  - `USE_OPTUNA = False` (기본, Optuna 비활성)
  - `USE_SAVED_PARAMS = True`  
    - `best_hyperparameters.json`에 저장된 파라미터를 불러와 사용
  - `N_TRIALS`, `OPTUNA_SAMPLE_SIZE` 등으로 탐색 설정 조정 가능
  - `USE_PERMUTATION_IMPORTANCE`로 permutation importance 계산 여부 제어

---

## main.py 실행 흐름

1. **데이터 로드**
   - `load_data(data_dir=project_root)`  
   - 로컬 `data/` 디렉토리에서 `train.csv`, `test.csv`, `sample_submission.csv` 로드
2. **전처리 (`prepare_data`)**
   - ID 컬럼 제거 (`ID_COL='id'`)
   - 타겟 분리 (`TARGET_COL='Churn'`)
   - 수치형/범주형 컬럼 분류
   - Feature Engineering 1단계 (인코딩 전: 이상치 클리핑, 숫자 매핑, 상호작용, 통계 특성 등)
   - 범주형 인코딩 (OneHot, Ordinal)
   - Feature Engineering 2단계 (인코딩 후 상호작용, 비율, 추가 상호작용)
   - 원본 범주형 삭제 여부 처리 (`drop_original` 설정)
   - 결측치 처리 (수치형 평균 대체)
3. **EDA 실행**
   - 전처리 완료된 `X_train` + `y_train`을 결합해 `run_eda_visualization(df_for_vis, target_col=TARGET_COL)` 호출
   - 결과는 `eda_results/`에 PNG로 저장
4. **모델 학습 (`ModelTrainer`)**
   - CatBoost / LightGBM / XGBoost 각각에 대해:
     - K-Fold CV 학습
     - OOF 예측, Fold별 점수·CV 평균/표준편차 계산
     - `auc_history.csv`에 로그 추가
     - 테스트 데이터 예측 저장
5. **Feature Importance / Permutation Importance**
   - `analyze_feature_importance`로 모델별 중요도 분석 및 저장
   - `USE_PERMUTATION_IMPORTANCE=True`일 때 permutation importance 추가 계산
6. **앙상블 (`EnsembleModel`)**
   - OOF 예측 딕셔너리(모델별)를 입력으로 앙상블 학습
   - `ENSEMBLE_METHOD`, `RIDGE_ALPHA`, `OPTUNA_RIDGE_ALPHA` 설정에 따라
     - 가중 평균 최적화 또는 Ridge 메타모델 사용
   - 앙상블 AUC를 `auc_history.csv`에 추가
   - `analyze_ensemble_feature_importance`, `compare_model_and_ensemble_importance`로 중요도 분석
7. **제출 파일 생성**
   - 앙상블 예측에서 양성 클래스(Churn=1) 확률을 사용
   - `submission.csv` 생성:
     - 컬럼: `id`, `Churn`

---

## EDA / 시각화 (visualization.py)

- **`run_eda_visualization` 주요 기능**
  - Boxplot, Histogram (수치형 컬럼) — 컬럼 수가 많으면 여러 파일로 분할 저장
  - 그룹별 Histogram / Boxplot:
    - 예: `tenure` vs `Churn`, `MonthlyCharges` vs `Churn`, `Contract` vs `Churn`,
      `InternetService` vs `Churn`, `SeniorCitizen`/`Dependents` 등
  - 범주형 분포 시각화 (`plot_categorical`)
  - 상관계수 히트맵 및 타겟 상관관계:
    - `common.eda.correlation.calculate_correlation_matrix`
    - `plot_correlation_heatmap`, `plot_correlation_with_target`
- **실행 방법**
  - `main.py` 실행 시 전처리된 데이터 기준으로 자동 호출
  - 원본 train 기준 EDA만 별도로 보고 싶다면:
    ```bash
    cd 202603_playground
    python visualization.py
    ```

---

## 공통 모듈 사용

| 모듈 | 용도 |
|------|------|
| `common.data.loader` | `load_data()` — 로컬 데이터 디렉토리에서 CSV 로드 |
| `common.preprocess.encoder` | `fit_encoder`, `transform_with_encoder` — OneHot / Ordinal 인코딩 |
| `common.preprocess.feature_engineering` | 이상치 클리핑, 상호작용, 비율, 통계 특성 등 |
| `common.modeling.model` | `ModelTrainer`, `EnsembleModel` — CV 학습, OOF 예측, 앙상블 |
| `common.modeling.hyperparameter` | `optimize_hyperparameters`, `save_hyperparameters`, `optimize_ridge_alpha` |
| `common.eda.feature_importance` | `analyze_feature_importance`, `analyze_ensemble_feature_importance`, `compare_model_and_ensemble_importance` |
| `common.eda.error_analysis` | `analyze_high_error_samples` (고오차 샘플 분석) |
| `common.eda.visualization` / `common.eda.correlation` | EDA 기본 시각화 및 상관관계 분석 |

---

## 실행 방법 요약

- **모델 학습 + 앙상블 + 제출 + EDA**

```bash
cd 202603_playground
python main.py
```

- **EDA만 별도 실행 (원본 train 기준)**

```bash
cd 202603_playground
python visualization.py
```

---

## 참고

- **하이퍼파라미터 / 성능 로그**
  - `best_hyperparameters.json`: 현재 사용 중인 베스트 파라미터 세트
  - `auc_history.csv`: CatBoost/LightGBM/XGBoost/앙상블의 CV 점수 히스토리
- **워크스페이스 공통 구조**
  - 루트 `STRUCTURE.md`
- **이전 Playground 참고**
  - 전체적인 흐름은 `202601_playground`와 유사하되,  
    데이터 컬럼과 Feature Engineering 내용은 현재 텔레콤 Churn 대회에 맞게 조정되어 있음.
