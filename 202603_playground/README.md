## 202603_playground

Kaggle Playground 텔레콤 Churn 분류 프로젝트입니다.  
`common/` 공통 모듈을 사용하는 **이진 분류(Churn)** 파이프라인이며, 평가·제출은 **ROC AUC 기준 확률 예측**으로 통일되어 있습니다.  
CatBoost, LightGBM, XGBoost(및 선택 시 **MLP**)를 K-Fold CV로 학습하고, **앙상블(가중 평균 / Ridge 메타모델)**, **하이퍼파라미터 재사용**, **Feature Importance / EDA 시각화**를 지원합니다.

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
   - CatBoost, LightGBM, XGBoost, (옵션) MLP
   - OOF 예측 및 CV 점수 기록
   ↓
5. 앙상블 (가중 평균 / Ridge 메타모델)
   ↓
6. Feature Importance / Permutation Importance / 앙상블 중요도 분석
   ↓
6-1. (옵션) MLP 기반 분석 초점 수치화: 특성 중요도·특성 쌍 상호작용·트리와 비교(USE_MLP_FOCUS_ANALYSIS=True)
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
├── feature_importance_results/  # 모델·앙상블 Feature Importance, MLP 분석 초점(mlp_feature_focus.csv, mlp_interaction_strength.csv, mlp_vs_tree_focus.csv)
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

- **사용 모델**: `MODEL_TYPES = ['catboost', 'lightgbm', 'xgboost']` (특성 간 관계 보완용으로 `'mlp'` 추가 가능)
- **교차검증 (본 학습)**: `N_FOLDS = 3` (Classification → StratifiedKFold)
- **Optuna 전용 교차검증**: `OPTUNA_N_FOLDS`  
  - 기본값은 `N_FOLDS`와 동일하게 두고, **Optuna 최적화 속도를 위해 fold 수를 줄이고 싶을 때만 별도로 조정**할 수 있음.
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
    - `MonthlyCharges` × `tenure` (평균 대비 현재 요금 비율 계산용)
- **비율 특성 (`create_ratios`)**
  - `flag = True`
  - `tenure/Contract_numeric`, `TotalCharges/(MonthlyCharges×Contract_numeric)` 등 계약·요금·기간 관계 비율
  - 이름 규칙: `ratio_feature_names = '_ratio'`
- **Target Encoding (`target_encoding`)**
  - `flag = False` (기본값, 필요 시 True로 변경)
  - 예: `PaymentMethod`, `InternetService` 등 일부 범주형 컬럼에 대해 **K-Fold 기반 Smooth Target Encoding** 적용
  - 설정: `cols`, `n_splits`(기본 `N_FOLDS`), `smoothing`(기본 10.0)
- **tenure / MonthlyCharges / TotalCharges 사칙연산 (`create_tenure_monthly_total_arithmetic`)**
  - `flag = True`
  - 세 수치형 간 쌍별 add, multiply, subtract, divide(양방향)로 18개 신규 특성 생성.
- **tenure 파생 (`create_tenure_derived`)**
  - `flag = True`
  - `tenure_years`: 가입 개월 수를 년 단위로 `floor(tenure/12)`.
  - `tenure_segment`: 구간 0–12, 12–24, 24–48, 48–72(및 72 초과)로 구간화, 레이블 0/1/2/3. `tenure_segment_bins = [0, 12, 24, 48, 72]`.
- **통계 특성 (`create_statistical_features`)**
  - `flag = False` (필요 시 True로 변경)
  - 그룹: `['tenure', 'MonthlyCharges', 'TotalCharges']`
  - 통계량: `['mean']`
- **이용 인터넷 서비스 개수 (`num_internet_services`)**
  - `OnlineSecurity_numeric`, `OnlineBackup_numeric`, `DeviceProtection_numeric`, `TechSupport_numeric`, `StreamingTV_numeric`, `StreamingMovies_numeric` (Yes=1) 합산.
  - 0~6 정수: 가입한 인터넷 부가 서비스 개수.
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
  - `USE_MLP_FOCUS_ANALYSIS=True`로 MLP 기반 분석 초점 수치화(특성 중요도·상호작용·트리 비교) 실행
  - `USE_MLP_FOCUS_ONLY=True`로 모델 학습 없이 전처리 후 MLP 분석만 실행 (트리 비교 없음)

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

### 신규 특성 추가 후 정확도가 안 오를 때 / Feature Importance 확인

신규 특성을 넣었는데 CV 점수(AUC 등)가 거의 안 오르는 경우, **Feature Importance 수치를 확인하는 것이 좋습니다.**

**가능한 원인**
1. **이미 원본 특성으로 설명됨**  
   GBDT는 `tenure`, `MonthlyCharges`, `TotalCharges`만으로도 비선형·조합을 잘 잡습니다. 비율/상호작용이 “추가 정보”가 아니라 **원본의 선형 결합**에 가까우면 이득이 작을 수 있습니다.
2. **중복(다중공선성)**  
   새 특성이 기존 특성과 높은 상관이면, 모델이 둘 중 하나만 써도 되므로 새 특성 중요도가 낮게 나옵니다.
3. **효과가 작음**  
   새 특성이 약한 신호만 주면, CV 분산 안에서 점수 개선이 눈에 안 띌 수 있습니다.
4. **Fold별로 쓰임이 다름**  
   평균 importance는 낮은데 일부 Fold에서는 쓰일 수 있어, 제거하면 오히려 떨어질 수 있습니다.

**확인 방법**  
- `main.py` 실행 후 `feature_importance_results/` 아래에  
  `*_feature_importance.csv`, `ensemble_feature_importance.csv` 가 생성됩니다.
- **어떤 특성이 쓰였는지**: Importance > 0 인지, 상위 몇 %인지 확인.
- **신규 특성만 보기**: CSV에서 `MonthlyCharges_TotalCharges_ratio`, `MonthlyCharges_x_tenure_FE_TotalCharges_ratio` 등 이름으로 검색해 순위와 수치를 봅니다.

예: CatBoost 기준으로  
- `MonthlyCharges_x_tenure_FE_TotalCharges_ratio` 는 어느 정도 importance를 가지는 반면,  
- `MonthlyCharges_TotalCharges_ratio` 는 매우 낮게 나올 수 있습니다.  
→ “현재 요금 / 과거 평균”은 보조로 쓰이지만, “현재/누적” 비율은 기존 특성으로 이미 설명되는 경우가 많다는 해석이 가능합니다.

**정리**  
- **Importance를 확인하면** “실제로 쓰였는지, 얼마나 기여했는지”를 알 수 있어, 신규 특성 유지/제거/다른 설계로 갈지 결정하는 데 도움이 됩니다.
- 새 특성이 중요도가 거의 0이면 **제거해도 성능은 비슷**할 가능성이 크고, 낮지만 0이 아니면 **유지**하는 편이 안전합니다.

### 노이즈를 줄이는 방법

| 구분 | 방법 | 현재 프로젝트 |
|------|------|----------------|
| **특성** | 이상치 클리핑(분위수) | ✅ `clip_outliers` (tenure, MonthlyCharges, TotalCharges) |
| **특성** | importance 0/극소 특성 제거 | ✅ 비율·상호작용·Partner_numeric 등 정리함 |
| **특성** | 상관 높은 특성 중 일부 제거 또는 요약 | `eda_results/correlation_heatmap_after_fe.png` 참고 후 선택 |
| **특성** | 스케일링(StandardScaler 등) | GBDT는 스케일에 강함, 필수 아님 |
| **샘플** | 이상치/극단 샘플 제거 또는 가중치 감소 | 현재 미적용, 필요 시 학습 시 `sample_weight` 등 |
| **샘플** | 라벨 노이즈(오표기) 정제 | 도메인/전수 검토 필요 |
| **모델** | 정규화·조기 종료 | ✅ GBDT `early_stopping`, max_depth 등으로 과적합 억제 |
| **모델** | 앙상블 | ✅ Ridge 메타·가중 평균으로 분산 감소 |

- **상관계수 히트맵**  
  `main.py` 실행 시 `prepare_data` 직후, **학습에 쓰이는 수치형 특성만**으로 상관 행렬을 계산해  
  `eda_results/correlation_heatmap_after_fe.png` 에 저장합니다.  
  상관이 매우 높은 쌍(예: 0.9 이상)은 중복·노이즈 후보로 보고, importance와 함께 판단해 제거할 수 있습니다.

#### min_frequency / max_categories (OneHotEncoder) 설명

sklearn의 **OneHotEncoder**에서 범주형 변수의 **희소/과다 카테고리**를 줄여 노이즈를 낮추는 옵션입니다.

| 파라미터 | 타입 | 의미 |
|----------|------|------|
| **min_frequency** | float 또는 int | **비율(float, 0~1)** 이면: 해당 변수에서 **상대 빈도가 이 값보다 작은** 카테고리들을 하나의 “infrequent” 그룹으로 묶습니다. 예: `0.01` → 1% 미만인 레벨을 묶음. **정수(int)** 이면: **건수가 이 값보다 작은** 카테고리를 묶습니다. 예: `5` → 5건 미만인 레벨을 묶음. |
| **max_categories** | int | 변수당 **최대 허용 카테고리 수**. 빈도 순으로 상위 `max_categories`개만 그대로 두고, 나머지는 모두 하나의 “infrequent” 카테고리로 묶습니다. 예: `10` → 최대 10개 레벨만 유지, 나머지는 묶음. |

**효과**
- **희소 레벨**이 one-hot 컬럼을 많이 만들고, 학습 데이터에만 나온 레벨은 **노이즈·과적합** 요인이 됩니다.  
  `min_frequency` 또는 `max_categories`를 주면 **드문 레벨을 “기타”처럼 하나로 묶어** one-hot 컬럼 수가 줄고, 일반화에 도움이 될 수 있습니다.
- **drop**과 함께 쓰면: 먼저 infrequent 묶기가 적용된 뒤, `drop='first'` 등이 적용됩니다.

**설정 예 (onehot_params)**  
- `min_frequency=0.01` → 1% 미만 빈도 카테고리 묶기  
- `max_categories=10` → 변수당 최대 10개 카테고리만 유지  

현재 프로젝트의 인코더(`common.preprocess.encoder`)에서 OneHotEncoder에 `**onehot_params`를 넘기므로, config의 `onehot_params`에 `min_frequency` 또는 `max_categories`를 넣으면 적용됩니다.  
범주 수가 많은 변수(PaymentMethod, InternetService 등)에 특히 유용합니다.  
**이 프로젝트(Churn)**: 범주형 변수의 레벨 수가 많지 않아 기본값(None)으로 두고 있으며, 설정하지 않아도 됩니다.

### 전체 특성 판단 (importance·상관·역할 기준)

아래는 현재 파이프라인의 **모든 특성**을 importance(CatBoost·LightGBM·앙상블), 상관 heatmap, 모델별 입력 구조를 보고 정리한 판단입니다.

---

#### 1. 유지 (핵심·높은 기여)

| 특성 | 판단 근거 |
|------|-----------|
| **tenure** | CatBoost 1위, 앙상블·LGB에서도 상위. 이탈과 직결된 “가입 기간” 신호. |
| **MonthlyCharges_x_tenure_FE_TotalCharges_ratio** | CatBoost 2위, “현재 요금/과거 평균” 비율. 앙상블에서도 사용. |
| **Contract_numeric** | Cat/앙상블/LGB 모두 상위. 계약 기간(1/12/24)으로 이탈과 강한 관계. |
| **MonthlyCharges, TotalCharges** | 상관은 높지만(heatmap ~0.69) 각각 importance 있음. GBDT가 구간별로 다르게 사용. |
| **high_risk_combo** | CatBoost 3위. 월간+전자결제 조합으로 도메인 신호. LGB는 극소→모델별 차이 있으나 Cat에서 핵심. |
| **has_internet** | Cat·앙상블에서 사용. 인터넷 유무 이진 특성. |
| **tenure_MonthlyCharges_mean** | 앙상블·LGB 상위. (tenure, MonthlyCharges, TotalCharges) 그룹 평균. TotalCharges와 상관 높아도 “요약 축”으로 쓰임. |
| **인코딩된 범주형** (InternetService_No_encoded, gender_Female_encoded, PaymentMethod_* 등) | LGB/앙상블에서 상위. OneHot 결과끼리 상관은 음수인 것이 정상. |
| **OnlineSecurity, TechSupport, StreamingTV_numeric, StreamingMovies_numeric, Dependents_numeric, MultipleLines** | Cat·앙상블에서 일정 이상 기여. StreamingTV–StreamingMovies 상관 ~0.66이지만 둘 다 사용됨. |

---

#### 2. 상관은 높지만 둘 다 유지

| 관계 | 판단 근거 |
|------|-----------|
| **tenure ↔ TotalCharges** | heatmap ~0.77. 둘 다 importance 있음. GBDT가 구간·조합을 다르게 써서 둘 다 두는 편이 유리. |
| **tenure_MonthlyCharges_mean ↔ TotalCharges** | 상관 매우 높음(~0.99 수준). mean은 “세 변수 요약” 역할로 앙상블/LGB에서 쓰이므로, TotalCharges만으로 대체하지 않고 **둘 다 유지** 권장. |
| **StreamingTV_numeric ↔ StreamingMovies_numeric** | ~0.66. 둘 다 사용되며 의미가 다소 달라 유지. |

---

#### 3. 제거·축소 후보 (근거 요약)

| 특성 | importance | 판단 근거 |
|------|------------|-----------|
| **TotalCharges_tenure_ratio** | Cat 극소, LGB 0, 앙상블 ≈0 | TotalCharges, tenure, tenure_MonthlyCharges_mean으로 이미 설명 가능. “월평균 누적” 정보 중복. **제거 권장.** |
| **tenure_Contract_numeric_ratio** | Cat 0.00095, LGB 0, 앙상블 0.00011 | tenure, Contract_numeric 각각 이미 사용. 비율의 추가 이득 거의 없음. **제거 검토.** |
| **MonthlyCharges_x_tenure_FE** (단독) | Cat 극소, LGB 0 | ratio(MonthlyCharges_x_tenure_FE_TotalCharges_ratio) 계산용으로만 필요. **컬럼은 유지**(ratio 생성에 필요), “단독 특성”으로서는 기대하지 않음. |
| **PaperlessBilling_encoded, PhoneService_encoded, Contract_encoded** | 앙상블 0 | LGB/XGB는 이 인코딩 컬럼을 쓰지만 기여도 0. CatBoost는 원본 범주 사용. 파이프라인 통일을 위해 **그대로 두어도 되고**, 실험으로 제거 시 성능 변화만 확인하면 됨. |

---

#### 4. 정리

- **반드시 유지**: tenure, Contract_numeric, MonthlyCharges, TotalCharges, MonthlyCharges_x_tenure_FE_TotalCharges_ratio, high_risk_combo, has_internet, tenure_MonthlyCharges_mean, 인코딩된 InternetService·PaymentMethod·gender, 서비스·가구 특성(OnlineSecurity, StreamingTV_numeric 등).
- **제거 권장(설정 변경)**: **TotalCharges_tenure_ratio** — create_ratios에서 해당 쌍 제거.
- **선택 제거(실험 권장)**: **tenure_Contract_numeric_ratio** — 제거 후 CV 비교해 보면 됨.
- **상관만 높고 둘 다 쓰이는 특성**은 제거하지 않고 유지하는 편이 안전함.

---

- **하이퍼파라미터 / 성능 로그**
  - `best_hyperparameters.json`: 현재 사용 중인 베스트 파라미터 세트  
    - Optuna를 다시 돌릴 때 **기존 파일을 덮어쓰지 않고**, 새로 최적화된 모델의 하이퍼파라미터만 해당 모델 키에 대해 갱신한다.  
    - 예를 들어 CatBoost만 다시 튜닝하면 `hyperparameters.catboost`만 업데이트되고, LightGBM/XGBoost/MLP 설정은 그대로 유지된다.
  - `auc_history.csv`: CatBoost/LightGBM/XGBoost/앙상블의 CV 점수 히스토리
- **워크스페이스 공통 구조**
  - 루트 `STRUCTURE.md`
- **이전 Playground 참고**
  - 전체적인 흐름은 `202601_playground`와 유사하되,  
    데이터 컬럼과 Feature Engineering 내용은 현재 텔레콤 Churn 대회에 맞게 조정되어 있음.
