# 202603_playground

Kaggle Playground 프로젝트. `common/` 공통 모듈을 사용하는 **분류(Churn)** 학습 파이프라인입니다. 평가·제출은 **ROC AUC용 확률**로 통일되어 있으며, CatBoost, LightGBM, XGBoost 앙상블과 Optuna 하이퍼파라미터 최적화를 지원합니다.

## 프로세스 개요

```
1. 데이터 로드 (Kaggle/로컬 환경 자동 감지)
   ↓
2. 전처리 (prepare_data)
   - ID·타겟 분리, 컬럼 타입 분류
   - Feature Engineering (이상치 클리핑, 상호작용·비율·범주형 상호작용)
   - 범주형 인코딩 (OneHot, Ordinal)
   - 결측치 처리
   ↓
3. 하이퍼파라미터 최적화 (Optuna, 선택)
   ↓
4. 모델 학습 (K-Fold CV)
   - CatBoost, LightGBM, XGBoost
   ↓
5. 앙상블 (config 선택: 가중 평균 / 단순 평균 / Ridge 메타모델)
   ↓
6. 제출 파일 생성 (submission.csv)
```

## 프로젝트 구조

```
202603_playground/
├── main.py                    # 메인 실행 파일
├── config.py                  # 프로젝트별 설정 (타겟, 인코딩, FE, Optuna 등)
├── README.md                  # 프로젝트 설명
├── data/                      # 데이터 (프로젝트 루트 또는 Kaggle input)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── best_hyperparameters.json  # Optuna 최적화 결과 (선택)
├── submission.csv             # 최종 제출 파일
└── training_summary.txt       # 학습 요약 (선택)
```

## 설정 요약 (config.py)

### 데이터
- **타겟**: Churn (분류, 이탈 여부)
- **ID**: `id`
- **제출**: ROC AUC용 **확률** (양성 클래스 확률)

### 인코딩
- **OneHot**: `gender`, `course`, `internet_access`, `study_method`
- **Ordinal**: `facility_rating`, `sleep_quality`, `exam_difficulty`  
  - 순서: `sleep_quality` (poor → average → good), `facility_rating` (low → medium → high), `exam_difficulty` (easy → moderate → hard)

### Feature Engineering
- **이상치 클리핑**: `study_hours`, `class_attendance`, `sleep_hours`, `age` (99% 분위)
- **인코딩 전 상호작용**: `class_attendance×study_hours`, `study_hours×sleep_hours` (multiply, add)
- **인코딩 후 상호작용**: `class_attendance×sleep_quality_encoded`, `study_hours×sleep_quality_encoded`
- **비율 특성**: `study_hours / (study_hours + sleep_hours)` 등 (`ratio_feature_names` 사용)
- **범주형 상호작용**: `facility_rating_exam_difficulty`, `sleep_quality_exam_difficulty`

### 학습
- **태스크**: Classification (`TASK_TYPE`, `SCORING_METRIC`으로 문제별 설정)
- **이번 대회**: `SCORING_METRIC='auc'` → 확률 예측·AUC 평가
- **K-Fold**: 5
- **GPU**: `USE_GPU = True` (지원 시 사용)
- **Optuna**: `USE_OPTUNA = False` 기본, `USE_SAVED_PARAMS`로 저장 파라미터 사용 가능

## 실행 방법

### 로컬
```bash
cd 202603_playground
python main.py
```

### EDA 시각화
```bash
python visualization.py
```
- `data/`에서 train 로드 후 boxplot, histogram, 범주형 분포, 상관계수 히트맵·타겟 상관관계 생성
- 결과는 `eda_results/`에 저장 (boxplot.png, histogram.png, categorical.png, correlation_heatmap.png, correlation_with_target.png)
- 옵션은 `run_eda_visualization()` 인자로 지정 (figsize, bins, show_outliers, show_stats, correlation_method 등)

### 데이터 위치
- **로컬**: 프로젝트 내 `data/` 디렉토리에 `train.csv`, `test.csv`, `sample_submission.csv` 배치
- **Kaggle**: `/kaggle/input/...` 에서 `common.data.loader.load_data(project_root=...)` 가 자동 감지

## 공통 모듈 사용

| 모듈 | 용도 |
|------|------|
| `common.data.loader` | `load_data()` — Kaggle/로컬 경로 자동 감지 |
| `common.preprocess.encoder` | `fit_encoder`, `transform_with_encoder` — OneHot/Ordinal |
| `common.preprocess.feature_engineering` | 이상치 클리핑, 상호작용, 비율, 범주형 상호작용 |
| `common.modeling.model` | `ModelTrainer`, `EnsembleModel` — CV 학습 및 앙상블 |
| `common.modeling.hyperparameter` | `optimize_hyperparameters`, `save_hyperparameters` — Optuna |
| `common.eda.feature_importance` | `analyze_feature_importance` |
| `common.eda.error_analysis` | `analyze_high_error_samples` |

## main.py 흐름

1. **전처리 (prepare_data)**: ID/타겟 분리, FE 1단계(클리핑·인코딩 전 상호작용) → 인코딩 → FE 2단계(인코딩 후 상호작용, 비율, 범주형 상호작용) → 결측치 처리
2. **특성 공학**: `config.py`의 `FEATURE_ENGINEERING_CONFIG`에 따라 적용
3. **인코딩**: OneHot / Ordinal — CatBoost는 원본 범주형, LightGBM/XGBoost는 인코딩된 컬럼 사용
4. **모델링**: `ModelTrainer`로 CatBoost, LightGBM, XGBoost 각각 K-Fold 학습
5. **앙상블**: OOF 예측으로 앙상블 — `ENSEMBLE_METHOD`에 따라 가중 평균(최적화)·단순 평균 또는 **Ridge 메타모델**(2차 스태킹) 적용
6. **제출**: `submission.csv` 생성 (id, Churn 확률)

## 옵션 (config.py)

- **ENSEMBLE_METHOD**: `'weighted_average'`(가중치 최적화) | `'simple_average'` | `'ridge_meta'`(Ridge 2차 메타모델)
- **RIDGE_ALPHA**: `ridge_meta` 사용 시 Ridge 정규화 강도 (Optuna 미사용 시 적용, 기본 1.0)
- **OPTUNA_RIDGE_ALPHA**: `True`면 Ridge alpha를 Optuna로 최적화
- **RIDGE_ALPHA_N_TRIALS**: Ridge alpha Optuna 시도 횟수 (기본 20)
- **USE_OPTUNA**: `True` 시 Optuna로 하이퍼파라미터 탐색
- **USE_SAVED_PARAMS**: `True`면 `best_hyperparameters.json` 사용, `None`이면 파일 존재 시 자동 사용
- **OPTUNA_SAMPLE_SIZE**: 대용량일 때 샘플 수 제한 (예: 50000)
- **USE_PERMUTATION_IMPORTANCE**: Permutation Importance 분석 실행 여부

## 참고

- 상세 파이프라인·인코딩 전략·모델별 특성 선택은 `202601_playground/README.md` 참고
- 워크스페이스 공통 구조는 루트 `STRUCTURE.md` 참고
