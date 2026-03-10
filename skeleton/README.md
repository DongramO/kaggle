# skeleton

데이터 로드 → 전처리 → Optuna 튜닝 → K-Fold + 앙상블 → 제출까지 한 디렉토리 안에서 처리합니다.

## 구성

| 파일 | 역할 |
|------|------|
| `data_loader.py` | `train.csv`, `test.csv`, `sample_submission.csv` 로드 (`load_all`) |
| `prepare_data.py` | 결측·클리핑·인코딩·간단 FE, `get_feature_columns` 제공 |
| `eda.py` / `run_eda.py` | 필요 시 EDA 전용 실행(메인 파이프라인과 분리) |
| `optuna.py` | LightGBM / CatBoost / XGBoost Optuna 튜닝 → `best_hyperparameters.json` 생성·갱신 |
| `modeling.py` | 모델 생성(`train_model`), 평가(`evaluate`), K-Fold 앙상블(`run_ensemble_models`) |
| `main.py` | 최종 학습·예측·앙상블·`submission_ensemble.csv` 생성 진입점 |
| `best_hyperparameters.json` | 모델별 최적 하이퍼파라미터 + GPU/로그 옵션 저장 |

## 동작 흐름

1. **Optuna 튜닝 (선행 작업)**  
   - `python optuna.py`  
   - LightGBM / CatBoost / XGBoost를 각각 `N_FOLDS` K-Fold AUC 기준으로 최적화  
   - 결과를 `best_hyperparameters.json`에 `{모델이름: 파라미터 딕셔너리}` 형식으로 저장

2. **메인 파이프라인 (`python main.py`)**  
   - `data_loader.load_all`로 train/test/submission 로드  
   - `prepare_data.prepare_data`로 전처리, `get_feature_columns`로 feature 목록 결정  
   - `best_hyperparameters.json`을 읽어 모델별 하이퍼파라미터 로드  
   - `modeling.run_ensemble_models`에서  
     - StratifiedKFold(기본 5-fold)로 각 모델 OOF AUC 계산  
     - `weights=None` 이므로 **OOF AUC 비율로 자동 가중치** 생성  
     - 모델별 테스트 예측을 가중 평균해 앙상블 확률 산출  
   - 최종 예측을 `submission_ensemble.csv`에 저장 (컬럼 이름: `target_col`, 예: `Churn`)

## 실행 요약

- **하이퍼파라미터 탐색**: `python optuna.py`  
- **학습 + 앙상블 + 제출 생성**: `python main.py`  
- (옵션) **EDA 전용 실행**: `python run_eda.py` (EDA 코드 작성 후 사용)
