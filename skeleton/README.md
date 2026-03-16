# Kaggle Skeleton

Kaggle Playground 대회에서 반복 사용하는 파이프라인 템플릿입니다.
새 대회 시작 시 이 디렉토리를 복사한 뒤, **문제별 수정 영역**만 채워 사용합니다.

---

## 파일 구조

```
skeleton/
├── data_loader.py           # train / test / sample_submission 로드
├── prepare_data.py          # 전처리 + 인코딩 + FE 뼈대 (★ 문제별 수정)
├── eda.py                   # 상관분석, 결측 리포트 등 EDA 유틸
├── run_eda.py               # EDA 독립 실행 진입점
├── modeling.py              # 모델 학습 / 스태킹 앙상블 / 진단
├── history.py               # AUC 로그·FI 누적 저장 및 시각화
├── main.py                  # 전체 파이프라인 진입점 (★ target_col 등 수정)
├── run_optuna.py            # Optuna 하이퍼파라미터 탐색 (★ 문제별 수정)
├── best_hyperparameters.json  # Optuna 결과 저장 (run_optuna.py가 생성)
├── auc_history.json         # OOF AUC 누적 로그 (자동 생성)
├── fi_history.json          # Feature Importance 누적 로그 (자동 생성)
└── data/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```

---

## 실행 순서

### 1. Optuna 하이퍼파라미터 탐색

```bash
python run_optuna.py
```

- `prepare_data.py`와 동일한 전처리 파이프라인 적용
- CatBoost / LightGBM / XGBoost 각각 최적화
- 결과를 `best_hyperparameters.json`에 저장 (기존 값 보존, Optuna 결과만 갱신)

### 2. 메인 파이프라인

```bash
python main.py
```

전체 흐름:

```
데이터 로드
  → 전처리 (prepare_data)
  → 상관 피처 필터링
  → 모델별 FE 적용 (FE_FLAGS_PER_MODEL)
  → 멀티시드 스태킹 앙상블 (Level 0: 트리 + 선택적 뉴럴넷 → Level 1: Ridge)
  → OOF AUC 기록 (auc_history.json)
  → Feature Importance 저장 (fi_history.json)
  → 스태킹 진단 차트 (stacking_diagnosis.png)
  → submission_ensemble.csv 저장
  → 역대 최고 AUC 경신 시 자동 git commit
```

### 3. 단일 모델 실행 (선택)

```bash
python main.py catboost    # catboost / lightgbm / xgboost
```

### 4. EDA (선택)

```bash
python run_eda.py
```

---

## 문제별 수정 영역

새 대회마다 수정이 필요한 위치는 `# ===== 문제별 수정 영역 =====` 주석으로 표시되어 있습니다.

| 파일 | 수정 내용 |
|---|---|
| `prepare_data.py` | `ONEHOT_COLS`, `DROP_COLS`, `ORDINAL_COLS`, `DIRECT_MAPPING_COLS`, `NGRAM_TE_COLS`, `FE_FLAGS_PER_MODEL`, `apply_model_fe()` 내 FE 로직 |
| `run_optuna.py` | `TARGET_COL`, `ID_COL`, `N_TRIALS`, `MODEL_TYPES`, objective 함수 파라미터 범위 |
| `main.py` | `TARGET_COL`, `ID_COL`, `run_stacking_ensemble` 호출 인자 (`seeds`, `use_mlp` 등) |
| `run_eda.py` | `TARGET_COL` |

---

## 모델링 구조

### 스태킹 앙상블 (`run_stacking_ensemble`)

- **Level 0**: CatBoost / LightGBM / XGBoost (+ 선택: MLP, TabNet, FT-Transformer)
- **Level 1**: RidgeCV 메타모델 (또는 단순 평균)
- 멀티시드 지원: 여러 `random_state`로 OOF를 평균내어 분산 감소
- 모델별 독립적인 피처셋 지원 (`X_per_model`)

### 진단 (`diagnose_stacking`)

실행 후 `stacking_diagnosis.png` 자동 저장:
- 모델별 OOF AUC 비교
- OOF 예측 상관관계 히트맵 (다양성 확인)
- 메타 모델 계수 (역기여 모델 탐지)

### Feature Importance 추적 (`history.py`)

- 매 실행마다 `fi_history.json`에 누적
- `plot_fi_history()`로 시간에 따른 중요도 변화 시각화 (`fi_history.png`)

---

## best_hyperparameters.json 구조

```json
{
  "catboost": {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "task_type": "GPU",
    "bootstrap_type": "Bernoulli",
    "verbose": 0,
    ...
  },
  "lightgbm": { ... },
  "xgboost": { ... }
}
```

> GPU/verbose/bootstrap_type 등 Optuna 탐색 대상이 아닌 옵션은 JSON에 직접 추가 후 유지됩니다.
