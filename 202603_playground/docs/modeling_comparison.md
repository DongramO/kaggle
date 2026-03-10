# 모델링 비교: 제공 XGB 학습 코드 vs 202603_playground

제공된 XGBoost 학습 스니펫과 현재 프로젝트(202603_playground)의 모델링 방식 차이를 정리한 문서입니다.

---

## 1. 제공 코드 요약 (XGB 단일 모델)

- **검증 방식**: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- **학습 API**: `xgb.train()` + `xgb.DMatrix`, `enable_categorical=True`
- **Early stopping**: 50 round
- **평가 지표**: `eval_metric='auc'`
- **테스트 예측**: Fold별 예측을 N_SPLITS로 나눈 평균 (K-Fold 평균)
- **출력**: OOF AUC, Fold별 Val AUC, Mean/Std

---

## 2. 202603_playground 모델링 요약

- **검증 방식**: StratifiedKFold(n_splits=N_FOLDS=5, shuffle=True, random_state=RANDOM_STATE=42) — 동일
- **학습 API**: 공통 래퍼 `XGBoostModel` → 내부에서 `xgb.DMatrix` + `xgb.train()`, **enable_categorical 미사용**(인코딩된 수치만 입력)
- **Early stopping**: config `EARLY_STOPPING_ROUNDS=100` (제공 코드는 50)
- **평가 지표**: config `SCORING_METRIC='auc'`로 평가·제출 통일. XGB **내부** `eval_metric`은 기본 `logloss`(common/modeling/model.py `_task_params`). AUC는 OOF/CV 계산 시에만 사용.
- **다중 모델**: CatBoost, LightGBM, XGBoost 동시 학습 후 앙상블(단순/가중 평균 또는 Ridge 메타)
- **테스트 예측**: 모델별 K-Fold 평균 → 앙상블 가중치 또는 Ridge 메타로 최종 예측

---

## 3. 항목별 차이

| 항목 | 제공 XGB 코드 | 202603_playground |
|------|----------------|-------------------|
| **K-Fold / Split** | StratifiedKFold(5), 동일 | StratifiedKFold(N_FOLDS=5), 동일 |
| **학습 루프** | 직접 for fold, skf.split → dtrain/dval/dtest 생성 | ModelTrainer.train_with_cv()가 동일 split, create_model → _fit_fold 호출 |
| **데이터 구조** | `xgb.DMatrix(..., enable_categorical=True)` | `xgb.DMatrix(X_train, label=y_train)` — enable_categorical 없음. 범주형은 미리 OneHot/Ordinal 인코딩된 DataFrame만 입력 |
| **XGB 파라미터** | n_estimators=2000, lr=0.05, max_depth=5, min_child_weight=20, subsample/colsample=0.8, reg_alpha=0.1, reg_lambda=1.0, device='cuda', max_bin=512, tree_method='hist' | best_hyperparameters.json 또는 기본값 사용. 예: n_estimators=400, max_depth=4, lr=0.04, min_child_weight=12, reg_alpha/reg_lambda=8, gamma=1.0. GPU는 setup_gpu_params()로 tree_method='hist', device='cuda:0' 등 주입 |
| **eval_metric (XGB 내부)** | `'auc'` | `_task_params`에서 분류 시 `'logloss'`. OOF/CV 점수는 config의 SCORING_METRIC('auc')로 별도 계산 |
| **early_stopping_rounds** | 50 | config EARLY_STOPPING_ROUNDS=100 |
| **num_boost_round** | params_xgb['n_estimators'] (=2000) | model_params의 n_estimators(또는 num_boost_round). 저장된 JSON 기준 400 등 |
| **OOF / 테스트 저장** | oof_preds_xgb, test_preds_xgb 수동 누적 | ModelTrainer.oof_predictions[model_type], predict_test()로 K-Fold 평균 반환 |
| **단일 vs 앙상블** | XGB만 학습·예측 | CatBoost + LightGBM + XGB 학습 후 앙상블(ridge_meta 등). 단일 XGB만 쓸 수는 없음(설정으로 모델 목록만 바꿈) |
| **하이퍼파라미터 출처** | 코드에 하드코딩 | config + best_hyperparameters.json(Optuna 결과). USE_OPTUNA/USE_SAVED_PARAMS로 제어 |
| **메모리/GC** | Fold 끝날 때마다 model, dtrain, dval, dtest 삭제 후 gc.collect() | 명시적 삭제/GC 없음. Fold별로 새 모델 생성 후 리스트에 보관 |
| **verbose** | verbose_eval=400 | verbose_eval=False |

---

## 4. 요약

- **구조**: 제공 코드는 **XGB 단일 모델 + 수동 K-Fold 루프**. 202603은 **CatBoost/LightGBM/XGB 공통 인터페이스 + ModelTrainer**로 K-Fold·OOF·테스트 예측을 한 곳에서 처리하고, 그 위에 **앙상블**을 올림.
- **데이터**: 제공 코드는 `enable_categorical=True`로 XGB 내부 범주형 처리 가능. 202603은 범주형을 전부 전처리 단계에서 인코딩하고, XGB에는 수치형만 넣음.
- **파라미터**: 제공 코드는 2000 round, early 50, auc, GPU 등 한 세트로 고정. 202603은 config + JSON으로 바꿀 수 있고, eval_metric은 내부는 logloss·평가는 auc로 분리.
- **앙상블**: 제공 코드에는 없음. 202603은 다중 모델 + Ridge 메타(또는 가중/단순 평균)로 최종 예측.

원하면 제공 코드 스타일(예: eval_metric='auc', early_stopping_rounds=50, max_bin=512)을 config/기본값에 반영하는 방법도 정리해 줄 수 있다.
