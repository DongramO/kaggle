# skeleton

기존 common/playground와 **연결하지 않는** 독립적인 4단계 코드 구조.

## 구성

| 파일 | 역할 |
|------|------|
| `data_loader.py` | train / test / sample_submission 로드 |
| `prepare_data.py` | 결측 처리, 기본 전처리, 특성 컬럼 목록 |
| `eda.py` | 요약 통계, 결측 리포트, EDA 실행 |
| `modeling.py` | 학습 / 예측 / 평가 진입점 |

## 사용 흐름

1. **data_loader** → 원시 데이터 로드  
2. **prepare_data** → 전처리 후 feature/target 분리  
3. **eda** → 탐색 분석 (선택)  
4. **modeling** → 학습 → 예측 → 평가 → 제출 파일 생성  

