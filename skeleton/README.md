# skeleton

기존 common/playground와 **연결하지 않는** 독립적인 4단계 코드 구조.

## 구성

| 파일 | 역할 |
|------|------|
| `data_loader.py` | train / test / sample_submission 로드 |
| `prepare_data.py` | 결측 처리, 기본 전처리, 특성 컬럼 목록 |
| `eda.py` | 요약 통계, 결측 리포트, EDA 실행 |
| `run_eda.py` | EDA 독립 실행 진입점 (`python run_eda.py`) |
| `modeling.py` | 학습 / 예측 / 평가 진입점 |
| `main.py` | 전체 파이프라인 (EDA는 기본 미포함) |

## 사용 흐름

1. **data_loader** → 원시 데이터 로드  
2. **prepare_data** → 전처리 후 feature/target 분리  
3. **eda** → 탐색 분석 (별도 실행 또는 main에서 선택 호출)  
4. **modeling** → 학습 → 예측 → 평가 → 제출 파일 생성  

## 실행 방법

- **전체 파이프라인**: `python main.py` (EDA 미포함)
- **EDA만 실행**: `python run_eda.py`
- **main에서 EDA 포함 실행**: `main(run_eda=True)` 호출  

