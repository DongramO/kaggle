# Feature Engineering 비교: Kaggle 노트북 vs 202603_playground vs 제공 FE 코드

Kaggle 노트북 패턴, 현재 프로젝트(202603_playground), 그리고 **제공된 FE 함수**(`feature_eng`: Count/Target Encoding, 구간화, 반올림·자릿수·소수부, 수치 상호작용 등)를 함께 비교한 문서입니다.

---

## 1. Kaggle 노트북에서 흔한 FE 패턴

(노트북 원문을 직접 가져오지 못해, Churn 예측 노트북들에서 공통적으로 등장하는 패턴 기준으로 정리)

| 구분 | 내용 |
|------|------|
| **결측 처리** | `TotalCharges` 등 결측을 0 또는 중앙값/평균으로 채우기 |
| **범주형 → 수치** | Contract를 Month-to-month=1, One year=12, Two year=24 등으로 매핑 (드물게 명시적 매핑) |
| **Yes/No 변수** | Yes=1, No=0, "No internet service"=0 등 단순 치환 (여러 노트북에서 반복 작성) |
| **파생 변수** | `tenure * MonthlyCharges`, `TotalCharges / tenure` 등 1~3개 수준으로 직접 계산 |
| **인코딩** | LabelEncoder 또는 One-Hot을 한두 셀에서 처리, FE와 순서가 섞여 있는 경우 많음 |
| **구조** | 셀 단위로 ad-hoc 코드, 설정 파일 없음, 인코딩 전/후 단계 구분이 없는 경우가 많음 |

**특징 요약**: 필요한 것만 몇 개 만들어 쓰는 방식. 재사용·설정 분리·단계 구분은 weak.

---

## 2. 202603_playground FE 구조

### 2.1 전체 흐름

- **1단계: 인코딩 전 FE** → **범주형 인코딩(OneHot/Ordinal)** → **2단계: 인코딩 후 FE**
- 설정은 `config.py`의 `FEATURE_ENGINEERING_CONFIG`에서 플래그와 파라미터로 제어.
- 구현은 `common/preprocess/feature_engineering.py` + `main.py`의 `prepare_data()`.

### 2.2 1단계 (인코딩 전)

| 항목 | config 키 | 설명 | 202603 설정 예시 |
|------|-----------|------|------------------|
| 이상치 클리핑 | `clip_outliers` | 수치형 컬럼 quantile 클리핑 | `tenure`, `MonthlyCharges`, `TotalCharges` @ 0.99 |
| 순서형 범주 → 수치 | `convert_ordered_categorical_to_numeric` | Contract, MultipleLines, OnlineSecurity 등 매핑 | Contract 1/12/24, Yes/No/No service → 1/0/0 |
| 인코딩 전 상호작용 | `create_interactions_before_encoding` | 특성 쌍 × 연산(multiply 등) | (인코딩 후 컬럼 사용 시 2단계와 연계) |
| 통계 특성 | `create_statistical_features` | 그룹별 mean/std 등 | flag=False |
| 파생 플래그 | (main.py 하드코딩) | 인터넷 사용 여부 | `has_internet` = (InternetService != 'No') |

### 2.3 2단계 (인코딩 후)

| 항목 | config 키 | 설명 | 202603 설정 예시 |
|------|-----------|------|------------------|
| 수치 변환 | `transform_numeric_features` | log, sqrt, square 등 | flag=False, columns=['tenure'] |
| 인코딩 후 상호작용 | `create_interactions_after_encoding` | 수치×수치, 플래그×서비스 | MonthlyCharges×Contract_numeric, has_internet×OnlineSecurity_numeric 등 7개 |
| 비율 특성 | `create_ratios` | numerator/denominator | tenure/Contract_numeric, TotalCharges/(MonthlyCharges×Contract_numeric) |
| 파생 카운트 | (main.py) | 인터넷 서비스 이용 개수 | `num_internet_services` (0~6) |
| 파생 수치 | (main.py) | Contract 역수, 월약정×결제수단 | Contract_numeric_inv, is_month_to_month × Electronic check |
| 범주형 조합 | `create_categorical_interactions` | 범주+범주 문자열 결합 | flag=False |
| 도메인 특화 | (main.py) | 시니어×요금/가구/파트너 | SeniorCitizen×MonthlyCharges, ×Dependents, ×Partner 등 |

---

## 2.5 제공된 FE 코드 (`feature_eng`) 요약

아래는 사용자가 제공한 `feature_eng(train, test, ...)` 함수에서 수행하는 FE 종류와 설정입니다.  
(전제: `all_cols`에 원본+One-Hot 스타일 컬럼이 이미 포함된 상태로 들어온다고 가정)

| 항목 | 설명 | 설정/예시 |
|------|------|-----------|
| **Count Encoding (CE)** | 전체 컬럼에 대해 `value_counts().map()` → `{col}_CE` | 모든 `all_cols` (범주+수치 포함) |
| **Smooth Target Encoding (TE)** | K-Fold(5) 기반 타겟 인코딩 + smoothing | `smoothing=10`, `global_mean` 사용, `{col}_TE` |
| **Quantile binning** | 수치형만 10분위 구간화, train에서 bin 경계 추정 후 test에 동일 적용 | `num_cols_fe`: tenure, MonthlyCharges, TotalCharges → `{col}_bin` |
| **Rounding** | 지정 자리에서 반올림한 정수형 특성 | MonthlyCharges: `round(-1)`(10단위), TotalCharges: `round(-2)`, `round(-3)` → `_r-1`, `_r-2`, `_r-3` |
| **Digit extraction** | 수치를 10^k 자리에서 잘라 한 자리 숫자로 추출 | tenure: d-1,d0 / MonthlyCharges: d-2~d2 / TotalCharges: d-3~d2 → `{col}_d{k}` |
| **Tenure 파생** | tenure → 년수·구간 | `tenure_years` = floor(tenure/12), `tenure_segment` = [0,12,24,48,72] 4구간 |
| **Decimal (소수부)** | 금액의 소수부만 사용 | MonthlyCharges, TotalCharges → `{col}_decimal` = (col % 1).round(2) |
| **수치 상호작용** | 수치 3개 조합(2개씩)에 대해 add/mul/sub(양방향)/div(양방향) | tenure, MonthlyCharges, TotalCharges → 3C2 × 6연산 = 18개 특성 |
| **제곱** | 수치형 컬럼 제곱 | `{col}_sq` (tenure, MonthlyCharges, TotalCharges) |

**특징**: 타겟 인코딩·빈도(CE)·구간화·반올림·자릿수·소수부·수치 전조합 상호작용·제곱 등 **인코딩/수치 변형 위주**이고, 도메인 특화(계약·인터넷 서비스 개수 등)는 없음. Leakage 방지를 위해 TE는 StratifiedKFold로 OOF 방식 적용.

**전제 차이**: 제공 코드의 `all_cols`/`cat_cols_fe`에는 이미 One-Hot 스타일 이름(`InternetService_Fiber optic`, `Contract_One year` 등)이 들어가 있어, **원본 범주형을 One-Hot(또는 유사 인코딩)한 뒤** 그 결과 컬럼들에 CE/TE를 적용하는 구조다. 202603은 원본 범주형을 OneHot/Ordinal로 인코딩하고, 그 **이후** 단계에서 수치·도메인 FE만 추가한다.

---

## 3. 항목별 3-way 비교

| FE 유형 | Kaggle 노트북 (일반적) | 제공 FE 코드 (`feature_eng`) | 202603_playground |
|--------|------------------------|------------------------------|-------------------|
| **결측 처리** | TotalCharges 등 직접 fillna | 함수 전제에 따름 (별도 기술 없음) | 로더/전처리에서 분리, FE와 독립 |
| **이상치** | 거의 없음 또는 수동 | test 수치를 train bin 경계로 clip (구간화 시) | `clip_outliers` quantile 클리핑, 컬럼별 규칙 |
| **범주형→수치** | Contract 등 1~2개 수동 | 사용 안 함 (원본/One-Hot 컬럼 그대로 CE·TE 입력) | `convert_ordered_categorical_to_numeric` 8개, config 관리 |
| **빈도/Count 인코딩** | 거의 없음 | **전체 컬럼** `_CE` (value_counts 기반) | common에 `create_frequency_features` 있음, 202603 미사용 |
| **타겟 인코딩** | 드물게 사용 | **전체 컬럼** `_TE`, K-Fold+smoothing으로 leakage 방지 | 없음 (OneHot/Ordinal만 사용) |
| **구간화(binning)** | 거의 없음 | **수치 3개** qcut 10구간, train bin으로 test clip+cut | common `create_binning_features` 있음, 202603 flag=False |
| **반올림 특성** | 없음 | **있음** (10/100/1000 단위 등) `_r-1`, `_r-2`, `_r-3` | 없음 |
| **자릿수 추출** | 없음 | **있음** (tenure/Monthly/Total 자릿수별) `_d{k}` | 없음 |
| **소수부** | 없음 | **있음** MonthlyCharges/TotalCharges `_decimal` | 없음 |
| **tenure 파생** | 드물게 | **있음** tenure_years, tenure_segment(4구간) | Contract·비율로 간접 표현, 동일 개념 없음 |
| **수치 상호작용** | 1~3개 수식 | **수치 3개 전조합** add/mul/sub/div 양방향 + 제곱 | 인코딩 전/후로 **선택 쌍**만 multiply 위주, 비율 별도 |
| **비율 특성** | 있으면 1개 | div로 생성 (상호작용 안에 포함) | `create_ratios`로 tenure/Contract, TotalCharges/… 명시적 관리 |
| **수치 변환(log 등)** | 드물게 | 제곱만 (`_sq`) | `transform_numeric_features`(log/sqrt/square 등) 지원, 현재 off |
| **도메인 파생** | 플래그 1~2개 | 없음 | has_internet, num_internet_services, Contract 역수, 월약정×결제수단, 시니어×요금 등 |
| **설정 방식** | 셀에 직접 코드 | 함수 인자+하드코딩 리스트/딕셔너리 | config 기반, 플래그+리스트 |
| **재사용성** | 프로젝트별 복붙 | 컬럼 리스트만 맞추면 다른 데이터에도 사용 가능 | common 모듈 + config로 재사용 |
| **인코딩과의 관계** | 순서 혼재 가능 | One-Hot 등 이미 적용된 컬럼을 전제로 CE/TE 추가 | 인코딩 전 1단계 / 인코딩 / 인코딩 후 2단계 명확 분리 |

---

## 4. common 모듈에서 제공하지만 202603에서 미사용인 FE

`common/preprocess/feature_engineering.py`에는 있으나, 현재 `FEATURE_ENGINEERING_CONFIG` 또는 `main.py`에서 쓰지 않는 기능:

- **빈도 인코딩**: `create_frequency_features` — 제공 FE 코드의 Count Encoding(CE)과 동일 개념. 202603에서는 미사용.
- **구간화**: `create_binning_features` — 제공 FE의 quantile binning과 유사. strategy='quantile' 등으로 동일 효과 가능.
- **다항식**: `create_polynomial_features`
- **범주형×수치 인코딩 상호작용**: `create_categorical_encoded_interaction`
- **통계 특성**: `create_statistical_features` (config에 정의만 되어 있고 flag=False)

**제공 FE에만 있는 것**: 타겟 인코딩(TE), 반올림(`_r{k}`), 자릿수 추출(`_d{k}`), 소수부(`_decimal`), tenure_years/tenure_segment. 이 중 일부는 common에 새 함수로 추가해 config로 켜는 방식으로 통합할 수 있다.

---

## 5. 요약

- **Kaggle 노트북**: 최소한의 FE(결측, Contract/Yes·No 매핑, 소수 상호작용)·ad-hoc 코드·설정 없음이 많다.
- **제공 FE 코드 (`feature_eng`)**: **인코딩·수치 변형 중심** — CE, Smooth TE(K-Fold), 구간화, 반올림, 자릿수, 소수부, tenure 파생, 수치 전조합 상호작용·제곱. 타겟 인코딩으로 정보 활용이 크고, 반올림/자릿수/소수부는 우리 쪽에 없음. 도메인 파생(계약·인터넷 서비스 수 등)은 없음.
- **202603_playground**: **도메인 파생·구조 중심** — 인코딩 전/후 2단계, config 기반, 이상치·순서형 매핑·선택 상호작용·비율·has_internet·num_internet_services·Contract·시니어 등. CE/TE/구간화/반올림/자릿수/소수부는 없고, common에 빈도·구간화는 있으나 미사용.
- **차이**: 제공 FE는 **타겟·빈도·수치 변형(반올림/자릿수/소수부/전조합)** 이 강하고, 202603은 **도메인 특화·설정 분리·재사용성** 이 강함. 서로 보완 가능(예: 202603에 TE 또는 반올림/자릿수 FE 추가).
