# 종합실습 : 은행 고객 이탈 예측 모델 개발 및 최적화

은행 고객 데이터(`bank_churn_train.csv`)로 이탈 여부(`Exited`)를 예측하는 이진 분류 모델을 개발하고, **Validation / Test 평균 AUC**를 최대화합니다.
데이터 누수를 막는 것을 최우선 원칙으로 두고 전처리·변수선택·튜닝·모델선택 전 과정을 설계했습니다.

## 결과 요약

| 구분 | Validation | Test |
|---|---|---|
| Baseline (DecisionTree, max_depth=3) | 0.7820 | 0.7625 |
| **최종 모델 (CatBoost, tuned)** | **0.8802** | **0.8652** |

- 최종 평균 AUC : **0.8727** (Baseline 대비 약 +0.10)
- 최종 모델 Test 지표 : Accuracy 0.8690 / F1 0.4554 / Recall 0.3382 / Precision 0.6970

## 파일 구성

| 파일 | 설명 |
|---|---|
| `종합실습_결과물_모델개발및최적화_권예리.ipynb` | 전체 분석 노트북 |
| `종합실습_결과물_모델개발및최적화_권예리.xlsx` | 결과 정리 시트 |
| `bank_churn_train.csv` | 학습 데이터 (2,100행, cp949 인코딩) |

## 실행 방법

1. Colab에서 노트북을 열고 `bank_churn_train.csv`를 `/content/`에 업로드
2. 전체 셀 실행 (첫 셀에서 `xgboost`, `catboost`, `optuna` 설치)
3. 튜닝을 건너뛰려면 9단계의 `RUN_TUNING = False` → 저장된 최적 파라미터(`BEST_SAVED`)로 바로 재현

## 누수 방지 원칙

1. **결측 대체는 Pipeline 안에서** : `SimpleImputer(median)`을 모델과 묶어 CV의 각 fold train 부분에서만 fit
2. **Test는 마지막에 1회만 평가** : 변수선택·모델비교·튜닝은 CV와 Validation만 사용
3. **단계별로 다른 CV 분할** 사용해 선택 편향 완화

   | 단계 | CV | seed |
   |---|---|---|
   | 모델 후보 비교 / 변수선택 | RepeatedStratifiedKFold 5×3 | 0 |
   | XGBoost 튜닝 | RepeatedStratifiedKFold 5×3 | 42 |
   | CatBoost 튜닝 | StratifiedKFold 5 | 42 |
   | 최종 모델 선택 | RepeatedStratifiedKFold 5×3 | 2026 (미사용 분할) |

## 분석 절차

1. **데이터 분할** : Train / Valid / Test = 6 : 2 : 2 (`random_state=42`)
2. **데이터 점검** : `CustomerId` 전부 고유(1행=1고객), `Surname`은 식별자, `baseDate`는 상수. 결측은 `Geography` 약 2%, `EstimatedSalary` 약 4%. Train 이탈률 21.75%
3. **Feature Engineering**
   - 파생변수 후보 : `TenureYears`, `BalanceZero`, `BalSalRatio`, `ProdPerTenure`, `AgeActive`, `Age_Prod`
   - 식별자·상수·원본 날짜 제거, `Geography` 결측은 `'Unknown'` 범주로 처리
   - `Geography`, `Gender` One-Hot Encoding (Valid/Test는 Train 컬럼 구조로 reindex)
4. **모델 후보 비교** (CV AUC)

   | 모델 | CV | Valid |
   |---|---|---|
   | LogisticRegression | 0.7397 | 0.7700 |
   | RandomForest | 0.8319 | 0.8502 |
   | XGBoost | 0.8408 | 0.8723 |
   | CatBoost | 0.8416 | 0.8659 |

5. **변수 선택 (Ablation, XGB·CAT 평균 CV AUC)**

   | 조합 | 평균 CV |
   |---|---|
   | A. 원본 + TenureYears | 0.8412 |
   | B. A + 파생변수 5종 | 0.8400 |
   | C. A − TenureYears | 0.8486 |
   | **D. C + BalanceZero (선택, 14개 변수)** | **0.8487** |

6. **하이퍼파라미터 튜닝** : Optuna TPE (XGBoost 50 trials, CatBoost 25 trials)
   - XGBoost CV 0.8553 / CatBoost CV 0.8577
7. **최종 모델 선택** (새 CV 분할 기준)

   | 후보 | CV | Valid |
   |---|---|---|
   | XGBoost (tuned) | 0.8549 | 0.8774 |
   | **CatBoost (tuned)** | **0.8595** | 0.8802 |
   | XGB + CAT Soft Voting | 0.8586 | 0.8808 |

8. **최종 평가** : Test 1회 평가 → 평균 AUC 0.8727

## 변수 중요도 (사후 확인용, 선택에 미사용)

CatBoost 기준 상위 : `NumOfProducts`(0.337) > `Age`(0.282) > `IsActiveMember`(0.160) > `Balance`(0.066)
→ 보유 상품 수, 나이, 활동 여부가 이탈을 가장 잘 설명하며 도메인 상식과 일치합니다.

## 한계 및 개선 방향

- 기본 임계값(0.5)에서 Recall이 0.34로 낮음 → 이탈 고객 탐지가 목적이라면 임계값 조정 필요
- 데이터가 2,100행으로 작아 Valid/Test 간 AUC 차이(약 0.015)가 존재

## 환경

Python 3.13 (Colab) · scikit-learn · xgboost 3.4 · catboost 1.2 · optuna 5.0 · pandas · numpy
