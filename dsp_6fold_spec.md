# DSP 재실행용 6-fold 분할 정의

딥러닝 재학습(EfficientPhys, AIRFlowNet)에 사용한 분할입니다. DSP를 이 분할로 다시 돌리면 Table 2에서 두 계열을 같은 조건으로 비교할 수 있습니다.

원 출처: `run_cv_experiment.py:42-49`의 FOLD_SUBJECTS (DSP 쪽에서 먼저 정의한 것을 그대로 사용)
검증: 6개 fold의 test 피험자가 15명을 중복 없이 분할하며, test 클립 합계 374 = 전체와 일치

---

## 1. 분할표

| fold | test 피험자 | test 클립 | valid 피험자 | valid 클립 | train 피험자 | train 클립 |
|---|---|---|---|---|---|---|
| 1 | S01, S02 | 60 | S03, S12 | 25 | 11명 | 289 |
| 2 | S05, S11 | 66 | S13, S15 | 38 | 11명 | 270 |
| 3 | S06, S08 | 70 | S09, S10 | 40 | 11명 | 264 |
| 4 | S04, S09, S12 | 60 | S01, S08 | 40 | 10명 | 274 |
| 5 | S03, S07, S13 | 54 | S04, S11 | 46 | 10명 | 274 |
| 6 | S10, S14, S15 | 64 | S02, S07 | 65 | 10명 | 245 |

test 클립 합계 60+66+70+60+54+64 = 374

---

## 2. valid에 관한 참고

딥러닝은 에포크별 체크포인트 선택을 위해 valid가 필요해서 별도로 분리했습니다. valid 피험자는 다음 조건으로 고정했습니다.

- 해당 fold의 test와 겹치지 않음
- 6개 fold를 통틀어 valid로 두 번 이상 쓰이는 피험자 없음
- 클립 수가 적은 피험자를 우선 배정해 train을 크게 유지 (train 245~289)
- S05(45), S06(50), S14(25)는 valid로 쓰이지 않고 항상 train에 포함

DSP는 학습이 없으므로 valid가 불필요합니다. 하이퍼파라미터(threshold, weight)를 fold마다 고르신다면 **train + valid 13명을 선택용으로 쓰고 test 2~3명은 건드리지 않는** 방식이면 딥러닝과 정보 사용량이 같아집니다.

test만 건드리지 않으면 되므로, valid를 따로 두지 않고 train 13명 전체를 선택용으로 쓰셔도 무방합니다.

---

## 3. 코드에 넣을 형태

```python
FOLD_TEST = {
    1: ["S01", "S02"],
    2: ["S05", "S11"],
    3: ["S06", "S08"],
    4: ["S04", "S09", "S12"],
    5: ["S03", "S07", "S13"],
    6: ["S10", "S14", "S15"],
}

# 딥러닝이 체크포인트 선택에 쓴 valid (DSP에서는 선택적)
FOLD_VALID = {
    1: ["S03", "S12"],
    2: ["S13", "S15"],
    3: ["S09", "S10"],
    4: ["S01", "S08"],
    5: ["S04", "S11"],
    6: ["S02", "S07"],
}

ALL_SUBJECTS = [f"S{i:02d}" for i in range(1, 16)]

def split(fold):
    test = FOLD_TEST[fold]
    valid = FOLD_VALID[fold]
    train = sorted(set(ALL_SUBJECTS) - set(test) - set(valid))
    return train, valid, test
```

---

## 4. 로컬 ID ↔ 공식 ID 대응

`clip_remap.csv`의 `new_subject` 컬럼이 아래 로컬 ID입니다.

| 로컬 | 공식 | 클립 수 |
|---|---|---|
| S01 | AIR_400_S09 | 20 |
| S02 | AIR_400_S10 | 40 |
| S03 | AIR_400_S11 | 10 |
| S04 | AIR_400_S12 | 25 |
| S05 | AIR_400_S13 | 45 |
| S06 | AIR_400_S14 | 50 |
| S07 | AIR_400_S15 | 25 |
| S08 | AIR_400_S16 | 20 |
| S09 | AIR_400_S17 | 20 |
| S10 | AIR_400_S18 | 20 |
| S11 | AIR_125_S01 | 21 |
| S12 | AIR_125_S02 | 15 |
| S13 | AIR_125_S03 | 19 |
| S14 | AIR_125_S04 | 25 |
| S15 | AIR_125_S05 | 19 |

합계 374

---

## 5. 결과 제출 형식 요청

Table 2를 채우려면 아래 두 가지가 필요합니다.

### (1) fold별 요약

| fold | n | MAE | RMSE | PCC | ICC | SR_3BPM | 선택된 threshold | 선택된 W_flow |
|---|---|---|---|---|---|---|---|---|

6행 + 평균 + 표준편차. 표준편차는 Table 2에 평균 ± sd로 병기하기 위해 필요합니다.

### (2) 클립별 예측 CSV

컬럼: `new_index`, `pred_rr`, `gt_rr`

`new_index`는 clip_remap.csv의 값(예: S01_001)입니다. 이게 있으면 자세별 분해, 피험자별 분해, 딥러닝과의 클립 단위 대조를 이쪽에서 전부 처리할 수 있습니다.

딥러닝 쪽은 동일 형식으로 이미 준비되어 있습니다.
- `preds/pred_efficientphys_b2e15_all.csv`
- `preds/pred_virenet_b2e15_all.csv`

---

## 6. 함께 확인 부탁드릴 사항

1. **하이퍼파라미터 선택 범위**: threshold와 weight를 374개 전체에서 고른 것인지, fold별 학습 그룹에서 고른 것인지. 전자라면 test 정보가 선택에 반영되어 낙관적 편향이 생깁니다.

2. **자세별 클립 수**: 초안 3.1절은 Prone 29 / Side 115 / Supine 230, clip_remap.csv는 Prone 29 / Side 120 / Supine 225입니다. 어느 쪽이 맞는지 확인이 필요합니다.

3. **Table 2의 Ours 1.9417**: Table 3의 최적 조합(1.954), Table 4의 fold 평균(2.1155)과 모두 다릅니다. 산출 근거를 알려주시면 본문 서술을 맞추겠습니다.

4. **처리 시간**: 계산 비용 비교표에 넣을 DSP의 클립당 처리 시간. 딥러닝 쪽은 fold당 학습 2.5~5.5시간, 전처리 374클립 126분, 캐시 45.13GB로 확보되어 있습니다.
