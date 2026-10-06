
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

## 2. 코드에 넣을 형태

```python
FOLD_TEST = {
    1: ["S01", "S02"],
    2: ["S05", "S11"],
    3: ["S06", "S08"],
    4: ["S04", "S09", "S12"],
    5: ["S03", "S07", "S13"],
    6: ["S10", "S14", "S15"],
}


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