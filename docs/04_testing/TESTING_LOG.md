# Testing Log — MediFed

| Date | Phase | Test ID | Description | Expected | Actual | Result | Notes |
|---|---|---|---|---|---|---|---|
| 2026-10-10 | 0 | TC-CFG-001 | Class order is the fixed global order | `["COVID19","NORMAL","PNEUMONIA","TUBERCULOSIS"]` | Matches | PASS | |
| 2026-10-10 | 0 | TC-CFG-002 | CLASS_TO_INDEX consistent with CLASS_NAMES | Index per position, length 4 | Consistent, length 4 | PASS | |
| 2026-10-10 | 0 | TC-CFG-003 | HOSPITALS has exactly 3 entries | `len == 3` | 3 | PASS | |
| 2026-10-10 | 0 | TC-CFG-004 | Each hospital's classes are a subset of CLASS_NAMES | All subsets | All subsets | PASS | |
| 2026-10-10 | 0 | TC-CFG-005 | Hospital 3 has COVID19, NORMAL, PNEUMONIA | Sorted list matches | Matches | PASS | |
| 2026-10-10 | 0 | TC-CFG-006 | Hospital 2 has NORMAL, TUBERCULOSIS | Sorted list matches | Matches | PASS | |
| 2026-10-10 | 0 | TC-CFG-007 | SEED is an integer | `isinstance(SEED, int)` is True | True | PASS | |
| 2026-10-10 | 0 | TC-CFG-008 | PROJECT_ROOT/medifed/config.py exists | File found on disk | Found | PASS | |
