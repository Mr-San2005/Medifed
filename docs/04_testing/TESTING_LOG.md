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
| 2026-10-10 | 1 | TC-DATA-001 | `._` junk file not returned, counted in skipped['junk'] | junk >= 1 | junk = 1 | PASS | |
| 2026-10-10 | 1 | TC-DATA-002 | Corrupt image not returned, counted in skipped['unreadable'] | unreadable >= 1, corrupt.jpg absent | Correct | PASS | |
| 2026-10-10 | 1 | TC-DATA-003 | masks/ folder images excluded and counted | excluded >= 1, masked.png absent | Correct | PASS | |
| 2026-10-10 | 1 | TC-DATA-004 | Lung_Opacity/ folder images excluded | lo.png absent | Absent | PASS | |
| 2026-10-10 | 1 | TC-DATA-005 | __MACOSX/ folder images excluded | mac.png absent | Absent | PASS | |
| 2026-10-10 | 1 | TC-DATA-006 | Class from nearest ancestor (COVID19/images/x.png → COVID19) | nested.png in COVID19 | Correct | PASS | |
| 2026-10-10 | 1 | TC-DATA-007 | Per-class cap respected | no class > cap | All ≤ cap | PASS | |
| 2026-10-10 | 1 | TC-DATA-008 | Same seed gives identical sample twice | rows1 == rows2 | Equal | PASS | |
| 2026-10-10 | 1 | TC-DATA-009 | Hospital-2-style: NORMAL=1, TUBERCULOSIS=3 (global indices) | indices match global | Correct | PASS | |
| 2026-10-10 | 1 | TC-DATA-010 | Train/val no overlap, both non-empty | intersection = {} | Empty | PASS | |
| 2026-10-10 | 1 | TC-DATA-011 | Manifest write/read round-trip lossless, forward slashes | loaded == original, no backslashes | Lossless | PASS | |
| 2026-10-10 | 1 | TC-DATA-012 | Dataset item: float tensor 3×224×224, integer label | shape=(3,224,224), dtype=float32 | Correct | PASS | |
