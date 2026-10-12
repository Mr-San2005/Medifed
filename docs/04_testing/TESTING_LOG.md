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
| 2026-10-11 | 2 | TC-MODEL-001 | build_model output shape is (batch, 4) for 224×224 input | shape == (2, 4) | (2, 4) | PASS | |
| 2026-10-11 | 2 | TC-MODEL-002 | Final layer has 4 outputs and in_features=512 | fc.out=4, fc.in=512 | Correct | PASS | |
| 2026-10-11 | 2 | TC-MODEL-003 | get_device returns cuda (CUDA available) | device.type == "cuda" | "cuda" | PASS | |
| 2026-10-11 | 2 | TC-MODEL-004 | set_seed makes two models have identical weights | all params equal | Equal | PASS | |
| 2026-10-11 | 2 | TC-TRAIN-001 | 30 training steps lower the loss on 8 synthetic images | last_loss < first_loss | Confirmed | PASS | |
| 2026-10-11 | 2 | TC-TRAIN-002 | evaluate: accuracy in [0,1], num_samples=8, confusion sum=8 | All conditions | Correct | PASS | |
| 2026-10-11 | 2 | TC-TRAIN-003 | per_class_recall is None for classes with no true samples | recall[2]=None, recall[3]=None | None | PASS | |
| 2026-10-11 | 2 | TC-TRAIN-004 | Checkpoint save/load gives identical outputs | allclose(out1, out2) | True | PASS | |
| 2026-10-12 | 3a | TC-FED-001 | resolve_project_root returns the environment path when it is set | Path(env_value) | Path('E:/custom/root') | PASS | |
| 2026-10-12 | 3a | TC-FED-002 | resolve_project_root returns the fallback when the value is None or empty | fallback | fallback | PASS | |
| 2026-10-12 | 3a | TC-FED-003 | partition_to_hospital maps 0,1,2 to 1,2,3 and raises ValueError for 3 and -1 | 1,2,3 and ValueError | As expected | PASS | |
| 2026-10-12 | 3a | TC-FED-004 | local_update keeps state_dict keys/shapes; train_loss is finite | keys/shapes equal, finite loss | Equal, finite | PASS | |
| 2026-10-12 | 3a | TC-FED-005 | append_round_row writes the header exactly once and appends rows | 1 header + 2 data rows | Correct | PASS | |
| 2026-10-12 | 3a | TC-FED-006 | client_app and server_app import and expose app | hasattr app | Both expose app | PASS | |
| 2026-10-12 | 3a | TC-FED-007 | state_dict_to_cpu returns tensors that all live on cpu | device.type == cpu | cpu | PASS | |
