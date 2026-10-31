# Requirements Traceability Matrix — MediFed

| Requirement ID | Description | Test ID(s) | Phase |
|---|---|---|---|
| FR-01 | Load images with junk/corrupt/mask filtering | TC-DATA-001, TC-DATA-002, TC-DATA-003, TC-DATA-004, TC-DATA-005 | Phase 1 — Done |
| FR-02 | One fixed global class mapping for all hospitals | TC-CFG-001, TC-CFG-002 | Phase 0 — Done |
| FR-03 | Use every valid image and create a reproducible split | TC-DATA-008, TC-DATA-010, TC-DATA-013 | Phase 1 — implemented; uncapped pipeline verified |
| FR-04 | Train 4-class ResNet18 locally per hospital | TC-MODEL-001, TC-MODEL-002, TC-TRAIN-001, TC-TRAIN-002 | Phase 2 — Done |
| FR-05 | Federated FedAvg baseline | TC-FED-001–012; saved run evidence in `results/baseline_seed42_final_metrics.csv` | Phase 3 — seed 42 run evaluated; more seeds not present |
| FR-06 | Personalized FedBN mode | TC-FBN-001–008; saved run evidence in `results/personalized_seed42_final_metrics.csv` | Phase 4 — seed 42 run evaluated; more seeds not present |
| FR-07 | Saved baseline vs personalized comparison | Saved comparison in `results/comparison_methods.csv`; `TC-SUM-001–003` | Phase 4 — implemented for saved runs |
| FR-08 | Grad-CAM heatmap for top prediction | TC-CAM-001–006; TC-INF-001 | Phase 5 — implemented; visualized by `scripts/demo_gradcam.py` |
| FR-09 | Diagnosis Page (upload, predictions, heatmap) | TC-APP-005–010; TC-INF-001–005 | Phase 6 — implemented; browser workflow remains to be demonstrated |
| FR-10 | Dashboard (per-hospital training results) | TC-APP-001–005; TC-SUM-001–003; TC-PLOT-001–002 | Phase 6 — implemented; browser workflow remains to be demonstrated |
| NFR-01 | Raw images never leave a hospital client | TC-FED-003 | Phase 3 — Planned |
| NFR-02 | Runs on 4 GB GPU with CPU fallback | TC-MODEL-003 | Phase 2 — Done |
| NFR-03 | Reproducible runs (fixed seed) | TC-CFG-007, TC-DATA-007, TC-DATA-008, TC-MODEL-004 | Phase 0/1/2 — Done |
| NFR-04 | Every requirement covered by a documented test | This matrix | All phases |
| NFR-05 | Clear, uncluttered UI | TC-APP-005; `docs/02_design/UI_DESIGN_GUIDE.md` | Phase 6 — implementation complete; visual review pending |
| — | HOSPITALS has exactly 3 entries | TC-CFG-003 | Phase 0 — Done |
| — | Hospital classes are subsets of CLASS_NAMES | TC-CFG-004 | Phase 0 — Done |
| — | Hospital 3 has COVID19, NORMAL, PNEUMONIA | TC-CFG-005 | Phase 0 — Done |
| — | Hospital 2 has NORMAL, TUBERCULOSIS | TC-CFG-006 | Phase 0 — Done |
| — | PROJECT_ROOT/medifed/config.py exists | TC-CFG-008 | Phase 0 — Done |
| — | Class from nearest ancestor folder | TC-DATA-006 | Phase 1 — Done |
| — | Global indices used for all hospitals | TC-DATA-009 | Phase 1 — Done |
| — | Train/val split stratified, no overlap | TC-DATA-010 | Phase 1 — Done |
| — | Manifest I/O round-trip lossless | TC-DATA-011 | Phase 1 — Done |
| — | Dataset item shape and type correct | TC-DATA-012 | Phase 1 — Done |
| — | per_class_recall None for absent classes | TC-TRAIN-003 | Phase 2 — Done |
| — | Checkpoint save/load round-trip | TC-TRAIN-004 | Phase 2 — Done |
