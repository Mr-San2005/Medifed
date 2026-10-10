# Requirements Traceability Matrix — MediFed

| Requirement ID | Description | Test ID(s) | Phase |
|---|---|---|---|
| FR-01 | Load images with junk/corrupt/mask filtering | TC-DATA-001, TC-DATA-002, TC-DATA-003, TC-DATA-004 | Phase 1 — Planned |
| FR-02 | One fixed global class mapping for all hospitals | TC-CFG-001, TC-CFG-002 | Phase 0 — Done |
| FR-03 | Reproducible per-class cap with fixed seed | TC-DATA-005 | Phase 1 — Planned |
| FR-04 | Train 4-class ResNet18 locally per hospital | TC-MODEL-001, TC-TRAIN-001 | Phase 2 — Planned |
| FR-05 | Federated FedAvg baseline | TC-FED-001, TC-FED-002 | Phase 3 — Planned |
| FR-06 | Personalized FedBN mode | TC-PERS-001, TC-PERS-002 | Phase 4 — Planned |
| FR-07 | Saved baseline vs personalized comparison | TC-EVAL-001 | Phase 4 — Planned |
| FR-08 | Grad-CAM heatmap for top prediction | TC-GCAM-001 | Phase 5 — Planned |
| FR-09 | Diagnosis Page (upload, predictions, heatmap) | TC-APP-001 | Phase 6 — Planned |
| FR-10 | Dashboard (per-hospital training results) | TC-APP-002 | Phase 6 — Planned |
| NFR-01 | Raw images never leave a hospital client | TC-FED-003 | Phase 3 — Planned |
| NFR-02 | Runs on 4 GB GPU with CPU fallback | TC-MODEL-002 | Phase 2 — Planned |
| NFR-03 | Reproducible runs (fixed seed) | TC-CFG-007, TC-DATA-005 | TC-CFG-007 Phase 0 — Done; TC-DATA-005 Phase 1 — Planned |
| NFR-04 | Every requirement covered by a documented test | This matrix | All phases |
| NFR-05 | Clear, uncluttered UI | TC-APP-003 | Phase 6 — Planned |
| — | HOSPITALS has exactly 3 entries | TC-CFG-003 | Phase 0 — Done |
| — | Hospital classes are subsets of CLASS_NAMES | TC-CFG-004 | Phase 0 — Done |
| — | Hospital 3 has COVID19, NORMAL, PNEUMONIA | TC-CFG-005 | Phase 0 — Done |
| — | Hospital 2 has NORMAL, TUBERCULOSIS | TC-CFG-006 | Phase 0 — Done |
| — | PROJECT_ROOT/medifed/config.py exists | TC-CFG-008 | Phase 0 — Done |
