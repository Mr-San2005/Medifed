# Final Report Outline — MediFed

## Chapter Structure

### Abstract
One page. Problem, approach, key result (FedAvg vs FedBN comparison), conclusion.
- Source: `docs/04_testing/TESTING_LOG.md` (final results), `results/` (comparison figures).

### Chapter 1 — Introduction
Background on federated learning in healthcare, motivation, objectives, scope, report organisation.
- Source: `PROJECT_BRIEF.md` sections 1–2, `docs/01_requirements/SRS.md` sections 1–3.

### Chapter 2 — Literature Survey
Federated learning (FedAvg, FedBN), transfer learning with ResNet, Grad-CAM, related medical imaging systems.
- Source: team research notes (not in this repository).

### Chapter 3 — Requirements
Functional and non-functional requirements, constraints, assumptions, out-of-scope items.
- Source: `docs/01_requirements/SRS.md`.

### Chapter 4 — Design
System architecture, component descriptions, data flow, federated round sequence, baseline vs personalized mode, deployment view, UI design decisions.
- Source: `docs/02_design/HLD.md`, `docs/02_design/LLD.md`, `docs/02_design/UI_DESIGN_GUIDE.md`.

### Chapter 5 — Implementation
Phase-by-phase account of what was built, key code decisions, data pipeline details, model construction, Flower integration, web app.
- Source: `docs/03_implementation/phase_reports/PHASE_0.md` through `PHASE_7.md`.

### Chapter 6 — Testing
Test plan, test levels, test cases, execution results, defects found and fixed.
- Source: `docs/04_testing/TEST_PLAN.md`, `docs/04_testing/TESTING_LOG.md`, `docs/04_testing/TRACEABILITY.md`, `docs/04_testing/DEFECT_LOG.md`, `docs/04_testing/test_runs/`.

### Chapter 7 — Results and Discussion
FedAvg vs FedBN accuracy per hospital (table and chart), Grad-CAM examples, discussion of personalisation benefit, limitations.
- Source: `results/fedavg_results.json`, `results/fedbn_results.json`, `results/comparison.png`, `results/comparison.csv`.

### Chapter 8 — Conclusion and Future Work
Summary of achievements, limitations, directions for future work (real hospital deployment, more diseases, differential privacy).
- Source: synthesised from all phase reports and results.

### References
IEEE format. Federated learning papers (McMahan et al. FedAvg, Li et al. FedBN), ResNet, Grad-CAM, Flower framework, dataset citations.
- Source: team reference list (maintained separately).
