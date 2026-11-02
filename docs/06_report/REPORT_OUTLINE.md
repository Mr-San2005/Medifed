# Final Report Outline — MediFed

**Project Title:** Decentralized Medical Diagnostics via Federated Learning  
**Course:** BIS785 — Major Project, Semester VII, 2025-26 (ODD)  
**Institute:** The National Institute of Engineering, Dept. of Information Science & Engineering  
**Batch:** 19 | **Guide:** Pradeep HS, Assistant Professor  
**Team:** Rahul K (4NI23IS157), Sanjay S (4NI23IS182), Pawan S Menon (4NI23IS136), R Navaneeth Shenoy (4NI23IS152)

---

## Chapter map

### Abstract
One page. Problem (hospitals cannot share patient images), approach (federated learning with local AI agents + central aggregation server), key result (FedBN 94.9% vs FedAvg 93.8%, +1.1 pp mean), conclusion.
- Source: `results/comparison_methods.csv`, `docs/04_testing/TESTING_LOG.md`

### Chapter 1 — Introduction
Background on federated learning in healthcare, motivation (HIPAA/GDPR, data silos), objectives from synopsis, scope, report organisation.
- Source: `docs/01_requirements/SRS.md` sections 1–3

### Chapter 2 — Literature Survey
FedAvg (McMahan et al. 2017), FedBN (Li et al. 2021), ResNet transfer learning (He et al. 2016), Grad-CAM (Selvaraju et al. 2017), Flower framework (Beutel et al. 2022), related medical imaging FL systems.
- Source: team research notes + references listed in README.md

### Chapter 3 — Requirements
Functional requirements (FR-01 to FR-10), non-functional requirements (NFR-01 to NFR-05), constraints, assumptions, out-of-scope items.
- Source: `docs/01_requirements/SRS.md`

### Chapter 4 — System Design
System architecture (local AI agents + central aggregation server), component diagram, federated round sequence (broadcast → local train → aggregate), FedAvg vs FedBN mode difference, deployment view, UI design.
- Source: `docs/02_design/HLD.md`, `docs/02_design/LLD.md`, `docs/02_design/UI_DESIGN_GUIDE.md`

### Chapter 5 — Implementation
Phase-by-phase: data pipeline, ResNet18 model, Flower FedAvg baseline, FedBN personalization, Grad-CAM, web app.
- Source: `docs/03_implementation/phase_reports/PHASE_0.md` through `PHASE_6.md`
- Detail docs: `docs/03_implementation/DATA_PIPELINE.md`, `MODEL_AND_TRAINING.md`, `EXPLAINABILITY.md`, `WEB_APP.md`

### Chapter 6 — Testing
Test plan, 81 test cases, execution results, defects found and fixed, requirements traceability.
- Source: `docs/04_testing/TEST_PLAN.md`, `TESTING_LOG.md`, `TRACEABILITY.md`, `DEFECT_LOG.md`

### Chapter 7 — Results and Discussion
FedAvg vs FedBN accuracy per hospital (table + chart), round-by-round curves, cross-hospital generalization matrix, Grad-CAM examples, discussion of personalization benefit.

| Hospital | Local | FedAvg | FedBN | Gain |
|---|---:|---:|---:|---:|
| H1 — Guangzhou (NORMAL/PNEUMONIA) | 96.7% | 94.8% | 95.4% | +0.7 pp |
| H2 — Qatar/Dhaka (NORMAL/TB) | 99.6% | 99.2% | 100.0% | +0.8 pp |
| H3 — Qatar (COVID19/NORMAL/PNEUMONIA) | 87.5% | 87.5% | 89.2% | +1.7 pp |
| **Mean** | **94.6%** | **93.8%** | **94.9%** | **+1.1 pp** |

- Source: `results/comparison_methods.csv`, `results/summary_rounds.csv`, `results/crosshospital_fedbn.csv`, `results/chart_methods_comparison.png`, `results/gradcam_demo_*.png`

### Chapter 8 — Conclusion and Future Work
Summary of achievements against synopsis objectives, limitations (single seed, simulated hospitals, no DICOM), future work (real hospital deployment, differential privacy, more disease classes).
- Source: synthesised from phase reports and results

### References (IEEE format)
1. B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS, 2017.
2. X. Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization," ICLR, 2021.
3. K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
4. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV, 2017.
5. D. Beutel et al., "Flower: A Friendly Federated Learning Framework," arXiv:2007.14390, 2022.
6. D. Kermany et al., "Chest X-Ray Images (Pneumonia)," Kaggle, doi:10.17632/rscbjbr9sj.3.
7. T. Rahman et al., "Tuberculosis (TB) Chest X-ray Database," Kaggle, 2020.
8. T. Rahman et al., "COVID-19 Radiography Database," Kaggle, 2021.
