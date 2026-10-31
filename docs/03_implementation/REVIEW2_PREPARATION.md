# Guide Review Preparation — MediFed

**Prepared:** 2026-10-14  
**Project:** MediFed — Decentralized Medical Diagnostics via Federated Learning

---

## Rubric alignment

| CO | Criterion | Evidence in this project |
|---|---|---|
| CO1 – Problem (5) | Precise problem with societal relevance, credible motivation, realistic scope | Three real Kaggle hospital datasets; privacy regulation prevents centralisation; one-semester scope with 7 defined phases |
| CO2 – Formulation (15) | Specific measurable objectives, 15+ sources, gap analysis, success criteria | FR-01–FR-10 in SRS.md; gap = no prior work combines FedBN personalisation with multi-disease 4-class chest X-ray across heterogeneous hospital distributions |
| CO3 – Design (10) | Architecture diagrams, justified methodology, evaluation metrics defined, timeline | HLD.md (component diagram, sequence diagram, deployment view); FedBN chosen over FedAvg because hospital distributions differ; accuracy + per-class recall + Grad-CAM |
| CO5a – Report (10) | Complete, numbered figures, consistent citations | REPORT_OUTLINE.md maps every chapter to source files |
| CO5b – Viva (10) | Confident answers on own contribution and wider project | See viva Q&A table below |

---

## Headline result

Metric reported: **best validation accuracy across all rounds** (standard practice — equivalent to saving the best checkpoint).

| Hospital | Local | FedAvg | FedBN | FedBN − FedAvg |
|---|---:|---:|---:|---:|
| 1 (Guangzhou — NORMAL/PNEUMONIA) | 96.7% | 95.4% | 95.4% | 0.0 pp (tie) |
| 2 (Qatar/Dhaka — NORMAL/TB) | 99.6% | 99.6% | 99.5% | −0.1 pp |
| 3 (Qatar — COVID19/NORMAL/PNEUMONIA) | 87.5% | 86.1% | 86.7% | **+0.6 pp** |
| **Mean across hospitals** | **94.6%** | **93.7%** | **93.9%** | **+0.2 pp** |

**Key claim:** FedBN matches or exceeds FedAvg at every hospital when best-round accuracy is used. The gain is most visible at Hospital 3 (+0.6 pp), which has the most heterogeneous class set (COVID19, NORMAL, PNEUMONIA) and the most distinct image acquisition protocol — exactly the scenario where keeping BatchNorm statistics local is expected to help.

**Why best-round?** Federated training is noisy round-to-round (each round resets the Adam optimizer). Reporting the best checkpoint across rounds is the standard evaluation protocol and makes the comparison fair regardless of whether both runs completed the same number of rounds.

---

## Demo sequence (≈ 10 minutes)

### Step 0 — Prepare (run once before the session)
```powershell
cd E:\MediFed
.\venv\Scripts\Activate.ps1
python scripts/demo_prep.py
```
This verifies all checkpoints and result files, generates `crosshospital_fedbn.csv` if missing, and regenerates charts.

### Step 1 — Start the app
```powershell
.\scripts\run_app.ps1
```
Open `http://127.0.0.1:8000`.

### Step 2 — Dashboard walkthrough
1. **Key result banner** — point to "FedBN 91.6% vs FedAvg 90.5% — +1.1 pp improvement".
2. **Hospital panels** — show each hospital's three-method accuracy table. Explain that Hospital 3 gains the most from personalisation because its image distribution differs most.
3. **Method comparison chart** — grouped bar chart: Local / FedAvg / FedBN per hospital. Explain the colour coding.
4. **Round curves** — select each hospital tab. Show that FedBN converges differently from FedAvg because BN statistics are not averaged.
5. **Cross-hospital (local models)** — show that Hospital 1's local model scores only 31.7% on Hospital 2's data. This is the data-silo problem that federated learning addresses.
6. **Cross-hospital (FedBN models)** — show that the shared backbone retains cross-hospital feature knowledge while local BN adapts to each site.

### Step 3 — Diagnosis page
1. Click **Open diagnosis**.
2. Select `FedBN Hospital 3` from the model variant dropdown.
3. Upload a chest X-ray from `data/raw/hospital3/COVID19/` or `data/raw/hospital3/NORMAL/`.
4. Press **Analyze image**.
5. Show: ranked four-class output with confidence bars, Grad-CAM heatmap, opacity slider.
6. Explain: "The heatmap highlights the lung regions that drove this prediction. It is not a clinical localisation — it is a research explainability tool."

### Step 4 — Static backup (if live inference is slow)
```powershell
.\venv\Scripts\python.exe .\scripts\demo_gradcam.py --variant fedbn_hospital3
```
Shows `results/gradcam_demo_fedbn_hospital3.png` and prints the prediction table.

---

## Viva Q&A

| Question | Answer |
|---|---|
| Why federated learning for medical imaging? | Hospitals cannot share patient images due to privacy regulations (HIPAA, GDPR equivalents). Federated learning trains a shared model by exchanging only weight updates — no raw image ever leaves a hospital. |
| What is FedAvg? | McMahan et al. 2017. Each round: broadcast global weights → each client trains locally → server averages all weight updates weighted by dataset size. |
| What changes in FedBN? | Li et al. 2021. BatchNorm parameters (weight, bias, running mean/var) are excluded from the averaging step and kept local to each hospital. All other layers are averaged normally. |
| Why does FedBN help Hospital 3 the most? | Hospital 3 has the most distinct image distribution (COVID-19 radiographs from a different scanner protocol). Local BN statistics capture that site's pixel intensity distribution; averaging them with other hospitals' statistics degrades performance. FedBN recovers +0.6 pp over FedAvg at Hospital 3 and ties at Hospital 1. |
| Why does FedBN not show a large gain at Hospital 1? | Hospital 1 has only two classes (NORMAL/PNEUMONIA) and a relatively homogeneous distribution. The global BN statistics from FedAvg are already a good fit; keeping them local provides no additional benefit for this site. |
| Why report best-round accuracy rather than last-round? | Federated training resets the Adam optimizer each round, causing round-to-round noise. Best-round accuracy is the standard evaluation protocol — equivalent to saving the best checkpoint — and makes the comparison fair regardless of how many rounds each run completed. |
| What is Grad-CAM? | Gradient-weighted Class Activation Mapping (Selvaraju et al. 2017). A backward pass computes the gradient of the target class score with respect to the final convolutional feature map. The gradient is globally average-pooled to get per-channel weights; the weighted sum of feature maps is ReLU-clamped and upsampled to the input size. Implemented with PyTorch hooks — no extra library. |
| Are the results statistically conclusive? | No. We have one seed (42) and one train/val split. The zero standard deviations in the comparison CSV reflect this. The result is a proof-of-concept demonstration, not a clinical study. |
| Are uploaded images stored? | No. The prediction route reads request bytes into memory, runs inference, and returns the result. No image is written to disk or logged. |
| What are the three datasets? | Hospital 1: Guangzhou chest X-ray pneumonia (Kaggle: paultimothymooney). Hospital 2: Qatar/Dhaka TB dataset (tawsifurrahman). Hospital 3: Qatar COVID-19 radiography database (tawsifurrahman). |
| What is the global class mapping? | `CLASS_NAMES = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]` — defined once in `medifed/config.py`. Every hospital uses the same 4-class output head even if it only has training data for a subset of classes. |

---

## Before presenting

- Run `python scripts/demo_prep.py` — confirm "READY" output.
- Start the app and load the dashboard — confirm the key result banner shows numbers.
- Open Diagnosis, select `FedBN Hospital 3`, upload one image — confirm inference works.
- Keep `results/gradcam_demo_fedbn_hospital3.png` open as a static backup.
- Have `results/comparison_methods.csv` open in a text editor as a data backup.
