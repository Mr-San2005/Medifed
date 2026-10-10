# High-Level Design — MediFed

This document describes the system architecture approved by the project guide. It is kept aligned with that approved design and updated as each phase is implemented.

---

## 1. Components

| Component | Responsibility |
|---|---|
| Data Pipeline | Load, filter, cap and transform chest X-ray images per hospital |
| Model | ResNet18 with 4-class output head; shared weights across hospitals |
| Hospital Clients | Local training loop; send weight updates; keep BatchNorm local in FedBN mode |
| Aggregation Server | Collect updates, run FedAvg or FedBN aggregation, broadcast new weights |
| Evaluation | Per-hospital accuracy/loss; cross-hospital aggregate metrics |
| Explainability | Grad-CAM heatmap via PyTorch hooks on the final conv layer |
| Web Application | Dashboard (training results) and Diagnosis Page (upload, predict, heatmap) |

---

## 2. Data Flow

```mermaid
flowchart LR
    subgraph H1["Hospital 1"]
        D1[Raw images] --> P1[Data Pipeline] --> T1[Local Trainer]
    end
    subgraph H2["Hospital 2"]
        D2[Raw images] --> P2[Data Pipeline] --> T2[Local Trainer]
    end
    subgraph H3["Hospital 3"]
        D3[Raw images] --> P3[Data Pipeline] --> T3[Local Trainer]
    end
    T1 -- weights --> S[Aggregation Server]
    T2 -- weights --> S
    T3 -- weights --> S
    S -- averaged weights --> T1
    S -- averaged weights --> T2
    S -- averaged weights --> T3
    S --> E[Evaluation & Results]
    E --> W[Web App]
```

Raw images never leave their hospital subgraph. Only weight tensors cross the boundary to the server.

---

## 3. Federated Training Round (Step by Step)

```mermaid
sequenceDiagram
    participant S as Aggregation Server
    participant H1 as Hospital 1 Client
    participant H2 as Hospital 2 Client
    participant H3 as Hospital 3 Client

    S->>H1: broadcast global weights
    S->>H2: broadcast global weights
    S->>H3: broadcast global weights
    H1->>H1: local SGD for N epochs
    H2->>H2: local SGD for N epochs
    H3->>H3: local SGD for N epochs
    H1->>S: send updated weights + num_examples
    H2->>S: send updated weights + num_examples
    H3->>S: send updated weights + num_examples
    S->>S: weighted average (FedAvg) or partial average (FedBN)
    S->>S: evaluate on held-out set, log metrics
```

---

## 4. Baseline vs Personalized Mode

| Aspect | FedAvg (baseline) | FedBN (personalized) |
|---|---|---|
| Layers averaged | All layers | All layers except BatchNorm |
| BatchNorm state | Shared | Local to each hospital |
| Benefit | Simple, well-understood | Adapts to each hospital's image distribution |
| Saved output | `results/fedavg_results.json` | `results/fedbn_results.json` |

Both modes use the same Flower deployment runtime (flower-superlink + flower-supernode). The mode is selected via a config flag passed through `train_config`.

---

## 5. Deployment View

```mermaid
flowchart TD
    subgraph Local Machine
        SL[flower-superlink\n(aggregation server process)]
        SN1[flower-supernode --node-config partition-id=1]
        SN2[flower-supernode --node-config partition-id=2]
        SN3[flower-supernode --node-config partition-id=3]
        APP[Flask / web app\nlocalhost:5000]
    end
    SL <--> SN1
    SL <--> SN2
    SL <--> SN3
    SL --> CKPT[checkpoints/]
    CKPT --> APP
    APP --> BROWSER[Browser]
```

All processes run on the same Windows 11 machine. The web app reads saved checkpoints; it does not interact with the Flower runtime at inference time.

---

## 6. Module Dependency

```mermaid
flowchart LR
    config --> data
    config --> model
    data --> client_app
    model --> client_app
    model --> gradcam
    client_app --> server_app
    server_app --> evaluation
    evaluation --> results
    gradcam --> app
    results --> app
```
