# Local Dashboard and Diagnosis Application

## Architecture

The local FastAPI service in `app/server.py` serves static pages and JSON endpoints. `app/services.py` reads committed result CSVs and calls `medifed/inference.py`. The browser uses plain HTML, CSS, and JavaScript; no CDN or client framework is required.

The Dashboard reads hospital counts, final local/FedAvg/FedBN accuracy, round curves, per-class recall, and the saved local cross-hospital evaluation. Missing CSVs are represented by empty-state flags. No result value is embedded in the page markup.

The Diagnosis Page offers only variants backed by checkpoint files. It sends image bytes directly to the local service, which decodes and transforms the image in memory, runs the selected checkpoint, and returns ranked probabilities plus PNG previews. It does not write or log uploaded image bytes.

## Run locally

From the project folder in PowerShell:

```powershell
.\scripts\run_app.ps1
```

Open `http://127.0.0.1:8000`. Use `-Port 8001` if port 8000 is already occupied. The service binds only to loopback.

## Endpoints

| Method and path | Purpose |
|---|---|
| `GET /` | Results Dashboard |
| `GET /diagnose` | Diagnosis Page |
| `GET /api/health` | Local service status |
| `GET /api/results` | Results CSV aggregation |
| `GET /api/variants` | Checkpoint-backed model variants |
| `POST /api/predict?variant=fedbn_hospital3` | Raw image bytes (`Content-Type: image/*`) to ranked scores and image previews |

Prediction requests are limited to 10 MiB. Invalid images return HTTP 400, unknown variants return 404, missing known-variant checkpoints return 503, and oversized requests return 413. Prediction responses use `Cache-Control: no-store`; responses include `X-Content-Type-Options: nosniff`.

## Research notice and limitations

Both pages display “Research prototype — not for clinical use.” The app is a local demonstration, not an exposed network service. Results reflect the saved validation data and one saved seed. The diagnosis output and Grad-CAM map are not clinically validated.