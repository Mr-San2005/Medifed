"use strict";

const fileInput     = document.getElementById("image-file");
const analyzeButton = document.getElementById("analyze-button");
const form          = document.getElementById("diagnosis-form");
let selectedFile    = null;

function setError(msg) {
  document.getElementById("diagnosis-error").textContent = msg;
}

// Convert raw variant key to a readable label
function variantLabel(key) {
  if (key === "fedavg_global")       return "FedAvg — Global";
  const fedbn = key.match(/^fedbn_hospital(\d+)$/);
  if (fedbn)  return `FedBN — Hospital ${fedbn[1]}`;
  const local = key.match(/^local_hospital(\d+)$/);
  if (local)  return `Local — Hospital ${local[1]}`;
  return key;
}

function previewFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    selectedFile = null;
    analyzeButton.disabled = true;
    setError("Choose a valid image file.");
    return;
  }
  selectedFile = file;
  analyzeButton.disabled = false;
  setError("");
}

async function loadVariants() {
  const select = document.getElementById("variant-select");
  try {
    const data = await fetch("/api/variants").then((r) => r.json());
    select.replaceChildren();
    if (!data.variants.length) {
      select.append(Object.assign(document.createElement("option"), { value: "", textContent: "No checkpoints found" }));
      analyzeButton.disabled = true;
      return;
    }
    data.variants.forEach((key) => {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = variantLabel(key);
      select.append(opt);
    });
  } catch {
    setError("Could not load model variants.");
  }
}

function renderRanked(ranked) {
  const list = document.getElementById("ranked-results");
  list.replaceChildren();
  ranked.forEach((item) => {
    const li = document.createElement("li");
    const name  = Object.assign(document.createElement("span"), { className: "ranked-class",  textContent: item.class });
    const score = Object.assign(document.createElement("span"), { className: "ranked-score", textContent: `${(item.probability * 100).toFixed(1)}%` });
    const track = document.createElement("span");
    track.className = "rank-bar";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(item.probability * 100, 100))}%`;
    track.append(fill);
    li.append(name, score, track);
    list.append(li);
  });
}

fileInput.addEventListener("change", () => previewFile(fileInput.files[0]));

document.getElementById("opacity-range").addEventListener("input", (e) => {
  document.getElementById("heatmap-image").style.opacity = String(Number(e.target.value) / 100);
  document.getElementById("opacity-value").value = `${e.target.value}%`;
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing…";
  setError("");
  try {
    const variant  = document.getElementById("variant-select").value;
    const response = await fetch(`/api/predict?variant=${encodeURIComponent(variant)}`, {
      method: "POST",
      headers: { "Content-Type": selectedFile.type || "application/octet-stream" },
      body: selectedFile,
      cache: "no-store",
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");

    renderRanked(result.ranked);
    document.getElementById("prediction-title").textContent = result.top_class;
    document.getElementById("inference-time").textContent   = `${result.milliseconds.toFixed(0)} ms`;
    document.getElementById("original-image").src = `data:image/png;base64,${result.original_png}`;
    document.getElementById("heatmap-image").src  = `data:image/png;base64,${result.heatmap_png}`;
    document.getElementById("diagnosis-output").hidden = false;
  } catch (err) {
    setError(err.message);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze";
  }
});

loadVariants();
