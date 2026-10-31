"use strict";

const fileInput = document.getElementById("image-file");
const analyzeButton = document.getElementById("analyze-button");
const form = document.getElementById("diagnosis-form");
let selectedFile = null;

function setError(message) {
  document.getElementById("diagnosis-error").textContent = message;
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
  document.getElementById("original-image").src = URL.createObjectURL(file);
}

async function loadVariants() {
  const select = document.getElementById("variant-select");
  try {
    const response = await fetch("/api/variants");
    const data = await response.json();
    select.replaceChildren();
    data.variants.forEach((variant) => {
      const option = document.createElement("option");
      option.value = variant;
      option.textContent = variant.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
      select.append(option);
    });
    if (!data.variants.length) {
      const option = document.createElement("option");
      option.textContent = "No saved model checkpoints found";
      option.value = "";
      select.append(option);
      analyzeButton.disabled = true;
    }
  } catch (_error) {
    setError("Model options could not be loaded. Check that saved checkpoints exist.");
  }
}

function renderRanked(ranked) {
  const list = document.getElementById("ranked-results");
  list.replaceChildren();
  ranked.forEach((item) => {
    const row = document.createElement("li");
    const className = document.createElement("span");
    className.className = "ranked-class";
    className.textContent = item.class;
    const score = document.createElement("span");
    score.className = "ranked-score";
    score.textContent = `${(item.probability * 100).toFixed(1)}%`;
    const track = document.createElement("span");
    track.className = "rank-track";
    const fill = document.createElement("span");
    fill.style.width = `${Math.max(0, Math.min(item.probability * 100, 100))}%`;
    track.append(fill);
    row.append(className, score, track);
    list.append(row);
  });
}

fileInput.addEventListener("change", () => previewFile(fileInput.files[0]));
fileInput.addEventListener("dragover", (event) => event.preventDefault());
fileInput.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  previewFile(file);
});

document.getElementById("opacity-range").addEventListener("input", (event) => {
  const opacity = Number(event.target.value) / 100;
  document.getElementById("heatmap-image").style.opacity = String(opacity);
  document.getElementById("opacity-value").value = `${event.target.value}%`;
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile) return;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing…";
  setError("");
  try {
    const variant = document.getElementById("variant-select").value;
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
    document.getElementById("inference-time").textContent = `${result.milliseconds.toFixed(0)} ms`;
    document.getElementById("original-image").src = `data:image/png;base64,${result.original_png}`;
    document.getElementById("heatmap-image").src = `data:image/png;base64,${result.heatmap_png}`;
    document.getElementById("diagnosis-output").hidden = false;
  } catch (error) {
    setError(error.message);
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.innerHTML = '<span aria-hidden="true">↗</span> Analyze image';
  }
});

loadVariants();