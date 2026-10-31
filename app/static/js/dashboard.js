"use strict";

const pct = (v) => v == null ? "—" : `${(v * 100).toFixed(1)}%`;
const sum = (rows, key) => rows.reduce((t, r) => t + r[key], 0);

const HOSPITAL_COLORS = { 1: "#0d6b6e", 2: "#3b5998", 3: "#92400e" };

function renderHospital(hospital) {
  const panel = document.createElement("div");
  panel.className = "hospital-panel";

  const header = document.createElement("div");
  header.className = "hospital-panel-header";
  const dot = document.createElement("span");
  dot.className = "hospital-dot";
  dot.style.background = HOSPITAL_COLORS[hospital.id] || "#6b7280";
  const titleWrap = document.createElement("div");
  const title = document.createElement("div");
  title.className = "hospital-title";
  title.textContent = `Hospital ${hospital.id}`;
  const subtitle = document.createElement("div");
  subtitle.className = "hospital-subtitle";
  subtitle.textContent = hospital.name;
  titleWrap.append(title, subtitle);
  header.append(dot, titleWrap);

  const classes = document.createElement("div");
  classes.className = "hospital-classes";
  hospital.classes.forEach((name) => {
    const tag = document.createElement("span");
    tag.className = "class-label";
    tag.textContent = name;
    classes.append(tag);
  });

  const table = document.createElement("table");
  table.className = "mini-table";
  table.innerHTML = "<thead><tr><th>Method</th><th>Accuracy</th><th>Val images</th></tr></thead>";
  const tbody = document.createElement("tbody");
  const valCount = sum(hospital.counts, "val");
  const methods = [
    ["Local", hospital.local_accuracy],
    ["FedAvg", hospital.fedavg_accuracy],
    ["FedBN", hospital.fedbn_accuracy],
  ];
  const bestAcc = Math.max(...methods.map(([, v]) => v ?? -1));
  methods.forEach(([label, acc]) => {
    const tr = document.createElement("tr");
    if (acc != null && acc === bestAcc) tr.className = "best-row";
    tr.innerHTML = `<td>${label}</td><td>${pct(acc)}</td><td>${valCount}</td>`;
    tbody.append(tr);
  });
  table.append(tbody);
  panel.append(header, classes, table);
  return panel;
}

function renderSummary(data) {
  document.getElementById("hospital-count").textContent = data.hospitals.length;
  const valTotal = data.hospitals.reduce((t, h) => t + sum(h.counts, "val"), 0);
  document.getElementById("validation-count").textContent = valTotal.toLocaleString();
  const maxRound = data.rounds.reduce((m, r) => Math.max(m, r.round), 0);
  document.getElementById("round-count").textContent = maxRound;

  const fedbnVals = data.hospitals.map((h) => h.fedbn_accuracy).filter((v) => v != null);
  const fedavgVals = data.hospitals.map((h) => h.fedavg_accuracy).filter((v) => v != null);
  const fedbnMean = fedbnVals.length ? fedbnVals.reduce((t, v) => t + v, 0) / fedbnVals.length : null;
  const fedavgMean = fedavgVals.length ? fedavgVals.reduce((t, v) => t + v, 0) / fedavgVals.length : null;
  const delta = fedbnMean != null && fedavgMean != null ? fedbnMean - fedavgMean : null;
  const deltaEl = document.getElementById("fedbn-delta");
  if (deltaEl) deltaEl.textContent = delta != null ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp` : "—";
}

function renderDashboard(data) {
  window._dashData = data;
  renderSummary(data);

  const grid = document.getElementById("hospital-grid");
  if (data.hospitals.length) {
    grid.replaceChildren(...data.hospitals.map(renderHospital));
  } else {
    grid.innerHTML = "<p class='empty-state'>No hospital data available.</p>";
  }

  window.MediFedCharts.drawComparison(document.getElementById("comparison-chart"), data.hospitals);

  const tabs = document.getElementById("round-hospital-tabs");
  const hospitalIds = [...new Set(data.rounds.map((r) => r.hospital))].sort((a, b) => a - b);
  let selected = hospitalIds[0];
  const redraw = () => window.MediFedCharts.drawRounds(document.getElementById("rounds-chart"), data.rounds, selected);
  hospitalIds.forEach((hid) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tab-button";
    btn.textContent = `Hospital ${hid}`;
    btn.setAttribute("aria-pressed", String(hid === selected));
    btn.addEventListener("click", () => {
      selected = hid;
      tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      redraw();
    });
    tabs.append(btn);
  });
  redraw();

  window.MediFedCharts.renderCrossHospital(document.getElementById("cross-hospital-table"), data.cross_hospital.local);
  const fedbnEl = document.getElementById("cross-fedbn-table");
  if (fedbnEl) window.MediFedCharts.renderCrossHospital(fedbnEl, data.cross_hospital.fedbn || []);
}

async function loadDashboard() {
  try {
    const res = await fetch("/api/results");
    if (!res.ok) throw new Error("Could not load results.");
    renderDashboard(await res.json());
  } catch (err) {
    document.getElementById("dashboard-error").textContent = err.message;
  }
}

window.addEventListener("DOMContentLoaded", loadDashboard);
window.addEventListener("resize", () => {
  if (!window._dashData) return;
  window.MediFedCharts.drawComparison(document.getElementById("comparison-chart"), window._dashData.hospitals);
});
