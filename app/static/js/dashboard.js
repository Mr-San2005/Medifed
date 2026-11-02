"use strict";

const pct = (v) => v == null ? "—" : `${(v * 100).toFixed(1)}%`;
const mean = (arr) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;

const PALETTE = ["#0d6b6e", "#3b5998", "#92400e", "#6b3fa0", "#2e7d32", "#c62828"];
const hospitalColor = (id) => PALETTE[(id - 1) % PALETTE.length];

function renderHospital(hospital) {
  const panel = document.createElement("div");
  panel.className = "hospital-panel";

  const header = document.createElement("div");
  header.className = "hospital-panel-header";
  const dot = document.createElement("span");
  dot.className = "hospital-dot";
  dot.style.background = hospitalColor(hospital.id);
  const titleWrap = document.createElement("div");
  titleWrap.innerHTML = `<div class="hospital-title">Hospital ${hospital.id}</div><div class="hospital-subtitle">${hospital.name}</div>`;
  header.append(dot, titleWrap);

  const classes = document.createElement("div");
  classes.className = "hospital-classes";
  hospital.classes.forEach((name) => {
    const tag = document.createElement("span");
    tag.className = "class-label";
    tag.textContent = name;
    classes.append(tag);
  });

  const methods = [
    ["Local",   hospital.local_accuracy],
    ["FedAvg",  hospital.fedavg_accuracy],
    ["FedBN",   hospital.fedbn_accuracy],
  ];
  const bestAcc = Math.max(...methods.map(([, v]) => v ?? -1));

  const table = document.createElement("table");
  table.className = "mini-table";
  table.innerHTML = "<thead><tr><th>Method</th><th>Accuracy</th></tr></thead>";
  const tbody = document.createElement("tbody");
  methods.forEach(([label, acc]) => {
    const tr = document.createElement("tr");
    if (acc != null && acc === bestAcc) tr.className = "best-row";
    tr.innerHTML = `<td>${label}</td><td>${pct(acc)}</td>`;
    tbody.append(tr);
  });
  table.append(tbody);
  panel.append(header, classes, table);
  return panel;
}

function renderSummary(data) {
  // Inline counts in heading subtitle
  document.getElementById("hospital-count").textContent = data.hospitals.length;
  const maxRound = data.rounds.reduce((m, r) => Math.max(m, r.round), 0);
  document.getElementById("round-count").textContent = maxRound;

  const fedbnVals  = data.hospitals.map((h) => h.fedbn_accuracy).filter((v) => v != null);
  const fedavgVals = data.hospitals.map((h) => h.fedavg_accuracy).filter((v) => v != null);
  const localVals  = data.hospitals.map((h) => h.local_accuracy).filter((v) => v != null);
  const fedbnMean  = mean(fedbnVals);
  const fedavgMean = mean(fedavgVals);
  const localMean  = mean(localVals);
  const delta = fedbnMean != null && fedavgMean != null ? fedbnMean - fedavgMean : null;

  document.getElementById("fedbn-mean").textContent  = pct(fedbnMean);
  document.getElementById("fedavg-mean").textContent = pct(fedavgMean);
  document.getElementById("local-mean").textContent  = pct(localMean);
  document.getElementById("fedbn-delta").textContent = delta != null
    ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`
    : "—";
}

function renderCrossTabs(data) {
  const tabs   = document.getElementById("cross-tabs");
  const wrap   = document.getElementById("cross-table");
  const tables = {
    local: data.cross_hospital.local  || [],
    fedbn: data.cross_hospital.fedbn  || [],
  };
  const hospitalIds = data.hospitals.map((h) => h.id);

  function show(key) {
    tabs.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.tab === key))
    );
    window.MediFedCharts.renderCrossHospital(wrap, tables[key], hospitalIds);
  }

  tabs.querySelectorAll("button").forEach((btn) =>
    btn.addEventListener("click", () => show(btn.dataset.tab))
  );
  show("local");
}

function renderDashboard(data) {
  window._dashData = data;
  renderSummary(data);

  const grid = document.getElementById("hospital-grid");
  grid.replaceChildren(...(data.hospitals.length
    ? data.hospitals.map(renderHospital)
    : [Object.assign(document.createElement("p"), { className: "empty-state", textContent: "No hospital data." })]
  ));

  window.MediFedCharts.drawComparison(document.getElementById("comparison-chart"), data.hospitals);

  // Convergence tabs
  const tabs = document.getElementById("round-hospital-tabs");
  const hospitalIds = [...new Set(data.rounds.map((r) => r.hospital))].sort((a, b) => a - b);
  let selected = hospitalIds[0];
  const redraw = () => window.MediFedCharts.drawRounds(
    document.getElementById("rounds-chart"), data.rounds, selected
  );
  hospitalIds.forEach((hid) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tab-button";
    btn.textContent = `Hospital ${hid}`;
    btn.setAttribute("aria-pressed", String(hid === selected));
    btn.addEventListener("click", () => {
      selected = hid;
      tabs.querySelectorAll("button").forEach((b) =>
        b.setAttribute("aria-pressed", String(b === btn))
      );
      redraw();
    });
    tabs.append(btn);
  });
  redraw();

  renderCrossTabs(data);
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
  window.MediFedCharts.drawComparison(
    document.getElementById("comparison-chart"), window._dashData.hospitals
  );
});
