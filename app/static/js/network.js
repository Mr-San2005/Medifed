"use strict";

const PALETTE = ["#0d6b6e", "#3b5998", "#92400e", "#6b3fa0", "#2e7d32", "#c62828"];
const col = (id) => PALETTE[(id - 1) % PALETTE.length];
const pct = (v) => v == null ? "—" : `${(v * 100).toFixed(1)}%`;

function drawTopology(hospitals) {
  const container = document.getElementById("arch-canvas");
  if (!container) return;

  const W  = container.getBoundingClientRect().width || container.offsetWidth || 800;
  const n  = hospitals.length;
  // Height scales with node count if many hospitals
  const H  = Math.max(300, 120 + Math.ceil(n / 4) * 120);
  const cx = W / 2;

  const SVG_NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.width  = "100%";
  svg.style.height = `${H}px`;

  const mk = (tag, attrs, text) => {
    const e = document.createElementNS(SVG_NS, tag);
    Object.entries(attrs || {}).forEach(([k, v]) => e.setAttribute(k, v));
    if (text != null) e.textContent = text;
    return e;
  };

  // Server box
  const sW = 200, sH = 48, sY = 48;
  svg.append(mk("rect", { x: cx - sW / 2, y: sY - sH / 2, width: sW, height: sH, rx: 8, fill: "#0d6b6e" }));
  svg.append(mk("text", { x: cx, y: sY - 6, "text-anchor": "middle", fill: "white", "font-size": 12, "font-weight": 700, "font-family": "Inter,system-ui,sans-serif" }, "Aggregation Server"));
  svg.append(mk("text", { x: cx, y: sY + 11, "text-anchor": "middle", fill: "rgba(255,255,255,0.7)", "font-size": 10, "font-family": "Inter,system-ui,sans-serif" }, "FedAvg · Flower SuperLink"));

  // Nodes — wrap into rows of 4 if many hospitals
  const cols    = Math.min(n, 4);
  const rows    = Math.ceil(n / cols);
  const colW    = Math.min(180, (W - 80) / cols);
  const rowH    = 110;
  const nodeR   = 26;
  const startY  = sY + sH / 2 + 70;

  hospitals.forEach((h, i) => {
    const row = Math.floor(i / cols);
    const colIdx = i % cols;
    const rowCount = row === rows - 1 ? n - row * cols : cols;
    const rowStartX = cx - ((rowCount - 1) * colW) / 2;
    const nx = rowStartX + colIdx * colW;
    const ny = startY + row * rowH;

    // Line from server to node
    svg.append(mk("line", {
      x1: cx, y1: sY + sH / 2,
      x2: nx, y2: ny - nodeR,
      stroke: "#d1d5db", "stroke-width": 1.5, "stroke-dasharray": "5,3",
    }));

    // Node circle
    svg.append(mk("circle", { cx: nx, cy: ny, r: nodeR, fill: col(h.id) }));
    svg.append(mk("text", { x: nx, y: ny + 5, "text-anchor": "middle", fill: "white", "font-size": 13, "font-weight": 700, "font-family": "Inter,system-ui,sans-serif" }, `H${h.id}`));

    // Label below
    svg.append(mk("text", { x: nx, y: ny + nodeR + 16, "text-anchor": "middle", fill: "#374151", "font-size": 10, "font-weight": 600, "font-family": "Inter,system-ui,sans-serif" }, h.classes.join(" / ")));
  });

  // Legend
  const ly = H - 20;
  [["#d1d5db", "Weight exchange (dashed)"], ["#0d6b6e", "Aggregation server"], [PALETTE[1], "Hospital node"]].forEach(([c, label], i) => {
    const lx = cx - 220 + i * 160;
    svg.append(mk("rect", { x: lx, y: ly - 8, width: 10, height: 10, fill: c, rx: 2 }));
    svg.append(mk("text", { x: lx + 14, y: ly + 1, fill: "#9ca3af", "font-size": 10, "font-family": "Inter,system-ui,sans-serif" }, label));
  });

  container.replaceChildren(svg);
}

function renderNodes(hospitals) {
  const grid = document.getElementById("node-grid");
  if (!grid) return;
  grid.replaceChildren(...hospitals.map((h) => {
    const trainTotal = h.counts.reduce((t, c) => t + c.train, 0);
    const valTotal   = h.counts.reduce((t, c) => t + c.val,   0);
    const card = document.createElement("div");
    card.className = "node-card";
    card.innerHTML = `
      <div class="node-card-header">
        <span class="hospital-dot" style="background:${col(h.id)}"></span>
        <div>
          <div class="node-card-title">Hospital ${h.id}</div>
          <div class="node-card-sub">${h.name}</div>
        </div>
      </div>
      <div class="node-stat-row"><span class="node-stat-label">Classes</span><span class="node-stat-value">${h.classes.join(", ")}</span></div>
      <div class="node-stat-row"><span class="node-stat-label">Training images</span><span class="node-stat-value">${trainTotal.toLocaleString()}</span></div>
      <div class="node-stat-row"><span class="node-stat-label">Validation images</span><span class="node-stat-value">${valTotal.toLocaleString()}</span></div>
      <div class="node-stat-row"><span class="node-stat-label">FedBN accuracy</span><span class="node-stat-value" style="color:var(--teal)">${pct(h.fedbn_accuracy)}</span></div>
      <div class="node-stat-row"><span class="node-stat-label">Shared with server</span><span class="node-stat-value" style="color:var(--teal)">Weights only</span></div>`;
    return card;
  }));
}

function renderSummary(data) {
  const trainTotal = data.hospitals.reduce((t, h) => t + h.counts.reduce((s, c) => s + c.train, 0), 0);
  const maxRound   = data.rounds.reduce((m, r) => Math.max(m, r.round), 0);
  document.getElementById("net-hospital-count").textContent = data.hospitals.length;
  document.getElementById("net-train-count").textContent    = trainTotal.toLocaleString();
  document.getElementById("net-round-count").textContent    = maxRound || "—";
}

let _networkData = null;

async function loadNetwork() {
  try {
    const res  = await fetch("/api/results");
    if (!res.ok) throw new Error("Could not load results.");
    _networkData = await res.json();
    renderSummary(_networkData);
    renderNodes(_networkData.hospitals);
    drawTopology(_networkData.hospitals);
  } catch (err) {
    const el = document.getElementById("network-error");
    if (el) el.textContent = err.message;
  }
}

window.addEventListener("DOMContentLoaded", loadNetwork);
window.addEventListener("resize", () => {
  if (_networkData) drawTopology(_networkData.hospitals);
});
