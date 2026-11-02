"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const SERIES = [
  { key: "local_accuracy",  label: "Local",  color: "#9ca3af" },
  { key: "fedavg_accuracy", label: "FedAvg", color: "#3b5998" },
  { key: "fedbn_accuracy",  label: "FedBN",  color: "#0d6b6e" },
];

function svgEl(tag, attrs = {}, text = "") {
  const e = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, String(v)));
  if (text) e.textContent = text;
  return e;
}

// Interpolate red→yellow→green for a 0-1 accuracy value
function heatColor(v) {
  if (v == null) return "#f3f4f6";
  // below 0.5 = red→yellow, above 0.5 = yellow→green
  if (v < 0.5) {
    const t = v * 2;
    const r = 220, g = Math.round(t * 180), b = 50;
    return `rgba(${r},${g},${b},0.18)`;
  }
  const t = (v - 0.5) * 2;
  const r = Math.round((1 - t) * 180), g = 160, b = 50;
  return `rgba(${r},${g},${b},0.18)`;
}

function drawComparison(svg, hospitals) {
  svg.replaceChildren();
  const W = Math.max(svg.clientWidth || 700, 320);
  const H = svg.clientHeight || 320;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const m = { top: 24, right: 24, bottom: 56, left: 52 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;

  // Zoom Y axis to meaningful range
  const allVals = hospitals.flatMap((h) => SERIES.map((s) => h[s.key]).filter((v) => v != null));
  const yMin = allVals.length ? Math.max(0, Math.floor((Math.min(...allVals) - 0.05) * 10) / 10) : 0;
  const yMax = 1.0;
  const yRange = yMax - yMin;

  const toY = (v) => m.top + ph * (1 - (v - yMin) / yRange);

  // Grid lines
  const ticks = [];
  for (let t = yMin; t <= yMax + 0.001; t = Math.round((t + 0.1) * 10) / 10) ticks.push(t);
  ticks.forEach((t) => {
    const y = toY(t);
    svg.append(svgEl("line", { x1: m.left, x2: W - m.right, y1: y, y2: y, stroke: "#e5e7eb", "stroke-width": 1 }));
    svg.append(svgEl("text", { x: m.left - 8, y: y + 4, "text-anchor": "end", class: "svg-label" }, `${(t * 100).toFixed(0)}%`));
  });

  const gw = pw / Math.max(hospitals.length, 1);
  const bw = Math.min(32, gw / (SERIES.length + 1));
  const gap = 3;

  hospitals.forEach((h, gi) => {
    const cx = m.left + gw * (gi + 0.5);
    const totalW = SERIES.length * bw + (SERIES.length - 1) * gap;
    SERIES.forEach((s, si) => {
      const v = h[s.key];
      if (v == null) return;
      const x = cx - totalW / 2 + si * (bw + gap);
      const barH = ph * (v - yMin) / yRange;
      const y = toY(v);
      svg.append(svgEl("rect", { x, y, width: bw, height: barH, fill: s.color, rx: 3 }));
      svg.append(svgEl("text", { x: x + bw / 2, y: Math.max(y - 4, 12), "text-anchor": "middle", class: "svg-value" }, `${(v * 100).toFixed(1)}%`));
    });
    svg.append(svgEl("text", { x: cx, y: H - m.bottom + 16, "text-anchor": "middle", class: "svg-label" }, `H${h.id}`));
  });

  // Legend
  const lx = m.left + pw / 2 - (SERIES.length * 72) / 2;
  SERIES.forEach((s, i) => {
    const x = lx + i * 72;
    svg.append(svgEl("rect", { x, y: H - 16, width: 10, height: 10, fill: s.color, rx: 2 }));
    svg.append(svgEl("text", { x: x + 14, y: H - 7, class: "svg-label" }, s.label));
  });
}

function drawRounds(svg, rows, hospitalId) {
  svg.replaceChildren();
  const W = Math.max(svg.clientWidth || 700, 320);
  const H = svg.clientHeight || 320;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const selected = rows.filter((r) => r.hospital === hospitalId);
  if (!selected.length) {
    svg.append(svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "svg-label" }, "No data"));
    return;
  }
  const m = { top: 20, right: 72, bottom: 40, left: 52 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;
  const maxR = Math.max(...selected.map((r) => r.round));

  // Zoom Y
  const allAcc = selected.map((r) => r.accuracy);
  const yMin = Math.max(0, Math.floor((Math.min(...allAcc) - 0.05) * 10) / 10);
  const yMax = 1.0;
  const yRange = yMax - yMin;
  const toY = (v) => m.top + ph * (1 - (v - yMin) / yRange);

  const ticks = [];
  for (let t = yMin; t <= yMax + 0.001; t = Math.round((t + 0.1) * 10) / 10) ticks.push(t);
  ticks.forEach((t) => {
    const y = toY(t);
    svg.append(svgEl("line", { x1: m.left, x2: W - m.right, y1: y, y2: y, stroke: "#e5e7eb" }));
    svg.append(svgEl("text", { x: m.left - 8, y: y + 4, "text-anchor": "end", class: "svg-label" }, `${(t * 100).toFixed(0)}%`));
  });

  const methods = [
    { key: "baseline",     label: "FedAvg", color: "#3b5998" },
    { key: "personalized", label: "FedBN",  color: "#0d6b6e" },
  ];
  methods.forEach((method) => {
    const pts = selected.filter((r) => r.mode === method.key).sort((a, b) => a.round - b.round);
    if (!pts.length) return;
    const coords = pts.map((r) => ({
      x: m.left + (maxR <= 1 ? 0 : ((r.round - 1) / (maxR - 1)) * pw),
      y: toY(r.accuracy),
    }));
    svg.append(svgEl("polyline", { points: coords.map((p) => `${p.x},${p.y}`).join(" "), fill: "none", stroke: method.color, "stroke-width": 2.5, "stroke-linejoin": "round" }));
    coords.forEach((p) => svg.append(svgEl("circle", { cx: p.x, cy: p.y, r: 3.5, fill: method.color })));
    const last = coords[coords.length - 1];
    svg.append(svgEl("text", { x: last.x + 8, y: last.y + 4, fill: method.color, class: "svg-label", "font-weight": "600" }, method.label));
  });

  for (let r = 1; r <= maxR; r++) {
    if (r === 1 || r === maxR || r % 2 === 0) {
      const x = m.left + (maxR <= 1 ? 0 : ((r - 1) / (maxR - 1)) * pw);
      svg.append(svgEl("text", { x, y: H - m.bottom + 14, "text-anchor": "middle", class: "svg-label" }, String(r)));
    }
  }
  svg.append(svgEl("text", { x: m.left + pw / 2, y: H - 4, "text-anchor": "middle", class: "svg-label" }, "Round"));
}

function renderCrossHospital(container, rows, hospitalIds) {
  container.replaceChildren();
  if (!rows || !rows.length) {
    container.append(Object.assign(document.createElement("p"), { className: "empty-state", textContent: "No data available." }));
    return;
  }

  // Derive hospital IDs from data if not passed
  const ids = hospitalIds || [...new Set(rows.map((r) => r.eval_hospital))].sort((a, b) => a - b);

  const table = document.createElement("table");
  table.className = "data-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  const th0 = document.createElement("th");
  th0.textContent = "Trained at \\ Tested on";
  headerRow.append(th0);
  ids.forEach((id) => {
    const th = document.createElement("th");
    th.textContent = `H${id}`;
    headerRow.append(th);
  });
  thead.append(headerRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  ids.forEach((mh) => {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.textContent = `Hospital ${mh}`;
    tr.append(rowHeader);
    ids.forEach((eh) => {
      const td = document.createElement("td");
      const match = rows.find((r) => r.model_hospital === mh && r.eval_hospital === eh);
      const v = match ? match.accuracy : null;
      td.textContent = v != null ? `${(v * 100).toFixed(1)}%` : "—";
      td.style.background = heatColor(v);
      if (mh === eh) td.classList.add("diag");
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  container.append(table);
}

window.MediFedCharts = { drawComparison, drawRounds, renderCrossHospital };
