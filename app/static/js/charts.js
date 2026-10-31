"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const SERIES = [
  { key: "local_accuracy",  label: "Local",  color: "#6b7280" },
  { key: "fedavg_accuracy", label: "FedAvg", color: "#3b5998" },
  { key: "fedbn_accuracy",  label: "FedBN",  color: "#0d6b6e" },
];

function el(tag, attrs = {}, text = "") {
  const e = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, String(v)));
  if (text) e.textContent = text;
  return e;
}

function drawComparison(svg, hospitals) {
  svg.replaceChildren();
  const W = Math.max(svg.clientWidth || 700, 320);
  const H = svg.clientHeight || 320;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const m = { top: 24, right: 24, bottom: 60, left: 52 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;

  // grid lines
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = m.top + ph * (1 - t);
    svg.append(el("line", { x1: m.left, x2: W - m.right, y1: y, y2: y, stroke: "#e5e7eb", "stroke-width": 1 }));
    svg.append(el("text", { x: m.left - 8, y: y + 4, "text-anchor": "end", class: "svg-label" }, `${(t * 100).toFixed(0)}%`));
  });

  const gw = pw / Math.max(hospitals.length, 1);
  const bw = Math.min(28, gw / (SERIES.length + 1));
  const gap = 4;

  hospitals.forEach((h, gi) => {
    const cx = m.left + gw * (gi + 0.5);
    const totalW = SERIES.length * bw + (SERIES.length - 1) * gap;
    SERIES.forEach((s, si) => {
      const v = h[s.key];
      if (v == null) return;
      const x = cx - totalW / 2 + si * (bw + gap);
      const barH = ph * v;
      const y = m.top + ph - barH;
      svg.append(el("rect", { x, y, width: bw, height: barH, fill: s.color, rx: 3 }));
      svg.append(el("text", { x: x + bw / 2, y: Math.max(y - 4, 12), "text-anchor": "middle", class: "svg-value" }, `${(v * 100).toFixed(1)}%`));
    });
    svg.append(el("text", { x: cx, y: H - m.bottom + 16, "text-anchor": "middle", class: "svg-label" }, `H${h.id}`));
  });

  // legend
  const lx = m.left + pw / 2 - (SERIES.length * 72) / 2;
  SERIES.forEach((s, i) => {
    const x = lx + i * 72;
    svg.append(el("rect", { x, y: H - 18, width: 10, height: 10, fill: s.color, rx: 2 }));
    svg.append(el("text", { x: x + 14, y: H - 9, class: "svg-label" }, s.label));
  });
}

function drawRounds(svg, rows, hospitalId) {
  svg.replaceChildren();
  const W = Math.max(svg.clientWidth || 700, 320);
  const H = svg.clientHeight || 320;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const selected = rows.filter((r) => r.hospital === hospitalId);
  if (!selected.length) {
    svg.append(el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "svg-label" }, "No data"));
    return;
  }
  const m = { top: 20, right: 72, bottom: 40, left: 52 };
  const pw = W - m.left - m.right;
  const ph = H - m.top - m.bottom;
  const maxR = Math.max(...selected.map((r) => r.round));

  [0, 0.25, 0.5, 0.75, 1].forEach((t) => {
    const y = m.top + ph * (1 - t);
    svg.append(el("line", { x1: m.left, x2: W - m.right, y1: y, y2: y, stroke: "#e5e7eb" }));
    svg.append(el("text", { x: m.left - 8, y: y + 4, "text-anchor": "end", class: "svg-label" }, `${(t * 100).toFixed(0)}%`));
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
      y: m.top + ph * (1 - r.accuracy),
    }));
    svg.append(el("polyline", { points: coords.map((p) => `${p.x},${p.y}`).join(" "), fill: "none", stroke: method.color, "stroke-width": 2.5, "stroke-linejoin": "round" }));
    coords.forEach((p) => svg.append(el("circle", { cx: p.x, cy: p.y, r: 3.5, fill: method.color })));
    const last = coords[coords.length - 1];
    svg.append(el("text", { x: last.x + 8, y: last.y + 4, fill: method.color, class: "svg-label", "font-weight": "600" }, method.label));
  });

  for (let r = 1; r <= maxR; r++) {
    if (r === 1 || r === maxR || r % 2 === 0) {
      const x = m.left + (maxR <= 1 ? 0 : ((r - 1) / (maxR - 1)) * pw);
      svg.append(el("text", { x, y: H - m.bottom + 14, "text-anchor": "middle", class: "svg-label" }, String(r)));
    }
  }
  svg.append(el("text", { x: m.left + pw / 2, y: H - 4, "text-anchor": "middle", class: "svg-label" }, "Round"));
}

function renderCrossHospital(container, rows) {
  container.replaceChildren();
  if (!rows || !rows.length) {
    const p = document.createElement("p");
    p.className = "empty-state";
    p.textContent = "No cross-hospital data available.";
    container.append(p);
    return;
  }
  const table = document.createElement("table");
  table.className = "data-table";
  table.innerHTML = `<thead><tr>
    <th>Model trained at</th>
    <th>Hospital 1</th>
    <th>Hospital 2</th>
    <th>Hospital 3</th>
  </tr></thead>`;
  const tbody = document.createElement("tbody");
  [1, 2, 3].forEach((mh) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = `Hospital ${mh}`;
    tr.append(th);
    [1, 2, 3].forEach((eh) => {
      const td = document.createElement("td");
      const match = rows.find((r) => r.model_hospital === mh && r.eval_hospital === eh);
      td.textContent = match ? `${(match.accuracy * 100).toFixed(1)}%` : "—";
      if (mh === eh) td.className = "diag";
      tr.append(td);
    });
    tbody.append(tr);
  });
  table.append(tbody);
  container.append(table);
}

window.MediFedCharts = { drawComparison, drawRounds, renderCrossHospital };
