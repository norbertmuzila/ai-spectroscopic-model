/* =========================================================================
   USB4000 Spectral Console

   Plotting is hand-rolled on inline SVG rather than pulled from a CDN: this
   runs on a bench that may have no internet, and a chart library that fails to
   load would take the whole dashboard with it.
   ========================================================================= */
"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const cssvar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

async function api(path, opts) {
  const res = await fetch(path, Object.assign(
    { headers: { "Content-Type": "application/json" } }, opts || {}));
  let body = null;
  try { body = await res.json(); } catch (e) { /* no body */ }
  if (!res.ok) throw new Error((body && body.detail) || res.statusText);
  return body;
}

const S = {
  connected: false, simulated: false, streaming: false,
  analysis: null, live: null, sessionId: null, minerals: {},
};

/* ------------------------------------------------------------------ toast */
function toast(msg, kind) {
  const el = document.createElement("div");
  el.className = "toast " + (kind || "");
  el.textContent = msg;
  $("toasts").appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .3s"; el.style.opacity = "0";
    setTimeout(() => el.remove(), 320);
  }, kind === "bad" ? 8000 : 4000);
}

/* ----------------------------------------------------------- status chips */
const STATUS = {
  identified:               { cls: "good",     ic: "●", lab: "Identified" },
  ambiguous:                { cls: "warning",  ic: "◐", lab: "Ambiguous" },
  provisional_out_of_range: { cls: "serious",  ic: "△", lab: "Provisional" },
  degenerate:               { cls: "info",     ic: "≡", lab: "Group level only" },
  inconclusive:             { cls: "",         ic: "?",      lab: "Inconclusive" },
  rejected_quality:         { cls: "critical", ic: "✕", lab: "Rejected" },
};
const chip = (st) => {
  const s = STATUS[st] || STATUS.inconclusive;
  return `<span class="chip ${s.cls}"><span>${s.ic}</span>${s.lab}</span>`;
};
const covCls = (c) => (c >= 0.999 ? "full" : c > 0 ? "part" : "none");

/* ================================================================== charts */
const SVGNS = "http://www.w3.org/2000/svg";
const mk = (tag, attrs, txt) => {
  const n = document.createElementNS(SVGNS, tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (txt != null) n.textContent = txt;
  return n;
};

function ticks(lo, hi, n) {
  const raw = (hi - lo) / n;
  if (!isFinite(raw) || raw <= 0) return [lo];
  const mag = Math.pow(10, Math.floor(Math.log10(raw))), nm = raw / mag;
  const step = (nm < 1.5 ? 1 : nm < 3 ? 2 : nm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}

/* Wavelength to displayable colour, for the spectral strip under the axis.
   This is the one place the page shows what the instrument is looking at. */
function wlColour(wl) {
  let r = 0, g = 0, b = 0, f = 1;
  if (wl < 380)       { r = .28; g = 0; b = .42; f = .30 + .35 * (wl - 340) / 40; }
  else if (wl < 440)  { r = -(wl - 440) / 60; g = 0; b = 1; f = .32 + .68 * (wl - 380) / 60; }
  else if (wl < 490)  { r = 0; g = (wl - 440) / 50; b = 1; }
  else if (wl < 510)  { r = 0; g = 1; b = -(wl - 510) / 20; }
  else if (wl < 580)  { r = (wl - 510) / 70; g = 1; b = 0; }
  else if (wl < 645)  { r = 1; g = -(wl - 645) / 65; b = 0; }
  else if (wl <= 700) { r = 1; g = 0; b = 0; }
  else { const t = Math.min(1, (wl - 700) / 260); r = 1 - .72 * t; g = .10 * t; b = .12 * t; f = .62 - .24 * t; }
  f = Math.max(0, Math.min(1, f));
  const c = (v) => Math.round(255 * Math.pow(Math.max(0, Math.min(1, v)) * f, .85));
  return `rgb(${c(r)},${c(g)},${c(b)})`;
}

function chart(svg, o) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const vb = svg.viewBox.baseVal, W = vb.width, H = vb.height, m = o.margin;
  const x0 = m.l, x1 = W - m.r, y0 = m.t, y1 = H - m.b;
  const [xlo, xhi] = o.xDomain, [ylo, yhi] = o.yDomain;
  const px = (v) => x0 + (v - xlo) / (xhi - xlo) * (x1 - x0);
  const py = (v) => y1 - (v - ylo) / (yhi - ylo) * (y1 - y0);
  const ink3 = cssvar("--ink-3"), grid = cssvar("--grid"), line = cssvar("--line");
  const MONO = "IBM Plex Mono, ui-monospace, Consolas, monospace";

  for (const t of ticks(ylo, yhi, 4)) {
    const y = py(t);
    svg.appendChild(mk("line", { x1: x0, x2: x1, y1: y, y2: y, stroke: grid, "stroke-width": 1 }));
    svg.appendChild(mk("text", { x: x0 - 8, y: y + 3.5, "text-anchor": "end", fill: ink3,
      "font-family": MONO, "font-size": 10 }, o.yFmt(t)));
  }
  for (const t of ticks(xlo, xhi, 6)) {
    const x = px(t);
    svg.appendChild(mk("line", { x1: x, x2: x, y1: y0, y2: y1, stroke: grid, "stroke-width": 1 }));
    svg.appendChild(mk("text", { x: x, y: y1 + (o.strip ? 27 : 16), "text-anchor": "middle",
      fill: ink3, "font-family": MONO, "font-size": 10 }, Math.round(t)));
  }
  svg.appendChild(mk("text", { x: x0, y: 12, fill: ink3, "font-family": MONO,
    "font-size": 9.5, "letter-spacing": ".09em" }, o.yLabel));

  if (o.strip) {
    const defs = mk("defs", {}), gid = svg.id + "-spec";
    const lg = mk("linearGradient", { id: gid, x1: "0", x2: "1", y1: "0", y2: "0" });
    for (let i = 0; i <= 40; i++) {
      lg.appendChild(mk("stop", { offset: (i / 40 * 100) + "%",
        "stop-color": wlColour(xlo + (xhi - xlo) * (i / 40)) }));
    }
    defs.appendChild(lg); svg.appendChild(defs);
    svg.appendChild(mk("rect", { x: x0, y: y1 + 5, width: x1 - x0, height: 9, rx: 2,
      fill: `url(#${gid})`, stroke: line, "stroke-width": .5 }));
    svg.appendChild(mk("text", { x: x1, y: y1 + 41, "text-anchor": "end", fill: ink3,
      "font-family": MONO, "font-size": 9.5 }, "wavelength (nm) → near-infrared"));
  }

  if (o.rule != null) {
    svg.appendChild(mk("line", { x1: x0, x2: x1, y1: py(o.rule), y2: py(o.rule),
      stroke: ink3, "stroke-width": 1, "stroke-dasharray": "2 4", opacity: .7 }));
  }

  /* Expected-band labels take their own row above the measured ones. When a
     mineral absorbs exactly where it should the two markers coincide, and a
     single row would stack the labels illegibly on the very case most worth
     reading: the match. */
  (o.markers || []).forEach((k) => {
    if (k.x < xlo || k.x > xhi) return;
    const x = px(k.x);
    svg.appendChild(mk("line", { x1: x, x2: x, y1: k.row === 0 ? y0 + 12 : y0 + 24, y2: y1,
      stroke: k.color, "stroke-width": k.w || 1.5,
      "stroke-dasharray": k.dash || "none", opacity: k.op || 1 }));
    svg.appendChild(mk("text", { x: x, y: k.row === 0 ? y0 + 8 : y0 + 20, "text-anchor": "middle",
      fill: k.color, "font-family": MONO, "font-size": 9.5, "font-weight": 500 }, k.label));
  });

  const path = (xs, ys) => xs.map((v, i) =>
    `${i ? "L" : "M"}${px(v).toFixed(2)},${py(ys[i]).toFixed(2)}`).join("");
  o.series.forEach((s) => {
    if (!s.y || !s.y.length) return;
    svg.appendChild(mk("path", { d: path(o.x, s.y), fill: "none", stroke: s.color,
      "stroke-width": s.w || 2, "stroke-linejoin": "round", "stroke-linecap": "round",
      "stroke-dasharray": s.dash || "none", opacity: s.op == null ? 1 : s.op }));
  });
  svg.appendChild(mk("line", { x1: x0, x2: x1, y1: y1, y2: y1, stroke: line, "stroke-width": 1 }));

  /* hover layer */
  const cross = mk("line", { x1: 0, x2: 0, y1: y0, y2: y1, stroke: ink3, "stroke-width": 1, opacity: 0 });
  svg.appendChild(cross);
  const dots = o.series.filter((s) => s.dot !== false).map((s) => {
    const d = mk("circle", { r: 4, fill: s.color, stroke: cssvar("--panel"), "stroke-width": 2, opacity: 0 });
    svg.appendChild(d); return { s, d };
  });
  const hit = mk("rect", { x: x0, y: y0, width: x1 - x0, height: y1 - y0,
    fill: "transparent", style: "cursor:crosshair" });
  svg.appendChild(hit);

  const tip = o.tip;
  const hide = () => {
    cross.setAttribute("opacity", 0);
    dots.forEach((d) => d.d.setAttribute("opacity", 0));
    tip.classList.remove("on");
  };
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("pointermove", (ev) => {
    const r = svg.getBoundingClientRect();
    const wl = xlo + ((ev.clientX - r.left) / r.width * W - x0) / (x1 - x0) * (xhi - xlo);
    let bi = 0, bd = Infinity;
    o.x.forEach((v, i) => { const d = Math.abs(v - wl); if (d < bd) { bd = d; bi = i; } });
    const xv = px(o.x[bi]);
    cross.setAttribute("x1", xv); cross.setAttribute("x2", xv); cross.setAttribute("opacity", .45);
    dots.forEach((d) => {
      if (!d.s.y || !d.s.y.length) { d.d.setAttribute("opacity", 0); return; }
      d.d.setAttribute("cx", xv); d.d.setAttribute("cy", py(d.s.y[bi])); d.d.setAttribute("opacity", 1);
    });
    tip.innerHTML = `<div class="row"><span class="lab">${o.x[bi].toFixed(0)} nm</span></div>` +
      o.series.filter((s) => s.y && s.y.length && s.dot !== false).map((s) =>
        `<div class="row"><span class="lab">${s.name}</span><span>${o.tipFmt(s.y[bi])}</span></div>`).join("");
    tip.classList.add("on");
    const lx = ev.clientX - r.left;
    tip.style.left = Math.min(Math.max(lx + 14, 8), r.width - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.max(8, (ev.clientY - r.top) - tip.offsetHeight - 12) + "px";
  });
}

function blankChart(svg, msg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const vb = svg.viewBox.baseVal;
  svg.appendChild(mk("text", { x: vb.width / 2, y: vb.height / 2, "text-anchor": "middle",
    fill: cssvar("--ink-3"), "font-family": "IBM Plex Sans, system-ui, sans-serif",
    "font-size": 13 }, msg));
}

/* ================================================================ rendering */
function renderCharts() {
  const a = S.analysis;
  if (S.streaming && S.live && !a) {
    const wl = S.live.wavelength_nm, c = S.live.counts;
    chart($("svg1"), {
      margin: { l: 56, r: 16, t: 18, b: 50 }, strip: true,
      xDomain: [wl[0], wl[wl.length - 1]],
      yDomain: [0, Math.max(Math.max.apply(null, c) * 1.08, 1000)],
      x: wl, yLabel: "RAW COUNTS", yFmt: (v) => (v / 1000).toFixed(0) + "k",
      tipFmt: (v) => v.toFixed(0), tip: $("tip1"),
      series: [{ name: "Counts", y: c, color: cssvar("--live"), w: 1.6 }],
    });
    blankChart($("svg2"), "Run an analysis to see absorption evidence");
    return;
  }
  if (!a) {
    blankChart($("svg1"), "No spectrum yet — connect and run an analysis");
    blankChart($("svg2"), "No spectrum yet");
    return;
  }

  const sp = a.spectrum, wl = sp.wavelength_nm;
  const all = sp.reflectance.concat(sp.continuum, sp.modelled && sp.modelled.length ? sp.modelled : []);
  let ylo = Math.min.apply(null, all), yhi = Math.max.apply(null, all);
  const pad = (yhi - ylo) * 0.1 || 0.01;
  $("leg-fit").style.display = (sp.modelled && sp.modelled.length) ? "" : "none";

  chart($("svg1"), {
    margin: { l: 56, r: 16, t: 18, b: 50 }, strip: true,
    xDomain: [wl[0], wl[wl.length - 1]], yDomain: [Math.max(0, ylo - pad), yhi + pad],
    x: wl, yLabel: "REFLECTANCE", yFmt: (v) => v.toFixed(2),
    tipFmt: (v) => v.toFixed(4), tip: $("tip1"),
    series: [
      { name: "Continuum", y: sp.continuum, color: cssvar("--s-cont"), w: 1.4, dash: "5 4", dot: false },
      { name: "Model fit", y: sp.modelled, color: cssvar("--s2"), w: 1.8, op: .95 },
      { name: "Measured", y: sp.reflectance, color: cssvar("--s1"), w: 2.2 },
    ],
  });

  const meta = S.minerals[a.identification.mineral] || {};
  const inRange = (meta.diagnostic_bands_nm || []).filter(
    (c) => c >= wl[0] && c <= wl[wl.length - 1]);
  const markers = [];
  inRange.forEach((c) => markers.push({ x: c, label: "expect " + c,
    color: cssvar("--accent"), dash: "3 4", w: 1.4, op: .85, row: 0 }));
  a.bands.slice().sort((x, y) => y.depth - x.depth).slice(0, 6).forEach((b) =>
    markers.push({ x: b.centre_nm, label: Math.round(b.centre_nm),
      color: cssvar("--s2"), w: 1.6, row: 1 }));

  chart($("svg2"), {
    margin: { l: 56, r: 16, t: 34, b: 26 },
    xDomain: [wl[0], wl[wl.length - 1]],
    yDomain: [Math.max(0, Math.min.apply(null, sp.continuum_removed) - 0.04), 1.03], rule: 1,
    x: wl, yLabel: "CONTINUUM REMOVED", yFmt: (v) => v.toFixed(2),
    tipFmt: (v) => ((1 - v) * 100).toFixed(2) + "% deep", tip: $("tip2"),
    markers: markers,
    series: [{ name: "Depth", y: sp.continuum_removed, color: cssvar("--s-cr"), w: 2.2 }],
  });
}

function renderResult() {
  const a = S.analysis;
  if (!a) return;
  $("res-empty").classList.add("hidden");
  $("res-body").classList.remove("hidden");

  const id = a.identification, sp = a.spectrum;
  const meta = S.minerals[id.mineral] || {};
  $("res-status").innerHTML = chip(id.status);
  $("res-headline").textContent = id.headline || id.mineral || "No identification";
  const twins = id.indistinguishable_from || [];
  $("res-sub").innerHTML = twins.length
    ? `best fit in group <b>${esc(id.mineral)}</b> · ${twins.length + 1} phases indistinguishable here · deepest band ${((id.band_depth_in_range || 0) * 100).toFixed(1)}%`
    : `<b>${esc(id.formula || "")}</b> · ${esc(id.group || "")} · ${esc((id.environments || []).join(" / "))}${id.sam_deg != null ? " · spectral angle " + id.sam_deg + "°" : ""}`;
  $("res-conf").textContent = ((id.confidence || 0) * 100).toFixed(1) + "%";
  $("res-conf").style.color = id.confidence > .75 ? cssvar("--good")
    : id.confidence > .45 ? cssvar("--warning") : cssvar("--critical");
  $("res-bar").style.width = Math.max(2, (id.confidence || 0) * 100) + "%";

  const ps = a.prediction_set || {};
  const n = (ps.minerals || []).length;
  $("pset-k").textContent = Math.round((ps.coverage || .95) * 100) + "% prediction set · " + n;
  $("pset-v").textContent = (ps.minerals || []).slice(0, 3).join(", ") +
    (n > 3 ? `, +${n - 3} more` : "");

  const co = $("res-callouts"); co.innerHTML = "";
  if (n) {
    const note = n === 1 ? "Unambiguous at this confidence level."
      : n <= 8 ? "The measurement cannot separate these phases at this confidence level."
      : `Consistent with ${n} minerals — the measurement carries little discriminating information. Usually the diagnostic features lie outside this instrument's range, or SNR is too low.`;
    co.insertAdjacentHTML("beforeend",
      `<div class="callout ${n > 1 ? "amber" : ""}">
         <b>${Math.round((ps.coverage || .95) * 100)}% prediction set${n > 1 ? ` (${n} minerals)` : ""}:</b>
         ${(ps.minerals || []).slice(0, 8).map(esc).join(" · ")}${n > 8 ? ` <span style="opacity:.7">+ ${n - 8} more</span>` : ""}
         <div style="font-size:11.5px;opacity:.85;margin-top:5px">${note} ${esc(ps.method || "")}</div>
       </div>`);
  }
  (a.quality.errors || []).forEach((e) =>
    co.insertAdjacentHTML("beforeend", `<div class="callout red">${esc(e)}</div>`));
  (a.quality.warnings || []).forEach((w) =>
    co.insertAdjacentHTML("beforeend", `<div class="callout amber">${esc(w)}</div>`));

  $("res-notes").innerHTML = (a.interpretation || []).map((t) =>
    `<li class="${/cannot|outside|caution|artefact|under-determined|no usable|failure/i.test(t) ? "warn" : ""}">${esc(t)}</li>`
  ).join("") || "<li>&mdash;</li>";

  const maxP = Math.max.apply(null, (a.candidates || [{ probability: 1 }]).map((c) => c.probability));
  $("cands").innerHTML = (a.candidates || []).slice(0, 8).map((c) => `
    <div class="bar">
      <div class="track">
        <div class="fill${c.probability < .05 ? " dim" : ""}" style="width:${Math.max(1.5, c.probability / maxP * 100)}%"></div>
        <div class="lbl"><span class="nm">${esc(c.mineral)}</span>
          <span class="cov ${covCls(c.diagnostic_coverage)}" title="fraction of this mineral's diagnostic bands inside the measured range">${Math.round(c.diagnostic_coverage * 100)}%</span>
        </div>
      </div>
      <div class="val">${c.probability < .001 ? "&lt;0.1%" : (c.probability * 100).toFixed(1) + "%"}</div>
    </div>`).join("") || '<div class="empty">&mdash;</div>';

  const ab = (a.composition || {}).abundances || {}, keys = Object.keys(ab);
  $("comp-note").textContent = keys.length
    ? `${a.composition.mixing_model} mixing · RMSE ${a.composition.rmse}` : "";
  $("comp").innerHTML = keys.length ? keys.map((k) => `
    <div class="bar">
      <div class="track"><div class="fill" style="width:${Math.max(1.5, ab[k] * 100)}%"></div>
        <div class="lbl"><span class="nm">${esc(k)}</span></div></div>
      <div class="val">${(ab[k] * 100).toFixed(1)}%</div>
    </div>`).join("")
    : '<div class="empty">Single phase &mdash; no stable mixture solution</div>';

  const bs = (a.bands || []).slice().sort((x, y) => y.depth - x.depth).slice(0, 8);
  $("bands").innerHTML = bs.length ? bs.map((b) => `
    <tr><td>${b.centre_nm.toFixed(1)} nm</td>
      <td class="num">${(b.depth * 100).toFixed(2)}%</td>
      <td class="num">${b.width_nm.toFixed(0)} nm</td>
      <td class="num">${b.asymmetry >= 0 ? "+" : ""}${b.asymmetry.toFixed(2)}</td></tr>`).join("")
    : '<tr><td colspan="4" class="empty">No band clears the detection floor</td></tr>';

  const au = a.range_audit || {}, dg = au.degeneracy || {};
  $("audit-callout").innerHTML = `
    <div class="callout amber">
      Measured <b>${(au.measured_range_nm || [0, 0])[0].toFixed(0)}–${(au.measured_range_nm || [0, 0])[1].toFixed(0)} nm</b>.
      Of ${au.library_minerals} minerals in the library, <b>${au.fully_diagnosable_here}</b>
      have every diagnostic band inside this range, ${au.partially_diagnosable_here} have some,
      and <b>${au.not_diagnosable_here}</b> have none. Silicon stops responding past about
      1100 nm, so hydrated phases, sulfates, carbonates and most clays — diagnosed
      between 1400 and 2500 nm — cannot be confirmed here however well their curves correlate.
    </div>`;
  $("audit-kv").innerHTML = `
    <dt>Separable groups</dt><dd>${dg.n_groups || "—"} of ${au.library_minerals || "—"}</dd>
    <dt>Uniquely identifiable</dt><dd>${dg.n_singleton_groups || "—"}</dd>
    <dt>Largest degenerate group</dt><dd>${dg.largest_group || "—"} phases</dd>
    <dt>Analysis time</dt><dd>${(a.engine || {}).elapsed_ms || "—"} ms</dd>`;
  const blind = au.candidates_with_evidence_outside_range || [];
  $("audit-blind").innerHTML = blind.length
    ? `<div style="font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);margin-bottom:6px">Candidates with evidence out of reach</div>` +
      blind.slice(0, 5).map((b) => `<div style="font-size:12px;padding:3px 0;border-bottom:1px solid var(--line-soft)">
        ${esc(b.mineral)} <span style="font-family:var(--mono);font-size:10.5px;color:var(--ink-3)">
        — bands at ${b.bands_outside_range_nm.map((v) => v.toFixed(0)).join(", ")} nm</span></div>`).join("")
    : '<div style="color:var(--ink-3);font-size:12.5px">Every ranked candidate has its diagnostic evidence inside the measured range.</div>';

  const q = a.quality || {};
  const p = $("pill-qc");
  p.className = "pill " + (!q.passed ? "bad" : (q.warnings || []).length ? "warn" : "ok");
  p.querySelector("span").textContent = `SNR ${Math.round(q.snr || 0)} · QC ${q.passed ? "pass" : "fail"}`;

  ["btn-pdf", "btn-csv", "btn-json"].forEach((b) => { $(b).disabled = false; });
}

function renderAll() { renderCharts(); if (S.analysis) renderResult(); }

/* ============================================================= diagnostics */
function showDiagnostics(diag, reason) {
  const sev = diag.severity || "error";
  const cls = sev === "ok" ? "green" : sev === "warning" ? "amber" : "red";
  const sum = diag.summary || {};
  $("hw-card").classList.remove("hidden");
  $("hw-body").innerHTML = `
    <div class="callout ${cls}">
      <b>${esc(diag.verdict || reason || "Hardware not available")}</b>
      <ol>${(diag.next_steps || []).map((s) => `<li>${esc(s)}</li>`).join("")}</ol>
    </div>
    <div class="cols" style="margin-top:12px;gap:14px">
      <dl class="kv">
        <dt>Ocean devices on bus</dt><dd>${sum.ocean_devices_on_bus}</dd>
        <dt>Attached (Windows)</dt><dd>${sum.windows_present}</dd>
        <dt>Remembered only</dt><dd>${sum.windows_ghost_entries}</dd>
        <dt>seabreeze lists</dt><dd>${sum.seabreeze_lists}</dd>
        <dt>USB devices total</dt><dd>${sum.usb_devices_total}</dd>
      </dl>
      <dl class="kv">
        <dt>seabreeze</dt><dd>${diag.library && diag.library.seabreeze_installed ? esc(diag.library.version) : "NOT INSTALLED"}</dd>
        <dt>cseabreeze</dt><dd>${diag.library && diag.library.backends.cseabreeze && diag.library.backends.cseabreeze.available ? "available" : "unavailable"}</dd>
        <dt>pyseabreeze</dt><dd>${diag.library && diag.library.backends.pyseabreeze && diag.library.backends.pyseabreeze.available ? "available" : "unavailable"}</dd>
        <dt>libusb</dt><dd>${diag.usb_backend && diag.usb_backend.libusb ? esc(diag.usb_backend.libusb_source || "yes") : "missing"}</dd>
        <dt>platform</dt><dd>${esc(diag.platform || "")}</dd>
      </dl>
    </div>
    ${(diag.windows_pnp && diag.windows_pnp.ghost || []).length ? `
      <div class="callout" style="margin-top:12px">
        Windows is holding ${diag.windows_pnp.ghost.length} <b>remembered</b> USB4000
        record${diag.windows_pnp.ghost.length !== 1 ? "s" : ""} from earlier sessions.
        Device Manager shows these exactly like a live device, which is why an unplugged
        spectrometer so often looks like a driver problem.
      </div>` : ""}`;
  $("hw-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ================================================================== status */
async function refreshStatus() {
  try {
    const st = await api("/api/status");
    S.connected = st.connected; S.streaming = st.streaming;
    const dev = st.device || {};
    S.simulated = !!dev.simulated;

    const p = $("pill-device");
    p.className = "pill " + (st.connected ? (dev.simulated ? "warn" : "ok") : "");
    p.querySelector("span").textContent = st.connected
      ? `${dev.model} · ${dev.serial}` : "disconnected";

    // The wavelength range is a property of the grating in the attached unit,
    // not a constant - show what the instrument actually reports.
    const rng = st.engine.instrument_range_nm || [];
    const sub = $("brand-sub");
    if (sub && rng.length === 2) {
      sub.textContent = dev.wavelength_min_nm != null
        ? `${dev.model || "USB4000"} · detector ${dev.wavelength_min_nm.toFixed(0)}–${dev.wavelength_max_nm.toFixed(0)} nm · analysing ${rng[0].toFixed(0)}–${rng[1].toFixed(0)} nm`
        : `Ocean Optics USB4000 · analysing ${rng[0].toFixed(0)}–${rng[1].toFixed(0)} nm`;
    }

    const v = st.engine.validation || {};
    const m = $("pill-model");
    m.className = "pill " + (st.engine.model_loaded ? "ok" : "warn");
    m.querySelector("span").textContent = st.engine.model_loaded
      ? `model ${v.top1_accuracy ? (v.top1_accuracy * 100).toFixed(0) + "%" : "ready"}` : "matcher only";
    m.title = st.engine.model_loaded
      ? `top-1 ${((v.top1_accuracy || 0) * 100).toFixed(1)}%  top-3 ${((v.top3_accuracy || 0) * 100).toFixed(1)}%\n` +
        `calibration error ${v.expected_calibration_error}\n` +
        `conformal coverage ${((v.conformal_empirical_coverage || 0) * 100).toFixed(1)}% ` +
        `(target ${((v.conformal_target_coverage || 0) * 100).toFixed(0)}%)`
      : "Learned model not trained — run scripts/train.py. Physics matcher is active.";

    $("btn-connect").disabled = st.connected;
    $("btn-disconnect").disabled = !st.connected;
    $("grp-sim").classList.toggle("hidden", !dev.simulated);
    if (st.connected) $("hw-card").classList.add("hidden");

    const refs = dev.references || {};
    ["dark", "white"].forEach((k) => {
      const r = refs[k], el = $("ref-" + k);
      if (!r || !r.present) { el.textContent = "not taken"; el.style.color = ""; return; }
      el.textContent = `${r.integration_time_ms} ms · ${r.age_minutes.toFixed(1)} min ago` +
        (r.integration_time_mismatch ? "  MISMATCH" : "");
      el.style.color = r.integration_time_mismatch ? cssvar("--critical") : "";
    });

    $("lbl-count").textContent = st.stats.total_analyses;
    $("lbl-session").textContent = st.session_id ? "#" + st.session_id : "none";
    S.sessionId = st.session_id;
    $("btn-stream").textContent = st.streaming ? "Stop live" : "Live view";
  } catch (e) {
    const p = $("pill-device");
    p.className = "pill bad";
    p.querySelector("span").textContent = "backend unreachable";
  }
}

async function refreshHistory() {
  try {
    const { analyses } = await api("/api/analyses?limit=40");
    const host = $("history");
    if (!analyses.length) { host.innerHTML = '<div class="empty">No analyses yet</div>'; return; }
    host.innerHTML = analyses.map((a) => `
      <button class="hist" type="button" data-id="${esc(a.analysis_id)}"
        aria-current="${S.analysis && S.analysis.analysis_id === a.analysis_id}">
        <span class="stripe"></span>
        <span style="min-width:0">
          <span class="nm">${esc(a.sample_label || "—")}</span>
          <span class="res">${esc(a.mineral || a.status || "—")}</span>
        </span>
        <span class="pc">${a.confidence != null ? (a.confidence * 100).toFixed(0) + "%" : "—"}</span>
      </button>`).join("");
    host.querySelectorAll(".hist").forEach((el) => {
      el.onclick = async () => {
        S.analysis = await api("/api/analyses/" + el.dataset.id);
        renderAll(); refreshHistory();
      };
    });
  } catch (e) { /* backend down; status pill already reflects it */ }
}

async function loadMinerals() {
  try {
    const { minerals } = await api("/api/minerals");
    const groups = {};
    minerals.forEach((m) => {
      S.minerals[m.name] = m;
      (groups[m.group] = groups[m.group] || []).push(m);
    });
    $("sel-sample").innerHTML = Object.keys(groups).sort().map((g) =>
      `<optgroup label="${esc(g)}">` + groups[g].map((m) =>
        `<option value="${esc(m.name)}">${esc(m.name)}${
          m.diagnostic_coverage >= .999 ? "" : m.diagnostic_coverage > 0 ? " ◐" : " ○"
        }</option>`).join("") + "</optgroup>").join("");
    const i = [...$("sel-sample").options].findIndex((o) => o.value === "Hematite");
    if (i >= 0) $("sel-sample").selectedIndex = i;
  } catch (e) { /* non-fatal */ }
}

/* =============================================================== websocket */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  const p = $("pill-ws");
  ws.onopen = () => { p.className = "pill ok"; p.querySelector("span").textContent = "socket live"; };
  ws.onclose = () => {
    p.className = "pill bad"; p.querySelector("span").textContent = "socket down";
    setTimeout(connectWS, 2500);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "live") {
      S.live = msg.data; p.className = "pill ok live";
      if (S.streaming) renderCharts();
      if (msg.data.saturated) toast("Detector saturated — reduce integration time", "bad");
    } else if (msg.type === "analysis") {
      S.analysis = msg.data; renderAll(); refreshHistory(); refreshStatus();
      const id = msg.data.identification;
      toast(`${msg.data.sample_label}: ${id.headline || id.mineral || "no ID"} (${((id.confidence || 0) * 100).toFixed(0)}%)`,
        id.status === "identified" ? "good" : "");
    } else if (msg.type === "streaming") {
      S.streaming = msg.active; refreshStatus();
    } else if (msg.type === "error") {
      toast(msg.message, "bad");
    }
  };
}

/* ================================================================== wiring */
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

async function doConnect(simulator) {
  const b = $("btn-connect");
  b.disabled = true; b.textContent = "Connecting...";
  try {
    const r = await api("/api/device/connect", {
      method: "POST", body: JSON.stringify({ simulator: !!simulator }),
    });
    if (r.ok === false) {
      showDiagnostics(r.diagnostics || {}, r.reason);
      toast(r.reason || "Spectrometer not found", "bad");
    } else {
      $("hw-card").classList.add("hidden");
      const d = r.device;
      toast(d.simulated ? "Simulator started" : `Connected to ${d.model} (${d.serial})`,
        d.simulated ? "" : "good");
      $("rng-int").value = d.integration_time_ms;
      $("lbl-int").textContent = Math.round(d.integration_time_ms);
    }
  } catch (e) { toast("Connect failed: " + e.message, "bad"); }
  b.textContent = "Connect";
  refreshStatus();
}

$("btn-connect").onclick = () => doConnect($("chk-sim").checked);
$("hw-sim").onclick = () => { $("chk-sim").checked = true; doConnect(true); };
$("hw-recheck").onclick = async () => {
  $("hw-recheck").disabled = true;
  try { showDiagnostics(await api("/api/diagnostics")); }
  catch (e) { toast(e.message, "bad"); }
  $("hw-recheck").disabled = false;
};
$("btn-diag").onclick = $("hw-recheck").onclick;

$("btn-disconnect").onclick = async () => {
  await api("/api/device/disconnect", { method: "POST" });
  S.streaming = false; refreshStatus();
};

const pushSettings = debounce(async () => {
  if (!S.connected) return;
  try {
    await api("/api/device/settings", {
      method: "POST",
      body: JSON.stringify({
        integration_time_ms: +$("rng-int").value,
        scans_to_average: +$("rng-scans").value,
        boxcar_width: +$("rng-box").value,
      }),
    });
    refreshStatus();
  } catch (e) { /* transient */ }
}, 280);

$("rng-int").oninput = () => { $("lbl-int").textContent = $("rng-int").value; pushSettings(); };
$("rng-scans").oninput = () => { $("lbl-scans").textContent = $("rng-scans").value; pushSettings(); };
$("rng-box").oninput = () => { $("lbl-box").textContent = $("rng-box").value; pushSettings(); };
$("rng-grain").oninput = () => { $("lbl-grain").textContent = (+$("rng-grain").value).toFixed(1); };
$("rng-weather").oninput = () => { $("lbl-weather").textContent = $("rng-weather").value; };

["dark", "white"].forEach((kind) => {
  $("btn-" + kind).onclick = async () => {
    const b = $("btn-" + kind); b.disabled = true;
    try {
      const r = await api("/api/device/reference/" + kind, { method: "POST" });
      toast(`${kind} reference taken — peak ${Math.round(r.peak_counts)} counts`, "good");
    } catch (e) { toast(e.message, "bad"); }
    b.disabled = false; refreshStatus();
  };
});

$("btn-load").onclick = async () => {
  try {
    await api("/api/simulator/sample", {
      method: "POST",
      body: JSON.stringify({
        mixture: { [$("sel-sample").value]: 1.0 },
        grain_size: +$("rng-grain").value,
        weathering: +$("rng-weather").value / 100,
      }),
    });
    toast("Simulator loaded: " + $("sel-sample").value);
  } catch (e) { toast(e.message, "bad"); }
};

$("btn-measure").onclick = async () => {
  const b = $("btn-measure");
  b.disabled = true; b.textContent = "Measuring…";
  try {
    S.analysis = await api("/api/measure", {
      method: "POST",
      body: JSON.stringify({
        sample_label: $("inp-label").value || "Unlabelled sample",
        notes: $("inp-notes").value,
      }),
    });
    renderAll(); refreshHistory(); refreshStatus();
  } catch (e) { toast("Measurement failed: " + e.message, "bad"); }
  b.disabled = false; b.textContent = "Run analysis";
};

$("btn-stream").onclick = async () => {
  const action = S.streaming ? "stop" : "start";
  try {
    await api("/api/stream/" + action, { method: "POST" });
    S.streaming = action === "start";
    $("btn-stream").textContent = S.streaming ? "Stop live" : "Live view";
    $("leg-live").classList.toggle("hidden", !S.streaming);
    if (S.streaming) S.analysis = null;
    renderCharts();
  } catch (e) { toast(e.message, "bad"); }
};

$("btn-import").onclick = () => $("file-input").click();
$("file-input").onchange = async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const fd = new FormData(); fd.append("file", file);
  try {
    const res = await fetch("/api/import", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || res.statusText);
    S.analysis = body; renderAll(); refreshHistory();
    toast("Imported " + file.name, "good");
  } catch (e) { toast("Import failed: " + e.message, "bad"); }
  ev.target.value = "";
};

const dl = (fmt) => () => {
  if (!S.analysis) return;
  window.location = `/api/report/${S.analysis.analysis_id}/download?fmt=${fmt}`;
};
$("btn-pdf").onclick = dl("pdf");
$("btn-csv").onclick = dl("csv");
$("btn-json").onclick = dl("json");

$("btn-session").onclick = async () => {
  const name = $("inp-session").value.trim();
  if (!name) { toast("Enter a session name first"); return; }
  const r = await api("/api/session", { method: "POST", body: JSON.stringify({ name }) });
  S.sessionId = r.session_id; toast("Session started: " + name, "good"); refreshStatus();
};
$("btn-session-pdf").onclick = async () => {
  if (!S.sessionId) { toast("No active session"); return; }
  try {
    await api("/api/report/session/" + S.sessionId, { method: "POST" });
    window.location = `/api/report/session/${S.sessionId}/download`;
  } catch (e) { toast(e.message, "bad"); }
};
$("btn-refresh").onclick = refreshHistory;
window.addEventListener("resize", debounce(renderAll, 140));

/* ==================================================================== boot */
loadMinerals();
refreshStatus();
refreshHistory();
connectWS();
renderCharts();
setInterval(refreshStatus, 5000);
