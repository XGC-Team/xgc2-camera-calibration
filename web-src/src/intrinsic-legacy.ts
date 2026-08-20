// @ts-nocheck -- imperative canvas and ROS transport retained behind the React view.
"use strict";

const el = (id) => document.getElementById(id);
let autoRunning = false;

async function postJSON(path, body) {
  try {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : "{}",
    });
    if (!res.ok) {
      let message = "request failed";
      try { message = (await res.json()).error || message; } catch (err) { /* ignore */ }
      return { ok: false, error: message };
    }
    return await res.json();
  } catch (err) {
    return { ok: false, error: String(err) };
  }
}

function setStatus(msg) { el("status").textContent = msg || ""; }

function renderBars(progress) {
  const host = el("bars");
  // Rebuild only when the set of labels changes; otherwise update widths in place.
  if (host.childElementCount !== progress.length) {
    host.innerHTML = "";
    for (const bar of progress) {
      const wrap = document.createElement("div");
      wrap.className = "bar";
      wrap.innerHTML =
        `<div class="bar-label"><span>${bar.label}</span><span class="pct"></span></div>` +
        `<div class="bar-track"><div class="bar-fill"></div></div>`;
      host.appendChild(wrap);
    }
  }
  const wraps = host.children;
  for (let i = 0; i < progress.length; i++) {
    const pct = Math.round(progress[i].progress * 100);
    const fill = wraps[i].querySelector(".bar-fill");
    fill.style.width = Math.min(100, pct) + "%";
    fill.classList.toggle("full", progress[i].progress >= 1.0);
    wraps[i].querySelector(".pct").textContent = pct + "%";
  }
}

function renderResult(result) {
  const box = el("result");
  if (!result) { box.hidden = true; return; }
  box.textContent = [
    `fx = ${result.fx.toFixed(2)}   fy = ${result.fy.toFixed(2)}`,
    `cx = ${result.cx.toFixed(2)}   cy = ${result.cy.toFixed(2)}`,
    `rms = ${result.rms_reprojection_error_px.toFixed(3)} px   (${result.sample_count} samples)`,
    `saved: ${result.output_file}`,
  ].join("\n");
  box.hidden = false;
}

function renderPose(pose) {
  el("pose").textContent = pose
    ? `pos (${pose.x.toFixed(2)}, ${pose.y.toFixed(2)}, ${pose.z.toFixed(2)})`
    : "";
}

function renderDetection(detection, board) {
  const kind = board && board.type === "aprilgrid" ? "AprilGrid" : "Checkerboard";
  const status = el("detection-status");
  const corners = el("detection-corners");
  const frame = el("detection-frame");
  const metrics = el("detection-metrics");
  const sample = el("detection-sample");
  if (!detection || detection.status === "waiting") {
    status.textContent = "waiting";
    status.className = "legacy-state-source pill pill-off";
    corners.textContent = "No detection result yet";
    frame.textContent = "No captured frame";
    metrics.textContent = "";
    sample.textContent = "Capture a frame to run board detection.";
    return;
  }
  const detected = detection.status === "detected";
  status.textContent = detected ? `${kind} detected` : `${kind} not detected`;
  status.className = `legacy-state-source pill ${detected ? "pill-on" : "pill-off"}`;
  corners.textContent = `${detection.corner_count} / ${detection.expected_corner_count} corners`;
  frame.textContent = detection.frame_width && detection.frame_height
    ? `${detection.frame_width}×${detection.frame_height} · frame ${detection.sequence}`
    : `frame ${detection.sequence}`;
  metrics.textContent = (detection.metrics || [])
    .map((metric) => `${metric.label} ${Number(metric.value).toFixed(3)}`).join(" · ");
  sample.textContent = !detected
    ? `${kind} is not fully visible in this frame.`
    : detection.accepted
      ? "Accepted as a geometrically distinct sample."
      : detection.duplicate
        ? "Detected, but too similar to an existing sample."
        : "Detected and ready for inspection.";
}

function applyState(s) {
  const wasAutoRunning = autoRunning;
  const action = s.action || null;
  autoRunning = !!(action && action.name === "auto_run" && action.status === "running");
  if (action && action.status === "failed") {
    setStatus("Auto-run failed: " + (action.error || "camera control failed"));
  } else if (wasAutoRunning && !autoRunning) {
    setStatus(`Auto-run done — ${s.samples} samples. Press Calibrate.`);
  }
  el("conn").textContent = "connected";
  el("conn").className = "legacy-state-source pill pill-on";
  el("samples").textContent = `${s.samples} samples`;
  renderBars(s.coverage || []);

  el("btn-calibrate").disabled = autoRunning || !(s.goodenough && !s.calibrated);
  el("btn-reset").disabled = autoRunning;
  renderResult(s.result);
  renderPose(s.pose);
  renderDetection(s.detection, s.board);

  scene.control = !!s.camera_control;
  el("btn-reset-pose").disabled = !scene.control || autoRunning;
  el("btn-auto").disabled = !scene.control || autoRunning;
  scene.cam = s.pose || null;
  if (s.targets) scene.targets = s.targets;
  scene.next = (s.next === undefined ? null : s.next);
  drawScene();
  updateRefPanel();
}

async function poll() {
  try {
    const res = await fetch("api/v1/intrinsic/state", { cache: "no-store" });
    applyState(await res.json());
  } catch (err) {
    el("conn").textContent = "disconnected";
    el("conn").className = "legacy-state-source pill pill-off";
  }
}

function refreshImage() {
  el("stream").src = "api/v1/intrinsic/image.jpg?t=" + Date.now();
}

function wireButtons() {
  el("btn-calibrate").addEventListener("click", async () => {
    setStatus("Calibrating…");
    const r = await postJSON("api/v1/intrinsic/calibrate");
    setStatus(r.ok === false ? (r.error || "Not enough coverage yet.") : "Calibrated and saved.");
    poll();
  });
  el("btn-reset").addEventListener("click", async () => {
    await postJSON("api/v1/intrinsic/reset");
    setStatus("Coverage reset.");
    poll();
  });

  el("btn-reset-pose").addEventListener("click", () =>
    postJSON("api/v1/intrinsic/reset_pose").then((r) =>
      setStatus(r.ok === false ? r.error : "Camera reset to launch pose.")));
  el("btn-auto").addEventListener("click", async () => {
    autoRunning = true; el("btn-auto").disabled = true;
    setStatus("Auto-run: sweeping every pose…");
    const r = await postJSON("api/v1/intrinsic/auto_run");
    if (r && r.accepted) {
      autoRunning = true;
      setStatus("Auto-run accepted: sweeping every pose…");
    } else {
      autoRunning = false;
      setStatus(r && r.ok ? `Auto-run done — ${r.samples} samples. Press Calibrate.` : (r.error || "Auto-run failed."));
    }
    poll();
  });
}

/* ---------------- 3D sample guide ---------------- */
const scene = { canvas: null, ctx: null, board: null, targets: [], cam: null, next: null, selected: null,
                az: 2.2, el: 0.5, markers: [], dragging: false, moved: false, lastX: 0, lastY: 0, control: false };

const vsub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const vdot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const vcross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
function vnorm(a) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0]/l, a[1]/l, a[2]/l]; }

function sceneBasis() {
  const f = [Math.cos(scene.el)*Math.cos(scene.az), Math.cos(scene.el)*Math.sin(scene.az), Math.sin(scene.el)];
  let right = vnorm(vcross(f, [0, 0, 1]));
  if (!isFinite(right[0])) right = [1, 0, 0];
  return { f, right, up: vcross(right, f) };
}

function boardCorners() {
  const c = scene.board.center, w = scene.board.width/2, h = scene.board.height/2;
  // vertical plane at x = c[0], spanning y (width) and z (height)
  return [[c[0], c[1]-w, c[2]-h], [c[0], c[1]+w, c[2]-h], [c[0], c[1]+w, c[2]+h], [c[0], c[1]-w, c[2]+h]];
}

function drawScene() {
  const cv = scene.canvas, ctx = scene.ctx;
  if (!cv || !ctx || !scene.board) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (!W || !H) return;
  if (cv.width !== Math.round(W*dpr) || cv.height !== Math.round(H*dpr)) { cv.width = Math.round(W*dpr); cv.height = Math.round(H*dpr); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);

  const { right, up } = sceneBasis();
  const center = scene.board.center;
  const pts = boardCorners().concat((scene.targets || []).map(t => t.position));
  if (scene.cam) pts.push([scene.cam.x, scene.cam.y, scene.cam.z]);
  let maxR = 0.6;
  for (const p of pts) { const d = vsub(p, center); maxR = Math.max(maxR, Math.abs(vdot(d, right)), Math.abs(vdot(d, up))); }
  const s = 0.42 * Math.min(W, H) / maxR, cx = W/2, cy = H/2;
  const project = (p) => { const d = vsub(p, center); return [cx + s*vdot(d, right), cy - s*vdot(d, up)]; };

  // checkerboard
  const corners = boardCorners().map(project);
  ctx.beginPath(); ctx.moveTo(corners[0][0], corners[0][1]);
  for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
  ctx.closePath();
  ctx.fillStyle = "rgba(255,255,255,0.10)"; ctx.fill();
  ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 1.5; ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.beginPath();
  ctx.moveTo(corners[0][0], corners[0][1]); ctx.lineTo(corners[2][0], corners[2][1]);
  ctx.moveTo(corners[1][0], corners[1][1]); ctx.lineTo(corners[3][0], corners[3][1]); ctx.stroke();
  const bc = project(center);

  // target spheres: grey = pending, green = captured; ring = next / selected
  scene.markers = [];
  (scene.targets || []).forEach((t, i) => {
    const p = project(t.position);
    ctx.strokeStyle = "rgba(139,152,168,0.25)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(bc[0], bc[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(p[0], p[1], i === scene.selected ? 6 : 4.5, 0, Math.PI*2);
    ctx.fillStyle = t.done ? "#22c55e" : "#8b98a8"; ctx.fill();
    if (i === scene.next) {
      ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p[0], p[1], 8, 0, Math.PI*2); ctx.stroke();
    }
    if (i === scene.selected) {
      ctx.strokeStyle = "#e6edf3"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(p[0], p[1], 9, 0, Math.PI*2); ctx.stroke();
    }
    scene.markers.push({ x: p[0], y: p[1], index: i });
  });

  // live camera pose (simulation only)
  if (scene.cam) {
    const p = project([scene.cam.x, scene.cam.y, scene.cam.z]);
    ctx.strokeStyle = "rgba(59,130,246,0.7)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(bc[0], bc[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI*2); ctx.fillStyle = "#3b82f6"; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

function updateRefPanel() {
  const targets = scene.targets || [];
  const done = targets.filter((t) => t.done).length;
  const doneCount = el("done-count");
  if (doneCount) doneCount.textContent = targets.length ? `  ${done}/${targets.length} captured` : "";
  const idx = (scene.selected != null) ? scene.selected : scene.next;
  const nameEl = el("next-name"), img = el("ref-img"), hint = el("ref-hint");
  if (idx == null || !targets[idx]) {
    nameEl.textContent = (targets.length && done === targets.length) ? "all captured ✓" : "—";
    img.hidden = true; hint.hidden = false;
    hint.textContent = (targets.length && done === targets.length) ? "Coverage complete — Calibrate." : "Waiting…";
    return;
  }
  const t = targets[idx];
  nameEl.textContent = t.name + (scene.selected != null ? " (selected)" : "") + (t.done ? " ✓" : "");
  if (t.has_ref) {
    img.src = "api/v1/intrinsic/ref/" + idx + ".jpg?v=" + (t.done ? "1" : "0");
    img.hidden = false; hint.hidden = true;
  } else {
    img.hidden = true; hint.hidden = false;
    hint.textContent = scene.control ? "No reference yet — align here to record it." : "Move the camera near this pose.";
  }
}

function initScene() {
  scene.canvas = el("scene");
  if (!scene.canvas) return;
  scene.ctx = scene.canvas.getContext("2d");
  fetch("api/v1/intrinsic/targets").then((r) => r.json()).then((d) => {
    scene.board = d.board;
    if (!scene.targets.length) {
      scene.targets = (d.views || []).map((v) => ({ name: v.name, position: v.position, done: false, has_ref: false }));
    }
    drawScene();
  }).catch(() => {});
  const cv = scene.canvas;
  cv.addEventListener("pointerdown", (e) => { scene.dragging = true; scene.moved = false; scene.lastX = e.clientX; scene.lastY = e.clientY; cv.setPointerCapture(e.pointerId); });
  cv.addEventListener("pointermove", (e) => {
    if (!scene.dragging) return;
    const dx = e.clientX - scene.lastX, dy = e.clientY - scene.lastY;
    if (Math.abs(dx) + Math.abs(dy) > 3) scene.moved = true;
    scene.az -= dx * 0.01;
    scene.el = Math.max(-1.4, Math.min(1.4, scene.el + dy * 0.01));
    scene.lastX = e.clientX; scene.lastY = e.clientY; drawScene();
  });
  cv.addEventListener("pointerup", (e) => {
    scene.dragging = false;
    if (scene.moved) return;               // it was a drag, not a click
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    let best = null, bestDist = 16;
    for (const m of scene.markers) { const d = Math.hypot(m.x - mx, m.y - my); if (d < bestDist) { bestDist = d; best = m; } }
    scene.selected = best ? best.index : null;
    // In simulation, clicking a sphere flies the camera there (teleport + auto-aim).
    if (best && scene.control && !autoRunning) {
      postJSON("api/v1/intrinsic/goto", { index: best.index }).then((r) => setStatus(r && r.ok ? "Moved to " + r.name + "." : (r.error || "")));
    }
    updateRefPanel(); drawScene();
  });
  window.addEventListener("resize", drawScene);
}

wireButtons();
initScene();
poll();
refreshImage();
setInterval(poll, 200);
setInterval(refreshImage, 250);

export {}
