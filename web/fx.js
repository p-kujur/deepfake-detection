// Pixel-art overlays powered by vendored pixelfx (MIT). rAF runs only while an effect is active.
import { createFrame, clearFrame, createFx, composite } from "./vendor/pixelfx/core/fx.js";
import * as scanSweep from "./vendor/pixelfx/fx/ui/scan-sweep.js";
import * as verdictBurst from "./vendor/pixelfx/fx/ui/verdict-burst.js";

const SCAN_PALETTE = ["#5a3d42", "#8d6a6e", "#c9938a", "#d4a5b8", "#e8d4dc", "#f8e4dc", "#fffbf7"];
const ROSE_PALETTE = ["#6b3030", "#9a3b3b", "#c96b6b", "#c9938a", "#f8d9d4", "#fff6f2", "#ffffff"];
const SAGE_PALETTE = ["#1f4a32", "#2f6a45", "#6e9b7e", "#a3c4ae", "#d8ebd9", "#f4faf5", "#ffffff"];

const reduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function layer(canvas) {
  return {
    canvas,
    ctx: canvas.getContext("2d", { alpha: true }),
    frame: null,
    img: null,
    fx: null,
  };
}

function pixelSize(el, max = 96) {
  const r = el.getBoundingClientRect();
  const w = Math.max(1, r.width), h = Math.max(1, r.height);
  if (w >= h) return { w: max, h: Math.max(24, Math.round(max * h / w)) };
  return { w: Math.max(24, Math.round(max * w / h)), h: max };
}

function resize(layer, w, h) {
  if (layer.canvas.width !== w || layer.canvas.height !== h || !layer.frame) {
    layer.canvas.width = w;
    layer.canvas.height = h;
    layer.frame = createFrame(w, h);
    layer.img = new ImageData(w, h);
  }
}

function paint(layer) {
  clearFrame(layer.frame);
  layer.fx.draw(layer.frame);
  composite(layer.frame, layer.img.data, null);
  layer.ctx.putImageData(layer.img, 0, 0);
}

function hide(layer) {
  if (!layer.ctx) return;
  layer.fx = null;
  layer.ctx.clearRect(0, 0, layer.canvas.width, layer.canvas.height);
  layer.canvas.hidden = true;
}

let scan, burst, raf = 0, last = 0;

function tick(now) {
  raf = 0;
  const dt = Math.min(0.05, Math.max(0, (now - last) / 1000)) || 1 / 60;
  last = now;
  let alive = false;
  if (scan?.fx) {
    scan.fx.update(dt);
    if (scan.fx.done()) hide(scan);
    else {
      paint(scan);
      alive = true;
    }
  }
  if (burst?.fx) {
    burst.fx.update(dt);
    if (burst.fx.done()) hide(burst);
    else {
      paint(burst);
      alive = true;
    }
  }
  if (alive) raf = requestAnimationFrame(tick);
}

function kick() {
  if (!raf) {
    last = performance.now();
    raf = requestAnimationFrame(tick);
  }
}

export function initFx() {
  scan = layer(document.getElementById("scanFx"));
  burst = layer(document.getElementById("burstFx"));
}

export function startScan(host) {
  if (reduced() || !scan) return;
  const { w, h } = pixelSize(host, 96);
  resize(scan, w, h);
  scan.fx = createFx(scanSweep, {
    x: w / 2,
    y: h / 2,
    width: w,
    height: h,
    palette: SCAN_PALETTE,
    seed: 11,
    light: 0.35,
  });
  scan.canvas.hidden = false;
  paint(scan);
  kick();
}

export function stopScan() {
  if (!scan?.fx) {
    if (scan) hide(scan);
    return;
  }
  scan.fx.stop();
  kick();
}

export function startBurst(isFake) {
  if (reduced() || !burst) return;
  const w = 80, h = 80;
  resize(burst, w, h);
  burst.fx = createFx(verdictBurst, {
    x: w / 2,
    y: h / 2,
    palette: isFake ? ROSE_PALETTE : SAGE_PALETTE,
    seed: isFake ? 5 : 9,
    light: 0.35,
  });
  burst.canvas.hidden = false;
  paint(burst);
  kick();
}

export function stopAll() {
  if (scan) hide(scan);
  if (burst) hide(burst);
  if (raf) {
    cancelAnimationFrame(raf);
    raf = 0;
  }
}
