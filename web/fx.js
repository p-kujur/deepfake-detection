// Pixel-art overlays powered by vendored pixelfx (MIT). rAF runs only while an effect is active.
import { createFrame, clearFrame, createFx, composite } from "./vendor/pixelfx/core/fx.js";
import * as scanSweep from "./vendor/pixelfx/fx/ui/scan-sweep.js";
import * as verdictBurst from "./vendor/pixelfx/fx/ui/verdict-burst.js";

// Sweetie 16 ramps (dark -> bright)
const SCAN_PALETTE = ["#1a1c2c", "#29366f", "#3b5dc9", "#41a6f6", "#73eff7", "#f4f4f4", "#ffffff"];
const FAKE_PALETTE = ["#1a1c2c", "#5d275d", "#b13e53", "#ef7d57", "#ffcd75", "#f4f4f4", "#ffffff"];
const REAL_PALETTE = ["#1a1c2c", "#257179", "#38b764", "#a7f070", "#ffcd75", "#f4f4f4", "#ffffff"];

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
    light: 0.5,
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
    palette: isFake ? FAKE_PALETTE : REAL_PALETTE,
    seed: isFake ? 5 : 9,
    light: 0.6,
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
