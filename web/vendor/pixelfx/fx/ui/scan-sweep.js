// A soft scan bar for overlaying a still image: one clean bar, a glow trail, tiny wake sparkles.
import { TAU } from "../../core/math.js";
import { fit, WHITE } from "../../core/color.js";
import { cycle } from "../../core/particles.js";

export const meta = {
  name: "Scan sweep",
  desc: "A thin scan bar glides down a rectangle, leaving a short cooling glow and tiny star motes that wink in its wake.",
  tags: ["ui", "scan", "scanline", "sparkle", "loop"],
  size: [96, 72],
  origin: [48, 36],
  duration: 2.4,
  loop: true,
  fps: 30,
  bg: "dark",
  options: {
    palette: ["#5a3d42", "#8d6a6e", "#c9938a", "#d4a5b8", "#e8d4dc", "#f8e4dc", "#fffbf7"],
    seed: 11,
    width: 96,
    height: 72,
  },
};

function twinkle(p, x, y, arm, c, k) {
  const n = Math.round(arm), X = Math.floor(x), Y = Math.floor(y);
  if (n < 0) return;
  p.px(X, Y, n > 0 ? WHITE : c);
  if (n >= 1) {
    p.px(X + 1, Y, c);
    p.px(X - 1, Y, c);
    p.px(X, Y + 1, c);
    p.px(X, Y - 1, c);
  }
  if (n >= 2) {
    p.px(X + 2, Y, c);
    p.px(X - 2, Y, c);
    p.px(X, Y + 2, c);
    p.px(X, Y - 2, c);
  }
  if (k > 0.05) p.glow(X + 0.5, Y + 0.5, 2 + n * 1.8, c, k);
}

export function draw(p, t, o) {
  const C = fit(o.palette, 7);
  const W = o.width ?? meta.size[0];
  const H = o.height ?? meta.size[1];
  const x0 = -W / 2, y0 = -H / 2;
  const D = meta.duration;
  const u = (((t % D) + D) % D) / D;
  const y = y0 + 2 + u * (H - 4);
  const yb = Math.round(y);
  const xL = Math.round(x0 + 2), xR = Math.round(x0 + W - 3);

  // additive trail only — no low-alpha full-width lines (those Bayer-dither into a zipper)
  p.glowLine(xL, yb, xR, yb, 6, C[3], 0.2);
  p.glow(0, yb, 12, C[4], 0.18);

  // one clean core bar, slightly brighter in the middle
  p.line(xL, yb, xR, yb, (s) => (s > 0.38 && s < 0.62 ? WHITE : C[5]), 1);
  // a single solid tick under the bar (full alpha, one color) so it reads as a 2 px edge
  if (yb + 1 <= y0 + H - 3) p.line(xL + 1, yb + 1, xR - 1, yb + 1, C[3], 1);

  // end caps: short 2 px ticks
  p.px(xL, yb - 1, C[5]);
  p.px(xR, yb - 1, C[5]);
  p.px(xL, yb, WHITE);
  p.px(xR, yb, WHITE);

  // a couple of traveling hot pixels on the bar (seeded, not random)
  for (let i = 0; i < 3; i++) {
    const hx = xL + 4 + Math.floor((((u * (2 + i) + i * 0.37) % 1) * (xR - xL - 8)));
    p.px(hx, yb, WHITE);
    if (i === 0) p.px(hx, yb - 1, C[5]);
  }

  cycle(t, D, 16, o.seed, (r) => r.range(0.35, 0.7), (age, life, r, i) => {
    const k = 1 - age / life;
    const born = (((t - age) % D) + D) % D;
    const by = y0 + 2 + (born / D) * (H - 4) + age * 3.2;
    if (by < y0 + 2 || by > y0 + H - 3) return;
    const x = x0 + 7 + r.next() * (W - 14);
    const tw = 0.5 + 0.5 * Math.sin(age * 20 + i * 1.7);
    if (k < 0.3 && Math.sin(age * 34 + i) < 0) return;
    const arm = (i % 4 === 0 ? 2 : 1) * tw * Math.min(1, k * 2.4);
    twinkle(p, x, by, arm, i % 2 ? C[5] : C[3], 0.28 * k);
  });
}
