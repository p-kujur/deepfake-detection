// A short, centered sparkle burst for a verdict chip. No floor — stars fly out and wink off.
import { Lights } from "../../core/sim.js";
import { Rng, ease, TAU } from "../../core/math.js";
import { fit, WHITE } from "../../core/color.js";

export const meta = {
  name: "Verdict burst",
  desc: "A short sparkle burst around a verdict: a soft flash, a thin ring, and tiny stars thrown out on curving paths that hang and wink out.",
  tags: ["ui", "burst", "sparkle", "reveal", "one-shot"],
  size: [80, 80],
  origin: [40, 40],
  duration: 0.95,
  loop: false,
  fps: 30,
  bg: "dark",
  options: {
    palette: ["#1a1c2c", "#5d275d", "#b13e53", "#ef7d57", "#ffcd75", "#f4f4f4", "#ffffff"],
    seed: 5,
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
  if (k > 0.05) p.glow(X + 0.5, Y + 0.5, 2 + n * 1.6, c, k);
}

export function create(o) {
  const C = fit(o.palette, 7), R = new Rng(o.seed), lights = new Lights();
  lights.add(0, 0, 16, C[3], 0.4, 0.18);
  lights.add(0, 0, 7, C[5], 0.35, 0.06);
  const stars = Array.from({ length: 11 }, (_, i) => {
    const a = (i / 11) * TAU + R.range(-0.2, 0.2);
    const sp = R.range(55, 115) * (i % 3 === 0 ? 1.15 : 0.82);
    return {
      x: Math.cos(a) * R.range(4, 7),
      y: Math.sin(a) * R.range(4, 7),
      a,
      sp,
      turn: (i % 2 ? 1 : -1) * R.range(1.6, 3.6),
      life: R.range(0.48, 0.82),
      age: -R.range(0.05, 0.12),
      size: i % 5 === 0 ? 2 : 1,
      tr: [],
      c: i % 3 === 0 ? C[5] : i % 2 ? C[4] : C[3],
      ph: R.range(0, TAU),
    };
  });
  let t = 0;
  return {
    update(dt) {
      t += dt;
      for (const s of stars) {
        s.age += dt;
        if (s.age < 0 || s.age > s.life) continue;
        s.tr.unshift([s.x, s.y]);
        if (s.tr.length > 3) s.tr.pop();
        s.a += s.turn * dt * Math.max(0, 1 - s.age * 1.6);
        s.sp -= s.sp * 4.6 * dt;
        s.x += Math.cos(s.a) * s.sp * dt;
        s.y += Math.sin(s.a) * s.sp * dt;
      }
      lights.update(dt);
    },
    draw(p) {
      if (t < 0.05) p.circle(0, 0, 3.2 * (1 - t / 0.05) + 1, t < 0.02 ? WHITE : C[5], 0.9);
      if (t < 0.12) {
        const k = 1 - t / 0.12;
        p.glow(0, 0, 10 * k + 2, C[3], 0.32 * k);
      }
      const u1 = t / 0.26;
      if (u1 < 1) {
        const r = 3 + 15 * ease.outExpo(u1);
        p.ring(0, 0, r, 1, u1 < 0.12 ? WHITE : u1 < 0.4 ? C[5] : C[3], 1.05 - u1);
        p.glowRing(0, 0, r, 2.5, C[3], 0.22 * (1 - u1));
      }
      const u2 = (t - 0.04) / 0.38;
      if (u2 > 0 && u2 < 1) {
        const r = 3 + 10 * ease.outCubic(u2);
        p.ring(0, 0, r, 1, C[3], 0.55 * (1 - u2));
      }
      for (const s of stars) {
        if (s.age < 0 || s.age > s.life) continue;
        const k = 1 - s.age / s.life, tw = 0.65 + 0.35 * Math.sin(s.age * 24 + s.ph);
        s.tr.forEach(([x, y], i) => i > 0 && p.dot(x, y, i < 2 ? s.c : C[2], (1 - i / 3) * k));
        if (k < 0.28 && Math.sin(s.age * 38 + s.ph) < 0) continue;
        twinkle(p, s.x, s.y, s.size * tw * Math.min(1, k * 2.2), s.c, 0.26 * k);
      }
      lights.draw(p);
    },
  };
}
