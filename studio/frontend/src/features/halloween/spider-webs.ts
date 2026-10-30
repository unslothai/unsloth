// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type WebPlacement = "top-right" | "bottom-right" | "top";

export type SpiderWeb = {
  placement: WebPlacement;
  // Corner webs span `size` square; a top web is twice as wide as tall.
  size: number;
  left?: string;
  spokes: string;
  threads: string;
};

type Rand = () => number;

const between = (rand: Rand, min: number, max: number) =>
  min + rand() * (max - min);
const f = (n: number) => n.toFixed(1);

// 100-unit frame, anchor at the origin, opening towards +x/+y.
function drawWeb(rand: Rand, from: number, to: number) {
  const count = Math.round(between(rand, 6, 9));
  const spokes = Array.from({ length: count }, (_, i) => {
    const t = count === 1 ? 0.5 : i / (count - 1);
    const angle =
      ((from + (to - from) * t + between(rand, -3, 3)) * Math.PI) / 180;
    return {
      dx: Math.cos(angle),
      dy: Math.sin(angle),
      length: between(rand, 78, 100),
    };
  });
  const spokePath = spokes
    .map((s) => `M0 0L${f(s.dx * s.length)} ${f(s.dy * s.length)}`)
    .join("");

  const rings = Math.round(between(rand, 6, 9));
  let threads = "";
  for (let r = 1; r <= rings; r++) {
    const radius = 92 * (r / (rings + 0.6)) ** 1.08;
    for (let i = 0; i < spokes.length - 1; i++) {
      const a = spokes[i];
      const b = spokes[i + 1];
      if (radius > Math.min(a.length, b.length) - 2 || rand() < 0.06) {
        continue;
      }
      const ra = radius * between(rand, 0.96, 1.04);
      const rb = radius * between(rand, 0.96, 1.04);
      const ax = a.dx * ra;
      const ay = a.dy * ra;
      const bx = b.dx * rb;
      const by = b.dy * rb;
      // Silk between two spokes bows towards the hub.
      const sag = between(rand, 0.84, 0.92);
      threads += `M${f(ax)} ${f(ay)}Q${f(((ax + bx) / 2) * sag)} ${f(((ay + by) / 2) * sag)} ${f(bx)} ${f(by)}`;
    }
  }
  const first = spokes[0];
  const last = spokes[spokes.length - 1];
  threads += `M${f(first.dx * first.length)} ${f(first.dy * first.length)}Q${f(((first.dx + last.dx) / 2) * 70)} ${f(((first.dy + last.dy) / 2) * 70)} ${f(last.dx * last.length)} ${f(last.dy * last.length)}`;
  return { spokes: spokePath, threads };
}

export function rollSpiderWebs(rand: Rand = Math.random): SpiderWeb[] {
  if (rand() >= 0.5) {
    return [];
  }
  // No left corners: the sidebar is there.
  const spots: WebPlacement[] = ["top-right", "bottom-right", "top"];
  const count = rand() < 0.4 ? 2 : 1;
  const picks = Array.from(
    { length: count },
    () => spots.splice(Math.floor(rand() * spots.length), 1)[0],
  );
  return picks.map((placement) =>
    placement === "top"
      ? {
          placement,
          size: Math.round(between(rand, 70, 110)),
          left: `${between(rand, 35, 80).toFixed(1)}%`,
          ...drawWeb(rand, 12, 168),
        }
      : {
          placement,
          size: Math.round(between(rand, 95, 165)),
          ...drawWeb(rand, 4, 86),
        },
  );
}
