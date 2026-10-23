// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { memo, useEffect, useMemo, useRef, useState } from "react";

import { Prng, seedFrom } from "../../expr";
import type { StoryTransition } from "../../types";

const REDUCED = "(prefers-reduced-motion: reduce)";

export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" && !!window.matchMedia?.(REDUCED).matches,
  );
  useEffect(() => {
    const q = window.matchMedia?.(REDUCED);
    if (!q) return;
    const on = () => setReduced(q.matches);
    q.addEventListener("change", on);
    return () => q.removeEventListener("change", on);
  }, []);
  return reduced;
}

const SPRITE: Record<StoryTransition, [string, number]> = {
  dissolve: ["flows-sprite-dissolve", 500],
  fade: ["flows-sprite-dissolve", 900],
  flash: ["flows-sprite-flash", 450],
  slideleft: ["flows-sprite-slideleft", 550],
  slideright: ["flows-sprite-slideright", 550],
  none: ["", 0],
};

/** CSS animation for a sprite coming on (or going off, played in reverse and held). */
export function spriteAnimation(
  t: StoryTransition,
  out: boolean,
): string | undefined {
  const [name, ms] = SPRITE[t] ?? SPRITE.dissolve;
  if (!name) return undefined;
  return out
    ? `${name} ${ms}ms ease-in reverse forwards`
    : `${name} ${ms}ms ease-out both`;
}

const BG_MS: Record<StoryTransition, number> = {
  dissolve: 700,
  fade: 1000,
  flash: 500,
  slideleft: 650,
  slideright: 650,
  none: 0,
};

/** A soft painted-looking gradient with drifting light blobs, seeded by the scene, for scenes without art. */
export const ProceduralBackdrop = memo(function ProceduralBackdrop({
  seedKey,
}: { seedKey: string }) {
  const style = useMemo(() => {
    const rng = new Prng(seedFrom("backdrop", seedKey));
    const hue = Math.floor(rng.random() * 360);
    const hue2 = (hue + 30 + Math.floor(rng.random() * 90)) % 360;
    const blobs = Array.from({ length: 5 }, () => ({
      x: Math.round(rng.random() * 100),
      y: Math.round(rng.random() * 70),
      r: 18 + Math.round(rng.random() * 30),
      h: (hue + Math.floor(rng.random() * 120) - 60 + 360) % 360,
      a: 0.25 + rng.random() * 0.3,
    }));
    const layers = blobs.map(
      (b) =>
        `radial-gradient(circle at ${b.x}% ${b.y}%, hsla(${b.h} 80% 72% / ${b.a.toFixed(2)}) 0%, transparent ${b.r}%)`,
    );
    return {
      backgroundImage: [
        ...layers,
        `linear-gradient(180deg, hsl(${hue} 45% 22%) 0%, hsl(${hue2} 40% 14%) 62%, hsl(${hue2} 35% 8%) 100%)`,
      ].join(","),
    };
  }, [seedKey]);
  return (
    <div className="absolute inset-0 overflow-hidden" style={style}>
      <div
        className="absolute inset-x-0 bottom-0 h-1/3"
        style={{
          background: "linear-gradient(0deg, rgba(0,0,0,0.45), transparent)",
        }}
      />
    </div>
  );
});

interface Layer {
  key: string;
  url: string | null;
  shown: boolean;
  t: StoryTransition;
}

function layerStyle(l: Layer): React.CSSProperties {
  const ms = BG_MS[l.t] ?? 700;
  if (l.t === "slideleft" || l.t === "slideright") {
    const from = l.t === "slideleft" ? "100%" : "-100%";
    return {
      transform: l.shown ? "none" : `translateX(${from})`,
      transition: `transform ${ms}ms ease-out`,
    };
  }
  if (l.t === "fade" || l.t === "flash") {
    // Swap under the overlay at its peak.
    return {
      opacity: l.shown ? 1 : 0,
      transition: `opacity 1ms linear ${Math.round(ms * 0.45)}ms`,
    };
  }
  return {
    opacity: l.shown ? 1 : 0,
    transition: ms ? `opacity ${ms}ms ease-out` : "none",
  };
}

/** Changes scene backgrounds with a transition (default crossfade); `url` null shows the procedural backdrop. */
export function BackgroundStage({
  seedKey,
  url,
  painting,
  transition = "dissolve",
}: {
  seedKey: string;
  url: string | null;
  painting: boolean;
  transition?: StoryTransition;
}) {
  const key = url ?? `proc:${seedKey}`;
  const [layers, setLayers] = useState<Layer[]>([
    { key, url, shown: true, t: "none" },
  ]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // biome-ignore lint/correctness/useExhaustiveDependencies: the transition belongs to the change of `key`, not a trigger
  useEffect(() => {
    setLayers((prev) => {
      if (prev[prev.length - 1]?.key === key) return prev;
      const t = transition;
      return [...prev.slice(-1), { key, url, shown: t === "none", t }];
    });
    const raf = requestAnimationFrame(() =>
      setLayers((prev) =>
        prev.map((l) => (l.key === key ? { ...l, shown: true } : l)),
      ),
    );
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(
      () => setLayers((prev) => prev.filter((l) => l.key === key)),
      (BG_MS[transition] ?? 700) + 200,
    );
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, url]);
  const top = layers[layers.length - 1];
  const veil =
    layers.length > 1 && (top.t === "fade" || top.t === "flash") ? top : null;
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  return (
    <div className="absolute inset-0 overflow-hidden bg-black">
      {layers.map((l) => (
        <div key={l.key} className="absolute inset-0" style={layerStyle(l)}>
          {l.url ? (
            <img
              src={l.url}
              alt=""
              draggable={false}
              className="h-full w-full scale-[1.03] object-cover"
              style={{
                animation: "flows-kenburns 40s ease-in-out infinite alternate",
              }}
            />
          ) : (
            <ProceduralBackdrop seedKey={l.key.replace(/^proc:/, "")} />
          )}
        </div>
      ))}
      {veil ? (
        <div
          key={`veil:${veil.key}`}
          className="pointer-events-none absolute inset-0"
          style={{
            background: veil.t === "flash" ? "#fff" : "#000",
            animation: `flows-veil ${BG_MS[veil.t]}ms ease-in-out both`,
          }}
        />
      ) : null}
      {painting ? (
        <div
          className="pointer-events-none absolute inset-0 opacity-60"
          style={{
            background:
              "linear-gradient(110deg, transparent 30%, rgba(255,255,255,0.08) 45%, rgba(255,255,255,0.16) 50%, rgba(255,255,255,0.08) 55%, transparent 70%)",
            backgroundSize: "250% 100%",
            animation: "flows-shimmer 2.2s linear infinite",
          }}
        />
      ) : null}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.55) 100%)",
        }}
      />
    </div>
  );
}

/** Keyframes the players use, injected once (the feature is lazy, so no global CSS edit is needed). */
export function PlayerKeyframes() {
  return (
    <style>{`
@keyframes flows-kenburns { from { transform: scale(1.03) translate(0,0); } to { transform: scale(1.09) translate(-1.2%, -0.8%); } }
@keyframes flows-shimmer { from { background-position: 150% 0; } to { background-position: -100% 0; } }
@keyframes flows-bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(4px); } }
@keyframes flows-rise { from { opacity: 0; transform: translateY(14px) scale(0.98); } to { opacity: 1; transform: none; } }
@keyframes flows-dots { 0%,80%,100% { opacity: .2; } 40% { opacity: 1; } }
@keyframes flows-veil { 0%,100% { opacity: 0; } 45%,55% { opacity: 1; } }
@keyframes flows-swap { from { opacity: .35; } to { opacity: 1; } }
@keyframes flows-sprite-dissolve { from { opacity: 0; } to { opacity: 1; } }
@keyframes flows-sprite-flash { from { opacity: 0; filter: brightness(4); } to { opacity: 1; filter: none; } }
@keyframes flows-sprite-slideleft { from { opacity: 0; transform: translateX(60vw); } to { opacity: 1; transform: none; } }
@keyframes flows-sprite-slideright { from { opacity: 0; transform: translateX(-60vw); } to { opacity: 1; transform: none; } }
`}</style>
  );
}
