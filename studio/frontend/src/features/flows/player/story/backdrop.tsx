// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { memo, useEffect, useMemo, useRef, useState } from "react";

import { Prng, seedFrom } from "../../expr";

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
}

/** Crossfades between scene backgrounds; `url` null shows the procedural backdrop for `seedKey`. */
export function BackgroundStage({
  seedKey,
  url,
  painting,
}: { seedKey: string; url: string | null; painting: boolean }) {
  const key = url ?? `proc:${seedKey}`;
  const [layers, setLayers] = useState<Layer[]>([{ key, url, shown: true }]);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    setLayers((prev) => {
      if (prev[prev.length - 1]?.key === key) return prev;
      return [...prev.slice(-1), { key, url, shown: false }];
    });
    const raf = requestAnimationFrame(() =>
      setLayers((prev) =>
        prev.map((l) => (l.key === key ? { ...l, shown: true } : l)),
      ),
    );
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(
      () => setLayers((prev) => prev.filter((l) => l.key === key)),
      900,
    );
    return () => cancelAnimationFrame(raf);
  }, [key, url]);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    [],
  );
  return (
    <div className="absolute inset-0 overflow-hidden bg-black">
      {layers.map((l) => (
        <div
          key={l.key}
          className="absolute inset-0 transition-opacity duration-700 ease-out"
          style={{ opacity: l.shown ? 1 : 0 }}
        >
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
`}</style>
  );
}
