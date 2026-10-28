// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useRouterState } from "@tanstack/react-router";
import { useReducedMotion } from "motion/react";
import {
  type CSSProperties,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSpookyDecorations } from "./halloween-store";
import { type SpiderWeb, rollSpiderWebs } from "./spider-webs";

type Visitor =
  | { kind: "spider"; id: number; left: string; drop: number }
  | {
      kind: "bats";
      id: number;
      top: string;
      left: string;
      flip: boolean;
      bats: { delay: number; offset: number; scale: number }[];
    };

const SPIDER_MS = 9000;
const BATS_MS = 9000;

const between = (min: number, max: number) => min + Math.random() * (max - min);

const px = (n: number) => `calc(${n}px*var(--ui-space-scale,1))`;

function Web({ web }: { web: SpiderWeb }) {
  const top = web.placement === "top";
  return (
    <svg
      aria-hidden="true"
      viewBox={top ? "-100 0 200 100" : "0 0 100 100"}
      fill="none"
      stroke="currentColor"
      strokeLinecap="round"
      className={cn(
        "absolute opacity-30",
        top && "top-0 -translate-x-1/2",
        web.placement === "top-right" && "top-0 right-0 -scale-x-100",
        web.placement === "bottom-right" && "right-0 bottom-0 -scale-100",
      )}
      style={{
        left: web.left,
        width: px(top ? web.size * 2 : web.size),
        height: px(web.size),
      }}
    >
      <path d={web.spokes} strokeWidth="1" vectorEffect="non-scaling-stroke" />
      <path
        d={web.threads}
        strokeWidth="0.75"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

function Spider() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      className="spooky-spider-body"
      fill="currentColor"
    >
      <g
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        fill="none"
      >
        <path d="M9 11L4 8L2 10M9 13L3 13L1 15M9.5 15L5 18L4 21M10 16.5L7 21" />
        <path d="M15 11L20 8L22 10M15 13L21 13L23 15M14.5 15L19 18L20 21M14 16.5L17 21" />
      </g>
      <ellipse cx="12" cy="9.5" rx="2.4" ry="2.2" />
      <ellipse cx="12" cy="15" rx="3.4" ry="4" />
    </svg>
  );
}

function Bat() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 48 20"
      className="spooky-bat-wings"
      fill="currentColor"
    >
      <path d="M24 6c1.2-2 2.4-2 3 0c4-4 10-5 21-1c-5 1-7 4-8 8c-3-2-6-2-9 1c-1.5-2-3.5-3-7-3s-5.5 1-7 3c-3-3-6-3-9-1C7 9 5 6 0 5c11-4 17-3 21 1c.6-2 1.8-2 3 0Z" />
    </svg>
  );
}

/** Webs on half the pages, re-rolled on each navigation, plus a spider and bats now and then. Reduced
 *  motion keeps the visits but swaps the drop and the flight for a fade in place. */
export function SpookyDecorations() {
  const on = useSpookyDecorations();
  const reduced = useReducedMotion();
  const [visitors, setVisitors] = useState<Visitor[]>([]);
  const href = useRouterState({ select: (s) => s.location.href });
  // biome-ignore lint/correctness/useExhaustiveDependencies: each page rolls its own webs.
  const webs = useMemo(() => rollSpiderWebs(), [href]);
  const nextId = useRef(0);

  useEffect(() => {
    if (!on) {
      setVisitors([]);
      return;
    }
    const timers = new Set<ReturnType<typeof setTimeout>>();
    const later = (ms: number, fn: () => void) => {
      const t = setTimeout(() => {
        timers.delete(t);
        fn();
      }, ms);
      timers.add(t);
    };
    const visit = (v: Visitor, ms: number) => {
      // A hidden tab skips the visit instead of piling them up for later.
      if (document.hidden) {
        return;
      }
      setVisitors((list) => [...list, v]);
      later(ms + 500, () =>
        setVisitors((list) => list.filter((x) => x.id !== v.id)),
      );
    };
    const spider = () => {
      visit(
        {
          kind: "spider",
          id: nextId.current++,
          left: `${between(35, 92).toFixed(1)}%`,
          drop: Math.round(between(90, 230)),
        },
        SPIDER_MS,
      );
      later(between(25_000, 55_000), spider);
    };
    const bats = () => {
      const count = 1 + Math.floor(Math.random() * 3);
      visit(
        {
          kind: "bats",
          id: nextId.current++,
          top: `${between(8, 32).toFixed(1)}%`,
          left: `${between(20, 65).toFixed(1)}%`,
          flip: Math.random() < 0.5,
          bats: Array.from({ length: count }, (_, i) => ({
            delay: i * between(0.25, 0.6),
            offset: Math.round(between(-24, 24)),
            scale: between(0.7, 1.1),
          })),
        },
        BATS_MS + 2000,
      );
      later(between(30_000, 60_000), bats);
    };
    later(between(2000, 6000), spider);
    later(between(7000, 14_000), bats);
    return () => {
      for (const t of timers) {
        clearTimeout(t);
      }
    };
  }, [on]);

  if (!on) {
    return null;
  }
  return (
    <div
      aria-hidden="true"
      data-testid="spooky-decorations"
      className={cn(
        "pointer-events-none fixed inset-0 z-[45] overflow-hidden text-foreground",
        reduced && "spooky-still",
      )}
    >
      {webs.map((web) => (
        <Web key={web.placement} web={web} />
      ))}
      {visitors.map((v) =>
        v.kind === "spider" ? (
          <div
            key={v.id}
            className="spooky-spider"
            style={
              {
                left: v.left,
                "--spooky-drop": `${v.drop}px`,
                animationDuration: `${SPIDER_MS}ms`,
              } as CSSProperties
            }
          >
            <Spider />
          </div>
        ) : (
          <div
            key={v.id}
            className="spooky-bats"
            style={
              {
                top: v.top,
                "--spooky-bats-left": v.left,
                transform: v.flip ? "scaleX(-1)" : undefined,
              } as CSSProperties
            }
          >
            {v.bats.map((b, i) => (
              <div
                // biome-ignore lint/suspicious/noArrayIndexKey: a flock never reorders.
                key={i}
                className="spooky-bat"
                style={
                  {
                    "--spooky-bat-i": i,
                    animationDuration: `${BATS_MS}ms`,
                    animationDelay: `${b.delay}s`,
                    marginTop: b.offset,
                    scale: b.scale,
                  } as CSSProperties
                }
              >
                <Bat />
              </div>
            ))}
          </div>
        ),
      )}
    </div>
  );
}
