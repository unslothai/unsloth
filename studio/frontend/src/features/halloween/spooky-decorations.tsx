// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { useRouterState } from "@tanstack/react-router";
import { useReducedMotionConfig } from "motion/react";
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
      flip: boolean;
      bats: { delay: number; offset: number; scale: number }[];
    };

// The drop; the spider then hangs until the page changes.
const SPIDER_MS = 3500;
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

// Font Awesome 7 "spider" (solid), CC BY 4.0 (fontawesome.com).
function Spider() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 640 640"
      className="spooky-spider-body"
      fill="currentColor"
    >
      <path d="M433.3 33c11.9-3.5 24.4 2.7 29 13.9l.8 2.3l40 136c2.5 8.4.1 17.5-6.1 23.7L432.9 273l76.4-20.8l19.9-59.8l.9-2.3c5-11.1 17.7-16.8 29.5-12.9s18.5 16.1 15.8 28l-.7 2.4l-24 72c-2.5 7.6-8.7 13.4-16.5 15.6L443.1 320l91.1 24.8c7.7 2.1 13.9 8 16.5 15.6l24 72l.7 2.4c2.7 11.9-4.1 24.1-15.8 28c-11.8 3.9-24.5-1.8-29.5-12.9l-.9-2.3l-19.9-59.8l-76.4-20.8l64.1 64.1c6.1 6.1 8.5 15.1 6.2 23.4l-40 144l-.8 2.3c-4.4 11.3-16.8 17.7-28.8 14.4S414.3 600 416.4 588l.5-2.4l36.2-130.5l-37.2-37.2C415 470 372.4 512 320 512s-95-42-96-94.1l-37 37l36 122.3l.6 2.4c2.2 11.9-4.9 23.9-16.8 27.4s-24.4-2.7-29-13.9l-.8-2.3l-40-136c-2.5-8.4-.1-17.5 6.1-23.8l64.1-64.1l-76.4 20.8l-19.9 59.8l-.9 2.3c-5 11.1-17.7 16.8-29.5 12.9s-18.5-16.1-15.8-28l.7-2.4l24-72c2.5-7.6 8.7-13.4 16.5-15.6l91.1-24.8l-91.1-24.8c-7.7-2.1-13.9-8-16.5-15.6l-24-72l-.7-2.4c-2.7-11.9 4.1-24.1 15.8-28S105 178.9 110 190l.9 2.3l19.9 59.8l76.4 20.8l-64.1-64.1c-6.2-6.2-8.5-15.3-6.1-23.7l40-136l.8-2.3c4.6-11.3 17.1-17.4 29-13.9s19 15.5 16.8 27.4l-.6 2.4L187 185l53 53c1-43.3 36.4-78.1 79.9-78.1s78.9 34.8 79.9 78.2l53.1-53.1l-36-122.3l-.6-2.4c-2.2-12 4.9-23.9 16.8-27.4z" />
    </svg>
  );
}

// Material Design Icons "bat", Apache-2.0 (github.com/Templarian/MaterialDesign).
function Bat() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0.5 6.5 23 12.5"
      className="spooky-bat-wings"
      fill="currentColor"
    >
      <path d="M.75 8S5 7 8 9c0 0 .5 3.75 2.5 3.75V11s.5 1 1.5 1s1.5-1 1.5-1v1.75C15.5 12.75 16 9 16 9c3-2 7.25-1 7.25-1c-2 1-2.25 4.5-2.25 4.5c-4 0-4 3.25-4 3.25c-5-1-5 2.75-5 2.75s0-3.75-5-2.75c0 0 0-3.25-4-3.25C3 12.5 2.75 9 .75 8" />
    </svg>
  );
}

// Reduced motion keeps the visits but fades them in place.
export function SpookyDecorations() {
  const on = useSpookyDecorations();
  // Follows Studio's Reduce motion setting, which can override the system one.
  const reduced = useReducedMotionConfig();
  const [visitors, setVisitors] = useState<Visitor[]>([]);
  const href = useRouterState({ select: (s) => s.location.href });
  // biome-ignore lint/correctness/useExhaustiveDependencies: each page rolls its own webs.
  const webs = useMemo(() => rollSpiderWebs(), [href]);
  const nextId = useRef(0);

  // biome-ignore lint/correctness/useExhaustiveDependencies: each page rolls its own visitors.
  useEffect(() => {
    setVisitors([]);
    if (!on) {
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
    const visit = (v: Visitor, ms?: number) => {
      // Hidden tabs skip visits rather than queue them.
      if (document.hidden) {
        return;
      }
      setVisitors((list) => [...list, v]);
      if (ms !== undefined) {
        later(ms + 500, () =>
          setVisitors((list) => list.filter((x) => x.id !== v.id)),
        );
      }
    };
    const spider = () => {
      visit({
        kind: "spider",
        id: nextId.current++,
        left: `${between(35, 92).toFixed(1)}%`,
        drop: Math.round(between(90, 230)),
      });
    };
    const bats = () => {
      const count = 1 + Math.floor(Math.random() * 3);
      visit(
        {
          kind: "bats",
          id: nextId.current++,
          top: `${between(8, 32).toFixed(1)}%`,
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
    // A page gets a spider or bats, never both: a third of the time each.
    const roll = Math.random();
    if (roll < 1 / 3) {
      later(between(2000, 6000), spider);
    } else if (roll < 2 / 3) {
      later(between(2000, 6000), bats);
    }
    return () => {
      for (const t of timers) {
        clearTimeout(t);
      }
    };
  }, [on, href]);

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
                "--spooky-ms": `${SPIDER_MS}ms`,
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
                    "--spooky-ms": `${BATS_MS}ms`,
                    animationDuration: `${BATS_MS}ms`,
                    animationDelay: `${b.delay}s`,
                    marginTop: b.offset,
                  } as CSSProperties
                }
              >
                {/* Sized here, not on the flying element: scale there shortens the flight too. */}
                <div style={{ scale: b.scale }}>
                  <Bat />
                </div>
              </div>
            ))}
          </div>
        ),
      )}
    </div>
  );
}
