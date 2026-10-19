// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { emitTo, listen } from "@tauri-apps/api/event";
import { LogicalPosition, availableMonitors, getCurrentWindow } from "@tauri-apps/api/window";
import { Maximize2Icon, PlayIcon, SquareIcon } from "lucide-react";
import { useEffect, useRef, useState, type FC, type PointerEvent, type ReactNode } from "react";

import { OrbBall, type OrbStateName, type SpectrumReader } from "@/components/assistant-ui/voice-orb";
import {
  ORB_WIDGET_ACTION_EVENT,
  ORB_WIDGET_BALL,
  ORB_WIDGET_CENTER,
  ORB_WIDGET_POSITION_KEY,
  ORB_WIDGET_READY_EVENT,
  ORB_WIDGET_SPECTRUM_EVENT,
  ORB_WIDGET_STATE_EVENT,
  clampToScreens,
  unpackSpectrum,
  type Area,
  type OrbWidgetAction,
  type OrbWidgetSnapshot,
  type Point,
} from "@/features/chat/voice/orb-widget";
import { SPECTRUM_BANDS } from "@/features/chat/voice/voice-spectrum";
import { cn } from "@/lib/utils";

// The latest frame streamed from the main window, where the analysers live.
const remote = { bands: new Float32Array(SPECTRUM_BANDS), level: 0, side: null as string | null };
const readRemote: SpectrumReader = (side, out) => {
  if (remote.side !== side) {
    out.fill(0);
    return 0;
  }
  out.set(remote.bands);
  return remote.level;
};

async function screenAreas(): Promise<Area[]> {
  const monitors = await availableMonitors().catch(() => []);
  return monitors.map((m) => {
    const k = m.scaleFactor || 1;
    return { x: m.position.x / k, y: m.position.y / k, width: m.size.width / k, height: m.size.height / k };
  });
}

/**
 * Moves the window by hand rather than through startDragging: the system drag keeps
 * a window below the menu bar, and a desktop widget should reach the top of the screen.
 */
function useWindowDrag() {
  const drag = useRef<{ pointer: Point; origin: Point; screens: Area[]; queued: Point | null; busy: boolean } | null>(
    null,
  );
  const move = () => {
    const d = drag.current;
    if (!d || d.busy || !d.queued) return;
    const next = clampToScreens(d.queued, d.screens);
    d.queued = null;
    d.busy = true;
    void getCurrentWindow()
      .setPosition(new LogicalPosition(next.x, next.y))
      .catch(() => undefined)
      .finally(() => {
        d.busy = false;
        move();
      });
  };
  return {
    onPointerDown: (e: PointerEvent<HTMLDivElement>) => {
      if (e.button !== 0) return;
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      const pointer = { x: e.screenX, y: e.screenY };
      const win = getCurrentWindow();
      const started = { pointer, origin: pointer, screens: [] as Area[], queued: null, busy: true };
      drag.current = started;
      void Promise.all([win.outerPosition(), win.scaleFactor(), screenAreas()]).then(([pos, k, screens]) => {
        started.origin = { x: pos.x / k, y: pos.y / k };
        started.screens = screens;
        started.busy = false;
        move();
      });
    },
    onPointerMove: (e: PointerEvent<HTMLDivElement>) => {
      const d = drag.current;
      if (!d) return;
      d.queued = { x: d.origin.x + e.screenX - d.pointer.x, y: d.origin.y + e.screenY - d.pointer.y };
      move();
    },
    onPointerUp: () => {
      drag.current = null;
    },
    onPointerCancel: () => {
      drag.current = null;
    },
  };
}

const send = (action: OrbWidgetAction) => {
  void emitTo("main", ORB_WIDGET_ACTION_EVENT, action).catch(() => undefined);
};

const PauseBars: FC = () => (
  <svg aria-hidden={true} viewBox="0 0 16 16" className="size-3.5" fill="currentColor">
    <rect x="3.5" y="2.5" width="2.6" height="11" rx="1.3" />
    <rect x="9.9" y="2.5" width="2.6" height="11" rx="1.3" />
  </svg>
);

const BarButton: FC<{ label: string; onClick: () => void; children: ReactNode }> = ({
  label,
  onClick,
  children,
}) => (
  <button
    type="button"
    title={label}
    aria-label={label}
    onClick={onClick}
    className="grid size-7 place-items-center rounded-full text-foreground/75 transition-colors hover:bg-foreground/10 hover:text-foreground"
  >
    {children}
  </button>
);

/** The floating orb window's whole page: the ball, draggable, and a slim bar on hover. */
export const OrbWidgetView: FC = () => {
  const [state, setState] = useState<OrbWidgetSnapshot | null>(null);

  useEffect(() => {
    let off: (() => void) | null = null;
    let disposed = false;
    void listen<unknown>(ORB_WIDGET_SPECTRUM_EVENT, ({ payload }) => {
      remote.level = unpackSpectrum(payload, remote.bands);
      remote.side = (payload as { side?: string } | null)?.side ?? null;
    }).then((unlisten) => {
      if (disposed) unlisten();
      else {
        const prev = off;
        off = () => {
          prev?.();
          unlisten();
        };
      }
    });
    void listen<OrbWidgetSnapshot>(ORB_WIDGET_STATE_EVENT, ({ payload }) => setState(payload)).then(
      (unlisten) => {
        if (disposed) unlisten();
        else {
          const prev = off;
          off = () => {
            prev?.();
            unlisten();
          };
        }
        void emitTo("main", ORB_WIDGET_READY_EVENT).catch(() => undefined);
      },
    );
    return () => {
      disposed = true;
      off?.();
    };
  }, []);

  // Remember where it was dropped, in logical pixels, for the next time it opens.
  useEffect(() => {
    const win = getCurrentWindow();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let off: (() => void) | null = null;
    let disposed = false;
    void win
      .onMoved(({ payload }) => {
        clearTimeout(timer);
        timer = setTimeout(() => {
          void win.scaleFactor().then((k) => {
            try {
              localStorage.setItem(
                ORB_WIDGET_POSITION_KEY,
                JSON.stringify({ x: payload.x / k, y: payload.y / k }),
              );
            } catch {
              // Storage blocked: it opens in the default corner next time.
            }
          });
        }, 250);
      })
      .then((unlisten) => {
        if (disposed) unlisten();
        else off = unlisten;
      });
    return () => {
      disposed = true;
      clearTimeout(timer);
      off?.();
    };
  }, []);

  const dragHandlers = useWindowDrag();
  const paused = state?.paused ?? false;
  const orbState = (state?.orbState ?? null) as OrbStateName | null;
  return (
    <div className="group relative h-full w-full select-none">
      <div
        role="presentation"
        className="absolute cursor-grab active:cursor-grabbing"
        style={{
          left: ORB_WIDGET_CENTER.x - ORB_WIDGET_BALL / 2,
          top: ORB_WIDGET_CENTER.y - ORB_WIDGET_BALL / 2,
          touchAction: "none",
        }}
        {...dragHandlers}
        onDoubleClick={() => send("back")}
      >
        <OrbBall
          orbState={orbState}
          muted={state?.muted ?? false}
          size={ORB_WIDGET_BALL}
          read={readRemote}
        />
      </div>
      {/* Centred under the ball and tucked into its lower edge. */}
      <div
        className={cn(
          "absolute left-1/2 flex -translate-x-1/2 items-center gap-0.5 rounded-full border border-border/60 bg-background/85 px-1 py-0.5 shadow-sm backdrop-blur",
          "opacity-55 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100",
        )}
        style={{ top: ORB_WIDGET_CENTER.y + 26 }}
      >
        <BarButton label={paused ? "Resume" : "Pause"} onClick={() => send("toggle-pause")}>
          {paused ? <PlayIcon className="size-3.5 fill-current" /> : <PauseBars />}
        </BarButton>
        <BarButton label="Stop the reply" onClick={() => send("stop")}>
          <SquareIcon className="size-3 fill-current" />
        </BarButton>
        <BarButton label="Back to Unsloth" onClick={() => send("back")}>
          <Maximize2Icon className="size-3.5" />
        </BarButton>
      </div>
    </div>
  );
};
