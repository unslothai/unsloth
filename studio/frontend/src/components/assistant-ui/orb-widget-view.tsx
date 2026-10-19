// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { emitTo, listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Maximize2Icon, PlayIcon, SquareIcon } from "lucide-react";
import { useEffect, useState, type FC, type ReactNode } from "react";

import { OrbBall, type OrbStateName } from "@/components/assistant-ui/voice-orb";
import {
  ORB_WIDGET_ACTION_EVENT,
  ORB_WIDGET_POSITION_KEY,
  ORB_WIDGET_READY_EVENT,
  ORB_WIDGET_STATE_EVENT,
  type OrbWidgetAction,
  type OrbWidgetSnapshot,
} from "@/features/chat/voice/orb-widget";
import { cn } from "@/lib/utils";

const ORB_SIZE = 72;

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
    void listen<OrbWidgetSnapshot>(ORB_WIDGET_STATE_EVENT, ({ payload }) => setState(payload)).then(
      (unlisten) => {
        if (disposed) unlisten();
        else off = unlisten;
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

  const paused = state?.paused ?? false;
  const orbState = (state?.orbState ?? null) as OrbStateName | null;
  return (
    <div className="group flex h-full w-full select-none flex-col items-center justify-start pt-4.5">
      <div
        role="presentation"
        className="cursor-grab active:cursor-grabbing"
        onPointerDown={(e) => {
          if (e.button !== 0) return;
          e.preventDefault();
          void getCurrentWindow().startDragging();
        }}
        onDoubleClick={() => send("back")}
      >
        <OrbBall orbState={orbState} muted={state?.muted ?? false} size={ORB_SIZE} />
      </div>
      <div
        className={cn(
          "mt-3 flex items-center gap-0.5 rounded-full border border-border/60 bg-background/85 px-1 py-0.5 shadow-sm backdrop-blur",
          "opacity-55 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100",
        )}
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
