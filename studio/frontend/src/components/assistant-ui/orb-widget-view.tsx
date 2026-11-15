// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { emitTo, listen } from "@tauri-apps/api/event";
import {
  LogicalPosition,
  LogicalSize,
  PhysicalPosition,
  availableMonitors,
  getCurrentWindow,
} from "@tauri-apps/api/window";
import { Maximize2Icon, PlayIcon, SquareIcon } from "lucide-react";
import { type FC, type PointerEvent, type ReactNode, useEffect, useRef, useState } from "react";

import { OrbBall, type OrbStateName, type SpectrumReader } from "@/components/assistant-ui/voice-orb";
import { Button } from "@/components/ui/button";
import {
  type Area,
  type DragStart,
  ORB_WIDGET_ACTION_EVENT,
  ORB_WIDGET_APPROVAL_EVENT,
  ORB_WIDGET_BALL,
  ORB_WIDGET_BAR_TOP,
  ORB_WIDGET_CENTER,
  ORB_WIDGET_GROWN_HEIGHT,
  ORB_WIDGET_HEIGHT,
  ORB_WIDGET_POPUP_WIDTH,
  ORB_WIDGET_POSITION_KEY,
  ORB_WIDGET_READY_EVENT,
  ORB_WIDGET_SPECTRUM_EVENT,
  ORB_WIDGET_STATE_EVENT,
  ORB_WIDGET_WIDTH,
  type OrbWidgetAction,
  type OrbWidgetApproval,
  type OrbWidgetSnapshot,
  type Point,
  type PopupPlacement,
  applyOrbWidgetTheme,
  clampToScreens,
  dragTarget,
  placeApprovalPopup,
  restoreAfterPopup,
  sameApproval,
  sameLabels,
  sameTheme,
  unpackSpectrum,
} from "@/features/chat/voice/orb-widget";
import { SPECTRUM_BANDS } from "@/features/chat/voice/voice-spectrum";
import { cn } from "@/lib/utils";

const remote = {
  bands: new Float32Array(SPECTRUM_BANDS),
  level: 0,
  side: null as string | null,
};
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
    return {
      x: m.position.x / k,
      y: m.position.y / k,
      width: m.size.width / k,
      height: m.size.height / k,
      visibleBottom: (m.workArea.position.y + m.workArea.size.height) / k,
    };
  });
}

// By hand, not startDragging: the system drag keeps a window below the menu bar.
function useWindowDrag(pad: { current: number }) {
  const screens = useRef<Area[]>([]);
  const drag = useRef<{
    start: DragStart;
    moving: boolean;
    target: Point | null;
    sent: Point | null;
  } | null>(null);
  const frame = useRef(0);
  useEffect(() => {
    void screenAreas().then((s) => (screens.current = s));
    return () => cancelAnimationFrame(frame.current);
  }, []);
  const flush = () => {
    frame.current = 0;
    const d = drag.current;
    if (!d?.target) return;
    const next = clampToScreens(d.target, screens.current, {
      x: ORB_WIDGET_CENTER.x,
      y: ORB_WIDGET_CENTER.y + pad.current,
    });
    if (d.sent && d.sent.x === next.x && d.sent.y === next.y) return;
    d.sent = next;
    const t0 = performance.now();
    void getCurrentWindow()
      .setPosition(new LogicalPosition(next.x, next.y))
      .then(() => {
        if (import.meta.env.DEV) console.debug(`orb widget move ${(performance.now() - t0).toFixed(1)} ms`);
      })
      .catch(() => undefined);
  };
  const end = () => {
    drag.current = null;
  };
  return {
    onPointerDown: (e: PointerEvent<HTMLDivElement>) => {
      // The second press of a double-click is the double-click, not a new drag.
      if (e.button !== 0 || e.detail > 1) return;
      e.preventDefault();
      e.currentTarget.setPointerCapture(e.pointerId);
      drag.current = {
        start: {
          screen: { x: e.screenX, y: e.screenY },
          grab: { x: e.clientX, y: e.clientY },
        },
        moving: false,
        target: null,
        sent: null,
      };
      void screenAreas().then((s) => (screens.current = s));
    },
    onPointerMove: (e: PointerEvent<HTMLDivElement>) => {
      const d = drag.current;
      if (!d) return;
      const target = dragTarget(d.start, { x: e.screenX, y: e.screenY }, d.moving);
      if (!target) return;
      d.moving = true;
      d.target = target;
      if (!frame.current) frame.current = requestAnimationFrame(flush);
    },
    onPointerUp: end,
    onPointerCancel: end,
  };
}

const send = (action: OrbWidgetAction) => {
  void emitTo("main", ORB_WIDGET_ACTION_EVENT, action).catch(() => undefined);
};

const decide = (id: string, decision: "allow" | "deny") => {
  void emitTo("main", ORB_WIDGET_APPROVAL_EVENT, { id, decision }).catch(() => undefined);
};

async function visibleFrames(): Promise<Area[]> {
  const monitors = await availableMonitors().catch(() => []);
  return monitors.map((m) => ({
    x: m.workArea.position.x,
    y: m.workArea.position.y,
    width: m.workArea.size.width,
    height: m.workArea.size.height,
  }));
}

// Placed in physical pixels, so monitors at different scales keep the card inside the widget's frame.
function useApprovalRoom(
  showing: boolean,
  pad: { current: number },
  shift: { current: Point },
): PopupPlacement | null {
  const [place, setPlace] = useState<PopupPlacement | null>(null);
  useEffect(() => {
    let cancelled = false;
    const win = getCurrentWindow();
    void (async () => {
      const [k, at] = await Promise.all([win.scaleFactor(), win.outerPosition()]);
      if (cancelled) return;
      if (showing) {
        const next = placeApprovalPopup({ x: at.x, y: at.y }, await visibleFrames(), k);
        if (cancelled) return;
        // Set before the move, so the saved spot never takes the shifted one.
        shift.current = next.shift;
        await win.setSize(new LogicalSize(ORB_WIDGET_WIDTH, ORB_WIDGET_GROWN_HEIGHT));
        await win.setPosition(new PhysicalPosition(next.at.x, next.at.y));
        pad.current = next.pad;
        setPlace(next);
      } else {
        const moved = shift.current;
        shift.current = { x: 0, y: 0 };
        pad.current = 0;
        setPlace(null);
        if (moved.x || moved.y) {
          const back = restoreAfterPopup({ x: at.x, y: at.y }, moved);
          await win.setPosition(new PhysicalPosition(back.x, back.y));
        }
        await win.setSize(new LogicalSize(ORB_WIDGET_WIDTH, ORB_WIDGET_HEIGHT));
      }
    })().catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [showing, pad, shift]);
  return place;
}

const PILL = "h-[calc(26px*var(--ui-space-scale,1))] rounded-full px-2.5 text-ui-11";

const ApprovalPopup: FC<{
  approval: OrbWidgetApproval;
  place: PopupPlacement;
}> = ({ approval, place }) => (
  <div
    role="dialog"
    aria-label={approval.line}
    className="absolute flex flex-col gap-1 rounded-lg border border-border/40 bg-background/95 px-1.5 py-1 shadow-sm"
    style={{
      left: place.popupX,
      top: place.popupY,
      width: ORB_WIDGET_POPUP_WIDTH,
    }}
  >
    <p className="truncate px-0.5 text-ui-11 leading-4 text-foreground/80" title={approval.line}>
      {approval.brief}
    </p>
    <div className="flex items-center gap-1">
      <Button size="xs" className={PILL} onClick={() => decide(approval.id, "allow")}>
        {approval.allow}
      </Button>
      <Button size="xs" variant="destructive" className={PILL} onClick={() => decide(approval.id, "deny")}>
        {approval.deny}
      </Button>
      {approval.more ? <span className="ml-auto pr-0.5 text-ui-11 text-muted-foreground">{approval.more}</span> : null}
    </div>
  </div>
);

const PauseBars: FC = () => (
  <svg aria-hidden={true} viewBox="0 0 16 16" className="size-3.5" fill="currentColor">
    <rect x="3.5" y="2.5" width="2.6" height="11" rx="1.3" />
    <rect x="9.9" y="2.5" width="2.6" height="11" rx="1.3" />
  </svg>
);

const BarButton: FC<{
  label: string;
  onClick: () => void;
  children: ReactNode;
}> = ({ label, onClick, children }) => (
  <button
    type="button"
    title={label}
    aria-label={label}
    onClick={onClick}
    className="grid size-7 place-items-center rounded-full text-foreground/75 transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
  >
    {children}
  </button>
);

export const OrbWidgetView: FC = () => {
  const [state, setState] = useState<OrbWidgetSnapshot | null>(null);
  const pad = useRef(0);
  const shift = useRef<Point>({ x: 0, y: 0 });

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
    void listen<OrbWidgetSnapshot>(ORB_WIDGET_STATE_EVENT, ({ payload }) =>
      // Only a real change re-renders; the pulse and the wave never go through React.
      setState((prev) =>
        prev &&
        prev.orbState === payload.orbState &&
        prev.paused === payload.paused &&
        prev.muted === payload.muted &&
        prev.active === payload.active &&
        sameApproval(prev.approval ?? null, payload.approval ?? null) &&
        sameLabels(prev.labels, payload.labels) &&
        sameTheme(prev.theme, payload.theme)
          ? prev
          : payload,
      ),
    ).then((unlisten) => {
      if (disposed) unlisten();
      else {
        const prev = off;
        off = () => {
          prev?.();
          unlisten();
        };
      }
      void emitTo("main", ORB_WIDGET_READY_EVENT).catch(() => undefined);
    });
    return () => {
      disposed = true;
      off?.();
    };
  }, []);

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
                JSON.stringify({
                  x: (payload.x - shift.current.x) / k,
                  y: (payload.y - shift.current.y) / k,
                }),
              );
            } catch {
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

  const dark = state?.theme?.dark;
  const palette = state?.theme?.palette ?? null;
  useEffect(() => {
    if (dark === undefined) return;
    applyOrbWidgetTheme(document.documentElement, { dark, palette });
  }, [dark, palette]);

  const approval = state?.active ? (state.approval ?? null) : null;
  const place = useApprovalRoom(approval !== null, pad, shift);
  const labels = state?.labels;
  const offset = place?.pad ?? 0;
  const dragHandlers = useWindowDrag(pad);
  const paused = state?.paused ?? false;
  const orbState = (state?.orbState ?? null) as OrbStateName | null;
  return (
    <div className="relative h-full w-full select-none">
      {approval && place ? <ApprovalPopup approval={approval} place={place} /> : null}
      <div className="group absolute inset-x-0" style={{ top: offset, height: ORB_WIDGET_HEIGHT }}>
        <div
          className={cn(
            "absolute left-1/2 flex -translate-x-1/2 items-center gap-0.5 rounded-full border border-border/60 bg-background/85 px-1 py-0.5 shadow-sm",
            "opacity-55 transition-opacity duration-150 group-hover:opacity-100 focus-within:opacity-100",
          )}
          style={{ top: ORB_WIDGET_BAR_TOP }}
        >
          <BarButton label={(paused ? labels?.resume : labels?.pause) ?? ""} onClick={() => send("toggle-pause")}>
            {paused ? <PlayIcon className="size-3.5 fill-current" /> : <PauseBars />}
          </BarButton>
          <BarButton label={labels?.stop ?? ""} onClick={() => send("stop")}>
            <SquareIcon className="size-3 fill-current" />
          </BarButton>
          <BarButton label={labels?.back ?? ""} onClick={() => send("back")}>
            <Maximize2Icon className="size-3.5" />
          </BarButton>
        </div>
        <div
          role="presentation"
          className="absolute z-10 cursor-grab active:cursor-grabbing"
          style={{
            left: ORB_WIDGET_CENTER.x - ORB_WIDGET_BALL / 2,
            top: ORB_WIDGET_CENTER.y - ORB_WIDGET_BALL / 2,
            touchAction: "none",
          }}
          {...dragHandlers}
          onDoubleClick={() => send("back")}
        >
          <OrbBall orbState={orbState} muted={state?.muted ?? false} size={ORB_WIDGET_BALL} read={readRemote} />
        </div>
      </div>
    </div>
  );
};
