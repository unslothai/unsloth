// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  getAppliedInterfaceZoom,
  isImeComposing,
  subscribeAppliedInterfaceZoom,
  useInterfaceScaleStore,
} from "@/features/settings";
import {
  formatBindingValueLabel,
  isMacPlatform,
} from "@/features/settings/lib/keyboard-shortcuts";
import { INTERFACE_SCALE_RANGE } from "@/features/settings/stores/interface-scale-store";
import {
  shortcutOwningBinding,
  useKeyboardShortcutsStore,
} from "@/features/settings/stores/keyboard-shortcuts-store";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { Z_LAYER } from "@/lib/z-layers";
import { MinusIcon, PlusIcon } from "lucide-react";
import { useEffect, useState, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import {
  useZoomPopupStore,
  zoomInterface,
  zoomInterfaceFromChord,
} from "../lib/zoom-actions.ts";
import {
  ZOOM_CHORDS,
  type ZoomDirection,
  zoomChordTaken,
  zoomDirectionForKey,
} from "../lib/zoom-chords.ts";
import { createWheelZoomAccumulator } from "../lib/zoom-wheel.ts";

/** Hide delay after the last zoom, paused while hovered or focused. */
export const ZOOM_POPUP_HIDE_MS = 2500;

/** Same hover wash as the find bar. */
const ZOOM_BUTTON_CLASS =
  "hover:bg-[rgb(0_0_0_/_calc(0.06*var(--contrast-wash-gain,1)))] dark:hover:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))]";

/** Keeps focus where the user was typing. */
function keepFocus(event: { preventDefault: () => void }): void {
  event.preventDefault();
}

/** Tooltip with its chord, e.g. "Zoom in (⌘=)". */
function withChord(label: string, direction: ZoomDirection): string {
  const chord = formatBindingValueLabel(ZOOM_CHORDS[direction]);
  return chord ? `${label} (${chord})` : label;
}

function ZoomPopup() {
  const t = useT();
  const scale = useInterfaceScaleStore((s) => s.scale);
  const token = useZoomPopupStore((s) => s.token);
  const hide = useZoomPopupStore((s) => s.hide);
  // Divided back out below, so the popup keeps one on-screen size at every zoom.
  const pageZoom = useSyncExternalStore(
    subscribeAppliedInterfaceZoom,
    getAppliedInterfaceZoom,
  );
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const held = hovered || focused;

  // biome-ignore lint/correctness/useExhaustiveDependencies: each zoom restarts the timer.
  useEffect(() => {
    if (held) return;
    const timer = setTimeout(hide, ZOOM_POPUP_HIDE_MS);
    return () => clearTimeout(timer);
  }, [token, held, hide]);

  return (
    // In the find bar's corner, in front of it. pointer-events-auto survives a modal's body lock.
    <div
      className="interface-zoom-position pointer-events-auto fixed right-4"
      style={{ zIndex: Z_LAYER.ZOOM_POPUP }}
      // Keeps an open modal from treating this as an outside click.
      onPointerDown={(event) => event.stopPropagation()}
    >
      <div
        // biome-ignore lint/a11y/useSemanticElements: a labelled group of zoom controls, not a form.
        role="group"
        aria-label={t("shell.zoom.label")}
        data-find-skip=""
        data-testid="interface-zoom-popup"
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
        onFocus={() => setFocused(true)}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget))
            setFocused(false);
        }}
        onKeyDown={(event) => {
          if (event.key !== "Escape") return;
          event.preventDefault();
          event.stopPropagation();
          hide();
        }}
        style={pageZoom === 1 ? undefined : { zoom: 1 / pageZoom }}
        className="find-bar-surface flex h-11 items-center gap-0.5 rounded-full pr-1.5 pl-4 duration-100 animate-in fade-in-0"
      >
        <span className="min-w-[calc(2.875rem*var(--ui-space-scale,1))] pr-1.5 font-medium text-ui-14 tabular-nums">
          {scale}%
        </span>
        <Button
          variant="ghost"
          size="icon-sm"
          className={ZOOM_BUTTON_CLASS}
          disabled={scale <= INTERFACE_SCALE_RANGE.min}
          onMouseDown={keepFocus}
          onClick={() => zoomInterface(-1)}
          aria-label={t("shell.zoom.zoomOut")}
          title={withChord(t("shell.zoom.zoomOut"), -1)}
        >
          <MinusIcon
            strokeWidth={1.75}
            className="size-[calc(16px*var(--ui-space-scale,1))]"
          />
        </Button>
        <Button
          variant="ghost"
          size="icon-sm"
          className={ZOOM_BUTTON_CLASS}
          disabled={scale >= INTERFACE_SCALE_RANGE.max}
          onMouseDown={keepFocus}
          onClick={() => zoomInterface(1)}
          aria-label={t("shell.zoom.zoomIn")}
          title={withChord(t("shell.zoom.zoomIn"), 1)}
        >
          <PlusIcon
            strokeWidth={1.75}
            className="size-[calc(16px*var(--ui-space-scale,1))]"
          />
        </Button>
        <span aria-hidden="true" className="mx-1 h-5 w-px bg-border" />
        <Button
          variant="ghost"
          size="sm"
          className={`rounded-full px-2.5 text-muted-foreground text-ui-14 ${ZOOM_BUTTON_CLASS}`}
          disabled={scale === INTERFACE_SCALE_RANGE.default}
          onMouseDown={keepFocus}
          onClick={() => zoomInterface(0)}
          title={withChord(t("shell.zoom.reset"), 0)}
        >
          {t("shell.zoom.reset")}
        </Button>
      </div>
    </div>
  );
}

/** Announces each zoom. Always mounted, so the first zoom is announced too. */
function ZoomAnnouncer({ open }: { open: boolean }) {
  const t = useT();
  const scale = useInterfaceScaleStore((s) => s.scale);
  return (
    <output aria-live="polite" className="sr-only">
      {open ? t("shell.zoom.announce", { percent: String(scale) }) : ""}
    </output>
  );
}

/**
 * Desktop zoom: Cmd/Ctrl +, - and 0 on every platform, plus Ctrl+wheel off macOS. Each zoom shows
 * a popup in the find bar's corner. Browsers keep their own zoom.
 */
export function InterfaceZoom() {
  const open = useZoomPopupStore((s) => s.open);

  useEffect(() => {
    if (!isTauri) return;
    const mac = isMacPlatform();
    // Capture phase, so it also works from text fields.
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.defaultPrevented || isImeComposing(event)) return;
      const direction = zoomDirectionForKey(event, mac);
      if (direction === null) return;
      const { overrides } = useKeyboardShortcutsStore.getState();
      const owned = (value: string) =>
        shortcutOwningBinding(overrides, value) !== null;
      if (zoomChordTaken(event, direction, mac, owned)) return;
      event.preventDefault();
      zoomInterfaceFromChord(direction);
    };
    window.addEventListener("keydown", onKeyDown, true);
    // Bubble phase, so canvases with their own Ctrl+wheel zoom keep it.
    const wheelStep = createWheelZoomAccumulator();
    const onWheel = (event: WheelEvent) => {
      if (event.defaultPrevented || !event.ctrlKey) return;
      event.preventDefault();
      const direction = wheelStep(event, getAppliedInterfaceZoom());
      if (direction !== null) zoomInterface(direction);
    };
    if (!mac) window.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      window.removeEventListener("keydown", onKeyDown, true);
      window.removeEventListener("wheel", onWheel);
    };
  }, []);

  if (typeof document === "undefined") return null;
  // Portaled so a modal's inert/aria-hidden does not reach it.
  return createPortal(
    <>
      {isTauri && <ZoomAnnouncer open={open} />}
      {open && <ZoomPopup />}
    </>,
    document.body,
  );
}
