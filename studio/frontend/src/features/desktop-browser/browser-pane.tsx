// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getModalLayer,
  subscribeModalLayer,
} from "@/components/ui/tooltip-modal-layer";
import {
  getAppliedInterfaceZoom,
  subscribeAppliedInterfaceZoom,
} from "@/features/settings";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { openUrl } from "@tauri-apps/plugin-opener";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type BrowserRect,
  BrowserSession,
  type BrowserSnapshot,
  logicalRect,
  resolveAddress,
  shouldSuspendSlot,
} from "./browser-session";
import { nativeBrowserTransport } from "./native-transport";

const START_URL = "https://duckduckgo.com/";
const FLOATING_LAYERS =
  '[role="menu"], [role="dialog"], [data-radix-popper-content-wrapper], [data-slot="popover-content"], [data-slot="dropdown-menu-content"]';

export function BrowserPane({
  onClose,
  measureRef,
}: { onClose: () => void; measureRef: { current: (() => void) | null } }) {
  const sessionRef = useRef<BrowserSession | null>(null);
  const [snapshot, setSnapshot] = useState<BrowserSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [address, setAddress] = useState(START_URL);
  const editingRef = useRef(false);
  const slotRef = useRef<HTMLDivElement>(null);
  const scaleRef = useRef(1);
  const openedRef = useRef(false);

  const measure = useCallback(() => {
    const slot = slotRef.current;
    const session = sessionRef.current;
    if (!slot || !session) return;
    const css = slot.getBoundingClientRect();
    const group = slot.closest('[data-slot="resizable-panel-group"]');
    const covered = shouldSuspendSlot(
      { x: css.left, y: css.top, width: css.width, height: css.height },
      group?.getBoundingClientRect().width ?? 0,
      getModalLayer(),
      Array.from(document.querySelectorAll<HTMLElement>(FLOATING_LAYERS))
        .filter((node) => !slot.contains(node))
        .map((node) => {
          const bounds = node.getBoundingClientRect();
          return {
            x: bounds.left,
            y: bounds.top,
            width: bounds.width,
            height: bounds.height,
          };
        }),
    );
    const rect: BrowserRect = logicalRect(
      { x: css.left, y: css.top, width: css.width, height: css.height },
      getAppliedInterfaceZoom(),
      window.devicePixelRatio,
      scaleRef.current,
      /Windows/.test(navigator.userAgent),
    );
    session.setBounds(rect, !covered);
    if (!openedRef.current) {
      openedRef.current = true;
      void session.open(START_URL, rect);
    }
  }, []);
  useEffect(() => {
    // StrictMode replays effect setup/cleanup without discarding hook state. Each
    // setup must own a fresh session; the old async open may still complete later.
    const session = new BrowserSession(nativeBrowserTransport);
    sessionRef.current = session;
    openedRef.current = false;
    measureRef.current = measure;
    const unsubscribe = session.subscribe((next, message) => {
      setSnapshot(next);
      setError(message);
      if (next?.url)
        setAddress((current) => (editingRef.current ? current : next.url));
    });
    // The browser's native page is deliberately outside React. Observe its slot, not
    // the message tree; no state update is needed for dragging or window resizing.
    const slot = slotRef.current;
    const group = slot?.closest('[data-slot="resizable-panel-group"]');
    const observer = new ResizeObserver(measure);
    if (slot) observer.observe(slot);
    if (group) observer.observe(group);
    const unsubscribeZoom = subscribeAppliedInterfaceZoom(measure);
    const unsubscribeModal = subscribeModalLayer(measure);
    const onResize = () => measure();
    window.addEventListener("resize", onResize);
    void getCurrentWindow()
      .scaleFactor()
      .then((value) => {
        scaleRef.current = value;
        measure();
      })
      .catch(console.error);
    // Radix poppers and menus can overlap the native child without setting the
    // body's modal pointer-events. Only portal layer additions/removals resample.
    const layers = new MutationObserver((records) => {
      if (
        records.some((record) => {
          if (record.type === "attributes")
            return (record.target as Element).matches(FLOATING_LAYERS);
          return Array.from(record.addedNodes)
            .concat(Array.from(record.removedNodes))
            .some(
              (node) =>
                node instanceof Element &&
                (node.matches(FLOATING_LAYERS) ||
                  Boolean(node.querySelector(FLOATING_LAYERS)) ||
                  Boolean(node.closest(FLOATING_LAYERS))),
            );
        })
      )
        measure();
    });
    layers.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["data-state"],
    });
    measure();
    const poll = window.setInterval(() => {
      void session.poll();
    }, 400);
    return () => {
      // Geometry callbacks after unmount see a cleared slot ref.
      window.clearInterval(poll);
      layers.disconnect();
      observer.disconnect();
      unsubscribeModal();
      unsubscribeZoom();
      window.removeEventListener("resize", onResize);
      if (measureRef.current === measure) measureRef.current = null;
      sessionRef.current = null;
      openedRef.current = false;
      unsubscribe();
      session.close();
    };
  }, [measure, measureRef]);

  const go = (value: string) => {
    try {
      void sessionRef.current?.navigate(resolveAddress(value));
      editingRef.current = false;
      setError(null);
    } catch (cause) {
      setError(String(cause));
    }
  };
  const action = (
    value: "back" | "forward" | "reload" | "stop" | "dismiss-popup",
  ) => {
    void sessionRef.current?.action(value);
  };
  return (
    <div
      className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden border-l bg-background"
      data-testid="desktop-browser-pane"
    >
      <div
        className="flex shrink-0 items-center gap-1 border-b px-2 py-2 pt-[calc(0.5rem+var(--studio-content-top-inset,0px))] pr-[calc(0.5rem+var(--studio-window-control-inset,0px))]"
        aria-label="Browser controls"
      >
        <button
          type="button"
          title="Back"
          aria-label="Back"
          disabled={!snapshot?.canGoBack}
          onClick={() => action("back")}
          className="rounded px-2 py-1 disabled:opacity-40"
        >
          ←
        </button>
        <button
          type="button"
          title="Forward"
          aria-label="Forward"
          disabled={!snapshot?.canGoForward}
          onClick={() => action("forward")}
          className="rounded px-2 py-1 disabled:opacity-40"
        >
          →
        </button>
        <button
          type="button"
          title={snapshot?.loading ? "Stop" : "Reload"}
          aria-label={snapshot?.loading ? "Stop" : "Reload"}
          onClick={() => action(snapshot?.loading ? "stop" : "reload")}
          className="rounded px-2 py-1"
        >
          {snapshot?.loading ? "×" : "↻"}
        </button>
        <form
          className="min-w-0 flex-1"
          onSubmit={(event) => {
            event.preventDefault();
            go(address);
          }}
        >
          <input
            data-testid="desktop-browser-address"
            aria-label="Website address or search"
            value={address}
            onChange={(event) => {
              setAddress(event.target.value);
              editingRef.current = true;
            }}
            onFocus={() => {
              editingRef.current = true;
            }}
            onBlur={() => {
              editingRef.current = false;
            }}
            className="h-8 w-full min-w-0 rounded border bg-background px-2 text-sm"
            placeholder="Website or search DuckDuckGo"
          />
        </form>
        <button
          type="button"
          title="Open externally"
          aria-label="Open externally"
          onClick={() => {
            try {
              void openUrl(snapshot?.url || resolveAddress(address)).catch(
                (cause) => setError(String(cause)),
              );
            } catch (cause) {
              setError(String(cause));
            }
          }}
          className="rounded px-2 py-1"
        >
          ↗
        </button>
        <button
          type="button"
          data-testid="desktop-browser-close"
          title="Close browser"
          aria-label="Close browser"
          onClick={onClose}
          className="rounded px-2 py-1"
        >
          ×
        </button>
      </div>
      <div
        data-testid="desktop-browser-title"
        className="shrink-0 truncate px-3 pb-1 text-xs text-muted-foreground"
        title={snapshot?.title || snapshot?.url || "Browser"}
      >
        {snapshot?.title || snapshot?.url || "Browser"}
      </div>
      {snapshot?.popupUrl ? (
        <output className="block shrink-0 border-b px-3 py-2 text-xs">
          <p className="truncate" title={snapshot.popupUrl}>
            Page requested a popup: {snapshot.popupUrl}
          </p>
          <p>
            Sign-in popups that need their opener may only work in your external
            browser.
          </p>
          <button
            type="button"
            onClick={() => {
              if (snapshot?.popupUrl) go(snapshot.popupUrl);
            }}
            className="mr-3 underline"
          >
            Open in this pane
          </button>
          <button
            type="button"
            onClick={() => {
              if (snapshot?.popupUrl) {
                void openUrl(snapshot.popupUrl).catch((cause) =>
                  setError(String(cause)),
                );
                action("dismiss-popup");
              }
            }}
            className="mr-3 underline"
          >
            Open externally
          </button>
          <button
            type="button"
            onClick={() => action("dismiss-popup")}
            className="underline"
          >
            Dismiss
          </button>
        </output>
      ) : null}
      {error || snapshot?.error ? (
        <div
          role="alert"
          className="shrink-0 px-3 py-1 text-xs text-destructive"
        >
          {error || snapshot?.error}
          {!snapshot && error ? (
            <button
              type="button"
              className="ml-2 underline"
              onClick={() => {
                openedRef.current = false;
                measure();
              }}
            >
              Retry
            </button>
          ) : null}
        </div>
      ) : null}
      <div
        className="min-h-0 flex-1 p-0 pr-2 pb-2"
        data-testid="desktop-browser-slot-container"
      >
        <div
          ref={slotRef}
          data-testid="desktop-browser-slot"
          data-session-id={snapshot?.sessionId}
          className="h-full w-full"
          aria-label="Native browser page"
        />
      </div>
    </div>
  );
}
