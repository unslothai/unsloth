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
import { RefreshGlyph } from "@/lib/refresh-icon";
import { cn } from "@/lib/utils";
import {
  ArrowExpand01Icon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  ArrowShrink01Icon,
  ArrowUpRight01Icon,
  Cancel01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { openUrl } from "@tauri-apps/plugin-opener";
import {
  type ComponentProps,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import {
  type BrowserRect,
  BrowserSession,
  type BrowserSnapshot,
  displayAddress,
  logicalRect,
  resolveAddress,
  shouldSuspendSlot,
} from "./browser-session";
import { useDesktopBrowserStore } from "./browser-store";
import { nativeBrowserTransport } from "./native-transport";

const START_URL = "https://duckduckgo.com/";

function ToolbarButton({
  label,
  className,
  children,
  ...props
}: ComponentProps<"button"> & { label: string; children: ReactNode }) {
  // a native title, not a Radix tooltip: a popper over the page suspends the native view.
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "flex size-8 shrink-0 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground disabled:pointer-events-none disabled:opacity-40",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}
const FLOATING_LAYERS =
  '[role="menu"], [role="dialog"], [data-radix-popper-content-wrapper], [data-slot="popover-content"], [data-slot="dropdown-menu-content"]';

export function BrowserPane({
  onClose,
  measureRef,
  expanded,
  onToggleExpanded,
}: {
  onClose: () => void;
  measureRef: { current: (() => void) | null };
  expanded: boolean;
  onToggleExpanded: () => void;
}) {
  const sessionRef = useRef<BrowserSession | null>(null);
  const [snapshot, setSnapshot] = useState<BrowserSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [address, setAddress] = useState(START_URL);
  const [addressFocused, setAddressFocused] = useState(false);
  const editingRef = useRef(false);
  const slotRef = useRef<HTMLDivElement>(null);
  const scaleRef = useRef(1);
  const openedRef = useRef(false);
  const activity = useDesktopBrowserStore((state) => state.activity);

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
    useDesktopBrowserStore.getState().attachSession(session);
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
      if (useDesktopBrowserStore.getState().session === session) {
        useDesktopBrowserStore.getState().attachSession(null);
      }
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
  const loading = Boolean(snapshot?.loading);
  return (
    <div
      className="flex h-full min-h-0 min-w-0 flex-col overflow-hidden border-l bg-background"
      data-testid="desktop-browser-pane"
    >
      <div
        className="flex shrink-0 items-center gap-1 px-2 py-2 pt-[calc(0.5rem+var(--studio-content-top-inset,0px))] pr-[calc(0.5rem+var(--studio-chat-header-right-inset,var(--studio-window-control-inset,0px)))]"
        aria-label="Browser controls"
      >
        <ToolbarButton
          label="Back"
          disabled={!snapshot?.canGoBack}
          onClick={() => action("back")}
        >
          <HugeiconsIcon
            icon={ArrowLeft01Icon}
            strokeWidth={2}
            className="size-4"
          />
        </ToolbarButton>
        <ToolbarButton
          label="Forward"
          disabled={!snapshot?.canGoForward}
          onClick={() => action("forward")}
        >
          <HugeiconsIcon
            icon={ArrowRight01Icon}
            strokeWidth={2}
            className="size-4"
          />
        </ToolbarButton>
        <ToolbarButton
          label={loading ? "Stop" : "Reload"}
          onClick={() => action(loading ? "stop" : "reload")}
        >
          {loading ? (
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={2}
              className="size-4"
            />
          ) : (
            <RefreshGlyph className="size-4" />
          )}
        </ToolbarButton>
        <form
          className="min-w-0 flex-1 px-1"
          onSubmit={(event) => {
            event.preventDefault();
            go(address);
          }}
        >
          <input
            data-testid="desktop-browser-address"
            aria-label="Website address or search"
            value={addressFocused ? address : displayAddress(address)}
            onChange={(event) => {
              setAddress(event.target.value);
              editingRef.current = true;
            }}
            onFocus={(event) => {
              editingRef.current = true;
              setAddressFocused(true);
              // select on the next frame, after the full address replaces the short one, so the selection covers the new text.
              const input = event.currentTarget;
              requestAnimationFrame(() => input.select());
            }}
            onBlur={() => {
              editingRef.current = false;
              setAddressFocused(false);
            }}
            className="h-8 w-full min-w-0 rounded-full border border-transparent bg-muted/40 px-3 text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground hover:bg-muted/60 focus:border-ring focus:bg-background"
            placeholder="Website or search DuckDuckGo"
            spellCheck={false}
          />
        </form>
        <ToolbarButton
          label="Open externally"
          onClick={() => {
            try {
              void openUrl(snapshot?.url || resolveAddress(address)).catch(
                (cause) => setError(String(cause)),
              );
            } catch (cause) {
              setError(String(cause));
            }
          }}
        >
          <HugeiconsIcon
            icon={ArrowUpRight01Icon}
            strokeWidth={2}
            className="size-4"
          />
        </ToolbarButton>
        <ToolbarButton
          label={expanded ? "Enter split view" : "Enter full view"}
          aria-pressed={expanded}
          onClick={onToggleExpanded}
        >
          <HugeiconsIcon
            icon={expanded ? ArrowShrink01Icon : ArrowExpand01Icon}
            strokeWidth={2}
            className="size-4"
          />
        </ToolbarButton>
        <ToolbarButton
          label="Close browser"
          data-testid="desktop-browser-close"
          onClick={onClose}
        >
          <HugeiconsIcon
            icon={Cancel01Icon}
            strokeWidth={2}
            className="size-4"
          />
        </ToolbarButton>
      </div>
      {/* one fixed-height row for the title or the agent's activity, since resizing the slot would reflow the page under every action. */}
      <div className="flex h-6 shrink-0 items-center gap-2 px-4 pb-1 text-xs">
        {activity ? (
          <output
            data-testid="desktop-browser-agent-activity"
            className="flex min-w-0 flex-1 items-center gap-2 text-primary"
          >
            <span
              aria-hidden={true}
              className="size-1.5 shrink-0 animate-pulse rounded-full bg-primary motion-reduce:animate-none"
            />
            <span className="min-w-0 flex-1 truncate font-medium">
              {activity.label}
            </span>
            {activity.done ? (
              <button
                type="button"
                data-testid="desktop-browser-handoff-done"
                onClick={activity.done}
                className="shrink-0 rounded-full bg-primary px-2.5 font-medium text-primary-foreground transition-colors hover:bg-primary/90"
              >
                I&apos;m done
              </button>
            ) : null}
            {activity.stop ? (
              <button
                type="button"
                aria-label="Stop the model"
                onClick={activity.stop}
                className="shrink-0 rounded-full px-2 font-medium transition-colors hover:bg-primary/10"
              >
                Stop
              </button>
            ) : null}
          </output>
        ) : (
          <span
            data-testid="desktop-browser-title"
            className="min-w-0 flex-1 truncate text-muted-foreground"
            title={snapshot?.title || snapshot?.url || "Browser"}
          >
            {snapshot?.title || snapshot?.url || "Browser"}
          </span>
        )}
      </div>
      {snapshot?.popupUrl ? (
        <output className="mx-2 mb-2 block shrink-0 rounded-xl bg-muted/40 px-3 py-2 text-xs">
          <p className="truncate" title={snapshot.popupUrl}>
            Page requested a popup: {snapshot.popupUrl}
          </p>
          <p className="text-muted-foreground">
            Sign-in popups that need their opener may only work in your external
            browser.
          </p>
          <div className="mt-1.5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => {
                if (snapshot?.popupUrl) go(snapshot.popupUrl);
              }}
              className="font-medium text-primary hover:underline"
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
              className="font-medium text-primary hover:underline"
            >
              Open externally
            </button>
            <button
              type="button"
              onClick={() => action("dismiss-popup")}
              className="text-muted-foreground hover:text-foreground"
            >
              Dismiss
            </button>
          </div>
        </output>
      ) : null}
      {error || snapshot?.error ? (
        <div
          role="alert"
          className="shrink-0 px-4 pb-1.5 text-xs text-destructive"
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
          data-agent-active={activity ? "true" : undefined}
          // the page is a native view over this box, so the agent's outline sits just outside it.
          className={cn(
            "h-full w-full rounded-sm transition-shadow",
            activity && "ring-2 ring-primary/50",
          )}
          aria-label="Native browser page"
        />
      </div>
    </div>
  );
}
