// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useEffect, useRef, useState } from "react";
import { isWebUrl } from "./address";
import { browserFrameUrl } from "./api";
import { type AnnotateEvent, type FrameMessage, parseFrameMessage } from "./frame-message";

export type { FrameMessage };

let loadCounter = 0;

const frames = new Map<string, HTMLIFrameElement>();

export type FrameCommand =
  | { command: "find"; query: string }
  | { command: "findStep"; delta: -1 | 1 }
  | { command: "snapshot" }
  | { command: "zoom"; value: number }
  | { command: "mute"; on: boolean }
  | { command: "annotateInstall"; code: string }
  | { command: "annotate"; on: boolean; color?: string }
  | { command: "annotateForget"; id: number }
  | { command: "annotateNumbers"; numbers: Array<[number, number]> }
  | { command: "agent"; id: string; op: Record<string, unknown>; code?: string }
  | { command: "resourceResult"; id: string; result: unknown };

type AgentReply = { installed: boolean; result: Record<string, unknown> | null };
const agentRequests = new Map<string, (reply: AgentReply | null) => void>();
let agentCounter = 0;

/** Runs a browser agent command in `tabId`'s page; null when the page didn't answer. */
export function requestFrameAgent(
  tabId: string,
  op: Record<string, unknown>,
  code?: string,
  timeoutMs = 5000,
): Promise<AgentReply | null> {
  return new Promise((resolve) => {
    // Not randomUUID: Studio on a LAN address is not a secure context.
    const id = `a${++agentCounter}-${Math.random().toString(36).slice(2)}`;
    const finish = (reply: AgentReply | null) => {
      clearTimeout(timer);
      agentRequests.delete(id);
      resolve(reply);
    };
    const timer = setTimeout(() => finish(null), timeoutMs);
    agentRequests.set(id, finish);
    if (!sendFrameCommand(tabId, { command: "agent", id, op, ...(code ? { code } : {}) })) finish(null);
  });
}

const annotateListeners = new Map<string, (event: AnnotateEvent) => void>();

export function onFrameAnnotate(tabId: string, listener: (event: AnnotateEvent) => void): () => void {
  annotateListeners.set(tabId, listener);
  return () => {
    if (annotateListeners.get(tabId) === listener) annotateListeners.delete(tabId);
  };
}

const snapshotRequests = new Map<string, (html: string | null) => void>();

/** The page's markup as it is now (see `snapshot` in routes/browser.py); null if it can't say. */
export function requestFrameSnapshot(tabId: string): Promise<string | null> {
  return new Promise((resolve) => {
    snapshotRequests.get(tabId)?.(null);
    const finish = (html: string | null) => {
      clearTimeout(timer);
      if (snapshotRequests.get(tabId) === finish) snapshotRequests.delete(tabId);
      resolve(html);
    };
    const timer = setTimeout(() => finish(null), 5000);
    snapshotRequests.set(tabId, finish);
    if (!sendFrameCommand(tabId, { command: "snapshot" })) finish(null);
  });
}

export function frameRect(tabId: string): DOMRect | null {
  return frames.get(tabId)?.getBoundingClientRect() ?? null;
}

export function sendFrameCommand(tabId: string, command: FrameCommand): boolean {
  const target = frames.get(tabId)?.contentWindow;
  if (!target) return false;
  target.postMessage({ type: "unsloth:browser-command", ...command }, "*");
  return true;
}

// A page's own API reads at once: more wait their turn rather than fail.
const RESOURCE_SLOTS = 4;
const RESOURCE_QUEUE = 64;
const RESOURCE_TIMEOUT_MS = 45_000;

function sameWebOrigin(url: string, pageUrl: string): boolean {
  try {
    return isWebUrl(url) && new URL(url).origin === new URL(pageUrl).origin;
  } catch {
    return false;
  }
}

/** The page's same-origin GETs (routes/browser_network.py), read by the backend with Studio's auth, which
 *  never enters the sandbox. `pageUrl` is the panel's address for the page, not one the page claims. */
function resourceRelay(reply: (id: string, result: unknown) => void) {
  const active = new Map<string, AbortController>();
  const waiting: Array<{ id: string; run: () => void }> = [];
  let closed = false;
  const next = () => {
    while (!closed && active.size < RESOURCE_SLOTS && waiting.length) waiting.shift()?.run();
  };
  return {
    read(message: { id: string; url: string; accept: string }, pageUrl: string | null) {
      const { id, url, accept } = message;
      if (!pageUrl || !sameWebOrigin(url, pageUrl)) {
        reply(id, { error: "not this page's site" });
        return;
      }
      if (active.has(id) || waiting.some((item) => item.id === id)) return;
      if (waiting.length >= RESOURCE_QUEUE) {
        reply(id, { error: "too many requests" });
        return;
      }
      waiting.push({
        id,
        run: () => {
          const controller = new AbortController();
          let timedOut = false;
          // Its own timer, since a timeout still answers the page while the page's own cancel doesn't.
          const timer = setTimeout(() => {
            timedOut = true;
            controller.abort();
          }, RESOURCE_TIMEOUT_MS);
          active.set(id, controller);
          authFetch("/api/browser/resource", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ page_url: pageUrl, url, accept }),
            signal: controller.signal,
          })
            .then(async (response) => (response.ok ? response.json() : { error: `HTTP ${response.status}` }))
            .catch(() => ({ error: "failed" }))
            .then((result) => {
              // Cancelled by the page: it no longer waits. Timed out: it still does.
              if (!closed && (timedOut || !controller.signal.aborted)) reply(id, result);
            })
            .finally(() => {
              clearTimeout(timer);
              active.delete(id);
              next();
            });
        },
      });
      next();
    },
    cancel(id: string) {
      const queued = waiting.findIndex((item) => item.id === id);
      if (queued >= 0) waiting.splice(queued, 1);
      active.get(id)?.abort();
    },
    close() {
      closed = true;
      waiting.length = 0;
      for (const controller of active.values()) controller.abort();
    },
  };
}

// Messages that open tabs or apps, or run shortcuts. Pages can post anything, so check them here.
const isUserAction = (message: FrameMessage) =>
  (message.type === "navigate" && message.newTab === true) ||
  message.type === "external" ||
  message.type === "shortcut";

// Activation outlives the click that loaded the page, so user actions are also rate limited, per page.
const USER_ACTION_INTERVAL_MS = 1000;

/** Recent user activation in the focused frame; without the API, focus opens a tab but never the system browser. */
function userActive(frame: HTMLIFrameElement, message: FrameMessage): boolean {
  if (document.activeElement !== frame) return false;
  const active = navigator.userActivation?.isActive;
  return active ?? message.type !== "external";
}

/** A page written into the sandbox shell (opaque origin, so it can't reach Studio's storage or API). */
export function PageFrame({
  html,
  url,
  base,
  refresh,
  title,
  tabId,
  zoom = 1,
  muted = false,
  onMessage,
}: {
  tabId: string;
  zoom?: number;
  muted?: boolean;
  html: string;
  url: string | null;
  base: string | null;
  refresh?: { delay: number; url: string } | null;
  title: string;
  onMessage: (message: FrameMessage) => void;
}) {
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const onMessageRef = useRef(onMessage);
  const tabIdRef = useRef(tabId);
  const urlRef = useRef(url);
  useEffect(() => {
    onMessageRef.current = onMessage;
    tabIdRef.current = tabId;
    urlRef.current = url;
  });
  const [loadId] = useState(() => String(++loadCounter));
  const postedRef = useRef(false);

  useEffect(() => {
    // A page loads and leaves once; repeats are spam.
    let loaded = false;
    let left = false;
    let lastUserAction = Number.NEGATIVE_INFINITY;
    const relay = resourceRelay((id, result) => sendFrameCommand(tabIdRef.current, { command: "resourceResult", id, result }));
    const listener = (event: MessageEvent) => {
      const frame = frameRef.current;
      if (!frame || event.source !== frame.contentWindow) return;
      let message = parseFrameMessage(event.data);
      if (!message) return;
      if (message.type === "resource") {
        relay.read(message, urlRef.current);
        return;
      }
      if (message.type === "resourceCancel") {
        relay.cancel(message.id);
        return;
      }
      if (message.type === "annotate") {
        annotateListeners.get(tabIdRef.current)?.(message.event);
        return;
      }
      if (message.type === "snapshot") {
        snapshotRequests.get(tabIdRef.current)?.(message.html);
        return;
      }
      if (message.type === "agent") {
        agentRequests.get(message.id)?.({ installed: message.installed, result: message.result });
        return;
      }
      if (isUserAction(message)) {
        if (!userActive(frame, message)) return;
        // Shortcuts too: a script can queue many before the first one moves focus off the page.
        const now = performance.now();
        if (now - lastUserAction < USER_ACTION_INTERVAL_MS) return;
        lastUserAction = now;
      }
      if (message.type === "loaded") {
        if (loaded) message = { type: "title", title: message.title };
        loaded = true;
      } else if ((message.type === "navigate" && !message.newTab) || message.type === "reload") {
        if (left) return;
        left = true;
        // An unrequested redirect replaces this page, as browsers keep it off Back.
        if (message.type === "navigate" && !userActive(frame, message)) message = { ...message, replace: true };
      }
      onMessageRef.current(message);
    };
    window.addEventListener("message", listener);
    return () => {
      window.removeEventListener("message", listener);
      relay.close();
    };
  }, []);

  useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;
    frames.set(tabId, frame);
    return () => {
      if (frames.get(tabId) === frame) frames.delete(tabId);
    };
  }, [tabId]);

  // The first zoom goes with the page; later changes are commands.
  const zoomRef = useRef(zoom);
  useEffect(() => {
    if (zoomRef.current === zoom) return;
    zoomRef.current = zoom;
    if (postedRef.current) sendFrameCommand(tabId, { command: "zoom", value: zoom });
  }, [tabId, zoom]);

  // Muting goes with the page too, so a muted tab's next page loads muted.
  const mutedRef = useRef(muted);
  useEffect(() => {
    if (mutedRef.current === muted) return;
    mutedRef.current = muted;
    if (postedRef.current) sendFrameCommand(tabId, { command: "mute", on: muted });
  }, [tabId, muted]);

  return (
    <iframe
      ref={frameRef}
      title={title}
      src={browserFrameUrl(loadId)}
      sandbox="allow-scripts allow-forms"
      referrerPolicy="no-referrer"
      className="size-full border-0 bg-white"
      onLoad={() => {
        if (postedRef.current) return;
        postedRef.current = true;
        frameRef.current?.contentWindow?.postMessage(
          {
            type: "unsloth:browser-html",
            html,
            url,
            base,
            refresh: refresh ?? null,
            zoom: zoomRef.current,
            muted: mutedRef.current,
          },
          "*",
        );
      }}
    />
  );
}
