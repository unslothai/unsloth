// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { browserFrameUrl } from "./api";
import { type FrameMessage, parseFrameMessage } from "./frame-message";

export type { FrameMessage };

let loadCounter = 0;

// The page frame showing each tab, for commands from the panel (find, zoom).
const frames = new Map<string, HTMLIFrameElement>();

export type FrameCommand = { command: "find"; query: string; backwards?: boolean } | { command: "zoom"; value: number };

/** Send a command to a tab's page; false if the tab shows no page. */
export function sendFrameCommand(tabId: string, command: FrameCommand): boolean {
  const target = frames.get(tabId)?.contentWindow;
  if (!target) return false;
  target.postMessage({ type: "unsloth:browser-command", ...command }, "*");
  return true;
}

// Messages that open tabs or apps, or run shortcuts. Pages can post anything, so check them here.
const isUserAction = (message: FrameMessage) =>
  (message.type === "navigate" && message.newTab === true) ||
  message.type === "external" ||
  message.type === "shortcut";

// Activation outlives the click that loaded the page, so also rate limit.
const USER_ACTION_INTERVAL_MS = 1000;
let lastUserAction = Number.NEGATIVE_INFINITY;

/** True when the frame has focus and the user just clicked or typed. */
function allowUserAction(frame: HTMLIFrameElement, message: FrameMessage): boolean {
  if (document.activeElement !== frame) return false;
  if (!(navigator.userActivation?.isActive ?? true)) return false;
  // Shortcuts aren't rate limited.
  if (message.type === "shortcut") return true;
  const now = performance.now();
  if (now - lastUserAction < USER_ACTION_INTERVAL_MS) return false;
  lastUserAction = now;
  return true;
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
  onMessage,
}: {
  tabId: string;
  zoom?: number;
  html: string;
  url: string | null;
  base: string | null;
  refresh?: { delay: number; url: string } | null;
  title: string;
  onMessage: (message: FrameMessage) => void;
}) {
  const frameRef = useRef<HTMLIFrameElement | null>(null);
  const onMessageRef = useRef(onMessage);
  useEffect(() => {
    onMessageRef.current = onMessage;
  });
  // One shell per page.
  const [loadId] = useState(() => String(++loadCounter));
  const postedRef = useRef(false);

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      const frame = frameRef.current;
      if (!frame || event.source !== frame.contentWindow) return;
      const message = parseFrameMessage(event.data);
      if (!message) return;
      if (isUserAction(message) && !allowUserAction(frame, message)) return;
      onMessageRef.current(message);
    };
    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
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

  return (
    <iframe
      ref={frameRef}
      title={title}
      src={browserFrameUrl(loadId)}
      sandbox="allow-scripts allow-forms"
      referrerPolicy="no-referrer"
      className="size-full border-0 bg-white"
      onLoad={() => {
        // Post once.
        if (postedRef.current) return;
        postedRef.current = true;
        frameRef.current?.contentWindow?.postMessage(
          { type: "unsloth:browser-html", html, url, base, refresh: refresh ?? null, zoom: zoomRef.current },
          "*",
        );
      }}
    />
  );
}
