// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { browserFrameUrl } from "./api";

export type FrameMessage =
  | {
      type: "navigate";
      url: string;
      newTab?: boolean;
      background?: boolean;
      replace?: boolean;
      method?: "GET" | "POST";
      body?: string;
    }
  | { type: "external"; url: string }
  | { type: "loaded"; title: string; favicon: string | null }
  | { type: "title"; title: string }
  | { type: "url"; url: string }
  | { type: "reload" }
  | { type: "shortcut"; key: string; shift: boolean };

let loadCounter = 0;

/** A page written into the sandbox shell (opaque origin, so it can't reach Studio's storage or API). */
export function PageFrame({
  html,
  url,
  base,
  refresh,
  title,
  onMessage,
}: {
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
  // One shell per page: document.open() drops the shell's listener.
  const [loadId] = useState(() => String(++loadCounter));
  const postedRef = useRef(false);

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      if (!frameRef.current || event.source !== frameRef.current.contentWindow) return;
      const data = event.data as ({ source?: string } & FrameMessage) | null;
      if (!data || data.source !== "unsloth-browser") return;
      onMessageRef.current(data);
    };
    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
  }, []);

  return (
    <iframe
      ref={frameRef}
      title={title}
      src={browserFrameUrl(loadId)}
      sandbox="allow-scripts allow-forms"
      referrerPolicy="no-referrer"
      className="size-full border-0 bg-white"
      onLoad={() => {
        // Post once; the written page fires load again.
        if (postedRef.current) return;
        postedRef.current = true;
        frameRef.current?.contentWindow?.postMessage(
          { type: "unsloth:browser-html", html, url, base, refresh: refresh ?? null },
          "*",
        );
      }}
    />
  );
}
