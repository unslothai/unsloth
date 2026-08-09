// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { apiUrl } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { hashArtifactCode } from "./types";

const HTML_FRAME_DEFAULT_HEIGHT = 400;
const HTML_FRAME_MAX_HEIGHT = 900;
const BLOCKED_HOSTS_SHOWN = 3;
// Far above what a real page trips, so the banner's count stays exact in
// practice; it only saturates for a canvas manufacturing violations.
const BLOCKED_URIS_TRACKED = 100;

type BlockedState = { code: string; uris: string[]; hosts: string[] };

const NOTHING_BLOCKED: BlockedState = { code: "", uris: [], hosts: [] };

// Reports from before a swap belong to the old canvas, so start over rather
// than appending to them. The cap is checked BEFORE the duplicate scan: past it
// this is O(1) per message, so a canvas posting unique URIs cannot make the
// parent rescan every stored string. Returning `current` unchanged is what lets
// React bail out of the re-render as well as the allocation.
function appendBlocked(
  current: BlockedState,
  code: string,
  uri: string,
  host: string,
): BlockedState {
  const mine = current.code === code ? current : { code, uris: [], hosts: [] };
  if (mine.uris.length >= BLOCKED_URIS_TRACKED || mine.uris.includes(uri)) {
    return mine === current ? current : mine;
  }
  return {
    code,
    uris: [...mine.uris, uri],
    hosts: mine.hosts.includes(host) ? mine.hosts : [...mine.hosts, host],
  };
}

function blockedHost(uri: string): string | null {
  try {
    return new URL(uri).host || null;
  } catch {
    return null;
  }
}

export type ArtifactViewMode = "preview" | "source";
export const ARTIFACT_VIEW_MODES: readonly ArtifactViewMode[] = [
  "preview",
  "source",
];

export function isArtifactViewMode(value: string): value is ArtifactViewMode {
  return (ARTIFACT_VIEW_MODES as readonly string[]).includes(value);
}

export function buildArtifactSrcDoc(code: string): string {
  const resizeScript = `<script>(()=>{const post=()=>parent.postMessage({chatArtifactHeight:document.documentElement.scrollHeight},"*");new ResizeObserver(post).observe(document.documentElement);window.addEventListener("load",post);post();})();</script>`;
  return `${code}\n${resizeScript}`;
}

// Preview iframes intentionally omit allow-downloads: generated canvases can
// offer their own UI, but downloads must go through Unsloth's explicit
// copy/download controls outside the no-same-origin sandbox.
export function ArtifactHtmlFrame({
  code,
  title = "HTML canvas preview",
  className,
  fill = false,
}: {
  code: string;
  title?: string;
  className?: string;
  fill?: boolean;
}) {
  const t = useT();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  // Every canvas honors this, fence or tool. Off by default; the standing half
  // of the gate, alongside the per-canvas grant below.
  const networkAccessEnabled = useChatRuntimeStore(
    (state) => state.allowArtifactNetworkAccess,
  );
  const [height, setHeight] = useState(HTML_FRAME_DEFAULT_HEIGHT);
  // Carries the code it was reported for, so a canvas swapped in place cannot
  // inherit the previous one's banner. Same reason the grant below stores code:
  // the [src] effect that used to clear this runs a render too late, and the
  // button in that stale render already closes over the new code.
  const [blocked, setBlocked] = useState<BlockedState>({
    code,
    uris: [],
    hosts: [],
  });
  const blockedForCanvas = blocked.code === code ? blocked : NOTHING_BLOCKED;
  // Granted by the banner button alone, and only for the exact code on screen
  // when it was clicked. Nothing the canvas sends may set it, or a blocked page
  // could talk its way onto the network. Comparing against the current code here
  // rather than resetting in an effect is what keeps the grant from leaking: an
  // effect runs after the DOM is updated, so the first render carrying new code
  // would still build src with allow_network=1 from the previous canvas' grant.
  const [grantedCode, setGrantedCode] = useState<string | null>(null);
  const grantedForCanvas = grantedCode === code;
  const networkAllowed = networkAccessEnabled || grantedForCanvas;
  const artifactHtml = useMemo(() => buildArtifactSrcDoc(code), [code]);
  // Identifies this load to the frame, which stamps its blocked reports with it.
  const codeVersion = useMemo(() => hashArtifactCode(code), [code]);
  const src = useMemo(() => {
    const query = new URLSearchParams({ v: codeVersion });
    // Never put the auth token in the URL: in-frame code can read location.href.
    if (networkAllowed) {
      query.set("allow_network", "1");
    }
    return apiUrl(`/api/inference/artifact-preview-frame?${query.toString()}`);
  }, [networkAllowed, codeVersion]);
  // Feed only parent-initiated loads, so a self-navigated frame can't self-upgrade.
  const pendingPostRef = useRef(false);
  useEffect(() => {
    pendingPostRef.current = true;
  }, [src]);
  const postArtifactHtml = useCallback(() => {
    if (!pendingPostRef.current) return;
    pendingPostRef.current = false;
    // Sandboxed frame has an opaque origin ("null"), so a wildcard target is
    // required; the payload only reaches this iframe's contentWindow.
    iframeRef.current?.contentWindow?.postMessage(
      { type: "unsloth:artifact-html", html: artifactHtml },
      "*",
    );
  }, [artifactHtml]);

  useEffect(() => {
    const handler = (event: MessageEvent) => {
      if (event.source !== iframeRef.current?.contentWindow) return;
      if (event.origin !== "null") return;
      if (event.data?.type === "unsloth:artifact-blocked") {
        // event.source survives the swap navigation, so a report from the
        // outgoing canvas would otherwise be tagged with the incoming code and
        // prompt a grant for a canvas that never hit the CSP. The frame stamps
        // the load it was served for; anything else is from a document we have
        // already navigated away from.
        if (event.data.v !== codeVersion) return;
        const uri = event.data.blockedURI;
        const host = blockedHost(uri);
        if (!host) return;
        setBlocked((current) => appendBlocked(current, code, uri, host));
        return;
      }
      if (typeof event.data?.chatArtifactHeight !== "number") return;
      setHeight(
        Math.min(
          Math.max(event.data.chatArtifactHeight, 160),
          HTML_FRAME_MAX_HEIGHT,
        ),
      );
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
    // `code`/`codeVersion` are listed so the handler always closes over the
    // canvas on screen, rather than relying on postArtifactHtml changing.
  }, [postArtifactHtml, code, codeVersion]);

  const showBlockedBanner = !networkAllowed && blockedForCanvas.uris.length > 0;
  const shownHosts = blockedForCanvas.hosts
    .slice(0, BLOCKED_HOSTS_SHOWN)
    .join(", ");
  const blockedFrom =
    blockedForCanvas.hosts.length > BLOCKED_HOSTS_SHOWN
      ? `${shownHosts}…`
      : shownHosts;

  return (
    <div className={cn("relative", fill ? "h-full" : undefined)}>
      <iframe
        ref={iframeRef}
        src={src}
        sandbox="allow-scripts"
        referrerPolicy="no-referrer"
        onLoad={postArtifactHtml}
        className={cn("block w-full border-0 bg-background", className)}
        style={{ height: fill ? "100%" : height }}
        title={title}
      />
      {showBlockedBanner ? (
        <div className="absolute inset-x-0 bottom-0 flex flex-wrap items-center justify-between gap-2 border-t bg-background/95 px-3 py-2 text-xs backdrop-blur">
          <span className="text-muted-foreground">
            {t(
              blockedForCanvas.uris.length === 1
                ? "settings.chat.artifacts.blockedBanner"
                : "settings.chat.artifacts.blockedBannerPlural",
              { count: blockedForCanvas.uris.length, hosts: blockedFrom },
            )}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setGrantedCode(code)}
          >
            {t("settings.chat.artifacts.blockedBannerAction")}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
