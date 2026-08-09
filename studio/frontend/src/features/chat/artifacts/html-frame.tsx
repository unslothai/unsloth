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
  const [blocked, setBlocked] = useState<{ uris: string[]; hosts: string[] }>({
    uris: [],
    hosts: [],
  });
  // Granted by the banner button alone, for this code only. Nothing the canvas
  // sends may set it, or a blocked page could talk its way onto the network.
  const [grantedForCanvas, setGrantedForCanvas] = useState(false);
  useEffect(() => {
    setGrantedForCanvas(false);
  }, [code]);
  const networkAllowed = networkAccessEnabled || grantedForCanvas;
  const artifactHtml = useMemo(() => buildArtifactSrcDoc(code), [code]);
  const src = useMemo(() => {
    const query = new URLSearchParams({ v: hashArtifactCode(code) });
    // Never put the auth token in the URL: in-frame code can read location.href.
    if (networkAllowed) {
      query.set("allow_network", "1");
    }
    return apiUrl(`/api/inference/artifact-preview-frame?${query.toString()}`);
  }, [networkAllowed, code]);
  // Feed only parent-initiated loads, so a self-navigated frame can't self-upgrade.
  const pendingPostRef = useRef(false);
  useEffect(() => {
    pendingPostRef.current = true;
    setBlocked({ uris: [], hosts: [] });
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
        const uri = event.data.blockedURI;
        const host = blockedHost(uri);
        if (!host) return;
        setBlocked((current) =>
          current.uris.includes(uri)
            ? current
            : {
                uris: [...current.uris, uri],
                hosts: current.hosts.includes(host)
                  ? current.hosts
                  : [...current.hosts, host],
              },
        );
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
  }, [postArtifactHtml]);

  const showBlockedBanner = !networkAllowed && blocked.uris.length > 0;
  const shownHosts = blocked.hosts.slice(0, BLOCKED_HOSTS_SHOWN).join(", ");
  const blockedFrom =
    blocked.hosts.length > BLOCKED_HOSTS_SHOWN ? `${shownHosts}…` : shownHosts;

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
              blocked.uris.length === 1
                ? "settings.chat.artifacts.blockedBanner"
                : "settings.chat.artifacts.blockedBannerPlural",
              { count: blocked.uris.length, hosts: blockedFrom },
            )}
          </span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => setGrantedForCanvas(true)}
          >
            {t("settings.chat.artifacts.blockedBannerAction")}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
