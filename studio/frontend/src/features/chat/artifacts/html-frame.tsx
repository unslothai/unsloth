// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { apiUrl } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { hashArtifactCode } from "./types";

const HTML_FRAME_DEFAULT_HEIGHT = 400;
const HTML_FRAME_MAX_HEIGHT = 900;

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
// offer their own UI, but downloads must go through Studio's explicit
// copy/download controls outside the no-same-origin sandbox.
export function ArtifactHtmlFrame({
  code,
  title = "HTML canvas preview",
  className,
  fill = false,
  // Only artifacts the user explicitly rendered (the render_html tool) may opt
  // into network mode. Auto-extracted fence cards default to no network so
  // prompt-injected assistant text can never reach the network-enabled frame.
  allowNetworkAccess = false,
}: {
  code: string;
  title?: string;
  className?: string;
  fill?: boolean;
  allowNetworkAccess?: boolean;
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const networkAccessEnabled = useChatRuntimeStore(
    (state) => state.allowArtifactNetworkAccess,
  );
  const [height, setHeight] = useState(HTML_FRAME_DEFAULT_HEIGHT);
  const artifactHtml = useMemo(() => buildArtifactSrcDoc(code), [code]);
  const src = useMemo(() => {
    const query = new URLSearchParams({ v: hashArtifactCode(code) });
    // No auth token is ever placed in the frame URL: untrusted canvas HTML runs
    // in this frame and could read it back via window.location.href. The
    // backend selects the network CSP from allow_network alone.
    if (allowNetworkAccess && networkAccessEnabled) {
      query.set("allow_network", "1");
    }
    return apiUrl(`/api/inference/artifact-preview-frame?${query.toString()}`);
  }, [allowNetworkAccess, networkAccessEnabled, code]);
  // Only feed loads WE initiated (mount or a src change). Untrusted canvas code
  // can self-navigate the frame to ?allow_network=1, which also fires onLoad;
  // reposting there would let a strict frame self-upgrade into a network one.
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

  return (
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
  );
}
