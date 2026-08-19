// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { useTheme } from "@/features/settings/stores/theme-store";
import { apiUrl, isTauri } from "@/lib/api-base";
import { openLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  callMcpUiTool,
  readMcpUiResource,
  type McpUiResource,
} from "../api/mcp-servers-api";
import type { McpUiEnvelope } from "../api/chat-adapter";

// Reported in the ui/initialize result so a view can adapt rather than guess.
const UI_PROTOCOL_VERSION = "2026-01-26";
const HOST_NAME = "Unsloth";
// No build-stamped version here, so this tracks the bridge itself.
const HOST_VERSION = "1.0.0";

const DEFAULT_HEIGHT = 320;
const MIN_HEIGHT = 120;
// Past this a widget scrolls rather than pushing the conversation off screen.
const MAX_HEIGHT = 900;

// The standard JSON-RPC codes; the spec adds none.
const INVALID_PARAMS = -32602;
const METHOD_NOT_FOUND = -32601;
const INTERNAL_ERROR = -32603;

type JsonRpcId = string | number;

interface JsonRpcMessage {
  jsonrpc: "2.0";
  id?: JsonRpcId;
  method?: string;
  params?: Record<string, unknown>;
}

function isJsonRpc(data: unknown): data is JsonRpcMessage {
  return (
    typeof data === "object" &&
    data !== null &&
    (data as { jsonrpc?: unknown }).jsonrpc === "2.0"
  );
}

// Height fallback for views that never send ui/notifications/size-changed; a
// reported size always wins over it.
const RESIZE_FALLBACK = `<script>(()=>{const post=()=>parent.postMessage({mcpAppHeight:document.documentElement.scrollHeight},"*");new ResizeObserver(post).observe(document.documentElement);window.addEventListener("load",post);post();})();</script>`;

/** Comma-joined declared domains for one CSP directive, or "" when undeclared. */
function domainParam(values: string[] | undefined): string {
  return Array.isArray(values) ? values.filter(Boolean).join(",") : "";
}

export interface McpAppFrameProps {
  /** Every call the widget makes is scoped to this server. */
  serverId: string;
  toolName: string;
  ui: McpUiEnvelope;
  /** The arguments the model called the tool with. */
  toolArgs?: Record<string, unknown>;
  /** The text the tool returned, replayed as the view's result content. */
  resultText?: string;
  /** Images the tool returned, replayed alongside that text. */
  resultImages?: { data: string; mimeType: string }[];
  /** Scopes stdio sessions to the conversation's own server process. */
  threadId?: string;
  sessionId?: string;
  className?: string;
}

export function McpAppFrame({
  serverId,
  toolName,
  ui,
  toolArgs,
  resultText,
  resultImages,
  threadId,
  sessionId,
  className,
}: McpAppFrameProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const { resolved: theme } = useTheme();
  const [resource, setResource] = useState<McpUiResource | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);

  const { resourceUri } = ui;

  useEffect(() => {
    let cancelled = false;
    setResource(null);
    setError(null);
    readMcpUiResource(serverId, resourceUri, { threadId, sessionId })
      .then((loaded) => {
        if (!cancelled) setResource(loaded);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [serverId, resourceUri, threadId, sessionId]);

  // The shell's CSP is fixed at request time, so declared domains ride the URL.
  const src = useMemo(() => {
    if (!resource) return null;
    const csp = resource.ui?.csp ?? {};
    const query = new URLSearchParams();
    const directives: [string, string][] = [
      ["connect", domainParam(csp.connectDomains)],
      ["resource", domainParam(csp.resourceDomains)],
      ["frame", domainParam(csp.frameDomains)],
      ["base_uri", domainParam(csp.baseUriDomains)],
    ];
    for (const [key, value] of directives) {
      if (value) query.set(key, value);
    }
    // Never put the auth token in the URL: in-frame code reads location.href.
    return apiUrl(
      `/api/inference/mcp-app-frame${query.size ? `?${query.toString()}` : ""}`,
    );
  }, [resource]);

  const html = useMemo(
    () => (resource ? `${resource.text}\n${RESIZE_FALLBACK}` : null),
    [resource],
  );

  // Only a parent-initiated load is fed, so a self-navigated frame can't ask to
  // be re-seeded.
  const pendingPostRef = useRef(false);
  // Once the view reports its own size the measured fallback is ignored for
  // good, or it would drag a self-sized widget back on every content change.
  const viewOwnsSizeRef = useRef(false);
  useEffect(() => {
    pendingPostRef.current = true;
    viewOwnsSizeRef.current = false;
    setHeight(DEFAULT_HEIGHT);
  }, [src, html]);

  const postToView = useCallback((message: unknown) => {
    // Opaque origin, so a wildcard target is required; it still only reaches
    // this iframe's contentWindow.
    iframeRef.current?.contentWindow?.postMessage(message, "*");
  }, []);

  const seedView = useCallback(() => {
    // Nothing may be sent before `initialized`, and tool-input precedes result.
    postToView({
      jsonrpc: "2.0",
      method: "ui/notifications/tool-input",
      params: { arguments: toolArgs ?? {} },
    });
    // The view is sent the tool's whole result: dropping the images would show
    // it a different result than the card beside it renders.
    const seedText = ui.text ?? resultText;
    const content: Record<string, unknown>[] = seedText
      ? [{ type: "text", text: seedText }]
      : [];
    for (const image of resultImages ?? []) {
      content.push({
        type: "image",
        data: image.data,
        mimeType: image.mimeType,
      });
    }
    postToView({
      jsonrpc: "2.0",
      method: "ui/notifications/tool-result",
      params: {
        content,
        ...(ui.structuredContent !== undefined
          ? { structuredContent: ui.structuredContent }
          : {}),
        ...(ui._meta ? { _meta: ui._meta } : {}),
      },
    });
  }, [
    postToView,
    toolArgs,
    resultText,
    resultImages,
    ui.text,
    ui.structuredContent,
    ui._meta,
  ]);

  const onLoad = useCallback(() => {
    if (!pendingPostRef.current || !html) return;
    pendingPostRef.current = false;
    postToView({ type: "unsloth:artifact-html", html });
  }, [html, postToView]);

  // Theme flips reach a live widget as a partial host-context update.
  const initializedRef = useRef(false);
  useEffect(() => {
    initializedRef.current = false;
  }, [src, html]);
  useEffect(() => {
    if (!initializedRef.current) return;
    postToView({
      jsonrpc: "2.0",
      method: "ui/notifications/host-context-changed",
      params: { theme },
    });
  }, [theme, postToView]);

  useEffect(() => {
    const respond = (id: JsonRpcId, result: unknown) =>
      postToView({ jsonrpc: "2.0", id, result });
    const fail = (id: JsonRpcId, code: number, message: string) =>
      postToView({ jsonrpc: "2.0", id, error: { code, message } });

    const handler = (event: MessageEvent) => {
      // Every sandboxed frame reports origin "null", so identity is the check.
      if (event.source !== iframeRef.current?.contentWindow) return;
      if (event.origin !== "null") return;

      const data = event.data;
      // The resize fallback, which is not part of the widget protocol.
      if (typeof data?.mcpAppHeight === "number") {
        if (viewOwnsSizeRef.current) return;
        setHeight(Math.min(Math.max(data.mcpAppHeight, MIN_HEIGHT), MAX_HEIGHT));
        return;
      }
      if (!isJsonRpc(data) || typeof data.method !== "string") return;

      const { method, params, id } = data;

      switch (method) {
        case "ui/initialize": {
          if (id === undefined) return;
          respond(id, {
            protocolVersion: UI_PROTOCOL_VERSION,
            hostInfo: { name: HOST_NAME, version: HOST_VERSION },
            hostCapabilities: {
              // Only what this host actually implements.
              openLinks: {},
              serverTools: { listChanged: false },
              logging: {},
            },
            hostContext: {
              theme,
              displayMode: "inline",
              availableDisplayModes: ["inline"],
              containerDimensions: { maxHeight: MAX_HEIGHT },
              locale:
                typeof navigator === "undefined" ? "en" : navigator.language,
              timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
              platform: isTauri ? "desktop" : "web",
              deviceCapabilities: {
                touch:
                  typeof window !== "undefined" && "ontouchstart" in window,
                hover:
                  typeof window !== "undefined" &&
                  window.matchMedia("(hover: hover)").matches,
              },
            },
          });
          return;
        }

        case "ui/notifications/initialized": {
          // Only now is the view ready for host-context updates.
          initializedRef.current = true;
          seedView();
          return;
        }

        case "ui/notifications/size-changed": {
          const reported = (params as { height?: unknown } | undefined)?.height;
          if (typeof reported === "number" && Number.isFinite(reported)) {
            viewOwnsSizeRef.current = true;
            setHeight(Math.min(Math.max(reported, MIN_HEIGHT), MAX_HEIGHT));
          }
          return;
        }

        case "tools/call": {
          if (id === undefined) return;
          const name = (params as { name?: unknown } | undefined)?.name;
          if (typeof name !== "string" || !name) {
            fail(id, INVALID_PARAMS, "tools/call requires a tool name");
            return;
          }
          const args = (params as { arguments?: unknown } | undefined)
            ?.arguments;
          // serverId is the host's, from the tool part that drew the frame.
          callMcpUiTool(serverId, {
            toolName: name,
            arguments:
              typeof args === "object" && args !== null
                ? (args as Record<string, unknown>)
                : {},
            threadId,
            sessionId,
          })
            .then((res) => {
              respond(id, {
                content: res.content ?? [],
                ...(res.structured_content !== null
                  ? { structuredContent: res.structured_content }
                  : {}),
                isError: res.is_error,
                ...(res.meta ? { _meta: res.meta } : {}),
              });
            })
            .catch((err: unknown) => {
              fail(
                id,
                INTERNAL_ERROR,
                err instanceof Error ? err.message : String(err),
              );
            });
          return;
        }

        case "resources/read": {
          if (id === undefined) return;
          const uri = (params as { uri?: unknown } | undefined)?.uri;
          if (typeof uri !== "string" || !uri) {
            fail(id, INVALID_PARAMS, "resources/read requires a uri");
            return;
          }
          // The backend restricts this to templates the server declared.
          readMcpUiResource(serverId, uri, { threadId, sessionId })
            .then((res) => {
              respond(id, {
                contents: [
                  { uri: res.uri, mimeType: res.mime_type, text: res.text },
                ],
              });
            })
            .catch((err: unknown) => {
              fail(
                id,
                INTERNAL_ERROR,
                err instanceof Error ? err.message : String(err),
              );
            });
          return;
        }

        case "ui/open-link": {
          const url = (params as { url?: unknown } | undefined)?.url;
          if (typeof url !== "string") {
            if (id !== undefined) {
              fail(id, INVALID_PARAMS, "ui/open-link requires a url");
            }
            return;
          }
          // http(s) only: never open a javascript:, data: or file: URL.
          let safe = false;
          try {
            safe = ["http:", "https:"].includes(new URL(url).protocol);
          } catch {
            safe = false;
          }
          if (!safe) {
            if (id !== undefined) {
              fail(id, INVALID_PARAMS, "Only http(s) links can be opened");
            }
            return;
          }
          openLink(url);
          if (id !== undefined) respond(id, {});
          return;
        }

        case "ui/request-display-mode": {
          if (id === undefined) return;
          // Inline is the only mode here, and the resulting mode is always returned.
          respond(id, { mode: "inline" });
          return;
        }

        case "notifications/message": {
          const level = (params as { level?: unknown } | undefined)?.level;
          const text = (params as { text?: unknown } | undefined)?.text;
          console[level === "error" ? "error" : "info"](
            `[mcp-app ${toolName}]`,
            text,
          );
          return;
        }

        default: {
          // Notifications get no reply, but an unknown request must not hang.
          if (id !== undefined) {
            fail(id, METHOD_NOT_FOUND, `Unsupported method: ${method}`);
          }
        }
      }
    };

    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [postToView, seedView, serverId, threadId, sessionId, theme, toolName]);

  if (error) {
    return (
      <div className="mt-2 rounded border border-border bg-muted/30 px-3 py-2 text-ui-12p5 text-muted-foreground">
        Could not load this MCP app's interface: {error}
      </div>
    );
  }

  if (!src || !html) {
    return (
      <div
        className="mt-2 animate-pulse rounded border border-border bg-muted/30"
        style={{ height: MIN_HEIGHT }}
      />
    );
  }

  return (
    <iframe
      ref={iframeRef}
      src={src}
      // No allow-same-origin, so the widget reaches neither this app's storage
      // nor its cookies. No allow-downloads, as with the HTML canvas.
      sandbox="allow-scripts"
      referrerPolicy="no-referrer"
      onLoad={onLoad}
      style={{ height }}
      title={`${toolName} app`}
      className={cn(
        "mt-2 block w-full rounded border border-border bg-background",
        className,
      )}
    />
  );
}
