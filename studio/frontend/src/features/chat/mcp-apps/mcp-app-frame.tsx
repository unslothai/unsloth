// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { useTheme } from "@/features/settings/stores/theme-store";
import { apiUrl, isTauri } from "@/lib/api-base";
import { openLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  McpUiApprovalRequired,
  callMcpUiTool,
  readMcpUiResource,
  type McpUiResource,
  type McpUiToolCallResult,
} from "../api/mcp-servers-api";
import type { McpUiEnvelope } from "../api/chat-adapter";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  MCP_APP_TOOL_DECLINED,
  mcpAppApprovalScope,
  mcpAppArgsPreview,
  mcpAppToolKey,
} from "./tool-approval";

const MAX_PENDING_TOOL_CALLS = 8;

interface PendingToolCall {
  key: number;
  name: string;
  args: Record<string, unknown>;
  decide: (allow: boolean) => void;
}

const UI_PROTOCOL_VERSION = "2026-01-26";
const HOST_NAME = "Unsloth";
const HOST_VERSION = "1.0.0";

const DEFAULT_HEIGHT = 320;
const MIN_HEIGHT = 120;
const MAX_HEIGHT = 900;

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

// Fallback height; a reported size always wins.
const RESIZE_FALLBACK = `<script>(()=>{const post=()=>parent.postMessage({mcpAppHeight:document.documentElement.scrollHeight},"*");new ResizeObserver(post).observe(document.documentElement);window.addEventListener("load",post);post();})();</script>`;

export function bridgeShim(token: string): string {
  const hostOrigin = typeof window === "undefined" ? "" : window.location.origin;
  // A MessageChannel port bound to THIS document (a navigated page cannot use it), exposed as window.parent; postMessage widened to Window's signature.
  return `(() => {
  try {
    const real = window.parent;
    const channel = new MessageChannel();
    const port = channel.port1;
    const raw = port.postMessage.bind(port);
    Object.defineProperty(port, "postMessage", {
      value: (message, a, b) =>
        raw(message, Array.isArray(a) ? a : Array.isArray(b) ? b : []),
      configurable: true,
      writable: true,
    });
    port.onmessage = (event) => {
      window.dispatchEvent(new MessageEvent("message", {
        data: event.data, source: port, origin: ${JSON.stringify(hostOrigin)},
      }));
    };
    for (const name of ["parent", "top"]) {
      try {
        Object.defineProperty(window, name, { value: port, configurable: true });
      } catch (e) {}
    }
    real.postMessage(
      { __unslothMcpApp: ${JSON.stringify(token)}, __unslothMcpAppPort: true },
      "*",
      [channel.port2],
    );
  } catch (e) {}
})();`;
}

/** getRandomValues fallback: randomUUID is undefined over plain-HTTP LAN; null when no Web Crypto. */
export function newBridgeToken(): string | null {
  const webCrypto = globalThis.crypto;
  if (typeof webCrypto?.randomUUID === "function") return webCrypto.randomUUID();
  if (typeof webCrypto?.getRandomValues === "function") {
    return Array.from(webCrypto.getRandomValues(new Uint8Array(16)), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
  }
  // A guessable token is worse than none, so the caller shows the failure.
  return null;
}

/** Parsed, not pattern-matched: a textual <head> can sit inside a comment or script string. */
export function withBridgeShim(html: string, shim: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const script = doc.createElement("script");
  script.textContent = shim;
  const parent = doc.head ?? doc.documentElement;
  parent.insertBefore(script, parent.firstChild);
  // Rebuilt, not outerHTML alone: adding a doctype to a quirks-mode template changes its layout.
  const doctype = doc.doctype ? `<!DOCTYPE ${doc.doctype.name}>\n` : "";
  return doctype + doc.documentElement.outerHTML;
}

function domainParam(values: string[] | undefined): string {
  return Array.isArray(values) ? values.filter(Boolean).join(",") : "";
}

export interface McpAppFrameProps {
  serverId: string;
  toolName: string;
  ui: McpUiEnvelope;
  toolArgs?: Record<string, unknown>;
  resultImages?: { data: string; mimeType: string }[];
  threadId?: string;
  sessionId?: string;
  className?: string;
}

export function McpAppFrame({
  serverId,
  toolName,
  ui,
  toolArgs,
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
  const [pendingCalls, setPendingCalls] = useState<PendingToolCall[]>([]);
  const pendingKeyRef = useRef(0);
  const pendingCallsRef = useRef(pendingCalls);
  pendingCallsRef.current = pendingCalls;
  const allowToolAlways = useChatRuntimeStore((s) => s.allowToolAlways);
  const approvalScope = mcpAppApprovalScope(sessionId, threadId);

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

  // One token per fetched template, so re-seeding cannot be replayed either.
  const bridgeToken = useMemo(
    () => (resource ? newBridgeToken() : null),
    [resource],
  );

  const html = useMemo(
    () =>
      resource && bridgeToken
        ? withBridgeShim(
            `${resource.text}\n${RESIZE_FALLBACK}`,
            bridgeShim(bridgeToken),
          )
        : null,
    [resource, bridgeToken],
  );

  // Only a parent-initiated load is fed, so a self-navigated frame cannot be re-seeded.
  const pendingPostRef = useRef(false);
  // Once the view reports its size, the measured fallback is ignored for good.
  const viewOwnsSizeRef = useRef(false);
  const initializedRef = useRef(false);
  // Layout, not passive: must arm before the iframe's onLoad, which a passive effect is unordered against.
  const viewPortRef = useRef<MessagePort | null>(null);
  useLayoutEffect(() => {
    pendingPostRef.current = true;
    viewOwnsSizeRef.current = false;
    initializedRef.current = false;
    viewPortRef.current?.close();
    viewPortRef.current = null;
    setHeight(DEFAULT_HEIGHT);
  }, [src, html]);

  // Via the seeded document's port, never contentWindow: the window survives navigation and would leak replies.
  const postToView = useCallback((message: unknown) => {
    viewPortRef.current?.postMessage(message);
  }, []);

  // The exception: the HTML delivery precedes the port; it carries only the template.
  const postTemplate = useCallback((message: unknown) => {
    // Opaque origin requires a wildcard target; it still only reaches this iframe.
    iframeRef.current?.contentWindow?.postMessage(message, "*");
  }, []);

  const seedView = useCallback(() => {
    // Nothing may be sent before `initialized`, and tool-input precedes result.
    postToView({
      jsonrpc: "2.0",
      method: "ui/notifications/tool-input",
      params: { arguments: toolArgs ?? {} },
    });
    // Server blocks, image bytes put back from the sentinel; never host prose from the flattened body.
    const images = [...(resultImages ?? [])];
    const content: Record<string, unknown>[] = [];
    for (const block of ui.content ?? []) {
      if (block?.type === "image" && block.data === undefined) {
        const image = images.shift();
        if (!image) continue;
        content.push({ ...block, data: image.data, mimeType: image.mimeType });
        continue;
      }
      content.push({ ...block });
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
    resultImages,
    ui.content,
    ui.structuredContent,
    ui._meta,
  ]);

  const onLoad = useCallback(() => {
    if (!pendingPostRef.current || !html) return;
    pendingPostRef.current = false;
    postTemplate({ type: "unsloth:artifact-html", html });
  }, [html, postTemplate]);

  useEffect(() => {
    if (!initializedRef.current) return;
    postToView({
      jsonrpc: "2.0",
      method: "ui/notifications/host-context-changed",
      params: { theme },
    });
  }, [theme, postToView]);

  // Layout, as above: the listener must be attached before the view's first message.
  useLayoutEffect(() => {
    const respond = (id: JsonRpcId, result: unknown) =>
      postToView({ jsonrpc: "2.0", id, result });
    const fail = (id: JsonRpcId, code: number, message: string) =>
      postToView({ jsonrpc: "2.0", id, error: { code, message } });

    // Only the seeded document holds the port; source and origin survive navigation and prove nothing.
    const handler = (event: MessageEvent) => {
      const data = event.data;
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
          const callArgs =
            typeof args === "object" && args !== null
              ? (args as Record<string, unknown>)
              : {};
          // serverId is the host's, from the tool part that drew the frame.
          const send = (approved: boolean) =>
            callMcpUiTool(serverId, {
              toolName: name,
              arguments: callArgs,
              threadId,
              sessionId,
              permissionMode: useChatRuntimeStore.getState().permissionMode,
              approved,
            });
          const deliver = (res: McpUiToolCallResult) =>
            respond(id, {
              content: res.content ?? [],
              ...(res.structured_content !== null
                ? { structuredContent: res.structured_content }
                : {}),
              isError: res.is_error,
              ...(res.meta ? { _meta: res.meta } : {}),
            });
          const refuse = (err: unknown) =>
            fail(
              id,
              INTERNAL_ERROR,
              err instanceof Error ? err.message : String(err),
            );
          const alwaysAllowed =
            useChatRuntimeStore
              .getState()
              .alwaysAllowToolsBySession.get(approvalScope)
              ?.has(mcpAppToolKey(serverId, name)) ?? false;
          send(alwaysAllowed)
            .then(deliver)
            .catch((err: unknown) => {
              if (!(err instanceof McpUiApprovalRequired)) {
                refuse(err);
                return;
              }
              if (pendingCallsRef.current.length >= MAX_PENDING_TOOL_CALLS) {
                fail(id, INTERNAL_ERROR, "Too many tool requests are waiting");
                return;
              }
              // Untrusted HTML: same confirm gate as the model's call.
              pendingKeyRef.current += 1;
              setPendingCalls((queue) => [
                ...queue,
                {
                  key: pendingKeyRef.current,
                  name,
                  args: callArgs,
                  decide: (allow) => {
                    if (allow) {
                      send(true).then(deliver).catch(refuse);
                      return;
                    }
                    respond(id, {
                      content: [{ type: "text", text: MCP_APP_TOOL_DECLINED }],
                      isError: true,
                    });
                  },
                },
              ]);
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

    // The handshake delivers the port, so it stays on the window, token-checked.
    const onHandshake = (event: MessageEvent) => {
      if (event.source !== iframeRef.current?.contentWindow) return;
      if (event.origin !== "null") return;
      const envelope = event.data as {
        __unslothMcpApp?: unknown;
        __unslothMcpAppPort?: unknown;
      };
      if (
        !bridgeToken ||
        typeof envelope !== "object" ||
        envelope === null ||
        envelope.__unslothMcpApp !== bridgeToken ||
        envelope.__unslothMcpAppPort !== true
      ) {
        return;
      }
      const port = event.ports[0];
      if (!port) return;
      viewPortRef.current?.close();
      viewPortRef.current = port;
      setPendingCalls([]);
      port.onmessage = handler;
    };

    // Re-point the live port's handler on rebuild, or it answers from a stale closure.
    if (viewPortRef.current) viewPortRef.current.onmessage = handler;

    window.addEventListener("message", onHandshake);
    return () => window.removeEventListener("message", onHandshake);
  }, [
    bridgeToken,
    postToView,
    seedView,
    serverId,
    threadId,
    sessionId,
    approvalScope,
    theme,
    toolName,
  ]);

  const failure =
    error ??
    (resource && !bridgeToken
      ? "this browser has no Web Crypto to isolate it with"
      : null);

  if (failure) {
    return (
      <div className="mt-2 rounded border border-border bg-muted/30 px-3 py-2 text-ui-12p5 text-muted-foreground">
        Could not load this MCP app's interface: {failure}
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

  const asking = pendingCalls[0];
  const answer = (allow: boolean, always = false) => {
    if (!asking) return;
    if (always) allowToolAlways(approvalScope, mcpAppToolKey(serverId, asking.name));
    setPendingCalls((queue) => queue.filter((call) => call.key !== asking.key));
    asking.decide(allow);
  };
  const askingArgs = asking ? mcpAppArgsPreview(asking.args) : "";

  return (
    <>
      <iframe
        ref={iframeRef}
        src={src}
        // No allow-same-origin (app storage/cookies), no allow-downloads.
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
      {asking ? (
        <div
          role="group"
          aria-label="Tool request from this app"
          className="mt-1 rounded border border-border bg-muted/30 px-3 py-2 text-ui-12p5"
        >
          <div>
            This app wants to run{" "}
            <span className="font-mono">{asking.name}</span>
            {pendingCalls.length > 1
              ? ` (+${pendingCalls.length - 1} more waiting)`
              : ""}
          </div>
          {askingArgs ? (
            <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-all text-muted-foreground">
              {askingArgs}
            </pre>
          ) : null}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button size="xs" onClick={() => answer(true)}>
              Allow
            </Button>
            <Button
              size="xs"
              variant="outline"
              onClick={() => answer(true, true)}
            >
              Always allow
            </Button>
            <Button
              size="xs"
              variant="destructive"
              onClick={() => answer(false)}
            >
              Deny
            </Button>
          </div>
        </div>
      ) : null}
    </>
  );
}
