// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Button } from "@/components/ui/button";
import { useTheme } from "@/features/settings/stores/theme-store";
import { apiUrl, isTauri } from "@/lib/api-base";
import { openLink } from "@/lib/open-link";
import { cn } from "@/lib/utils";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import {
  type McpUiResource,
  callMcpUiTool,
  readMcpUiResource,
} from "../api/mcp-servers-api";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  type McpUiEnvelope,
  RESIZE_FALLBACK,
  bridgeShim,
  cspFrameQuery,
  newBridgeToken,
  toolApprovalScope,
  toolResultParams,
  withBridgeShim,
} from "./mcp-ui";

const MAX_PENDING_TOOL_CALLS = 8;
// Each call can hold a server worker for a minute, so cap in-flight calls.
const MAX_IN_FLIGHT_SERVER_CALLS = 8;
const SERVER_METHODS = new Set(["tools/call", "resources/read"]);
const DEFAULT_HEIGHT = 320;
const MIN_HEIGHT = 120;
const MAX_HEIGHT = 900;
const INVALID_PARAMS = -32602;
const METHOD_NOT_FOUND = -32601;
const INTERNAL_ERROR = -32603;
const DECLINED = "The user declined to run this tool call.";

class RpcError extends Error {
  code: number;
  constructor(message: string, code = INTERNAL_ERROR) {
    super(message);
    this.code = code;
  }
}

function isHttpUrl(url: string): boolean {
  try {
    return /^https?:$/.test(new URL(url).protocol);
  } catch {
    return false;
  }
}

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

interface FrameScope {
  serverId: string;
  threadId?: string;
  sessionId?: string;
}

interface PendingToolCall {
  name: string;
  argsPreview: string;
  scope: string;
  toolKey: string;
  /** Opened from the Open click itself so the browser sees a user gesture. */
  link?: string;
  decide: (allow: boolean, always: boolean) => void;
}

const clampHeight = (h: number) =>
  Math.min(Math.max(h, MIN_HEIGHT), MAX_HEIGHT);

function argsPreview(args: Record<string, unknown>): string {
  if (Object.keys(args).length === 0) return "";
  try {
    const text = JSON.stringify(args, null, 2);
    return text.length > 600 ? `${text.slice(0, 600)}…` : text;
  } catch {
    return "";
  }
}

// CSP is fixed at request time, so declared domains ride the URL; never the auth token.
function frameSrc(csp: McpUiResource["ui"]["csp"] | null): string {
  const query = cspFrameQuery(csp);
  return apiUrl(`/api/inference/mcp-app-frame${query ? `?${query}` : ""}`);
}

export interface McpAppFrameProps {
  /** From the host's tool part; never taken from the widget. */
  serverId: string;
  toolName: string;
  ui: McpUiEnvelope;
  argsText?: string;
  resultImages?: { data: string; mimeType: string }[];
  threadId?: string;
  sessionId?: string;
  className?: string;
}

export function McpAppFrame(props: McpAppFrameProps) {
  const { serverId, toolName, ui, threadId, sessionId, className } = props;
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const { resolved: theme } = useTheme();
  const [resource, setResource] = useState<
    (McpUiResource & { scope: FrameScope }) | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [height, setHeight] = useState(DEFAULT_HEIGHT);
  const [pendingCalls, setPendingCalls] = useState<PendingToolCall[]>([]);
  const latest = useRef({ props, theme });
  latest.current = { props, theme };
  const pendingRef = useRef<PendingToolCall[]>([]);
  const portRef = useRef<MessagePort | null>(null);
  const initializedRef = useRef(false);

  useEffect(() => {
    let cancelled = false;
    setResource(null);
    setError(null);
    readMcpUiResource(serverId, ui.resourceUri, { threadId, sessionId }).then(
      (loaded) =>
        !cancelled &&
        setResource({ ...loaded, scope: { serverId, threadId, sessionId } }),
      (err: unknown) =>
        !cancelled &&
        setError(err instanceof Error ? err.message : String(err)),
    );
    return () => {
      cancelled = true;
    };
  }, [serverId, ui.resourceUri, threadId, sessionId]);

  // One token per fetched template, so a re-seed cannot be replayed.
  const frame = useMemo(() => {
    const token = resource ? newBridgeToken() : null;
    if (!resource || !token) return null;
    const shim = bridgeShim(token, window.location.origin);
    const html = withBridgeShim(`${resource.text}\n${RESIZE_FALLBACK}`, shim);
    return {
      token,
      html,
      src: frameSrc(resource.ui?.csp),
      scope: resource.scope,
      fed: false,
    };
  }, [resource]);

  // Layout effect: the listener must be armed before the iframe's onLoad.
  useLayoutEffect(() => {
    const setQueue = (queue: PendingToolCall[]) => {
      pendingRef.current = queue;
      setPendingCalls(queue);
    };
    let viewOwnsSize = false;
    initializedRef.current = false;
    portRef.current?.close();
    portRef.current = null;
    setQueue([]);
    setHeight(DEFAULT_HEIGHT);
    if (!frame) return;

    const callTool = async (name: unknown, rawArgs: unknown) => {
      if (typeof name !== "string" || !name) {
        throw new RpcError("tools/call requires a tool name", INVALID_PARAMS);
      }
      const args = isObject(rawArgs) ? rawArgs : {};
      const { serverId, threadId, sessionId } = frame.scope;
      const scope = toolApprovalScope(sessionId, threadId);
      const toolKey = `mcp__${serverId}__${name}`;
      const send = (approved: boolean) =>
        callMcpUiTool(serverId, {
          tool_name: name,
          arguments: args,
          thread_id: threadId ?? null,
          session_id: sessionId ?? null,
          permission_mode: useChatRuntimeStore.getState().permissionMode,
          approved,
        }).then((res) => ({
          content: res.content ?? [],
          ...(res.structured_content !== null
            ? { structuredContent: res.structured_content }
            : {}),
          isError: res.is_error,
          ...(res.meta ? { _meta: res.meta } : {}),
        }));
      const { alwaysAllowToolsBySession } = useChatRuntimeStore.getState();
      const alwaysAllowed =
        alwaysAllowToolsBySession.get(scope)?.has(toolKey) ?? false;
      try {
        return await send(alwaysAllowed);
      } catch (err) {
        if (!(err instanceof Error && err.message === "approval_required")) {
          throw err;
        }
      }
      if (pendingRef.current.length >= MAX_PENDING_TOOL_CALLS) {
        throw new RpcError("Too many tool requests are waiting");
      }
      const preview = argsPreview(args);
      const [allow, always] = await new Promise<[boolean, boolean]>((done) =>
        setQueue([
          ...pendingRef.current,
          {
            name,
            argsPreview: preview,
            scope,
            toolKey,
            decide: (a, b) => done([a, b]),
          },
        ]),
      );
      if (!allow) {
        return { content: [{ type: "text", text: DECLINED }], isError: true };
      }
      const result = await send(true);
      // After the call: a failed press must not auto-approve later calls.
      if (always) {
        useChatRuntimeStore.getState().allowToolAlways(scope, toolKey);
      }
      return result;
    };

    const request = async (method: string, params: Record<string, unknown>) => {
      const { theme: nowTheme } = latest.current;
      switch (method) {
        case "ui/initialize":
          return {
            protocolVersion: "2026-01-26",
            hostInfo: { name: "Unsloth", version: "1.0.0" },
            hostCapabilities: {
              openLinks: {},
              serverTools: { listChanged: false },
              serverResources: { listChanged: false },
              logging: {},
            },
            hostContext: {
              theme: nowTheme,
              displayMode: "inline",
              availableDisplayModes: ["inline"],
              containerDimensions: { maxHeight: MAX_HEIGHT },
              locale: navigator.language,
              timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
              platform: isTauri ? "desktop" : "web",
            },
          };
        case "tools/call":
          return callTool(params.name, params.arguments);
        case "resources/read": {
          if (typeof params.uri !== "string" || !params.uri) {
            throw new RpcError("resources/read requires a uri", INVALID_PARAMS);
          }
          const { serverId, threadId, sessionId } = frame.scope;
          const res = await readMcpUiResource(serverId, params.uri, {
            threadId,
            sessionId,
          });
          if (res.contents?.length) return { contents: res.contents };
          const body = res.blob ? { blob: res.blob } : { text: res.text };
          return {
            contents: [{ uri: res.uri, mimeType: res.mime_type, ...body }],
          };
        }
        case "ui/open-link": {
          const url = String(params.url);
          if (!isHttpUrl(url)) {
            throw new RpcError(
              "Only http(s) links can be opened",
              INVALID_PARAMS,
            );
          }
          if (pendingRef.current.length >= MAX_PENDING_TOOL_CALLS) {
            throw new RpcError("Too many requests are waiting");
          }
          // Widget is untrusted and Desktop has no popup blocker, so ask first.
          const opened = await new Promise<boolean>((done) =>
            setQueue([
              ...pendingRef.current,
              {
                name: url,
                argsPreview: "",
                scope: "",
                toolKey: "",
                link: url,
                decide: (allow) => done(allow),
              },
            ]),
          );
          return opened ? {} : { isError: true };
        }
        case "ui/request-display-mode":
          return { mode: "inline" };
        default:
          throw new RpcError(`Unsupported method: ${method}`, METHOD_NOT_FOUND);
      }
    };

    let inFlight = 0;
    const handler = (event: MessageEvent) => {
      // Reply on the port, never contentWindow: the window survives navigation.
      const port = event.target as MessagePort;
      const data = event.data;
      if (typeof data?.mcpAppHeight === "number") {
        if (!viewOwnsSize) setHeight(clampHeight(data.mcpAppHeight));
        return;
      }
      if (data?.jsonrpc !== "2.0" || typeof data.method !== "string") return;
      const params = isObject(data.params) ? data.params : {};
      const now = latest.current.props;
      if (data.id !== undefined) {
        const counted = SERVER_METHODS.has(data.method);
        let run: Promise<unknown>;
        if (counted && inFlight >= MAX_IN_FLIGHT_SERVER_CALLS) {
          run = Promise.reject(new RpcError("Too many requests in flight"));
        } else {
          run = request(data.method, params);
          if (counted) {
            inFlight += 1;
            run
              .finally(() => {
                inFlight -= 1;
              })
              .catch(() => {});
          }
        }
        run.then(
          (result) => port.postMessage({ jsonrpc: "2.0", id: data.id, result }),
          (err: unknown) =>
            port.postMessage({
              jsonrpc: "2.0",
              id: data.id,
              error: {
                code: err instanceof RpcError ? err.code : INTERNAL_ERROR,
                message: err instanceof Error ? err.message : String(err),
              },
            }),
        );
      } else if (data.method === "ui/notifications/initialized") {
        // Nothing may be sent before `initialized`; tool-input precedes the result.
        initializedRef.current = true;
        let args: unknown = {};
        try {
          args = JSON.parse(now.argsText || "{}");
        } catch {
          // Streaming can leave argsText partial; send {} rather than a half-parse.
        }
        port.postMessage({
          jsonrpc: "2.0",
          method: "ui/notifications/tool-input",
          params: { arguments: isObject(args) ? args : {} },
        });
        port.postMessage({
          jsonrpc: "2.0",
          method: "ui/notifications/tool-result",
          params: toolResultParams(now.ui, now.resultImages),
        });
      } else if (
        data.method === "ui/notifications/size-changed" &&
        Number.isFinite(params.height)
      ) {
        viewOwnsSize = true;
        setHeight(clampHeight(params.height as number));
      } else if (data.method === "notifications/message") {
        const log = params.level === "error" ? console.error : console.info;
        log(`[mcp-app ${now.toolName}]`, params.text);
      }
    };

    // Only the token-checked handshake is read off the window.
    const onHandshake = (event: MessageEvent) => {
      const data = event.data;
      const port = event.ports[0];
      if (
        event.source !== iframeRef.current?.contentWindow ||
        event.origin !== "null" ||
        data?.__unslothMcpApp !== frame.token ||
        data.__unslothMcpAppPort !== true ||
        !port
      ) {
        return;
      }
      portRef.current?.close();
      portRef.current = port;
      initializedRef.current = false;
      setQueue([]);
      port.onmessage = handler;
    };
    window.addEventListener("message", onHandshake);
    return () => {
      window.removeEventListener("message", onHandshake);
      portRef.current?.close();
      portRef.current = null;
    };
  }, [frame]);

  useEffect(() => {
    if (!initializedRef.current) return;
    portRef.current?.postMessage({
      jsonrpc: "2.0",
      method: "ui/notifications/host-context-changed",
      params: { theme },
    });
  }, [theme]);

  // Only the first parent-initiated load is seeded.
  const onLoad = () => {
    if (!frame || frame.fed) return;
    frame.fed = true;
    // Opaque origin requires a wildcard target; it still only reaches this iframe.
    iframeRef.current?.contentWindow?.postMessage(
      { type: "unsloth:artifact-html", html: frame.html },
      "*",
    );
  };

  const failure =
    error ??
    (resource && !frame
      ? "this browser has no Web Crypto to isolate it with"
      : null);
  if (failure) {
    return (
      <div className="mt-2 rounded border border-border bg-muted/30 px-3 py-2 text-ui-12p5 text-muted-foreground">
        Could not load this MCP app's interface: {failure}
      </div>
    );
  }
  if (!frame) {
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
    const rest = pendingRef.current.filter((call) => call !== asking);
    pendingRef.current = rest;
    setPendingCalls(rest);
    if (allow && asking.link) openLink(asking.link);
    asking.decide(allow, always);
  };

  return (
    <>
      <iframe
        ref={iframeRef}
        src={frame.src}
        // No allow-same-origin or allow-downloads.
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
            {asking.link ? "This app wants to open " : "This app wants to run "}
            <span className="font-mono break-all">{asking.name}</span>
            {pendingCalls.length > 1
              ? ` (+${pendingCalls.length - 1} more waiting)`
              : ""}
          </div>
          {asking.argsPreview ? (
            <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap break-all text-muted-foreground">
              {asking.argsPreview}
            </pre>
          ) : null}
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Button size="xs" onClick={() => answer(true)}>
              {asking.link ? "Open" : "Allow"}
            </Button>
            {asking.link ? null : (
              <Button
                size="xs"
                variant="outline"
                onClick={() => answer(true, true)}
              >
                Always allow
              </Button>
            )}
            <Button
              size="xs"
              variant="destructive"
              onClick={() => answer(false)}
            >
              {asking.link ? "Cancel" : "Deny"}
            </Button>
          </div>
        </div>
      ) : null}
    </>
  );
}
