// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

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

export function bridgeShim(token: string): string {
  const hostOrigin = typeof window === "undefined" ? "" : window.location.origin;
  // The view's handle on the host is a MessageChannel port, not the frame's
  // parent window, and everything follows from that:
  //
  //   - it is bound to THIS document, so a page the frame navigates to can
  //     neither send over it nor receive an in-flight reply on it;
  //   - it IS `window.parent` here, so a view that filters responses on
  //     `event.source === window.parent` -- the defensive habit, and what a
  //     postMessage transport does by default -- still matches;
  //   - `event.source.postMessage(...)` reaches the host for the same reason.
  //
  // A Window takes (message, targetOrigin, transfer) and a port takes
  // (message, transfer), so the port's own postMessage is widened to accept the
  // call a view actually writes.
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

/** A fresh bridge token, or null when nothing here can make an unguessable one.
 *
 * crypto.randomUUID needs a secure context and Studio is reachable over plain
 * HTTP on a LAN address (`-H 0.0.0.0`), where it is simply undefined. The other
 * chat call sites fall back to Date.now()+Math.random(), which is fine for an
 * attachment id and not for this: the token is what stops a page the frame
 * navigated to from installing a port of its own. getRandomValues is the right
 * fallback -- unlike randomUUID it is not secure-context gated.
 */
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

/** Put `shim` where it runs before any of the view's own script.
 *
 * Parsed, not pattern-matched: the first textual `<head>` in a template can sit
 * inside a comment or a script string (`<!-- template has no <head> -->`), and a
 * shim inserted there never runs, which reads downstream as a view that simply
 * never initializes. A parse finds the element the browser will find.
 */
export function withBridgeShim(html: string, shim: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const script = doc.createElement("script");
  script.textContent = shim;
  const parent = doc.head ?? doc.documentElement;
  parent.insertBefore(script, parent.firstChild);
  // Rebuilt rather than round-tripped through outerHTML alone: a template with no
  // doctype is asking for quirks mode, and adding one would change how it lays out.
  const doctype = doc.doctype ? `<!DOCTYPE ${doc.doctype.name}>\n` : "";
  return doctype + doc.documentElement.outerHTML;
}

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

  // Only a parent-initiated load is fed, so a self-navigated frame can't ask to
  // be re-seeded.
  const pendingPostRef = useRef(false);
  // Once the view reports its own size the measured fallback is ignored for
  // good, or it would drag a self-sized widget back on every content change.
  const viewOwnsSizeRef = useRef(false);
  // The view is ready for host-context updates only after it says `initialized`.
  const initializedRef = useRef(false);
  // Layout, not passive: this arms the state onLoad reads, and the iframe starts
  // fetching the moment it is committed. A passive effect is queued during that
  // same commit and so normally wins, but the two are different task sources and
  // nothing orders them; losing once means onLoad declines to post and the widget
  // sits on the empty shell for good, with nothing to retry it. A layout effect
  // runs inside the commit, before the browser can dispatch anything.
  // The seeded document's own reply channel, handed over by the shim.
  const viewPortRef = useRef<MessagePort | null>(null);
  useLayoutEffect(() => {
    pendingPostRef.current = true;
    viewOwnsSizeRef.current = false;
    initializedRef.current = false;
    viewPortRef.current?.close();
    viewPortRef.current = null;
    setHeight(DEFAULT_HEIGHT);
  }, [src, html]);

  // Down the seeded document's own channel, never the frame's contentWindow: the
  // window survives a navigation and would hand an in-flight tool result or
  // resource body to whatever page the frame moved to. A port cannot outlive the
  // document that made it, so there is nowhere for a reply to leak to.
  const postToView = useCallback((message: unknown) => {
    viewPortRef.current?.postMessage(message);
  }, []);

  // The one exception, and it has to be: the shim only exists inside the HTML
  // this delivers, so there is no port yet. It carries the template the host
  // just fetched and nothing about the conversation.
  const postTemplate = useCallback((message: unknown) => {
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
    // The server's own blocks, in order, with the image bytes put back: the
    // envelope leaves those to the image sentinel rather than carrying a second
    // copy, so an image block arrives with its mimeType and no data. Anything
    // the flattened body would have shown instead is host prose -- an
    // "[1 image attached...]" note, or a Python repr of structuredContent --
    // and no part of what the server returned.
    const images = [...(resultImages ?? [])];
    const content: Record<string, unknown>[] = [];
    for (const block of ui.content ?? []) {
      if (block?.type === "image" && block.data === undefined) {
        const image = images.shift();
        // Dropped by the payload budget upstream; the card says so too.
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

  // Theme flips reach a live widget as a partial host-context update.
  useEffect(() => {
    if (!initializedRef.current) return;
    postToView({
      jsonrpc: "2.0",
      method: "ui/notifications/host-context-changed",
      params: { theme },
    });
  }, [theme, postToView]);

  // Layout, for the same reason as the arming above: the view's first message can
  // only follow the HTML onLoad posts, but the listener must already be attached
  // when it lands, and a passive effect is not ordered against that.
  useLayoutEffect(() => {
    const respond = (id: JsonRpcId, result: unknown) =>
      postToView({ jsonrpc: "2.0", id, result });
    const fail = (id: JsonRpcId, code: number, message: string) =>
      postToView({ jsonrpc: "2.0", id, error: { code, message } });

    // Everything the view says arrives on its port, which only the document the
    // host seeded holds. Sender identity and the opaque origin both survive a
    // navigation and so prove nothing; holding the port is the proof.
    const handler = (event: MessageEvent) => {
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

    // The handshake is the one thing that cannot come over the port, since it is
    // what delivers the port. Checked the old way, plus the token, so only the
    // document the host seeded can install one.
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
      port.onmessage = handler;
    };

    // A live port keeps the handler it was given, so re-point it whenever this
    // effect rebuilds one: otherwise a theme change leaves the view answered by
    // a closure describing the previous theme.
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
