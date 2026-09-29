// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No imports: the chat adapter, the tool card and the search index share it, and node tests load it as is.

export interface McpUiEnvelope {
  resourceUri: string;
  /** Image blocks carry no `data`: it rides the image envelope, in order. */
  content?: { type?: string; data?: string; [key: string]: unknown }[];
  structuredContent?: unknown;
  _meta?: Record<string, unknown>;
  /** Seed data was too large to persist; the widget must fetch it itself. */
  structuredContentOmitted?: boolean;
}

export interface McpUiToolResult {
  text: string;
  ui: McpUiEnvelope;
  images?: { data: string; mimeType: string }[];
}

const MCP_UI_MARKER = "\n__MCP_UI__:";

function isEnvelope(value: unknown): value is McpUiEnvelope {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { resourceUri?: unknown }).resourceUri === "string"
  );
}

/** Pulls the last __MCP_UI__ line off an MCP result; one JSON line, so an image envelope may follow. */
export function extractMcpUiEnvelope(
  raw: string,
  toolName: string,
): { text: string; ui: McpUiEnvelope | null } {
  const start = toolName.startsWith("mcp__")
    ? raw.lastIndexOf(MCP_UI_MARKER)
    : -1;
  if (start === -1) return { text: raw, ui: null };
  const lineEnd = raw.indexOf("\n", start + MCP_UI_MARKER.length);
  const end = lineEnd === -1 ? raw.length : lineEnd;
  let ui: unknown = null;
  try {
    ui = JSON.parse(raw.slice(start + MCP_UI_MARKER.length, end));
  } catch {
    // A tool that merely prints the marker keeps its whole output.
  }
  return isEnvelope(ui)
    ? { text: raw.slice(0, start) + raw.slice(end), ui }
    : { text: raw, ui: null };
}

/** Name-gated: an imported conversation can store any object under any tool name. */
export function isMcpUiToolResult(
  val: unknown,
  toolName?: string,
): val is McpUiToolResult {
  if (toolName !== undefined && !toolName.startsWith("mcp__")) return false;
  return (
    typeof val === "object" &&
    val !== null &&
    typeof (val as { text?: unknown }).text === "string" &&
    isEnvelope((val as { ui?: unknown }).ui)
  );
}

/** The scope the chat adapter records "Always allow" under, shared so it covers model and widget calls. */
export function toolApprovalScope(
  sessionId: string | null | undefined,
  threadId: string | null | undefined,
): string {
  const session = sessionId || "_default";
  return threadId ? `${session}:${threadId}` : session;
}

/** The tool-result seed: server blocks, image bytes put back from the images envelope in order. */
export function toolResultParams(
  ui: McpUiEnvelope,
  images: { data: string; mimeType: string }[] = [],
): Record<string, unknown> {
  let next = 0;
  const content: Record<string, unknown>[] = [];
  for (const block of ui.content ?? []) {
    if (block?.type !== "image" || block.data !== undefined) {
      content.push({ ...block });
      continue;
    }
    const image = images[next++];
    if (image) content.push({ ...block, ...image });
  }
  return {
    content,
    ...(ui.structuredContent !== undefined
      ? { structuredContent: ui.structuredContent }
      : {}),
    ...(ui._meta ? { _meta: ui._meta } : {}),
  };
}

/** getRandomValues, not randomUUID (missing over plain-HTTP LAN); null without Web Crypto. */
export function newBridgeToken(): string | null {
  const bytes = globalThis.crypto?.getRandomValues?.(new Uint8Array(16));
  return bytes
    ? Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")
    : null;
}

/** A MessageChannel port bound to THIS document (a navigated page cannot use it), exposed as window.parent/top. */
export function bridgeShim(token: string, hostOrigin: string): string {
  // postMessage widened to Window's (message, targetOrigin, transfer); replies re-dispatched as the port's.
  return `(() => { try {
  const real = window.parent, channel = new MessageChannel(), port = channel.port1;
  const raw = port.postMessage.bind(port);
  Object.defineProperty(port, "postMessage", { configurable: true, writable: true,
    value: (m, a, b) => raw(m, Array.isArray(a) ? a : Array.isArray(b) ? b : []) });
  port.onmessage = (e) => window.dispatchEvent(new MessageEvent("message",
    { data: e.data, source: port, origin: ${JSON.stringify(hostOrigin)} }));
  for (const name of ["parent", "top"]) {
    try { Object.defineProperty(window, name, { value: port, configurable: true }); } catch (e) {}
  }
  real.postMessage({ __unslothMcpApp: ${JSON.stringify(token)}, __unslothMcpAppPort: true }, "*", [channel.port2]);
} catch (e) {} })();`;
}

/** Parsed, not pattern-matched: a textual <head> can sit inside a comment or script string. */
export function withBridgeShim(html: string, shim: string): string {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const script = doc.createElement("script");
  script.textContent = shim;
  const parent = doc.head ?? doc.documentElement;
  parent.insertBefore(script, parent.firstChild);
  // Only an existing doctype: adding one to a quirks-mode template changes its layout.
  const doctype = doc.doctype ? `<!DOCTYPE ${doc.doctype.name}>\n` : "";
  return doctype + doc.documentElement.outerHTML;
}

// Height for a view that never reports one. html is measured at max-content (as ext-apps' App does):
// documentElement.scrollHeight never drops below the frame, so a widget could grow but never shrink.
export const RESIZE_FALLBACK = `<script>(()=>{let last=0;const post=()=>{const html=document.documentElement,prev=html.style.height;html.style.height="max-content";const h=Math.ceil(html.getBoundingClientRect().height);html.style.height=prev;if(h!==last){last=h;parent.postMessage({mcpAppHeight:h},"*");}};const ro=new ResizeObserver(post);ro.observe(document.documentElement);if(document.body)ro.observe(document.body);addEventListener("load",post);post();})();</script>`;
