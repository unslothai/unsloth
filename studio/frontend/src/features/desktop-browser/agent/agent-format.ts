// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isBrowserToolName } from "@/lib/browser-tool-names";

export type PageBlockKind = "snapshot" | "text" | "find";

// the backend finds blocks by these tags, so page text containing one could close the untrusted block early or be stubbed out.
const PAGE_TAG = /<(\/?)\s*browser_page/gi;

export function escapePageText(text: string): string {
  return text.replace(PAGE_TAG, "<$1browser-page");
}

/** untrusted page content, marked so the model and the backend can tell it from Studio's own. */
export function pageBlock(kind: PageBlockKind, body: string): string {
  return `<browser_page kind="${kind}">\n${escapePageText(body.trim())}\n</browser_page>`;
}

export function joinResult(
  ...parts: (string | null | undefined | false)[]
): string {
  return parts.filter((part): part is string => Boolean(part)).join("\n");
}

/** cut by code point, so an emoji never leaves a lone surrogate that fails the next llama-server request. */
export function quoteForSummary(text: string, max = 60): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const chars = Array.from(flat);
  return chars.length > max ? `${chars.slice(0, max - 1).join("")}…` : flat;
}

export function siteOf(url: string | null | undefined): string | null {
  if (!url) return null;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:")
      return null;
    return parsed.hostname.replace(/^www\./, "") || null;
  } catch {
    return null;
  }
}

/** the first line of a browser result is the action summary the executor wrote. */
export function resultSummary(result: unknown): string {
  const text =
    typeof result === "string"
      ? result
      : result && typeof result === "object" && "text" in result
        ? String((result as { text: unknown }).text ?? "")
        : "";
  const line = text.split("\n", 1)[0]?.trim() ?? "";
  if (!line.startsWith("Error:")) return line.replace(/\.$/, "");
  // what went wrong is the first sentence; the rest is advice to the model.
  const first = line
    .replace(/^Error:\s*/, "")
    .split(/\.\s+(?=[A-Z])/, 1)[0]
    .replace(/\.$/, "");
  return first.charAt(0).toUpperCase() + first.slice(1);
}

// mirrors the backend stub for turns outside the tool loop; a block cut to fit the window has no closing tag, so it runs to the end.
const SNAPSHOT_BLOCK =
  /<browser_page kind="snapshot"([^>]*)>[\s\S]*?(?:<\/browser_page>|$(?![\s\S]))/g;
export const SUPERSEDED_SNAPSHOT =
  '<browser_page kind="snapshot" superseded="true">(older view of the page; see the latest snapshot)</browser_page>';

/** only a browser tool's result has its page text escaped, so only those can hold a real block. */
function browserResult(message: {
  role?: string;
  name?: unknown;
  content?: unknown;
}): message is { role: "tool"; name: string; content: string } {
  return (
    message?.role === "tool" &&
    typeof message.content === "string" &&
    isBrowserToolName(message.name)
  );
}

export function supersedeBrowserSnapshots<
  T extends { role?: string; name?: unknown; content?: unknown },
>(messages: readonly T[]): T[] {
  let newest: { index: number; offset: number } | null = null;
  for (let index = messages.length - 1; index >= 0 && !newest; index--) {
    const message = messages[index];
    if (!browserResult(message)) continue;
    let last: number | null = null;
    for (const match of message.content.matchAll(SNAPSHOT_BLOCK)) {
      if (!match[1].includes('superseded="true"')) last = match.index ?? 0;
    }
    if (last !== null) newest = { index, offset: last };
  }
  if (!newest) return messages.slice();
  return messages.map((message, index) => {
    if (!browserResult(message)) return message;
    const content = message.content.replace(
      SNAPSHOT_BLOCK,
      (block: string, attributes: string, offset: number) =>
        attributes.includes('superseded="true"') ||
        (index === newest.index && offset === newest.offset)
          ? block
          : SUPERSEDED_SNAPSHOT,
    );
    return content === message.content ? message : { ...message, content };
  });
}
