// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Shared fenced-code helpers for the HTML-artifact render paths, hoisted from
// markdown-text.tsx so the in-place collapse and the post-message auto-render
// agree on what counts as a renderable HTML fence.

export type CodeFence = {
  language: string | null;
  source: string;
};

// Matches one fenced block spanning the whole string (one pre-split block).
export const CODE_FENCE_RE = /^```([^\r\n`]*)\r?\n([\s\S]*?)\r?\n?```$/;

export type ToolCallPartLike = {
  type?: string;
  toolName?: string;
  args?: unknown;
  result?: unknown;
};

// True when a part is a render_html tool call with usable code or a non-error result.
export function isRenderableRenderHtmlToolPart(part: unknown): boolean {
  const toolPart = part as ToolCallPartLike;
  if (toolPart.type !== "tool-call" || toolPart.toolName !== "render_html") {
    return false;
  }
  if (
    typeof toolPart.result === "string" &&
    toolPart.result.startsWith("Error:")
  ) {
    return false;
  }
  if (
    typeof toolPart.result === "string" &&
    toolPart.result.startsWith("Rendered HTML canvas")
  ) {
    return true;
  }
  const args = toolPart.args as { code?: unknown } | undefined;
  return typeof args?.code === "string" && args.code.trim().length > 0;
}

export function getCodeFence(blockContent: string): CodeFence | null {
  const match = blockContent.trimEnd().match(CODE_FENCE_RE);
  if (!match) {
    return null;
  }

  return {
    language: match[1]?.trim() || null,
    source: match[2],
  };
}

export function isSvgFence(codeFence: CodeFence): boolean {
  const lang = codeFence.language?.toLowerCase() ?? "";
  if (lang === "svg") return true;
  if (lang === "xml" || lang === "html") {
    const trimmed = codeFence.source.trimStart();
    // <svg directly, or <?xml ...?> then <svg
    if (trimmed.startsWith("<svg")) return true;
    if (trimmed.startsWith("<?xml") && trimmed.includes("<svg")) return true;
  }
  return false;
}

export function isHtmlFence(codeFence: CodeFence): boolean {
  const lang = codeFence.language?.toLowerCase() ?? "";
  return lang === "html" && !isSvgFence(codeFence);
}

export function isFullHtmlDocument(source: string): boolean {
  const trimmed = source.trimStart();
  return /^<!doctype\s+html\b/i.test(trimmed) || /^<html[\s>]/i.test(trimmed);
}

export interface HtmlFence {
  source: string;
  isFullDocument: boolean;
  // Plain 3-backtick unindented fence: the only form the in-place collapser
  // (CODE_FENCE_RE) recognizes, so only these may be skipped as already shown.
  isPlainFence: boolean;
  index: number;
}

// Opening fence: up to 3 leading spaces, >=3 backticks, then a backtick-free info string.
const FENCE_OPEN_RE = /^( {0,3})(`{3,})([^`\r\n]*)$/;

// Scan a full message for every closed ```html fence. Line-based so multiple
// fences are found and backticks inside a <script> string never split a block
// (a close must be its own fence line). Drops unterminated/SVG fences.
export function extractHtmlFences(text: string): HtmlFence[] {
  const lines = text.split(/\r?\n/);
  const fences: HtmlFence[] = [];
  let i = 0;
  let index = 0;

  while (i < lines.length) {
    const open = lines[i].match(FENCE_OPEN_RE);
    if (!open) {
      i++;
      continue;
    }
    const indent = open[1].length;
    const ticks = open[2].length;
    const lang = open[3].trim().split(/\s+/)[0]?.toLowerCase() ?? "";

    const closeRe = new RegExp(`^ {0,3}\`{${ticks},}\\s*$`);
    // Strip up to `indent` leading spaces (CommonMark fence indentation).
    const indentRe = indent > 0 ? new RegExp(`^ {0,${indent}}`) : null;
    let j = i + 1;
    const body: string[] = [];
    let closed = false;
    while (j < lines.length) {
      if (closeRe.test(lines[j])) {
        closed = true;
        break;
      }
      body.push(indentRe ? lines[j].replace(indentRe, "") : lines[j]);
      j++;
    }

    if (!closed) {
      break; // everything after an unterminated open fence is inside it
    }

    if (lang === "html") {
      const source = body.join("\n");
      if (!isSvgFence({ language: "html", source })) {
        fences.push({
          source,
          isFullDocument: isFullHtmlDocument(source),
          isPlainFence: indent === 0 && ticks === 3,
          index: index++,
        });
      }
    }

    i = j + 1;
  }

  return fences;
}
