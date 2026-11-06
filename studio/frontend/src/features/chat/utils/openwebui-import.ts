// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** `history.messages` is the authoritative DAG and `currentId` its active branch. */

import type { MessageRecord, ParsedConversation } from "../types";

type Dict = Record<string, unknown>;

interface Node {
  id: string;
  parentId: string | null;
  raw: Dict;
}

/** Guards against cyclic parent chains. */
const MAX_MESSAGES_PER_CHAT = 200_000;

function isDict(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

const MAX_EPOCH_MS = 8.64e15;

/** Open WebUI stores seconds on the record and milliseconds on `chat.timestamp`. */
function epochMs(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  const ms = value < 1e12 ? Math.round(value * 1000) : Math.round(value);
  // Out-of-range stamps are dropped: past 2^53 `previousTs + 1` stops advancing.
  return ms > MAX_EPOCH_MS ? null : ms;
}

function chatBlob(record: unknown): Dict | null {
  if (!isDict(record)) return null;
  if (isDict(record.chat)) return record.chat;
  return record;
}

function looksLikeOpenWebUIMessage(value: unknown): boolean {
  if (!isDict(value)) return false;
  if (typeof value.role !== "string") return false;
  return (
    "parentId" in value ||
    "childrenIds" in value ||
    "modelName" in value ||
    "modelIdx" in value ||
    "done" in value ||
    "output" in value ||
    ("timestamp" in value && typeof value.id === "string")
  );
}

function looksLikeOpenAIToolMessage(value: unknown): boolean {
  if (!isDict(value)) return false;
  return value.role === "tool" || "tool_calls" in value || "tool_call_id" in value;
}

export function isOpenWebUIRecord(value: unknown): boolean {
  if (!isDict(value)) return false;
  if (isDict(value.chat)) {
    const chat = value.chat;
    return (
      isDict(chat.history) || Array.isArray(chat.messages) || "models" in chat
    );
  }
  const blob = chatBlob(value);
  if (!blob) return false;
  if (isDict(blob.history) && isDict(blob.history.messages)) return true;
  if (!Array.isArray(blob.messages)) return false;
  // Open WebUI never stores tool calls as sibling Chat Completions turns.
  if (blob.messages.some(looksLikeOpenAIToolMessage)) return false;
  return blob.messages.some(looksLikeOpenWebUIMessage);
}

// Unclosed fences are anchored to line start; unanchored, a stray ``` swallowed the message.
const FENCED_CODE =
  /```[\s\S]*?```|~~~[\s\S]*?~~~|(?:^|\n)[ \t]{0,3}(?:```|~~~)[\s\S]*$/g;
const DETAILS_BLOCK = /<details\b([^>]*)>([\s\S]*?)<\/details>/gi;
const ATTRIBUTE = /([\w-]+)="([^"]*)"/g;

function numericEntity(digits: string, radix: number): string | null {
  const code = Number.parseInt(digits, radix);
  if (!Number.isFinite(code) || code < 0 || code > 0x10ffff) return null;
  if (code >= 0xd800 && code <= 0xdfff) return null;
  return String.fromCodePoint(code);
}

function unescapeHtml(value: string): string {
  return value
    // Numeric escapes first and `&amp;` last, so double escapes decode only once.
    .replace(/&#x([0-9a-f]+);/gi, (whole, hex: string) => numericEntity(hex, 16) ?? whole)
    .replace(/&#(\d+);/g, (whole, dec: string) => numericEntity(dec, 10) ?? whole)
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&amp;/g, "&");
}

function detailsAttributes(rawAttributes: string): Record<string, string> {
  const attributes: Record<string, string> = {};
  ATTRIBUTE.lastIndex = 0;
  let match = ATTRIBUTE.exec(rawAttributes);
  while (match !== null) {
    attributes[match[1]] = unescapeHtml(match[2]);
    match = ATTRIBUTE.exec(rawAttributes);
  }
  return attributes;
}

function parseJsonLoose(value: string | undefined): unknown {
  if (!value) return {};
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function toolArgs(value: unknown): Dict {
  if (isDict(value)) return value;
  if (typeof value === "string" && value) return { arguments: value };
  return {};
}

function detailsBody(body: string, stripQuoteMarkers = false): string {
  const withoutSummary = body.replace(/<summary>[\s\S]*?<\/summary>/i, "");
  if (!stripQuoteMarkers) return withoutSummary.trim();
  return withoutSummary
    .split("\n")
    .map((line) => line.replace(/^\s*>\s?/, ""))
    .join("\n")
    .trim();
}

function pushText(parts: unknown[], text: string): void {
  const trimmed = text.trim();
  if (!trimmed) return;
  const last = parts.at(-1) as Dict | undefined;
  if (last && last.type === "text") {
    last.text = `${last.text as string}\n\n${trimmed}`;
    return;
  }
  parts.push({ type: "text", text: trimmed });
}

function fencedRanges(content: string): Array<[number, number]> {
  const ranges: Array<[number, number]> = [];
  FENCED_CODE.lastIndex = 0;
  let match = FENCED_CODE.exec(content);
  while (match !== null) {
    ranges.push([match.index, match.index + match[0].length]);
    match = FENCED_CODE.exec(content);
  }
  return ranges;
}

function contentWithDetailsToParts(content: string): unknown[] {
  const parts: unknown[] = [];
  const fences = fencedRanges(content);
  const quotedInFence = (start: number, end: number) =>
    fences.some(([from, to]) => start >= from && end <= to);

  let cursor = 0;
  DETAILS_BLOCK.lastIndex = 0;
  let match = DETAILS_BLOCK.exec(content);
  while (match !== null) {
    const start = match.index;
    const end = start + match[0].length;
    if (quotedInFence(start, end)) {
      match = DETAILS_BLOCK.exec(content);
      continue;
    }

    pushText(parts, content.slice(cursor, start));
    cursor = end;

    const attributes = detailsAttributes(match[1]);
    const body = detailsBody(match[2], attributes.type === "reasoning");
    if (attributes.type === undefined) {
      pushText(parts, match[0]);
    } else if (attributes.type === "tool_calls") {
      const result =
        attributes.result !== undefined ? parseJsonLoose(attributes.result) : body || undefined;
      parts.push({
        type: "tool-call",
        toolCallId: attributes.id || crypto.randomUUID(),
        toolName: attributes.name || "unknown",
        args: toolArgs(parseJsonLoose(attributes.arguments)),
        ...(result !== undefined ? { result } : {}),
      });
    } else if (attributes.type === "reasoning") {
      if (body) parts.push({ type: "reasoning", text: body });
    } else if (body) {
      pushText(parts, body);
    }

    match = DETAILS_BLOCK.exec(content);
  }
  pushText(parts, content.slice(cursor));

  return parts;
}

const PORTABLE_IMAGE = /^(?:https?:|data:)/i;

const TEXT_PART_TYPES = new Set(["input_text", "output_text", "text"]);

function reasoningText(source: unknown): string {
  if (!Array.isArray(source)) return "";
  return source
    .map((part) => (isDict(part) && typeof part.text === "string" ? part.text : ""))
    .join("")
    .trim();
}

const BUILTIN_TOOL_NAMES: Record<string, string> = { shell: "code_execution" };

function outputItemsToParts(output: unknown[]): { parts: unknown[]; sawMessage: boolean } {
  const parts: unknown[] = [];
  const toolCallIndex = new Map<string, Dict>();
  let sawMessage = false;

  for (const item of output) {
    if (!isDict(item)) continue;

    if (item.type === "reasoning") {
      // `summary` is an empty array when summaries are off.
      const summary = reasoningText(item.summary);
      const text = summary || reasoningText(item.content);
      if (text) parts.push({ type: "reasoning", text });
      continue;
    }

    if (item.type === "message") {
      const content = Array.isArray(item.content) ? item.content : [];
      const text = content
        .map((part) =>
          isDict(part) && part.type === "output_text" && typeof part.text === "string"
            ? part.text
            : "",
        )
        .join("");
      if (text.trim()) sawMessage = true;
      pushText(parts, text);
      continue;
    }

    if (item.type === "image_generation_call") {
      const encoded = str(item.result) ?? str(item.b64_json);
      if (encoded) {
        const format = str(item.output_format) ?? "png";
        // Never wrap a url as a base64 data url.
        parts.push({
          type: "image",
          image: PORTABLE_IMAGE.test(encoded) ? encoded : `data:image/${format};base64,${encoded}`,
        });
      }
      continue;
    }

    if (item.type === "function_call") {
      const callId = str(item.call_id) ?? crypto.randomUUID();
      const part: Dict = {
        type: "tool-call",
        toolCallId: callId,
        toolName: str(item.name) ?? "unknown",
        args: toolArgs(
          parseJsonLoose(typeof item.arguments === "string" ? item.arguments : undefined),
        ),
      };
      if (isDict(item.arguments)) {
        part.args = item.arguments;
      }
      toolCallIndex.set(callId, part);
      parts.push(part);
      continue;
    }

    const builtin = typeof item.type === "string" ? /^(\w+)_call$/.exec(item.type) : null;
    if (builtin) {
      const callId = str(item.call_id) ?? str(item.id) ?? crypto.randomUUID();
      const part: Dict = {
        type: "tool-call",
        toolCallId: callId,
        toolName: BUILTIN_TOOL_NAMES[builtin[1]] ?? builtin[1],
        args: isDict(item.action) ? item.action : {},
      };
      toolCallIndex.set(callId, part);
      parts.push(part);
      continue;
    }

    if (typeof item.type === "string" && item.type.endsWith("_call_output")) {
      const callId = str(item.call_id);
      const outputParts = Array.isArray(item.output) ? item.output : [];
      let resultText = typeof item.output === "string" ? item.output : "";
      const images: string[] = [];
      for (const part of outputParts) {
        if (!isDict(part)) continue;
        if (TEXT_PART_TYPES.has(part.type as string) && typeof part.text === "string") {
          resultText += part.text;
        } else if (part.type === "input_image") {
          const url = str(part.image_url);
          if (url) images.push(url);
        }
      }
      const target = callId ? toolCallIndex.get(callId) : undefined;
      if (target) {
        // Leave result unset, not "": an empty string renders an empty Result heading.
        if (resultText) target.result = resultText;
      } else if (resultText.trim()) {
        pushText(parts, resultText);
      }
      for (const image of images) {
        if (PORTABLE_IMAGE.test(image)) parts.push({ type: "image", image });
      }
    }
  }

  return { parts, sawMessage };
}

/** Drop extracted document text, which would otherwise be resent to the model. */
function filesToParts(files: unknown): { parts: unknown[]; attachments: unknown[] } {
  const parts: unknown[] = [];
  const attachments: unknown[] = [];
  if (!Array.isArray(files)) return { parts, attachments };

  for (const file of files) {
    if (!isDict(file)) continue;
    const url = str(file.url);
    if (file.type === "image") {
      // A bare `/api/v1/files/<id>` url is dead outside Open WebUI; only inline data survives.
      if (url?.startsWith("data:")) parts.push({ type: "image", image: url });
      continue;
    }
    const name = str(file.name) ?? str((isDict(file.file) ? file.file.filename : null)) ?? "file";
    attachments.push({
      id: str(file.id) ?? crypto.randomUUID(),
      type: "document",
      name,
      contentType: str(file.content_type) ?? "application/octet-stream",
      content: [],
      status: { type: "complete" },
    });
  }

  return { parts, attachments };
}

function arrayContentToParts(raw: unknown[]): unknown[] {
  return raw.flatMap((entry): unknown[] => {
    if (!isDict(entry)) return [];
    if (entry.type === "text" && typeof entry.text === "string") {
      return entry.text.trim() ? [{ type: "text", text: entry.text }] : [];
    }
    if (entry.type === "image_url") {
      const url = str((isDict(entry.image_url) ? entry.image_url.url : null));
      return url ? [{ type: "image", image: url }] : [];
    }
    return [];
  });
}

function messageParts(
  message: Dict,
  role: MessageRecord["role"],
): { content: unknown[]; attachments: unknown[] } {
  const parts: unknown[] = [];
  let sawMessage = false;

  if (Array.isArray(message.output)) {
    const converted = outputItemsToParts(message.output);
    parts.push(...converted.parts);
    sawMessage = converted.sawMessage;
  }

  if (!sawMessage && Array.isArray(message.content)) {
    parts.push(...arrayContentToParts(message.content));
  }

  const content = typeof message.content === "string" ? message.content : "";
  // Use `content` only when no output message item produced the answer.
  if (content && !sawMessage) {
    if (role === "assistant") parts.push(...contentWithDetailsToParts(content));
    else if (content.trim()) parts.push({ type: "text", text: content });
  }

  const { parts: fileParts, attachments } = filesToParts(message.files);
  parts.push(...fileParts);

  return { content: parts, attachments };
}

function activePath(byId: Map<string, Node>, currentId: unknown): Set<string> {
  const path = new Set<string>();
  let cursor = str(currentId);
  while (cursor) {
    if (path.has(cursor)) break; // cycle
    const node = byId.get(cursor);
    if (!node) break;
    path.add(cursor);
    cursor = node.parentId;
  }
  return path;
}

function collectNodes(chat: Dict): Node[] {
  const history = isDict(chat.history) ? chat.history : null;
  const historyMessages = history && isDict(history.messages) ? history.messages : null;

  const byId = new Map<string, Node>();
  for (const [id, raw] of Object.entries(historyMessages ?? {})) {
    if (!isDict(raw)) continue;
    byId.set(id, { id, parentId: str(raw.parentId), raw });
  }

  if (byId.size === 0) {
    const flat = Array.isArray(chat.messages) ? chat.messages : [];
    const nodes: Node[] = [];
    const seen = new Set<string>();
    let previousId: string | null = null;
    for (const raw of flat) {
      if (!isDict(raw)) continue;
      if (nodes.length >= MAX_MESSAGES_PER_CHAT) break;
      const id = str(raw.id) ?? crypto.randomUUID();
      if (seen.has(id)) continue; // duplicate ids after a bad merge
      seen.add(id);
      nodes.push({ id, parentId: previousId, raw });
      previousId = id;
    }
    return nodes;
  }

  // Re-derive children from parentId: `childrenIds` goes stale on edits.
  const children = new Map<string, Node[]>();
  const roots: Node[] = [];
  for (const node of byId.values()) {
    const parent = node.parentId ? byId.get(node.parentId) : undefined;
    if (!parent || parent.id === node.id) {
      node.parentId = null;
      roots.push(node);
      continue;
    }
    const siblings = children.get(parent.id);
    if (siblings) siblings.push(node);
    else children.set(parent.id, [node]);
  }

  const path = activePath(byId, history?.currentId);
  const ordered: Node[] = [];
  const visited = new Set<string>();

  // Depth-first, active branch last, since studio restores the head from the final message.
  // Iterative to avoid stack overflow on deep chains.
  const walk = (root: Node): void => {
    const stack: Node[] = [root];
    while (stack.length > 0) {
      const node = stack.pop() as Node;
      if (visited.has(node.id)) continue;
      if (ordered.length >= MAX_MESSAGES_PER_CHAT) return;
      visited.add(node.id);
      ordered.push(node);
      const kids = children.get(node.id) ?? [];
      const rest = kids.filter((kid) => !path.has(kid.id));
      const active = kids.filter((kid) => path.has(kid.id));
      for (let index = active.length - 1; index >= 0; index--) stack.push(active[index]);
      for (let index = rest.length - 1; index >= 0; index--) stack.push(rest[index]);
    }
  };

  // Walk cycle-trapped nodes before the selected branch so it is still emitted last.
  const reachable = new Set<string>();
  const pending = [...roots];
  while (pending.length > 0) {
    const node = pending.pop() as Node;
    if (reachable.has(node.id)) continue;
    reachable.add(node.id);
    for (const kid of children.get(node.id) ?? []) pending.push(kid);
  }

  const rootRest = roots.filter((node) => !path.has(node.id));
  const rootActive = roots.filter((node) => path.has(node.id));
  for (const node of rootRest) walk(node);
  for (const node of byId.values()) {
    if (!reachable.has(node.id) && !visited.has(node.id)) walk(node);
  }
  for (const node of rootActive) walk(node);
  for (const node of byId.values()) {
    if (!visited.has(node.id)) walk(node);
  }

  return ordered;
}

function earliestTimestamp(nodes: Node[]): number | null {
  let earliest: number | null = null;
  for (const node of nodes) {
    const ts = epochMs(node.raw.timestamp);
    if (ts !== null && (earliest === null || ts < earliest)) earliest = ts;
  }
  return earliest;
}

function roleOf(raw: Dict): MessageRecord["role"] {
  const role = typeof raw.role === "string" ? raw.role : "";
  if (role === "user") return "user";
  if (role === "system") return "system";
  return "assistant";
}

export function openWebUIRecordToConversation(
  record: unknown,
  fallbackTitle: string,
): ParsedConversation | null {
  const chat = chatBlob(record);
  if (!chat) return null;
  const outer = isDict(record) ? record : {};

  const threadId = crypto.randomUUID();
  const nodes = collectNodes(chat);
  // `updated_at` is the conversation end, so prefer the first message's time.
  const createdAt =
    epochMs(outer.created_at) ??
    epochMs(chat.timestamp) ??
    earliestTimestamp(nodes) ??
    epochMs(outer.updated_at) ??
    Date.now();

  const messages: MessageRecord[] = [];
  // Stored messages sort by createdAt, so timestamps must strictly increase.
  let previousTs = createdAt - 1;
  const keptIdByOriginal = new Map<string, string>();

  for (const node of nodes) {
    const role = roleOf(node.raw);
    const { content, attachments } = messageParts(node.raw, role);
    // Empty messages are dropped and children relink; attachment-only user turns still render.
    const renders = content.length > 0 || (role === "user" && attachments.length > 0);
    const parentId = node.parentId ? (keptIdByOriginal.get(node.parentId) ?? null) : null;
    if (!renders) {
      if (node.parentId) {
        const inherited = keptIdByOriginal.get(node.parentId);
        if (inherited) keptIdByOriginal.set(node.id, inherited);
      }
      continue;
    }

    const id = crypto.randomUUID();
    keptIdByOriginal.set(node.id, id);
    const sourceTimestamp = epochMs(node.raw.timestamp);
    const ts = Math.max(previousTs + 1, sourceTimestamp ?? 0);
    previousTs = ts;

    messages.push({
      id,
      threadId,
      parentId,
      role,
      content: content as MessageRecord["content"],
      ...(attachments.length > 0
        ? { attachments: attachments as MessageRecord["attachments"] }
        : {}),
      createdAt: ts,
      ...((sourceTimestamp === null || ts !== sourceTimestamp) && {
        metadata: { createdAtEstimated: true },
      }),
    });
  }

  if (messages.length === 0) return null;

  return {
    title: str(chat.title) ?? str(outer.title) ?? fallbackTitle,
    threadId,
    messages,
    archived: outer.archived === true,
    createdAt,
  };
}
