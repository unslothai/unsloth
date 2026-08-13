// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Converts Open WebUI exports to Studio conversations. `history.messages` is
 * the authoritative DAG and `currentId` selects its active branch; flat
 * `messages` is a legacy fallback. Both modern output items and legacy details
 * blocks are converted to Studio message parts.
 */

import type { MessageRecord, ParsedConversation } from "../types";

type Dict = Record<string, unknown>;

interface Node {
  id: string;
  parentId: string | null;
  raw: Dict;
}

/** A cyclic parent chain would otherwise spin forever; no real chat is deeper. */
const MAX_MESSAGES_PER_CHAT = 200_000;

function isDict(value: unknown): value is Dict {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function str(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

/** Open WebUI stores seconds on the record and milliseconds on `chat.timestamp`. */
function epochMs(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value < 1e12 ? Math.round(value * 1000) : Math.round(value);
}

/** The chat blob: under `chat` on an exported record, or the record itself on a legacy bare chat. */
function chatBlob(record: unknown): Dict | null {
  if (!isDict(record)) return null;
  if (isDict(record.chat)) return record.chat;
  return record;
}

/** Distinguish Open WebUI turns from plain OpenAI role/content messages. */
function looksLikeOpenWebUIMessage(value: unknown): boolean {
  if (!isDict(value)) return false;
  if (typeof value.role !== "string") return false;
  return (
    "parentId" in value ||
    "childrenIds" in value ||
    "timestamp" in value ||
    "modelName" in value ||
    "modelIdx" in value ||
    "done" in value ||
    "output" in value
  );
}

export function isOpenWebUIRecord(value: unknown): boolean {
  if (!isDict(value)) return false;
  // The wrapper is unambiguous: nothing else we import nests a chat blob.
  if (isDict(value.chat)) {
    const chat = value.chat;
    return (
      isDict(chat.history) || Array.isArray(chat.messages) || "models" in chat
    );
  }
  const blob = chatBlob(value);
  if (!blob) return false;
  if (isDict(blob.history) && isDict(blob.history.messages)) return true;
  return Array.isArray(blob.messages) && blob.messages.some(looksLikeOpenWebUIMessage);
}

// Content

const FENCED_CODE = /```[\s\S]*?```|~~~[\s\S]*?~~~/g;
const DETAILS_BLOCK = /<details\b([^>]*)>([\s\S]*?)<\/details>/gi;
const ATTRIBUTE = /([\w-]+)="([^"]*)"/g;

function unescapeHtml(value: string): string {
  return value
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
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

/** Body text of a `<details>`, minus its `<summary>`. */
function detailsBody(body: string, stripQuoteMarkers = false): string {
  const withoutSummary = body.replace(/<summary>[\s\S]*?<\/summary>/i, "");
  if (!stripQuoteMarkers) return withoutSummary.trim();
  // Open WebUI writes reasoning as a blockquote; a tool result keeps its own lines.
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
  // Text either side of a details block belongs to one bubble, not two.
  if (last && last.type === "text") {
    last.text = `${last.text as string}\n\n${trimmed}`;
    return;
  }
  parts.push({ type: "text", text: trimmed });
}

/** Byte ranges of fenced code in `content`, as [start, end) pairs. */
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

/**
 * Convert details blocks while preserving blocks quoted inside code fences.
 * A real tool result may itself contain fenced code.
 */
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
    if (attributes.type === "tool_calls") {
      const result =
        attributes.result !== undefined ? parseJsonLoose(attributes.result) : body || undefined;
      parts.push({
        type: "tool-call",
        toolCallId: attributes.id || crypto.randomUUID(),
        toolName: attributes.name || "unknown",
        args: parseJsonLoose(attributes.arguments),
        ...(result !== undefined ? { result } : {}),
      });
    } else if (attributes.type === "reasoning") {
      if (body) parts.push({ type: "reasoning", text: body });
    } else if (body) {
      // code_interpreter and any future block: keep the text, drop the markup.
      pushText(parts, body);
    }

    match = DETAILS_BLOCK.exec(content);
  }
  pushText(parts, content.slice(cursor));

  return parts;
}

/** Modern assistant turns: Responses-API items stored on `message.output`. */
function outputItemsToParts(output: unknown[]): unknown[] {
  const parts: unknown[] = [];
  const toolCallIndex = new Map<string, Dict>();

  for (const item of output) {
    if (!isDict(item)) continue;

    if (item.type === "reasoning") {
      const source = Array.isArray(item.summary)
        ? item.summary
        : Array.isArray(item.content)
          ? item.content
          : [];
      const text = source
        .map((part) => (isDict(part) && typeof part.text === "string" ? part.text : ""))
        .join("");
      if (text.trim()) parts.push({ type: "reasoning", text: text.trim() });
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
      pushText(parts, text);
      continue;
    }

    if (item.type === "function_call") {
      const callId = str(item.call_id) ?? crypto.randomUUID();
      const part: Dict = {
        type: "tool-call",
        toolCallId: callId,
        toolName: str(item.name) ?? "unknown",
        args: parseJsonLoose(typeof item.arguments === "string" ? item.arguments : undefined),
      };
      if (typeof item.arguments === "object" && item.arguments !== null) {
        part.args = item.arguments;
      }
      toolCallIndex.set(callId, part);
      parts.push(part);
      continue;
    }

    if (item.type === "function_call_output") {
      const callId = str(item.call_id);
      // `output` is a plain string for most tools and a content array only when
      // the tool returned images or files.
      const outputParts = Array.isArray(item.output) ? item.output : [];
      let resultText = typeof item.output === "string" ? item.output : "";
      const images: string[] = [];
      for (const part of outputParts) {
        if (!isDict(part)) continue;
        if (part.type === "input_text" && typeof part.text === "string") {
          resultText += part.text;
        } else if (part.type === "input_image") {
          const url = str(part.image_url);
          if (url) images.push(url);
        }
      }
      const target = callId ? toolCallIndex.get(callId) : undefined;
      if (target) {
        target.result = resultText;
      } else if (resultText.trim()) {
        pushText(parts, resultText);
      }
      // A tool that returned images: keep them as image parts so they still render.
      for (const image of images) {
        if (image.startsWith("data:")) parts.push({ type: "image", image });
      }
    }
  }

  return parts;
}

/**
 * Keep inline images. Keep document names without extracted text, which would
 * otherwise be resent to the model on the next turn.
 */
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

function messageParts(
  message: Dict,
  role: MessageRecord["role"],
): { content: unknown[]; attachments: unknown[] } {
  const parts: unknown[] = [];

  if (Array.isArray(message.output)) {
    parts.push(...outputItemsToParts(message.output));
  }

  const content = typeof message.content === "string" ? message.content : "";
  const hasText = parts.some((part) => isDict(part) && part.type === "text");
  // `content` mirrors the final assistant text that `output` already carries, so
  // it is only used when the items produced none.
  if (content && !hasText) {
    // Open WebUI writes those details blocks into its own assistant output. The
    // same markup in a prompt is text the user typed, not reasoning or a call.
    if (role === "assistant") parts.push(...contentWithDetailsToParts(content));
    else pushText(parts, content);
  }

  const { parts: fileParts, attachments } = filesToParts(message.files);
  parts.push(...fileParts);

  return { content: parts, attachments };
}

// Message graph

/** Message ids from `currentId` back to the root: the branch the user had open. */
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

  if (!historyMessages) {
    // No DAG: the flat active branch is all that survived.
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

  const byId = new Map<string, Node>();
  for (const [id, raw] of Object.entries(historyMessages)) {
    if (!isDict(raw)) continue;
    byId.set(id, { id, parentId: str(raw.parentId), raw });
  }

  // Re-derive children from parentId: `childrenIds` goes stale on edits, and an
  // orphan parent (deleted message) has to fall back to being a root.
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

  // Depth-first, active branch last: studio restores the head from the final
  // message, so the branch the user had open is the one that opens here too.
  // The traversal is iterative because a long chat is a chain thousands of
  // messages deep, which overflows the call stack long before the cap above.
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
      // Pushed back to front, so they pop in that same order.
      for (let index = active.length - 1; index >= 0; index--) stack.push(active[index]);
      for (let index = rest.length - 1; index >= 0; index--) stack.push(rest[index]);
    }
  };

  const rootRest = roots.filter((node) => !path.has(node.id));
  const rootActive = roots.filter((node) => path.has(node.id));
  for (const node of [...rootRest, ...rootActive]) walk(node);
  // Anything only reachable through a parent cycle still belongs in the thread.
  for (const node of byId.values()) {
    if (!visited.has(node.id)) walk(node);
  }

  return ordered;
}

function roleOf(raw: Dict): MessageRecord["role"] {
  const role = typeof raw.role === "string" ? raw.role : "";
  if (role === "user") return "user";
  if (role === "system") return "system";
  return "assistant";
}

/**
 * Convert one exported Open WebUI chat. Returns null when it holds no message
 * we can render (an empty chat, or one whose turns were all blank).
 */
export function openWebUIRecordToConversation(
  record: unknown,
  fallbackTitle: string,
): ParsedConversation | null {
  const chat = chatBlob(record);
  if (!chat) return null;
  const outer = isDict(record) ? record : {};

  const threadId = crypto.randomUUID();
  const createdAt =
    epochMs(outer.created_at) ?? epochMs(chat.timestamp) ?? epochMs(outer.updated_at) ?? Date.now();

  const messages: MessageRecord[] = [];
  // Studio sorts stored messages by createdAt, so the timeline has to be
  // strictly increasing or the depth-first order above would not survive a
  // reload. Real timestamps are kept whenever they already increase.
  let previousTs = createdAt - 1;
  const keptIdByOriginal = new Map<string, string>();

  for (const node of collectNodes(chat)) {
    const role = roleOf(node.raw);
    const { content, attachments } = messageParts(node.raw, role);
    // A message that renders to nothing (a failed turn holding only an error)
    // is dropped, and its children relink to the nearest ancestor that stayed.
    // A user turn that uploaded a file and typed nothing still renders, as
    // studio shows attachments on user messages.
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
    const ts = Math.max(previousTs + 1, epochMs(node.raw.timestamp) ?? 0);
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
