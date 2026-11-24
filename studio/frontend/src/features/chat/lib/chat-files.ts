// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getCodeFilename } from "../../../components/assistant-ui/code-filename.ts";
import {
  IMAGE_SENTINEL_TOOLS,
  SANDBOX_FILE_TOOLS,
  isSandboxToolResult,
} from "../../../components/assistant-ui/sandbox-files.ts";
import { isRenderableRenderHtmlToolPart } from "../artifacts/html-fences.ts";
import { getArtifactFilename } from "../artifacts/types.ts";

export type ChatFileKind = "code" | "html" | "image" | "file";

export type ChatFile = {
  id: string;
  name: string;
  kind: ChatFileKind;
  /** null: in the chat folder, but no message names it (uploaded, or written by a tool that did not list it). */
  messageId: string | null;
  text?: string;
  dataUrl?: string;
  sandbox?: { sessionId: string; name: string };
};

/** The chat folder listing (`GET /api/inference/sandbox/{id}`). */
export type ChatFolder = { sessionId: string; names: readonly string[] };

type PartLike = {
  type?: string;
  text?: unknown;
  toolName?: string;
  args?: unknown;
  result?: unknown;
};

type MessageLike = {
  id: string;
  role: string;
  content: readonly unknown[];
};

const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|avif|svg)$/i;
const FENCE_OPEN_RE = /^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)/;
// Mirrors _MAX_SNAPSHOT_FILES: a listing this long may be cut, so absence proves nothing.
const FOLDER_LISTING_CAP = 2000;

export function extractCodeFences(
  markdown: string,
): { language: string | null; source: string }[] {
  const blocks: { language: string | null; source: string }[] = [];
  const lines = markdown.split("\n");
  let i = 0;
  while (i < lines.length) {
    const open = FENCE_OPEN_RE.exec(lines[i]);
    if (!open) {
      i += 1;
      continue;
    }
    const fence = open[1];
    const closeRe = new RegExp(`^ {0,3}${fence[0]}{${fence.length},}\\s*$`);
    const body: string[] = [];
    i += 1;
    while (i < lines.length && !closeRe.test(lines[i])) {
      body.push(lines[i]);
      i += 1;
    }
    if (i >= lines.length) break;
    i += 1;
    const source = body.join("\n");
    if (source.trim()) blocks.push({ language: open[2] || null, source });
  }
  return blocks;
}

function baseName(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1) || path;
}

function extFromMime(mime: string): string {
  const sub = mime
    .split("/")[1]
    ?.split(/[+;]/)[0]
    ?.toLowerCase()
    .replace(/[^a-z0-9]/g, "");
  return sub === "jpeg" ? "jpg" : sub || "png";
}

function uniqueName(name: string, used: Map<string, number>): string {
  const count = (used.get(name) ?? 0) + 1;
  used.set(name, count);
  if (count === 1) return name;
  const dot = name.lastIndexOf(".");
  return dot > 0
    ? `${name.slice(0, dot)}-${count}${name.slice(dot)}`
    : `${name}-${count}`;
}

type Found = Omit<ChatFile, "name"> & { name: string };

const messageCache = new WeakMap<object, Found[]>();

function sandboxFound(
  id: string,
  messageId: string | null,
  sessionId: string,
  name: string,
): Found {
  return {
    id,
    name: baseName(name),
    kind: IMAGE_EXT.test(name) ? "image" : "file",
    messageId,
    sandbox: { sessionId, name },
  };
}

function collectMessageFiles(message: MessageLike): Found[] {
  const cached = messageCache.get(message);
  if (cached) return cached;
  const found: Found[] = [];
  message.content.forEach((raw, partIndex) => {
    const part = raw as PartLike;
    const base = `${message.id}:${partIndex}`;
    if (part.type === "text" && typeof part.text === "string") {
      extractCodeFences(part.text).forEach((block, blockIndex) => {
        const isHtml = block.language?.toLowerCase() === "html";
        found.push({
          id: `${base}:${blockIndex}`,
          name: isHtml ? "snippet.html" : getCodeFilename(block.language),
          kind: isHtml ? "html" : "code",
          messageId: message.id,
          text: block.source,
        });
      });
      return;
    }
    if (part.type !== "tool-call") return;
    const toolName = part.toolName ?? "";
    if (isRenderableRenderHtmlToolPart(part)) {
      const args = (part.args ?? {}) as { code?: unknown; title?: unknown };
      if (typeof args.code !== "string" || !args.code.trim()) return;
      const title =
        typeof args.title === "string" && args.title.trim()
          ? args.title
          : "page";
      found.push({
        id: base,
        name: getArtifactFilename({ title }),
        kind: "html",
        messageId: message.id,
        text: args.code,
      });
      return;
    }
    const result = part.result;
    if (
      (SANDBOX_FILE_TOOLS.has(toolName) ||
        IMAGE_SENTINEL_TOOLS.has(toolName)) &&
      // Hosted code execution plots live with the provider, not in the chat folder.
      toolName !== "code_execution" &&
      isSandboxToolResult(result)
    ) {
      const { sessionId } = result;
      const { images, files } = result as unknown as {
        images: unknown[];
        files?: { name: string }[] | null;
      };
      const names = [
        ...(files ?? []).map((entry) => entry.name),
        ...images.filter((entry): entry is string => typeof entry === "string"),
      ];
      names.forEach((name, index) => {
        found.push(
          sandboxFound(`${base}:${index}`, message.id, sessionId, name),
        );
      });
      return;
    }
    if (
      toolName === "image_generation" &&
      result &&
      typeof result === "object" &&
      typeof (result as { image_b64?: unknown }).image_b64 === "string"
    ) {
      const { image_b64, image_mime } = result as {
        image_b64: string;
        image_mime?: unknown;
      };
      const mime = typeof image_mime === "string" ? image_mime : "image/png";
      found.push({
        id: base,
        name: `generated-image.${extFromMime(mime)}`,
        kind: "image",
        messageId: message.id,
        dataUrl: `data:${mime};base64,${image_b64}`,
      });
    }
  });
  messageCache.set(message, found);
  return found;
}

const sandboxKey = (sessionId: string, name: string) =>
  `${sessionId}\u0000${name}`;

/**
 * Everything downloadable from a chat, in message order, then folder files no message names.
 * With the folder listed, a named file it no longer holds is dropped (its download would 404).
 */
export function collectChatFiles(
  messages: readonly MessageLike[],
  folder: ChatFolder | null = null,
): ChatFile[] {
  const folderId = folder?.sessionId;
  const present = new Set(folder?.names ?? []);
  const trustAbsence =
    folder !== null && folder.names.length < FOLDER_LISTING_CAP;
  const seen = new Set<string>();
  const found: Found[] = [];
  for (const message of messages) {
    if (message.role !== "assistant") {
      continue;
    }
    for (const file of collectMessageFiles(message)) {
      const key = file.sandbox
        ? sandboxKey(file.sandbox.sessionId, file.sandbox.name)
        : null;
      if (key !== null && seen.has(key)) {
        continue;
      }
      if (key !== null) {
        seen.add(key);
      }
      const gone =
        trustAbsence &&
        file.sandbox?.sessionId === folderId &&
        !present.has(file.sandbox?.name ?? "");
      if (!gone) {
        found.push(file);
      }
    }
  }
  if (folderId !== undefined) {
    for (const name of present) {
      if (!seen.has(sandboxKey(folderId, name))) {
        found.push(sandboxFound(`folder:${name}`, null, folderId, name));
      }
    }
  }
  const used = new Map<string, number>();
  return found.map((file) => ({ ...file, name: uniqueName(file.name, used) }));
}
