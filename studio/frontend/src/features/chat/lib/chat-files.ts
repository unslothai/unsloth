// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getCodeFilename } from "@/components/assistant-ui/code-filename";
import {
  IMAGE_SENTINEL_TOOLS,
  SANDBOX_FILE_TOOLS,
  isSandboxToolResult,
} from "@/components/assistant-ui/sandbox-files";
import { isRenderableRenderHtmlToolPart } from "../artifacts/html-fences";
import { getArtifactFilename } from "../artifacts/types";

export type ChatFileKind = "code" | "html" | "image" | "file";

export type ChatFile = {
  id: string;
  name: string;
  kind: ChatFileKind;
  messageId: string;
  text?: string;
  dataUrl?: string;
  sandbox?: { sessionId: string; name: string };
};

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
  const sub = mime.split("/")[1]?.split("+")[0];
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

type Found = { file: Omit<ChatFile, "name">; name: string; dedupeKey?: string };

const messageCache = new WeakMap<object, Found[]>();

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
          file: {
            id: `${base}:${blockIndex}`,
            kind: isHtml ? "html" : "code",
            messageId: message.id,
            text: block.source,
          },
          name: isHtml ? "snippet.html" : getCodeFilename(block.language),
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
        file: {
          id: base,
          kind: "html",
          messageId: message.id,
          text: args.code,
        },
        name: getArtifactFilename({ title }),
      });
      return;
    }
    const result = part.result;
    if (
      (SANDBOX_FILE_TOOLS.has(toolName) ||
        IMAGE_SENTINEL_TOOLS.has(toolName)) &&
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
        found.push({
          file: {
            id: `${base}:${index}`,
            kind: IMAGE_EXT.test(name) ? "image" : "file",
            messageId: message.id,
            sandbox: { sessionId, name },
          },
          name: baseName(name),
          dedupeKey: `${sessionId}\u0000${name}`,
        });
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
        file: {
          id: base,
          kind: "image",
          messageId: message.id,
          dataUrl: `data:${mime};base64,${image_b64}`,
        },
        name: `generated-image.${extFromMime(mime)}`,
      });
    }
  });
  messageCache.set(message, found);
  return found;
}

export function collectChatFiles(messages: readonly MessageLike[]): ChatFile[] {
  const seen = new Set<string>();
  const used = new Map<string, number>();
  const files: ChatFile[] = [];
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    for (const { file, name, dedupeKey } of collectMessageFiles(message)) {
      if (dedupeKey) {
        if (seen.has(dedupeKey)) continue;
        seen.add(dedupeKey);
      }
      files.push({ ...file, name: uniqueName(name, used) });
    }
  }
  return files;
}
