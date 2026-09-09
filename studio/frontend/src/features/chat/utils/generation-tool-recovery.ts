// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SANDBOX_FILE_TOOLS,
  extractCreatedFiles,
} from "@/components/assistant-ui/sandbox-files";
import {
  SEARCH_IMAGE_TOOL,
  extractSearchImages,
} from "../search-images/search-images";
import {
  mergedToolCallArgumentsText,
  toolCallArgumentsText,
} from "../tool-call-arguments";
import type { CarriedPart } from "./chat-generation-recovery";

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function recoveredToolResult(
  event: Record<string, unknown>,
  toolName: unknown,
  sessionId: string,
): unknown {
  if (toolName === "image_generation" && typeof event.image_b64 === "string") {
    return {
      image_b64: event.image_b64,
      image_mime: event.image_mime ?? "image/png",
      size: event.size,
      quality: event.quality,
      background: event.background,
      prompt: event.prompt,
    };
  }
  if (typeof event.result !== "string") {
    return event.result ?? "";
  }
  const sandbox =
    typeof toolName === "string" && SANDBOX_FILE_TOOLS.has(toolName);
  const { text, files } = sandbox
    ? extractCreatedFiles(event.result)
    : { text: event.result, files: [] };
  const mcpMarker = "\n__MCP_IMAGES__:";
  const mcpAt = text.lastIndexOf(mcpMarker);
  if (mcpAt !== -1) {
    try {
      const images: unknown = JSON.parse(text.slice(mcpAt + mcpMarker.length));
      if (
        Array.isArray(images) &&
        images.length > 0 &&
        images.every(
          (image) =>
            typeof record(image)?.data === "string" &&
            typeof record(image)?.mimeType === "string",
        )
      ) {
        return { text: text.slice(0, mcpAt), images };
      }
    } catch {
      // Keep malformed envelopes as text.
    }
  }
  const imageMarker = "\n__IMAGES__:";
  const imageAt = text.lastIndexOf(imageMarker);
  if (imageAt !== -1) {
    try {
      const images: unknown = JSON.parse(
        text.slice(imageAt + imageMarker.length),
      );
      if (
        Array.isArray(images) &&
        images.every((image) => typeof image === "string")
      ) {
        return { text: text.slice(0, imageAt), images, sessionId, files };
      }
    } catch {
      // Keep malformed envelopes as text.
    }
  }
  if (sandbox) {
    return { text, images: [], sessionId, files };
  }
  if (toolName === SEARCH_IMAGE_TOOL) {
    const search = extractSearchImages(text);
    if (search.images.length > 0) {
      return { text: search.text, webImages: search.images };
    }
  }
  return text;
}

/** Update carried cards as a stored run replays tool events. */
export function createGenerationToolRecovery(
  carried: CarriedPart[],
  runId: string,
): (payload: unknown, at: number, seq: number, sessionId?: string) => void {
  const pending = new Map<string, CarriedPart>();
  for (const entry of carried) {
    const part = record(entry.part);
    if (part?.type !== "tool-call" || part.result !== undefined) {
      continue;
    }
    const id = part.backendToolCallId ?? part.toolCallId;
    if (typeof id === "string") {
      pending.set(id.split(":")[0], entry);
    }
  }
  return (payload, at, seq, sessionId = "_default") => {
    const chunk = record(payload);
    const event = record(chunk?._toolEvent) ?? chunk;
    if (event?.type !== "tool_start" && event?.type !== "tool_end") {
      return;
    }
    const backendId =
      typeof event.tool_call_id === "string" ? event.tool_call_id : "";
    const toolName = typeof event.tool_name === "string" ? event.tool_name : "";
    let entry = pending.get(backendId);
    if (
      event.type === "tool_end" &&
      !(entry || backendId) &&
      pending.size === 1
    ) {
      entry = pending.values().next().value;
    }
    if (event.type === "tool_start") {
      if (!toolName) {
        return;
      }
      if (!entry) {
        entry = {
          at,
          part: {
            type: "tool-call",
            toolCallId: `${backendId || "tool"}:${runId}:${seq}`,
            backendToolCallId: backendId,
          },
        };
        carried.push(entry);
      }
      const args = record(event.arguments) ?? {};
      entry.part = {
        ...record(entry.part),
        toolName,
        args,
        argsText: toolCallArgumentsText(event.arguments_text, args),
        ...(record(event.provenance) ? { provenance: event.provenance } : {}),
      };
      pending.set(backendId, entry);
      return;
    }
    if (!entry) {
      return;
    }
    const part = record(entry.part);
    if (!part) {
      return;
    }
    const nextArgs = record(event.arguments);
    const args = { ...record(part.args), ...nextArgs };
    entry.part = {
      ...part,
      args,
      argsText: mergedToolCallArgumentsText(
        part.argsText,
        args,
        Object.keys(nextArgs ?? {}),
      ),
      result: recoveredToolResult(event, part.toolName, sessionId),
      ...(record(event.provenance)
        ? {
            provenance: {
              ...record(part.provenance),
              ...record(event.provenance),
            },
          }
        : {}),
    };
    for (const [id, active] of pending) {
      if (active === entry) {
        pending.delete(id);
      }
    }
  };
}
