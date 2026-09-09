// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SANDBOX_FILE_TOOLS,
  extractCreatedFiles,
} from "@/components/assistant-ui/sandbox-files";
import {
  SEARCH_IMAGE_TOOL,
  extractSearchImages,
  searchResultText,
} from "../search-images/search-images";
import {
  mergedToolCallArgumentsText,
  toolCallArgumentsText,
} from "../tool-call-arguments";
import type { CarriedPart } from "./chat-generation-recovery";
import {
  newDeepResearchHandoff,
  readDeepResearchToolEvent,
} from "./deep-research-handoff";
import {
  documentCitationToSource,
  parseSourcesFromResult,
} from "./document-citation-source";
import { mergeGoogleNativeParts } from "./google-native-parts";

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

export function createGenerationToolRecovery(
  carried: CarriedPart[],
  runId: string,
  snapshotSeq = 0,
) {
  const pending = new Map<string, CarriedPart>();
  const researchHandoff = newDeepResearchHandoff();
  const sourceIds = new Set(
    carried.flatMap(({ part }) => {
      const source = record(part);
      return source?.type === "source" && typeof source.id === "string"
        ? [source.id]
        : [];
    }),
  );
  const savedPending = carried.filter((entry) => {
    const part = record(entry.part);
    return part?.type === "tool-call" && part.result === undefined;
  });
  // Id-less cards share the empty id: one slot each, else all but the last stay running forever.
  let savedIdless = 0;
  for (const entry of savedPending) {
    const id = record(entry.part)?.backendToolCallId;
    if (typeof id === "string") {
      pending.set(id || `#idless:saved:${savedIdless++}`, entry);
    }
  }
  const replayFrom = savedPending.some(
    (entry) => typeof record(entry.part)?.backendToolCallId !== "string",
  )
    ? 0
    : snapshotSeq;
  const legacyPending = savedPending.filter(
    (entry) => typeof record(entry.part)?.backendToolCallId !== "string",
  );
  const claimLegacy = (entry: CarriedPart | undefined) => {
    const at = entry ? legacyPending.indexOf(entry) : -1;
    if (at !== -1) legacyPending.splice(at, 1);
    return entry;
  };
  // Seeded from saves: a reload between two completions of one card leaves it in no other lookup.
  const completed = new Map<string, CarriedPart>();
  /** Most recent finished card a provider gave no id, for a repeated id-less ending. */
  let lastIdless: CarriedPart | undefined;
  for (const entry of carried) {
    const part = record(entry.part);
    const id = part?.backendToolCallId;
    if (part?.type !== "tool-call" || part.result === undefined) continue;
    if (typeof id === "string" && id) completed.set(id, entry);
    else if (id === "") lastIdless = entry;
  }
  /** A card the previous frontend saved carries its backend id inside toolCallId and nowhere
   *  else, so a later completion has to recognise it the way the pending lookup already does. */
  const findCompletedLegacy = (backendId: string) => {
    if (!backendId) return undefined;
    for (let i = carried.length - 1; i >= 0; i--) {
      const part = record(carried[i].part);
      const id = part?.toolCallId;
      if (
        part?.type !== "tool-call" ||
        part.result === undefined ||
        part.backendToolCallId !== undefined ||
        typeof id !== "string"
      ) {
        continue;
      }
      if (id === backendId || id.startsWith(`${backendId}:`)) return carried[i];
    }
    return undefined;
  };
  const findSavedEntry = (backendId: string, approvalId: unknown) => {
    const matches = savedPending.filter((entry) => {
      const part = record(entry.part);
      const id = part?.toolCallId;
      if (!part || typeof id !== "string" || part.result !== undefined)
        return false;
      if (typeof approvalId === "string" && approvalId) {
        return (
          part.toolApprovalId === approvalId ||
          id === approvalId ||
          id.endsWith(`:${approvalId}`)
        );
      }
      return (
        Boolean(backendId) &&
        (id === backendId || id.startsWith(`${backendId}:`))
      );
    });
    return matches.length === 1 ? matches[0] : undefined;
  };
  let appliedSeq = 0;
  const apply = (
    payload: unknown,
    at: number,
    seq: number,
    sessionId = "_default",
  ) => {
    const chunk = record(payload);
    const event = record(chunk?._toolEvent) ?? chunk;
    if (
      event?.type !== "tool_start" &&
      event?.type !== "tool_end" &&
      event?.type !== "document_citations"
    ) {
      return;
    }
    if (seq <= appliedSeq) return;
    appliedSeq = seq;
    const backendId =
      typeof event.tool_call_id === "string" ? event.tool_call_id : "";
    if (event.tool_name === "deep_research") {
      if (event.type === "tool_start")
        researchHandoff.hiddenCallIds.delete(backendId);
      if (readDeepResearchToolEvent(researchHandoff, event)) return;
    }
    // Older approval cards need their original start event to recover the backend id.
    if (seq <= snapshotSeq) {
      const entry =
        event.type === "tool_start"
          ? claimLegacy(
              findSavedEntry(backendId, event.approval_id) ??
                (backendId ? undefined : legacyPending[0]),
            )
          : undefined;
      if (entry) {
        entry.part = {
          ...record(entry.part),
          backendToolCallId: backendId,
          generationToolCallId: `${runId}:${seq}`,
        };
        pending.set(backendId || `#idless:legacy:${seq}`, entry);
      }
      return;
    }
    if (event.type === "document_citations") {
      if (Array.isArray(event.citations)) {
        event.citations.forEach((value, index) => {
          const citation = record(value);
          const part = citation
            ? documentCitationToSource(citation, index)
            : null;
          if (!part || sourceIds.has(part.id)) return;
          carried.push({ at, part });
          sourceIds.add(part.id);
        });
      }
      return;
    }
    const toolName = typeof event.tool_name === "string" ? event.tool_name : "";
    let entry =
      (backendId ? pending.get(backendId) : undefined) ??
      findSavedEntry(backendId, event.approval_id);
    // Id-less end closes the most recent start, as the adapter does; else two calls recover as one.
    if (
      event.type === "tool_end" &&
      !(entry || backendId) &&
      pending.size > 0
    ) {
      for (const active of pending.values()) entry = active;
    }
    // OpenAI Responses ends a web search twice (placeholder, then citations); a start clears this.
    if (!entry && event.type === "tool_end") {
      entry = backendId
        ? (completed.get(backendId) ?? findCompletedLegacy(backendId))
        : lastIdless;
    }
    // Gemini can emit a second completion carrying a generated image.
    if (
      !entry &&
      event.type === "tool_end" &&
      record(record(event.google)?.native_part)
    ) {
      for (let i = carried.length - 1; i >= 0; i--) {
        const candidate = record(carried[i].part);
        if (
          candidate?.type !== "tool-call" ||
          !record(record(record(candidate.args)?.google)?.native_part)
        )
          continue;
        const id = candidate.toolCallId;
        if (
          backendId &&
          (candidate.backendToolCallId === backendId ||
            (candidate.backendToolCallId === undefined &&
              typeof id === "string" &&
              (id === backendId || id.startsWith(`${backendId}:`))))
        ) {
          entry = carried[i];
          break;
        }
      }
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
          },
        };
        carried.push(entry);
      }
      const args = record(event.arguments) ?? {};
      entry.part = {
        ...record(entry.part),
        backendToolCallId: backendId,
        generationToolCallId: `${runId}:${seq}`,
        ...(typeof event.approval_id === "string" && event.approval_id
          ? { toolApprovalId: event.approval_id }
          : {}),
        toolName,
        args,
        argsText: toolCallArgumentsText(event.arguments_text, args),
        ...(record(event.provenance) ? { provenance: event.provenance } : {}),
      };
      pending.set(backendId || `#idless:${runId}:${seq}`, entry);
      if (backendId) completed.delete(backendId);
      else lastIdless = undefined;
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
    const args = mergeGoogleNativeParts(
      { ...record(part.args), ...nextArgs },
      event.google,
    );
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
    if (backendId) completed.set(backendId, entry);
    else lastIdless = entry;
  };
  // Recovery never reaches the live path's end-of-stream source yield, so rebuild those entries.
  const withSources = <TPart>(parts: TPart[]): TPart[] => {
    const seen = new Set(sourceIds);
    const out: TPart[] = [...parts];
    for (const { part } of carried) {
      const card = record(part);
      if (
        card?.type !== "tool-call" ||
        card.result === undefined ||
        (card.toolName !== "web_search" && card.toolName !== "web_fetch")
      ) {
        continue;
      }
      for (const source of parseSourcesFromResult(
        searchResultText(card.result),
      )) {
        if (seen.has(source.id)) continue;
        seen.add(source.id);
        out.push(source as TPart);
      }
    }
    return out;
  };
  return { replayFrom, apply, withSources };
}
