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
import { extractMcpUiEnvelope } from "../mcp-apps/mcp-ui";
import {
  providerCompactionPart,
  providerCompactionReplayToolCallCount,
} from "./provider-compaction";

function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function mcpImages(
  text: string,
): { at: number; images: { data: string; mimeType: string }[] } | null {
  const marker = "\n__MCP_IMAGES__:";
  const at = text.lastIndexOf(marker);
  if (at === -1) return null;
  try {
    const images: unknown = JSON.parse(text.slice(at + marker.length));
    return Array.isArray(images) &&
      images.length > 0 &&
      images.every(
        (image) =>
          typeof record(image)?.data === "string" &&
          typeof record(image)?.mimeType === "string",
      )
      ? { at, images: images as { data: string; mimeType: string }[] }
      : null;
  } catch {
    return null;
  }
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
  const { text: withUi, files } = sandbox
    ? extractCreatedFiles(event.result)
    : { text: event.result, files: [] };
  const { text, ui } = extractMcpUiEnvelope(
    withUi,
    typeof toolName === "string" ? toolName : "",
  );
  if (ui) {
    const images = mcpImages(text);
    return images
      ? { text: text.slice(0, images.at), images: images.images, ui }
      : { text, ui };
  }
  const images = mcpImages(text);
  if (images) return { text: text.slice(0, images.at), images: images.images };
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

/** Re-arms a tool call still waiting on approval after a tab reopens; the caller owns the store.
 *  Both calls must be idempotent: a card can be re-raised from the seed and its frame. */
export type RecoveredToolConfirmations = {
  /** `partId` is the card's toolCallId; `sessionId` scopes the decision and "Always allow". */
  register: (partId: string, approvalId: string, sessionId: string) => void;
  resolve: (partId: string) => void;
};

export function createGenerationToolRecovery(
  carried: CarriedPart[],
  runId: string,
  snapshotSeq = 0,
  toolConfirmations?: RecoveredToolConfirmations,
) {
  const pending = new Map<string, CarriedPart>();
  /** The store is shared with other cards on screen, so only resolve the ones armed here. */
  const armed = new Set<string>();
  const armApproval = (
    entry: CarriedPart,
    approvalId: unknown,
    sessionId: string,
  ) => {
    if (!toolConfirmations || typeof approvalId !== "string" || !approvalId)
      return;
    const partId = record(entry.part)?.toolCallId;
    if (typeof partId === "string" && partId) {
      armed.add(partId);
      toolConfirmations.register(partId, approvalId, sessionId);
    }
  };
  const disarmApproval = (entry: CarriedPart) => {
    const partId = record(entry.part)?.toolCallId;
    if (!toolConfirmations || typeof partId !== "string" || !armed.has(partId))
      return;
    armed.delete(partId);
    toolConfirmations.resolve(partId);
  };
  /** For runs ending without tool_end (backend failed or restarted), which would orphan the card. */
  const disarmAll = () => {
    if (!toolConfirmations) return;
    for (const partId of armed) toolConfirmations.resolve(partId);
    armed.clear();
  };
  const researchHandoff = newDeepResearchHandoff();
  /** Parsed once per card; `apply` replaces cards wholesale, so identity implies same result. */
  const parsedSources = new WeakMap<
    object,
    { result: unknown; sources: ReturnType<typeof parseSourcesFromResult> }
  >();
  const searchCard = (part: unknown) => {
    const card = record(part);
    return card?.type === "tool-call" &&
      card.result !== undefined &&
      (card.toolName === "web_search" || card.toolName === "web_fetch")
      ? card
      : undefined;
  };
  const cardSources = (card: Record<string, unknown>) => {
    const hit = parsedSources.get(card);
    if (hit && hit.result === card.result) return hit.sources;
    const sources = parseSourcesFromResult(searchResultText(card.result));
    parsedSources.set(card, { result: card.result, sources });
    return sources;
  };
  // Drop rebuildable sources so withSources re-appends them at the end; citations stay in place.
  const rebuildableSourceIds = new Set(
    carried.flatMap(({ part }) => {
      const card = searchCard(part);
      return card ? cardSources(card).map((source) => source.id) : [];
    }),
  );
  for (let i = carried.length - 1; i >= 0; i--) {
    const part = record(carried[i].part);
    if (
      part?.type === "source" &&
      typeof part.id === "string" &&
      rebuildableSourceIds.has(part.id)
    ) {
      carried.splice(i, 1);
    }
  }
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
  /** Re-arm calls parked before the tab closed; the seed is the only place they still exist. */
  /*  `stillPending` avoids arming a call already answered; a saved card looks the same either way. */
  const armSeededApprovals = async (
    sessionId: unknown,
    stillPending?: (approvalId: string) => Promise<boolean>,
  ) => {
    if (!toolConfirmations) return;
    const session = typeof sessionId === "string" ? sessionId : "";
    for (const entry of savedPending) {
      const approvalId = record(entry.part)?.toolApprovalId;
      if (stillPending && typeof approvalId === "string" && approvalId) {
        // A failed check falls back to arming, so the user keeps their buttons.
        let pending = true;
        try {
          pending = await stillPending(approvalId);
        } catch {
          pending = true;
        }
        if (!pending) continue;
      }
      // Re-read after the await: tool_end can fold this card concurrently.
      if (record(entry.part)?.result !== undefined) continue;
      armApproval(entry, approvalId, session);
    }
  };
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
  const completed = new Map<string, CarriedPart>();
  let lastIdless: CarriedPart | undefined;
  for (const entry of carried) {
    const part = record(entry.part);
    const id = part?.backendToolCallId;
    if (part?.type !== "tool-call" || part.result === undefined) continue;
    if (typeof id === "string" && id) completed.set(id, entry);
    else if (id === "") lastIdless = entry;
  }
  /** Legacy saved cards carry their backend id only inside toolCallId. */
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
      event?.type !== "document_citations" &&
      event?.type !== "compaction_block"
    ) {
      return;
    }
    if (seq <= appliedSeq) return;
    appliedSeq = seq;
    if (event.type === "compaction_block") {
      const providerCompaction = providerCompactionPart(event);
      if (!providerCompaction) return;
      return {
        providerCompaction,
        providerCompactionAfterToolCalls:
          providerCompactionReplayToolCallCount(
            carried.map((entry) => entry.part),
          ),
      };
    }
    const backendId =
      typeof event.tool_call_id === "string" ? event.tool_call_id : "";
    if (event.tool_name === "deep_research") {
      if (event.type === "tool_start")
        researchHandoff.hiddenCallIds.delete(backendId);
      if (readDeepResearchToolEvent(researchHandoff, event)) return;
    }
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
      const toolCallId = `${backendId || "tool"}:${runId}:${seq}`;
      entry ??= carried.find(({ part }) => {
        const card = record(part);
        return (
          card?.type === "tool-call" &&
          (card.toolCallId === toolCallId ||
            card.generationToolCallId === `${runId}:${seq}`)
        );
      });
      if (!entry) {
        entry = { at, part: { type: "tool-call", toolCallId } };
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
      // The backend parks for the returning session, and this tab is that session.
      if (event.awaiting_confirmation === true) {
        armApproval(entry, event.approval_id, sessionId);
      }
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
    disarmApproval(entry);
    if (backendId) completed.set(backendId, entry);
    else lastIdless = entry;
  };
  // Recovery never hits the live end-of-stream source yield; rebuild per occurrence, not per url.
  const withSources = <TPart>(parts: TPart[]): TPart[] => {
    const out: TPart[] = [...parts];
    for (const { part } of carried) {
      const card = searchCard(part);
      if (!card) continue;
      for (const source of cardSources(card)) {
        // Copy cached parts so callers editing them cannot alter the next rebuild.
        out.push({
          ...source,
          ...(source.metadata ? { metadata: { ...source.metadata } } : {}),
        } as TPart);
      }
    }
    return out;
  };
  return { replayFrom, apply, withSources, armSeededApprovals, disarmAll };
}
