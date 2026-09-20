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

/** How a reopened tab re-arms a tool call that is still waiting on a human.
 *
 *  The live stream registers a parked call with the confirmation store, and `ToolConfirmationControls`
 *  renders Approve/Deny only for a card that has an entry there. Recovery rebuilds the card but has no
 *  store of its own, so without this the reopened tab shows the call spinning with no way to answer it
 *  while the backend sits parked -- see `state/tool_approvals.wait_tool_decision`, which waits for the
 *  returning session precisely so that tab can answer.
 *
 *  Passed in rather than imported so this module stays a pure util: the caller owns the store. Both
 *  calls must be idempotent -- a card can be re-raised from the seed AND re-folded from its frame. */
export type RecoveredToolConfirmations = {
  /** This card is waiting on a decision. `partId` is the card's own `toolCallId`, which is what the
   *  controls look themselves up by, whichever path minted it. `sessionId` is the run's sandbox
   *  session, which the decision is resolved against and which scopes "Always allow". */
  register: (partId: string, approvalId: string, sessionId: string) => void;
  /** It is no longer waiting: answered, denied, timed out, or finished. */
  resolve: (partId: string) => void;
};

export function createGenerationToolRecovery(
  carried: CarriedPart[],
  runId: string,
  snapshotSeq = 0,
  toolConfirmations?: RecoveredToolConfirmations,
) {
  const pending = new Map<string, CarriedPart>();
  /** Arm the card a frame or a seed says is parked. Reads the id off the part rather than taking one,
   *  so the seed path and the fold path cannot disagree about which card is being armed. */
  /** Cards this recovery armed, so it only ever resolves its own. The live adapter clears
   *  unconditionally because it owns every card in its stream; a recovery shares the store with
   *  whatever else is on screen, so reaching for a card it never raised is not its business. */
  const armed = new Set<string>();
  const armApproval = (entry: CarriedPart, approvalId: unknown, sessionId: string) => {
    if (!toolConfirmations || typeof approvalId !== "string" || !approvalId) return;
    const partId = record(entry.part)?.toolCallId;
    if (typeof partId === "string" && partId) {
      armed.add(partId);
      toolConfirmations.register(partId, approvalId, sessionId);
    }
  };
  const disarmApproval = (entry: CarriedPart) => {
    const partId = record(entry.part)?.toolCallId;
    if (!toolConfirmations || typeof partId !== "string" || !armed.has(partId)) return;
    armed.delete(partId);
    toolConfirmations.resolve(partId);
  };
  /** Drop every card this recovery armed, for the end of the run rather than the end of a call.
   *  A run that terminates WITHOUT a tool_end (the backend failed or restarted while the call was
   *  parked) never reaches disarmApproval, so the card would outlive its own run: buttons still on
   *  screen, tool group still open, and a decision that can only 404 because the backend's pending
   *  slot went with the restart. Still only this recovery's own cards, for the same reason
   *  disarmApproval is scoped that way. */
  const disarmAll = () => {
    if (!toolConfirmations) return;
    for (const partId of armed) toolConfirmations.resolve(partId);
    armed.clear();
  };
  const researchHandoff = newDeepResearchHandoff();
  /** The sources a finished search card yields, parsed once per card rather than per publish.
   *  Every rebuild used to re-run the parse over every finished search result in the turn,
   *  which measured as the bulk of a recovery's per-publish cost on a long tool-using reply.
   *  `apply` replaces a card object wholesale rather than editing one, so an entry that is
   *  still the same object still holds the same result; the result is compared as well, so
   *  an in-place edit somewhere else could not make this serve a stale list either. */
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
  // A source a previous recovery appended is carried at the offset it was appended AT, so text
  // replayed after it lands behind it and cuts the reply in two, breaking any markdown that
  // spans the cut. `withSources` rebuilds these from the card, so drop them and let every
  // rebuild re-append them, which is also where the live adapter puts them. A citation source
  // is anchored where it arrived and has no card to rebuild it, so it stays.
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
  /** Re-arm every call that parked BEFORE the tab closed.
   *
   *  Such a call was saved as an unresolved card and its `tool_start` sits at or below the cursor, so
   *  no frame re-folds it and the fold's own registration never fires for it: the seed is the only
   *  place it still exists. Called by the follower once the run's session is known rather than at
   *  construction, because the decision is resolved against that session. A call whose `tool_start`
   *  lands above the cursor arms itself as the frame folds; `register` is idempotent, so both paths
   *  can name the same pair without raising two cards. */
  const armSeededApprovals = (sessionId: unknown) => {
    if (!toolConfirmations) return;
    for (const entry of savedPending) {
      armApproval(
        entry,
        record(entry.part)?.toolApprovalId,
        typeof sessionId === "string" ? sessionId : "",
      );
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
      // The frame says this call is waiting on a human, so the card must offer the decision here too:
      // the backend parks for the returning session, and this tab IS the returning session.
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
    // The call is answered, denied, or timed out: the card now has a result, so the decision is gone.
    // Unconditional, as the live path's clearToolConfirmation is: a card that was never armed resolves
    // to a no-op, and a card left armed would offer buttons over a finished result.
    disarmApproval(entry);
    if (backendId) completed.set(backendId, entry);
    else lastIdless = entry;
  };
  // Recovery never reaches the live path's end-of-stream source yield, so rebuild those entries.
  // Per occurrence, not per url, because the live path flat-maps the cards: two rounds finding
  // the same page carry their own title and snippet, and the Sources panel lists both.
  const withSources = <TPart>(parts: TPart[]): TPart[] => {
    const out: TPart[] = [...parts];
    for (const { part } of carried) {
      const card = searchCard(part);
      if (!card) continue;
      for (const source of cardSources(card)) {
        // Copied rather than handed out from the cache, `metadata` included: before the
        // cache each rebuild yielded its own objects all the way down, and a caller that
        // edits a part it was given must not reach back into what the next rebuild yields.
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
