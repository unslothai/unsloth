// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// routes/rag.py reports a dead vec0 via a list marker and 503s; the client used to drop both.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

const RAG_API = readSrc("features/rag/api/rag-api.ts");

registerBundlerResolver();

const {
  hasRagAvailabilityMarker,
  noteRagAvailability,
  noteRagResponse,
  useRagAvailabilityStore,
} = await import("../src/features/rag/api/rag-availability.ts");

const BACKEND_REASON =
  "RAG is unavailable: the sqlite-vec extension could not be loaded.";

function resetAvailability() {
  useRagAvailabilityStore.setState({
    available: true,
    reason: null,
    answered: false,
  });
}

function functionBody(src: string, name: string): string {
  const start = src.search(new RegExp(`(async )?function\\*?\\s+${name}\\b`));
  assert.ok(start >= 0, `${name} is gone or was renamed`);
  const end = src.indexOf("\n}\n", start);
  assert.ok(end > start, `could not find the end of ${name}`);
  return src.slice(start, end);
}

test("nothing is gated before the backend has answered", () => {
  resetAvailability();
  const state = useRagAvailabilityStore.getState();
  assert.equal(
    state.isUnavailable(),
    false,
    "the dialog would gray itself out on a guess, the mistake this PR already fixed " +
      "for the Train and Video tabs",
  );
  assert.equal(state.availabilityUnknown(), true);
  assert.equal(state.unavailableReason(), null);
});

test("a list marker of false is a measured unavailable, with its reason", () => {
  resetAvailability();
  noteRagAvailability({
    knowledgeBases: [],
    ragAvailable: false,
    ragUnavailableReason: BACKEND_REASON,
  });
  const state = useRagAvailabilityStore.getState();
  assert.equal(state.isUnavailable(), true);
  assert.equal(state.availabilityUnknown(), false);
  assert.equal(
    state.unavailableReason(),
    BACKEND_REASON,
    "the empty state has nothing to show but the generic 'No knowledge bases yet.'",
  );
});

test("a healthy host stays optimistic and reports no reason", () => {
  resetAvailability();
  noteRagAvailability({
    knowledgeBases: [],
    ragAvailable: true,
    ragUnavailableReason: null,
  });
  const state = useRagAvailabilityStore.getState();
  assert.equal(state.isUnavailable(), false);
  assert.equal(state.availabilityUnknown(), false, "an answer did arrive");
  assert.equal(state.unavailableReason(), null);
});

test("a list with no marker leaves availability unknown", () => {
  resetAvailability();
  assert.equal(hasRagAvailabilityMarker({ knowledgeBases: [] }), false);
  noteRagAvailability({ knowledgeBases: [] });
  const state = useRagAvailabilityStore.getState();
  assert.equal(
    state.availabilityUnknown(),
    true,
    "a marker was invented for a backend that never sent one",
  );
  assert.equal(state.isUnavailable(), false);
});

test("a marker of true clears a stale unavailable", () => {
  resetAvailability();
  noteRagResponse(503, { detail: BACKEND_REASON });
  assert.equal(useRagAvailabilityStore.getState().isUnavailable(), true);
  noteRagAvailability({ knowledgeBases: [], ragAvailable: true });
  assert.equal(
    useRagAvailabilityStore.getState().isUnavailable(),
    false,
    "nothing can ever undo a 503, so one blip grays the dialog out for the session",
  );
});

test("a 503 from any endpoint marks unavailable, with the stated reason", () => {
  resetAvailability();
  noteRagResponse(503, { detail: BACKEND_REASON });
  const state = useRagAvailabilityStore.getState();
  assert.equal(state.isUnavailable(), true);
  assert.equal(state.unavailableReason(), BACKEND_REASON);
});

// A bodyless 503 comes from proxies too, so it is not a sqlite-vec verdict.
test("a 503 with no readable body is not a capability verdict", () => {
  resetAvailability();
  noteRagResponse(503, null);
  const state = useRagAvailabilityStore.getState();
  assert.equal(state.isUnavailable(), false);
  assert.equal(
    state.availabilityUnknown(),
    true,
    "a gateway 503 was recorded as a measured sqlite-vec failure",
  );
  assert.equal(state.unavailableReason(), null);
});

test("a non-503 failure does not mark unavailable", () => {
  for (const status of [401, 404, 422, 500]) {
    resetAvailability();
    noteRagResponse(status, { detail: "nope" });
    assert.equal(
      useRagAvailabilityStore.getState().availabilityUnknown(),
      true,
      `a ${status} was read as a verdict on the sqlite-vec extension`,
    );
  }
});

test("a success from a gated endpoint clears a stale unavailable", () => {
  resetAvailability();
  noteRagResponse(503, { detail: BACKEND_REASON });
  noteRagResponse(200, { documents: [] });
  assert.equal(
    useRagAvailabilityStore.getState().isUnavailable(),
    false,
    "every endpoint but the list gates on the extension, so a 2xx is proof it loads",
  );
  resetAvailability();
  noteRagResponse(503, { detail: BACKEND_REASON });
  noteRagResponse(204, null);
  assert.equal(
    useRagAvailabilityStore.getState().isUnavailable(),
    false,
    "the 204 early return skips the availability read",
  );
});

test("the KB list's own 200 does not overrule its marker", () => {
  resetAvailability();
  const body = {
    knowledgeBases: [],
    ragAvailable: false,
    ragUnavailableReason: BACKEND_REASON,
  };
  noteRagResponse(200, body);
  // Assert before the marker read: that write is unconditional and would hide the bug.
  assert.equal(
    useRagAvailabilityStore.getState().availabilityUnknown(),
    true,
    "the list's 200 was read as a verdict, racing its own marker back to available",
  );
  noteRagAvailability(body);
  assert.equal(
    useRagAvailabilityStore.getState().isUnavailable(),
    true,
    "the list's 200 marked RAG available on a host where it cannot run",
  );
});

// four call sites, because three of them bypass ragRequest
test("every RAG response path reports availability", async () => {
  assert.match(
    RAG_API,
    /import \{ noteRagAvailability, noteRagResponse \} from "\.\/rag-availability";/,
    "rag-api does not read the availability contract at all",
  );
  const request = functionBody(RAG_API, "ragRequest");
  assert.match(
    request,
    /noteRagResponse\(204, null\)/,
    "the 204 early return leaves before anything is recorded",
  );
  assert.match(
    request,
    /noteRagResponse\(response\.status, json\);\n\s*if \(!response\.ok\)/,
    "the 503 is turned straight into a thrown Error, so nothing records it",
  );
  const upload = functionBody(RAG_API, "ragUpload");
  assert.match(
    upload,
    /noteRagResponse\(response\.status, json\)/,
    "ragUpload bypasses ragRequest, so its 503 is still dropped",
  );
  const stream = functionBody(RAG_API, "openEventStream");
  assert.match(
    stream,
    /noteRagResponse\(response\.status, body\)/,
    "the SSE endpoints bypass ragRequest too",
  );
  assert.match(
    functionBody(RAG_API, "boundedEventStream"),
    /openEventStream\(/,
    "the budgeted stream opens without the availability check",
  );
  for (const generator of ["streamJobEvents", "streamFolderSyncJobEvents"]) {
    assert.match(
      functionBody(RAG_API, generator),
      /boundedEventStream</,
      `${generator} opens its stream outside the budgeted opener`,
    );
  }
});

test("listKnowledgeBases reads the marker before returning the array", async () => {
  const list = functionBody(RAG_API, "listKnowledgeBases");
  assert.match(
    list,
    /ragAvailable\?: boolean;/,
    "the response type still discards everything but the array",
  );
  assert.match(
    list,
    /noteRagAvailability\(data\);\n\s*return data\.knowledgeBases/,
    "the marker is read after the return, or not at all",
  );
});

test("the dialog stops offering a Create button that can only 503", async () => {
  const src = await readSrc("features/rag/components/knowledge-base-dialog.tsx");
  assert.match(
    src,
    /const ragUnavailable = useRagAvailabilityStore\(\(s\) => s\.isUnavailable\(\)\);/,
    "the dialog reads the raw flag rather than the measured gate, or nothing at all",
  );
  assert.match(
    src,
    /New knowledge base/,
    "the create entry point moved; this test no longer guards it",
  );
  assert.match(
    src,
    /disabled=\{ragUnavailable\}\n\s*title=\{ragUnavailableHint\}/,
    "'New knowledge base' is still enabled on a host where it can only 503",
  );
  assert.match(
    src,
    /const ragUnavailableHint = ragUnavailable\n\s*\? \(ragUnavailableReason \?\? undefined\)\n\s*: undefined;/,
    "the hover hint is not derived from the measured verdict",
  );
  assert.match(
    src,
    /disabled=\{saving \|\| ragUnavailable\}/,
    "Create/Save is still enabled",
  );
  const submit = functionBody(src, "submitForm");
  assert.match(
    submit,
    /if \(ragUnavailable\) \{/,
    "submitForm is reachable by keyboard with the button disabled",
  );
  assert.match(
    src,
    /\) : ragUnavailable \? \(/,
    "the empty state does not branch on availability",
  );
  assert.match(
    src,
    /\{ragUnavailableReason \?\? "Knowledge bases are unavailable\."\}/,
    "a machine where RAG cannot run still reads 'No knowledge bases yet.'",
  );
});

// The queue reschedules on probe failure with no cap, so a broken RAG host never dispatched.
test("a queued prompt is not held forever by a probe that can never succeed", async () => {
  const src = await readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    src,
    /import \{ useRagAvailabilityStore \} from "@\/features\/rag\/api\/rag-availability";/,
    "thread.tsx cannot tell a 503 apart from a transient failure",
  );
  const gate = functionBody(src, "targetHasIndexingDocuments");
  const cat = gate.slice(gate.indexOf("} catch {"));
  assert.ok(cat.length > 0, "the probe no longer has a catch to fix");
  assert.ok(
    !/\breturn true;/.test(cat),
    "a failed probe still reports 'still indexing' unconditionally, so the retry " +
      "in dispatchQueuedPrompt never ends",
  );
  assert.match(
    cat,
    /return !useRagAvailabilityStore\.getState\(\)\.isUnavailable\(\);/,
    "the catch does not break out on a measured unavailable",
  );
  resetAvailability();
  assert.equal(
    !useRagAvailabilityStore.getState().isUnavailable(),
    true,
    "an unknown verdict would hold the prompt as it does today",
  );
});

// useRagToolDisabled is a model-capability gate, deliberately false with no model loaded.
test("host availability is kept out of the model-capability gate", async () => {
  const src = await readSrc("features/chat/hooks/use-rag-tool-disabled.ts");
  assert.ok(
    !src.includes("rag-availability"),
    "the sqlite-vec verdict was folded into the tool-capability hook",
  );
});

test("a generic 503 from a proxy is not read as a RAG capability verdict", () => {
  useRagAvailabilityStore.setState({ available: true, reason: null, answered: false });

  for (const body of [
    { detail: "Service Temporarily Unavailable" },
    { detail: "upstream connect error" },
    null,
    {},
    "<html><body><h1>503 Service Unavailable</h1></body></html>",
    // Only the package name marks a capability verdict, since nothing upstream emits it by accident.
    { detail: "RAG is unavailable right now, try again shortly" },
    { detail: "RAG unavailable: upstream timeout" },
  ]) {
    noteRagResponse(503, body);
    assert.equal(
      useRagAvailabilityStore.getState().isUnavailable(),
      false,
      `a 503 carrying ${JSON.stringify(body)} was treated as a sqlite-vec verdict`,
    );
    assert.equal(
      useRagAvailabilityStore.getState().answered,
      false,
      "a generic 503 must leave the verdict unanswered, not answer it optimistically",
    );
  }
});

test("the backend's own 503 detail is still read as unavailable", () => {
  useRagAvailabilityStore.setState({ available: true, reason: null, answered: false });
  noteRagResponse(503, { detail: BACKEND_REASON });
  assert.equal(useRagAvailabilityStore.getState().isUnavailable(), true);
  assert.equal(useRagAvailabilityStore.getState().reason, BACKEND_REASON);
});
