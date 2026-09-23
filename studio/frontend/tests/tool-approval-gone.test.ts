// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An approval the backend is no longer holding is not a failed post. Both used to arrive at the
// card as one bare throw out of parseJsonOrThrow, so an expired request rendered "Could not send
// your decision. Try again." next to re-enabled buttons: advice that can never work, since the
// slot is gone and every retry 404s until tool_end clears the card. The 404 now carries a type.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Module = {
  resolveToolConfirmation: (
    sessionId: string,
    approvalId: string,
    decision: "allow" | "deny",
  ) => Promise<boolean>;
  ToolApprovalGoneError: new (message?: string) => Error;
};

function jsonResponse(status: number, body: unknown) {
  return {
    status,
    ok: status >= 200 && status < 300,
    headers: { get: () => null },
    async json() {
      return body;
    },
  };
}

function harness(response: ReturnType<typeof jsonResponse>) {
  const requests: { url: string; init?: RequestInit }[] = [];
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: async (url: string, init?: RequestInit) => {
          requests.push({ url, init });
          return response;
        },
      },
      "@/lib/format-fastapi-error": {
        formatApiErrorBody: (body: unknown) =>
          (body as { detail?: string } | null)?.detail ?? null,
      },
      "../types": {},
      "../types/api": {},
      "../utils/chat-history-revision": {
        notifyChatHistoryUpdated: () => {},
        isCoalescedHistoryEvent: () => false,
      },
      "../utils/load-warning-toast": { showLoadWarning: () => {} },
      "./generation-length.ts": {},
      "./gguf-variants-request": {},
      "./padded-response": { assertCompletedPaddedBody: () => {} },
      "@/features/hf-auth": { prepareHfTokenForUse: async () => undefined },
      "@/features/settings/low-disk-check": { checkDiskSpace: () => Promise.resolve() },
      "@/features/igpu-carveout": {
        dismissCarveoutAdviceForModel: () => {},
        showCarveoutAdvice: () => {},
      },
      "@/features/hub/lib/abort-signals": {},
      "@/features/hub/lib/hub-token-header": { hubTokenHeader: () => ({}) },
      "@/features/hub/lib/network": { isHuggingFaceOffline: () => false },
      "@/features/native-intents/api": { consumeNativePathToken: () => undefined },
      "@/lib/model-lifecycle-events": {},
    },
  );
  return { module, requests };
}

test("an approval that is no longer pending throws the typed error, not a bare one", async () => {
  const { module } = harness(
    jsonResponse(404, { detail: "No pending tool call confirmation" }),
  );
  const error = await module
    .resolveToolConfirmation("sess", "approval-1", "allow")
    .then(
      () => null,
      (err: unknown) => err,
    );
  assert.ok(
    error instanceof module.ToolApprovalGoneError,
    "a 404 must be distinguishable from a transport failure, or the card tells the user to retry something that can never succeed",
  );
});

test("a transport failure stays an ordinary error, so retrying is still offered", async () => {
  const { module } = harness(jsonResponse(500, { detail: "upstream exploded" }));
  const error = await module
    .resolveToolConfirmation("sess", "approval-1", "allow")
    .then(
      () => null,
      (err: unknown) => err,
    );
  assert.ok(error instanceof Error);
  assert.ok(
    !(error instanceof module.ToolApprovalGoneError),
    "only a 404 means the slot is gone",
  );
});

test("a matched decision still resolves true", async () => {
  const { module, requests } = harness(jsonResponse(200, { resolved: true }));
  assert.equal(await module.resolveToolConfirmation("sess", "a1", "deny"), true);
  assert.equal(requests.length, 1);
  assert.match(requests[0]!.url, /tool-confirm/);
});

// ── The card's own wiring, pinned at the source ──────────────────────────────
// The component is not mounted here (no DOM in this suite), so these read the source. Both are
// one-token regressions that a type check cannot catch and that would silently restore the bug.

const CONTROLS = await import("node:fs").then((fs) =>
  fs.readFileSync(
    new URL(
      "../src/components/assistant-ui/tool-confirmation-controls.tsx",
      import.meta.url,
    ),
    "utf-8",
  ),
);

test("the gone case gets its own copy and does not tell the user to retry", () => {
  assert.match(CONTROLS, /This request is no longer waiting for an answer\./);
  assert.match(
    CONTROLS,
    /failure === "gone"/,
    "the two failure kinds must stay distinguishable at the card",
  );
});

test("a decision that is gone disables the buttons instead of inviting another 404", () => {
  const disabled = CONTROLS.match(/disabled=\{pending !== null[^}]*\}/g) ?? [];
  assert.equal(disabled.length, 3, "all three buttons carry a disabled expression");
  for (const expr of disabled) {
    assert.match(
      expr,
      /failure === "gone"/,
      `a button stays pressable after the approval is gone: ${expr}`,
    );
  }
});

test("Always allow records its session grant only after the backend takes the decision", () => {
  // The regression: allowToolAlways ran in the onClick, before the post. A press that visibly
  // failed still auto-approved this tool for every later call in the session.
  assert.doesNotMatch(
    CONTROLS,
    /onClick=\{\(\) => \{\s*if \(autoAllowKey\) allowToolAlways/,
    "the session grant is being recorded on the click again, ahead of the decision landing",
  );
  assert.match(CONTROLS, /if \(alsoAlways && autoAllowKey\) allowToolAlways\(/);
  assert.match(
    CONTROLS,
    /onClick=\{\(\) => void resolve\("allow", true\)\}/,
    "Always allow must route through resolve so the grant follows the backend's answer",
  );
});
