// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// An expired approval 404s on every retry, so it must not render as a retryable failed post.

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";

type Module = {
  resolveToolConfirmation: (
    sessionId: string,
    approvalId: string,
    decision: "allow" | "deny",
    fullAccess?: boolean,
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

function harness(response: ReturnType<typeof jsonResponse>, epoch = { value: 0 }) {
  const requests: { url: string; init?: RequestInit }[] = [];
  const module = loadWithStubs<Module>(
    new URL("../src/features/chat/api/chat-api.ts", import.meta.url),
    {
      "./skill-load-event": { skillLoadCardEvent },
      "./thread-read-cache.ts": loadWithStubs(
        new URL("../src/features/chat/api/thread-read-cache.ts", import.meta.url),
        {},
      ),
      "@/features/auth": {
        getAuthSessionEpoch: () => epoch.value,
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
      "../utils/openapi-support": { readAdvertisedPatchGuards: () => null },
      "../utils/load-warning-toast": { showLoadWarning: () => {} },
      "../lib/load-request": {},
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

// No DOM in this suite, so these read the source.

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
  assert.equal(disabled.length, 4, "approval buttons and the Full Access shortcut share the gone gate");
  for (const expr of disabled) {
    assert.match(
      expr,
      /failure === "gone"/,
      `a button stays pressable after the approval is gone: ${expr}`,
    );
  }
});

test("Always allow records its session grant only after the backend takes the decision", () => {
  // allowToolAlways must run after a successful post, not in the onClick.
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


test("Full Access needs an explicit server acknowledgment, including on an older backend", async () => {
  for (const body of [{ resolved: true }, { resolved: false, full_access_granted: true }]) {
    const { module } = harness(jsonResponse(200, body));
    assert.equal(await module.resolveToolConfirmation("sess", "a1", "allow", true), false);
  }
  const { module, requests } = harness(jsonResponse(200, { resolved: true, full_access_granted: true }));
  assert.equal(await module.resolveToolConfirmation("sess", "a1", "allow", true), true);
  assert.deepEqual(JSON.parse(String(requests[0]!.init!.body)), {
    session_id: "sess", approval_id: "a1", decision: "allow", permission_mode: "full",
  });
});

test("a Full Access acknowledgment cannot cross an authentication session change", async () => {
  const epoch = { value: 0 };
  const response = jsonResponse(200, { resolved: true, full_access_granted: true });
  response.json = async () => { epoch.value += 1; return { resolved: true, full_access_granted: true }; };
  const { module } = harness(response, epoch);
  assert.equal(await module.resolveToolConfirmation("sess", "a1", "allow", true), false);
});
