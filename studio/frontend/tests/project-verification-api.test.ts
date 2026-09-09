// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { ProjectVerificationRun } from "../src/features/chat/api/project-verification-api.ts";

import { registerStoreStubResolver } from "./helpers/kit.ts";
import { setAuthFetchHandler } from "./helpers/store-stubs/auth.ts";

registerStoreStubResolver();

const verificationApi = await import(
  "../src/features/chat/api/project-verification-api.ts"
);

const check = {
  name: "Tests",
  kind: "test",
  command: "python -m pytest",
  required: true,
  timeoutSeconds: 300,
  logLimitBytes: 262144,
};

test("aggregate profile size counts normalized UTF-8 JSON including escapes", () => {
  const checks = Array.from({ length: 8 }, (_, index) => ({
    ...check,
    name: `check-${index}`,
    command: "🧪".repeat(4096),
  }));
  assert.match(
    verificationApi.projectVerificationChecksError(checks) ?? "",
    /128 KiB/,
  );
  assert.equal(
    verificationApi.projectVerificationChecksError(checks.slice(0, 7)),
    null,
  );
  assert.match(
    verificationApi.projectVerificationChecksError(
      checks.map((item) => ({
        ...item,
        command: '"'.repeat(9000),
      })),
    ) ?? "",
    /128 KiB/,
  );
});

const run: ProjectVerificationRun = {
  id: "run-one",
  projectId: "project one",
  status: "running",
  configRevision: 4,
  workspaceRevision: 8,
  evidenceRevision: 5,
  historySequence: null,
  checks: [check],
  results: [],
  error: null,
  startedAt: 100,
  updatedAt: 101,
  completedAt: null,
  sourceFreshness: "unverified" as const,
};

const summary = {
  id: run.id,
  projectId: run.projectId,
  status: run.status,
  configRevision: run.configRevision,
  workspaceRevision: run.workspaceRevision,
  evidenceRevision: run.evidenceRevision,
  historySequence: run.historySequence,
  cancelRequested: run.cancelRequested,
  error: run.error,
  startedAt: run.startedAt,
  updatedAt: run.updatedAt,
  completedAt: run.completedAt,
  sourceFreshness: run.sourceFreshness,
  evidenceStatus: "not_loaded" as const,
};
const wireSummary = {
  id: summary.id,
  projectId: summary.projectId,
  status: summary.status,
  configRevision: summary.configRevision,
  workspaceRevision: summary.workspaceRevision,
  evidenceRevision: summary.evidenceRevision,
  historySequence: summary.historySequence,
  cancelRequested: summary.cancelRequested,
  error: summary.error,
  startedAt: summary.startedAt,
  updatedAt: summary.updatedAt,
  completedAt: summary.completedAt,
  sourceFreshness: summary.sourceFreshness,
};

const config = {
  projectId: "project one",
  workspaceAvailable: true,
  workspaceRevision: 8,
  active: true,
  activeRun: run,
  checks: [check],
  revision: 4,
  updatedAt: 99,
  sourceFreshness: "unverified" as const,
  execution: { available: true, backend: "bubblewrap", reason: null },
};

function response(body: unknown, status = 200): Response {
  if (status === 204) {
    return new Response(null, { status });
  }
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test.beforeEach(() => setAuthFetchHandler(null));

test("verification client uses project-scoped CAS and run endpoints", async () => {
  const calls: Array<{
    input: string;
    method: string;
    cache: RequestCache | null;
    contentType: string | null;
    body: unknown;
  }> = [];
  setAuthFetchHandler((input, init) => {
    calls.push({
      input,
      method: init?.method ?? "GET",
      cache: init?.cache ?? null,
      contentType: new Headers(init?.headers).get("Content-Type"),
      body: init?.body ? JSON.parse(String(init.body)) : null,
    });
    if (input.endsWith("/cancel")) {
      return response({ cancelRequested: true, run });
    }
    if (input.includes("?afterEvidenceRevision=")) {
      return response(null, 204);
    }
    if (input.includes("?limit=")) {
      return response({ runs: [wireSummary] });
    }
    if (input.endsWith("/verification") && init?.method !== "PUT") {
      return response(config);
    }
    if (input.endsWith("/verification")) {
      return response(config);
    }
    return response(run);
  });

  await verificationApi.getProjectVerificationConfig("project one");
  await verificationApi.saveProjectVerificationConfig("project one", {
    checks: [check],
    expectedRevision: 3,
    workspaceRevision: 8,
  });
  await verificationApi.startProjectVerification("project one", {
    configRevision: 4,
    workspaceRevision: 8,
  });
  assert.deepEqual(
    await verificationApi.listProjectVerificationRuns("project one", 7),
    [summary],
  );
  assert.deepEqual(
    await verificationApi.getProjectVerificationRun("project one", "run one"),
    run,
  );
  assert.equal(
    await verificationApi.getProjectVerificationRun(
      "project one",
      "run one",
      5,
    ),
    null,
  );
  assert.deepEqual(
    await verificationApi.cancelProjectVerificationRun(
      "project one",
      "run one",
    ),
    { ...run, cancelRequested: true },
  );

  assert.deepEqual(calls, [
    {
      input: "/api/agent/projects/project%20one/verification",
      method: "GET",
      cache: "no-store",
      contentType: null,
      body: null,
    },
    {
      input: "/api/agent/projects/project%20one/verification",
      method: "PUT",
      cache: null,
      contentType: "application/json",
      body: { checks: [check], expectedRevision: 3, workspaceRevision: 8 },
    },
    {
      input: "/api/agent/projects/project%20one/verifications",
      method: "POST",
      cache: null,
      contentType: "application/json",
      body: { configRevision: 4, workspaceRevision: 8 },
    },
    {
      input: "/api/agent/projects/project%20one/verifications?limit=7",
      method: "GET",
      cache: "no-store",
      contentType: null,
      body: null,
    },
    {
      input: "/api/agent/projects/project%20one/verifications/run%20one",
      method: "GET",
      cache: "no-store",
      contentType: null,
      body: null,
    },
    {
      input:
        "/api/agent/projects/project%20one/verifications/run%20one?afterEvidenceRevision=5",
      method: "GET",
      cache: "no-store",
      contentType: null,
      body: null,
    },
    {
      input: "/api/agent/projects/project%20one/verifications/run%20one/cancel",
      method: "POST",
      cache: null,
      contentType: null,
      body: null,
    },
  ]);
});

test("history summaries strip evidence bodies without reading them", () => {
  const poisoned = { ...run };
  Object.defineProperties(poisoned, {
    checks: {
      get() {
        throw new Error("summary read checks");
      },
    },
    results: {
      get() {
        throw new Error("summary read results");
      },
    },
  });

  assert.deepEqual(verificationApi.projectVerificationRunSummary(poisoned), {
    ...summary,
  });
  assert.equal(
    verificationApi.projectVerificationRunHasDetails(summary),
    false,
  );
  assert.equal(verificationApi.projectVerificationRunHasDetails(run), true);
  assert.equal("checks" in summary, false);
  assert.equal("results" in summary, false);
});

test("profile text limits use UTF-8 bytes and leave Unicode casefold to the server", () => {
  const valid = { ...check };
  assert.equal(
    verificationApi.projectVerificationChecksError([
      {
        ...valid,
        kind: "é".repeat(32),
        command: "😀".repeat(4_096),
      },
    ]),
    null,
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, kind: "é".repeat(33) },
    ]),
    "Check kinds must be at most 64 UTF-8 bytes.",
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, command: `${"😀".repeat(4_096)}x` },
    ]),
    "Check commands must be at most 16384 UTF-8 bytes.",
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, name: "Straße" },
      { ...valid, name: "STRASSE" },
    ]),
    null,
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([valid, { ...valid }]),
    "Verification check names must be unique.",
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, name: "😀".repeat(120) },
    ]),
    null,
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, name: "😀".repeat(121) },
    ]),
    "Check names must be at most 120 characters.",
  );
  for (const field of ["name", "kind", "command"] as const) {
    assert.equal(
      verificationApi.projectVerificationChecksError([
        { ...valid, [field]: "safe\u202Ehidden" },
      ]),
      "Check names, kinds, and commands cannot contain Unicode format controls or default-ignorable code points.",
    );
    for (const surrogate of ["\uD800", "\uDFFF"]) {
      assert.equal(
        verificationApi.projectVerificationChecksError([
          { ...valid, [field]: `safe${surrogate}` },
        ]),
        "Check names, kinds, and commands cannot contain lone UTF-16 surrogates.",
      );
    }
  }
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, command: `${"x".repeat(16_385)}\uD800` },
    ]),
    "Check names, kinds, and commands cannot contain lone UTF-16 surrogates.",
  );
  for (const command of [
    "printf safe \u034F# comment; printf HACKED",
    "printf safe \uFE0F# comment; printf HACKED",
    "printf safe \u{E0100}# comment; printf HACKED",
    "printf safe \u{1343F}# comment; printf HACKED",
  ]) {
    assert.equal(
      verificationApi.projectVerificationChecksError([{ ...valid, command }]),
      "Check names, kinds, and commands cannot contain Unicode format controls or default-ignorable code points.",
    );
  }
  for (const command of ["printf safe\u001B", "printf safe\u0085"]) {
    assert.equal(
      verificationApi.projectVerificationChecksError([{ ...valid, command }]),
      "Check commands can contain tabs and newlines, but no other control characters.",
    );
  }
  for (const command of [
    "printf safe\u00A0# comment; printf HACKED",
    "printf safe\u2028printf HACKED",
    "printf safe\u2029printf HACKED",
  ]) {
    assert.equal(
      verificationApi.projectVerificationChecksError([{ ...valid, command }]),
      "Check commands cannot contain non-ASCII Unicode whitespace.",
    );
  }
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, command: "printf first\n\tprintf second" },
    ]),
    null,
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...valid, timeoutSeconds: 1.9 },
    ]),
    "Check timeouts must be whole seconds from 1 to 3600.",
  );
});

test("profile text rejects backend-equivalent blanks and identifier controls", () => {
  for (const field of ["name", "kind", "command"] as const) {
    const whitespaceOnly = field === "command" ? "\t\n" : "\u0085";
    assert.equal(
      verificationApi.projectVerificationChecksError([
        { ...check, [field]: whitespaceOnly },
      ]),
      "Every check needs a name, kind, and command.",
    );
  }

  for (const field of ["name", "kind"] as const) {
    for (const control of ["\u0000", "\u001F", "\u007F"]) {
      assert.equal(
        verificationApi.projectVerificationChecksError([
          { ...check, [field]: `safe${control}` },
        ]),
        "Check names and kinds cannot contain C0 control characters or DEL.",
      );
    }
    assert.equal(
      verificationApi.projectVerificationChecksError([
        { ...check, [field]: "safe\u0085text" },
      ]),
      null,
    );
  }

  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...check, command: "printf first\n\tprintf second" },
    ]),
    null,
  );
  assert.equal(
    verificationApi.projectVerificationChecksError([
      { ...check, command: "printf safe\u0085" },
    ]),
    "Check commands can contain tabs and newlines, but no other control characters.",
  );
});

test("only profile and workspace conflicts require a profile refresh", () => {
  assert.equal(
    verificationApi.verificationConflictRequiresRefresh(
      new verificationApi.ProjectVerificationApiError(
        409,
        "Project verification settings changed. Refresh and retry.",
      ),
    ),
    true,
  );
  assert.equal(
    verificationApi.verificationConflictRequiresRefresh(
      new verificationApi.ProjectVerificationApiError(
        409,
        "Verification evidence revision regressed. Refresh the run.",
      ),
    ),
    false,
  );
  assert.equal(
    verificationApi.verificationConflictRequiresRefresh(
      new verificationApi.ProjectVerificationApiError(
        422,
        "Verification check names are not unique after case folding.",
      ),
    ),
    false,
  );
});

test("an older save snapshot cannot replace a newer event snapshot", () => {
  assert.equal(
    verificationApi.projectVerificationConfigCanReplace(
      { projectId: "project", revision: 8 },
      { projectId: "project", revision: 7 },
    ),
    false,
  );
  assert.equal(
    verificationApi.projectVerificationConfigCanReplace(
      { projectId: "project", revision: 8 },
      { projectId: "project", revision: 8 },
    ),
    true,
  );
  assert.equal(
    verificationApi.projectVerificationConfigCanReplace(
      { projectId: "previous-project", revision: 20 },
      { projectId: "project", revision: 1 },
    ),
    true,
  );
});

test("an evidence revision regression clears only the conditional poll marker", () => {
  const regression = new verificationApi.ProjectVerificationApiError(
    409,
    "Verification evidence revision regressed. Refresh the run.",
  );
  assert.equal(
    verificationApi.projectVerificationPollingEvidenceMarker(12, regression),
    undefined,
  );
  assert.equal(
    verificationApi.projectVerificationPollingEvidenceRegressed(regression),
    true,
  );
  assert.equal(
    verificationApi.projectVerificationPollingEvidenceMarker(
      12,
      new verificationApi.ProjectVerificationApiError(
        409,
        "Verification run changed during cancellation.",
      ),
    ),
    12,
  );
});

test("verification output visibly escapes terminal, newline, and bidi controls", () => {
  assert.equal(
    verificationApi.visibleProjectVerificationText(
      "red\u001b[31m\\path\r\nnext\t\u202Eend\u00A0\u2028\u2029\u034F\uFE0F\u{E0100}\u{1343F}",
    ),
    "red\\u001B[31m\\\\path\\r\\n\nnext\\t\\u202Eend\\u00A0\\u2028\\u2029\\u034F\\uFE0F\\u{E0100}\\u{1343F}",
  );
  assert.equal(
    verificationApi.visibleProjectVerificationCommand(
      "caf\u00E9 \u2800 \uFF03 \\u00E9",
    ),
    "caf\\u00E9 \\u2800 \\uFF03 \\\\u00E9",
  );
  assert.equal(
    verificationApi.projectVerificationCommandNeedsExactPreview("printf ok"),
    false,
  );
  assert.equal(
    verificationApi.projectVerificationCommandNeedsExactPreview(
      "printf caf\u00E9",
    ),
    true,
  );
  assert.equal(verificationApi.visibleProjectVerificationText(null), "");
});

test("active status and request guard fail closed", () => {
  for (const status of ["queued", "running", "cancelling"]) {
    assert.equal(
      verificationApi.projectVerificationRunIsActive({ status }),
      true,
    );
  }
  for (const status of ["passed", "failed", "cancelled", "blocked"]) {
    assert.equal(
      verificationApi.projectVerificationRunIsActive({ status }),
      false,
    );
  }

  const guard = new verificationApi.ProjectVerificationRequestGuard();
  guard.activate();
  const stale = guard.begin();
  const current = guard.begin();
  assert.equal(guard.accepts(stale), false);
  assert.equal(guard.accepts(current), true);
  guard.retire();
  assert.equal(guard.accepts(current), false);
});

test("polling stops on authorization and missing-run responses", () => {
  for (const status of [401, 403, 404]) {
    const error = new verificationApi.ProjectVerificationApiError(
      status,
      `Polling failed (${status})`,
    );
    assert.equal(
      verificationApi.projectVerificationPollingMustStop(error),
      true,
    );
    assert.equal(
      new verificationApi.ProjectVerificationPollBackoff().failureDelay(error),
      null,
    );
  }
  assert.equal(
    verificationApi.projectVerificationPollingMustStop(
      new verificationApi.ProjectVerificationApiError(500, "Retry"),
    ),
    false,
  );
  assert.equal(
    verificationApi.projectVerificationPollingMustStop(new Error("Offline")),
    false,
  );
});

test("retryable polling failures use bounded exponential timers", (t) => {
  t.mock.timers.enable({ apis: ["setTimeout"] });
  t.after(() => t.mock.timers.reset());

  const backoff = new verificationApi.ProjectVerificationPollBackoff();
  const observed: number[] = [];
  const expected = [1_500, 3_000, 6_000, 12_000, 12_000];

  for (const delay of expected) {
    const actual = backoff.failureDelay(new Error("Temporarily offline"));
    assert.equal(actual, delay);
    const callbacksBeforeTimer = observed.length;
    setTimeout(() => observed.push(delay), delay);
    t.mock.timers.tick(delay - 1);
    assert.equal(observed.length, callbacksBeforeTimer);
    t.mock.timers.tick(1);
    assert.equal(observed.length, callbacksBeforeTimer + 1);
    assert.equal(observed.at(-1), delay);
  }

  assert.equal(
    backoff.successDelay(),
    verificationApi.PROJECT_VERIFICATION_POLL_INTERVAL_MS,
  );
  assert.equal(backoff.failureDelay(new Error("Retry again")), 1_500);

  const callbacksBeforeTerminalStatus = observed.length;
  const terminalDelay = backoff.failureDelay(
    new verificationApi.ProjectVerificationApiError(403, "Forbidden"),
  );
  if (terminalDelay !== null) {
    setTimeout(() => observed.push(terminalDelay), terminalDelay);
  }
  t.mock.timers.tick(
    verificationApi.PROJECT_VERIFICATION_POLL_MAX_BACKOFF_MS * 2,
  );
  assert.equal(observed.length, callbacksBeforeTerminalStatus);
});

test("verification errors retain HTTP status for CAS recovery", async () => {
  setAuthFetchHandler(() => response({ detail: "Verification changed." }, 409));

  await assert.rejects(
    verificationApi.saveProjectVerificationConfig("project", {
      checks: [],
      expectedRevision: 2,
      workspaceRevision: 3,
    }),
    (error: unknown) =>
      error instanceof verificationApi.ProjectVerificationApiError &&
      error.status === 409 &&
      error.message === "Verification changed.",
  );
});
