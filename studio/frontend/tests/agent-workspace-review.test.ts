// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { type ViteDevServer, createServer } from "vite";
import {
  agentPreparedCommitConfirmationRequest,
  agentPreparedCommitRequest,
  agentPullRequestHandoffConfirmationRequest,
  agentPullRequestHandoffRequest,
} from "../src/features/chat/api/agent-workspace-requests.ts";
import {
  preparedCommitConfirmation,
  pullRequestHandoffCanSubmit,
  pullRequestHandoffConfirmation,
  pullRequestSubmissionDisplay,
} from "../src/features/chat/components/agent-workspace-state.ts";

type ReviewApi = {
  agentWorkspaceMutationOutcomeUnknown(error: unknown): boolean;
  queueAgentTask(
    projectId: string,
    payload: {
      instruction: string;
      runtime: {
        kind: "local" | "provider";
        model: string;
        providerId?: string;
        permissionMode: "off" | "full";
        reasoningEffort?: string;
        maxOutputTokens: number;
      };
      start?: boolean;
    },
  ): Promise<Record<string, unknown>>;
  prepareAgentCommit(
    projectId: string,
    ownedPaths: string[],
    message: string,
  ): Promise<Record<string, unknown>>;
  confirmAgentPreparedCommit(
    projectId: string,
    preparationId: string,
    confirmationToken: string,
  ): Promise<Record<string, unknown>>;
  prepareAgentPullRequestHandoff(
    projectId: string,
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;
  confirmAgentPullRequestHandoff(
    projectId: string,
    handoffId: string,
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>>;
};

type CapturedFetch = {
  path: string;
  method: string;
  body: unknown;
};

const TOKEN = "confirmation-token-value-1234567890abcdef";
const DIGEST = "d".repeat(64);
const HEAD = "a".repeat(40);
const CHECK_GITHUB_DETAIL = /Check GitHub before creating another handoff/;

let vite: ViteDevServer;
let api: ReviewApi;

before(async () => {
  vite = await createServer({
    appType: "custom",
    server: { hmr: false, middlewareMode: true, ws: false },
  });
  api = (await vite.ssrLoadModule(
    "/src/features/chat/api/agent-workspace-api.ts",
  )) as ReviewApi;
});

after(async () => {
  await vite.close();
});

function stubFetch(
  responder: (call: CapturedFetch) => { status?: number; body: unknown },
): { calls: CapturedFetch[]; restore: () => void } {
  const original = globalThis.fetch;
  const calls: CapturedFetch[] = [];
  globalThis.fetch = ((input, init) => {
    const call = {
      path: String(input),
      method: init?.method ?? "GET",
      body: init?.body ? JSON.parse(String(init.body)) : null,
    };
    calls.push(call);
    const response = responder(call);
    return Promise.resolve(
      new Response(JSON.stringify(response.body), {
        status: response.status ?? 200,
        headers: { "content-type": "application/json" },
      }),
    );
  }) as typeof globalThis.fetch;
  return {
    calls,
    restore: () => {
      globalThis.fetch = original;
    },
  };
}

function commitPreview() {
  return {
    id: "preparation-1",
    projectId: "project-1",
    status: "awaiting_confirmation",
    branch: "feature/review",
    baseHead: HEAD,
    message: "Selected files",
    ownedPaths: ["src/a.ts", "src/b.ts"],
    sourceFingerprint: "f".repeat(64),
    createdAt: 1,
    expiresAt: 2,
    confirmationToken: TOKEN,
    files: [
      { code: " M", path: "src/a.ts" },
      { code: "M ", path: "src/b.ts" },
    ],
    diff: "bounded diff",
    diffTruncated: false,
  };
}

function handoffPreview() {
  return {
    id: "handoff-1",
    confirmationToken: TOKEN,
    requestDigest: DIGEST,
    expiresAt: 2,
    connector: { id: "github-connector", displayName: "GitHub" },
    request: {
      owner: "unslothai",
      repo: "unsloth",
      base: "main",
      head: "feature/review",
      title: "Workspace review",
      body: "Bounded review body",
      draft: true,
      maintainer_can_modify: true,
    },
    submitted: false as const,
  };
}

test("prepared commit builders preserve the exact path selection and one-use token", () => {
  const prepared = agentPreparedCommitRequest("project-1", {
    ownedPaths: ["src/a.ts", "src/b.ts", "src/a.ts"],
    message: "  Selected files  ",
  });
  assert.equal(
    prepared.path,
    "/api/agent-workspace/projects/project-1/git/commits/prepare",
  );
  assert.deepEqual(JSON.parse(String(prepared.init.body)), {
    ownedPaths: ["src/a.ts", "src/b.ts"],
    message: "Selected files",
  });

  const confirmed = agentPreparedCommitConfirmationRequest(
    "project-1",
    "preparation/1",
    TOKEN,
  );
  assert.equal(
    confirmed.path,
    "/api/agent-workspace/projects/project-1/git/commits/preparations/preparation%2F1/confirm",
  );
  assert.deepEqual(JSON.parse(String(confirmed.init.body)), {
    confirmationToken: TOKEN,
  });
});

test("GitHub handoff builders separate preview fields from confirmation secrets", () => {
  const prepared = agentPullRequestHandoffRequest("project-1", {
    serverId: " github-connector ",
    owner: " unslothai ",
    repository: " unsloth ",
    base: " main ",
    head: " feature/review ",
    draft: true,
  });
  assert.equal(
    prepared.path,
    "/api/agent-workspace/projects/project-1/review/pull-request-handoff/prepare",
  );
  assert.deepEqual(JSON.parse(String(prepared.init.body)), {
    serverId: "github-connector",
    owner: "unslothai",
    repository: "unsloth",
    base: "main",
    head: "feature/review",
    draft: true,
  });

  const confirmed = agentPullRequestHandoffConfirmationRequest(
    "project-1",
    "handoff/1",
    {
      serverId: "github-connector",
      confirmationToken: TOKEN,
      expectedRequestDigest: DIGEST,
    },
  );
  assert.equal(
    confirmed.path,
    "/api/agent-workspace/projects/project-1/review/pull-request-handoff/handoff%2F1/confirm",
  );
  assert.deepEqual(JSON.parse(String(confirmed.init.body)), {
    serverId: "github-connector",
    confirmationToken: TOKEN,
    expectedRequestDigest: DIGEST,
  });
});

test("real API functions issue exactly one prepare and one commit confirmation request", async () => {
  const mock = stubFetch((call) => {
    if (call.path.endsWith("/git/commits/prepare")) {
      return { body: commitPreview() };
    }
    return {
      body: {
        ...commitPreview(),
        confirmationToken: undefined,
        status: "confirmed",
        commitSha: "c".repeat(40),
        refName: "refs/unsloth-studio/prepared-commits/preparation-1",
      },
    };
  });
  try {
    const preview = await api.prepareAgentCommit(
      "project-1",
      ["src/a.ts", "src/b.ts"],
      "Selected files",
    );
    const secret = preparedCommitConfirmation(preview as never);
    await api.confirmAgentPreparedCommit(
      "project-1",
      secret.preparationId,
      secret.confirmationToken,
    );

    assert.equal(mock.calls.length, 2);
    assert.deepEqual(mock.calls[0], {
      path: "/api/agent-workspace/projects/project-1/git/commits/prepare",
      method: "POST",
      body: {
        ownedPaths: ["src/a.ts", "src/b.ts"],
        message: "Selected files",
      },
    });
    assert.deepEqual(mock.calls[1], {
      path: "/api/agent-workspace/projects/project-1/git/commits/preparations/preparation-1/confirm",
      method: "POST",
      body: { confirmationToken: TOKEN },
    });
  } finally {
    mock.restore();
  }
});

test("real background API sends an explicit credential-free provider runtime", async () => {
  const mock = stubFetch(() => ({
    body: { id: "background-1", status: "running" },
  }));
  try {
    await api.queueAgentTask("project-1", {
      instruction: "  Implement the selected plan task  ",
      runtime: {
        kind: "provider",
        model: "  gpt-5.6-codex  ",
        providerId: "  codex-account  ",
        permissionMode: "full",
        reasoningEffort: " high ",
        maxOutputTokens: 12_000,
        apiKey: "must-not-cross-the-request-boundary",
      } as never,
      start: true,
    });

    assert.deepEqual(mock.calls, [
      {
        path: "/api/agent-workspace/projects/project-1/background/agent",
        method: "POST",
        body: {
          instruction: "Implement the selected plan task",
          runtime: {
            kind: "provider",
            model: "gpt-5.6-codex",
            providerId: "codex-account",
            permissionMode: "full",
            reasoningEffort: "high",
            maxOutputTokens: 12_000,
          },
          start: true,
          cleanupWorktreeOnCancel: false,
        },
      },
    ]);
    const runtime = (mock.calls[0]?.body as { runtime: object }).runtime;
    assert.equal("apiKey" in runtime, false);
    assert.equal("token" in runtime, false);
  } finally {
    mock.restore();
  }
});

test("successful GitHub submit consumes preview state and cannot confirm twice", async () => {
  const fixture = handoffPreview();
  const mock = stubFetch((call) => {
    if (call.path.endsWith("/pull-request-handoff/prepare")) {
      return { body: fixture };
    }
    return {
      body: {
        id: fixture.id,
        requestDigest: DIGEST,
        connector: fixture.connector,
        submitted: true,
        connectorResult: "https://github.example.invalid/pull/1",
        connectorResultTruncated: false,
      },
    };
  });
  try {
    const preview = await api.prepareAgentPullRequestHandoff("project-1", {
      serverId: "github-connector",
      owner: "unslothai",
      repository: "unsloth",
      base: "main",
      head: "feature/review",
      draft: true,
    });
    assert.equal(pullRequestHandoffCanSubmit(preview as never, null), true);
    const confirmation = pullRequestHandoffConfirmation(preview as never);
    const visible = pullRequestSubmissionDisplay(
      preview as never,
      "submitting",
    );
    assert.equal(pullRequestHandoffCanSubmit(null, visible), false);
    assert.equal("confirmationToken" in visible, false);
    assert.equal("requestDigest" in visible, false);

    await api.confirmAgentPullRequestHandoff(
      "project-1",
      confirmation.handoffId,
      {
        serverId: confirmation.serverId,
        confirmationToken: confirmation.confirmationToken,
        expectedRequestDigest: confirmation.expectedRequestDigest,
      },
    );
    assert.equal(mock.calls.length, 2);
    assert.deepEqual(mock.calls[0], {
      path: "/api/agent-workspace/projects/project-1/review/pull-request-handoff/prepare",
      method: "POST",
      body: {
        serverId: "github-connector",
        owner: "unslothai",
        repository: "unsloth",
        base: "main",
        head: "feature/review",
        draft: true,
      },
    });
    assert.deepEqual(mock.calls[1], {
      path: "/api/agent-workspace/projects/project-1/review/pull-request-handoff/handoff-1/confirm",
      method: "POST",
      body: {
        serverId: "github-connector",
        confirmationToken: TOKEN,
        expectedRequestDigest: DIGEST,
      },
    });
    assert.equal(
      pullRequestHandoffCanSubmit(
        null,
        pullRequestSubmissionDisplay(preview as never, "submitted"),
      ),
      false,
    );
  } finally {
    mock.restore();
  }
});

test("connector failure becomes a token-free unknown outcome without retry", async () => {
  const fixture = handoffPreview();
  const mock = stubFetch((call) =>
    call.path.endsWith("/pull-request-handoff/prepare")
      ? { body: fixture }
      : {
          status: 502,
          body: {
            detail:
              "The connector did not confirm submission. Check GitHub before creating another handoff.",
          },
        },
  );
  try {
    const preview = await api.prepareAgentPullRequestHandoff("project-1", {
      serverId: "github-connector",
      owner: "unslothai",
      repository: "unsloth",
      base: "main",
      head: "feature/review",
      draft: true,
    });
    const confirmation = pullRequestHandoffConfirmation(preview as never);
    let display = pullRequestSubmissionDisplay(preview as never, "submitting");
    try {
      await api.confirmAgentPullRequestHandoff(
        "project-1",
        confirmation.handoffId,
        {
          serverId: confirmation.serverId,
          confirmationToken: confirmation.confirmationToken,
          expectedRequestDigest: confirmation.expectedRequestDigest,
        },
      );
      assert.fail("confirmation should fail");
    } catch (error) {
      assert.equal(api.agentWorkspaceMutationOutcomeUnknown(error), true);
      display = pullRequestSubmissionDisplay(preview as never, "unknown");
    }

    assert.equal(mock.calls.length, 2);
    assert.equal(display.status, "unknown");
    assert.match(display.detail, CHECK_GITHUB_DETAIL);
    assert.equal("confirmationToken" in display, false);
    assert.equal("requestDigest" in display, false);
    assert.equal(pullRequestHandoffCanSubmit(null, display), false);
  } finally {
    mock.restore();
  }
});

test("pre-dispatch rejection is a known non-submission outcome", async () => {
  const fixture = handoffPreview();
  const mock = stubFetch((call) =>
    call.path.endsWith("/pull-request-handoff/prepare")
      ? { body: fixture }
      : {
          status: 409,
          body: { detail: "The handoff expired before submission." },
        },
  );
  try {
    const preview = await api.prepareAgentPullRequestHandoff("project-1", {
      serverId: "github-connector",
      owner: "unslothai",
      repository: "unsloth",
      base: "main",
      head: "feature/review",
      draft: true,
    });
    const confirmation = pullRequestHandoffConfirmation(preview as never);

    await assert.rejects(
      api.confirmAgentPullRequestHandoff("project-1", confirmation.handoffId, {
        serverId: confirmation.serverId,
        confirmationToken: confirmation.confirmationToken,
        expectedRequestDigest: confirmation.expectedRequestDigest,
      }),
      (error) => {
        assert.equal(api.agentWorkspaceMutationOutcomeUnknown(error), false);
        assert.match(String(error), /expired before submission/);
        return true;
      },
    );
  } finally {
    mock.restore();
  }
});
