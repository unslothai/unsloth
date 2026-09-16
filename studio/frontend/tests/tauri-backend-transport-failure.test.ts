// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #10520. On the reported host a per-process firewall filter delayed the loopback handshake
// and Unsloth Desktop answered with "Unsloth isn't running -- please relaunch it." while the
// backend was listening and healthy. Two causes, one per suite below: the webview's fetch
// retry ladder ran out long before the launcher's own liveness budget, and the verdict was
// reached without ever asking the launcher.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type AuthApi = {
  authFetch: (
    input: string,
    init?: RequestInit,
    options?: { retryNetworkErrors?: boolean },
  ) => Promise<Response>;
  BACKEND_NOT_RUNNING_MESSAGE: string;
  BACKEND_NOT_ANSWERING_MESSAGE: string;
};

/** The stub map api.ts needs, with the Tauri-specific parts under the caller's control. */
function loadAuthApi(options: {
  port: number | null;
  checkHealth?: (port: number) => boolean | Promise<boolean>;
  onInvoke?: (command: string, args: Record<string, unknown>) => void;
}): AuthApi {
  return loadWithStubs<AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": {
        apiUrl: (path: string) =>
          options.port === null
            ? `http://127.0.0.1:0${path}`
            : `http://127.0.0.1:${options.port}${path}`,
        getApiPort: () => options.port,
        isTauri: true,
      },
      "@/lib/account-transition": { accountTransitionPending: () => false },
      "./session": {
        clearAuthTokens: () => {},
        getAuthToken: () => "access-token",
        getRefreshToken: () => null,
        mustChangePassword: () => false,
        setMustChangePassword: () => {},
        storeAuthTokens: () => {},
      },
      "@tauri-apps/api/core": {
        invoke: async (command: string, args: Record<string, unknown>) => {
          options.onInvoke?.(command, args);
          if (command !== "check_health") {
            throw new Error(`unexpected command ${command}`);
          }
          return options.checkHealth?.(args.port as number) ?? false;
        },
      },
    },
  );
}

/** A port nothing is listening on yet, and a server that starts accepting later. */
async function reserveLoopbackPort(): Promise<number> {
  const probe = createServer();
  const port = await new Promise<number>((resolve, reject) => {
    probe.once("error", reject);
    probe.listen(0, "127.0.0.1", () => {
      const address = probe.address();
      if (typeof address === "string" || address === null) {
        reject(new Error("no loopback address"));
        return;
      }
      resolve(address.port);
    });
  });
  await new Promise<void>((resolve) => probe.close(() => resolve()));
  return port;
}

function listenAfter(server: Server, port: number, delayMs: number): void {
  setTimeout(() => {
    server.listen(port, "127.0.0.1");
  }, delayMs).unref();
}

// The launcher spends 10s on a single liveness probe (HEALTH_PROBE_TIMEOUT in
// src-tauri/src/commands.rs) and three of those before it will call a backend dead. The
// ladder this exercises must reach past the old 250 + 750 + 1500ms, so the host below starts
// accepting at 3s: inside the new ladder, outside the old one. The Rust-side guard
// `the_frontend_retry_ladder_outlives_one_probe_budget` pins the arithmetic itself.
const SLOW_LOOPBACK_ACCEPT_DELAY_MS = 3_000;
const OLD_LADDER_ATTEMPTS = 4;

test("a backend that accepts late is reached instead of declared not running", async () => {
  const port = await reserveLoopbackPort();
  const server = createServer((_request, response) => {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ ok: true }));
  });
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  globalThis.fetch = async (input, init) => {
    attempts += 1;
    return await originalFetch(input, init);
  };

  try {
    listenAfter(server, port, SLOW_LOOPBACK_ACCEPT_DELAY_MS);
    const authApi = loadAuthApi({ port });
    const response = await authApi.authFetch("/api/models");

    assert.equal(response.status, 200);
    assert.ok(
      attempts > OLD_LADDER_ATTEMPTS,
      `the backend was only reached on attempt ${attempts}; a ladder of ` +
        `${OLD_LADDER_ATTEMPTS} attempts would have declared it not running`,
    );
  } finally {
    globalThis.fetch = originalFetch;
    await new Promise<void>((resolve) => server.close(() => resolve()));
  }
});

test("a backend the launcher still sees is not reported as not running", async () => {
  const port = 61799;
  const probed: Array<Record<string, unknown>> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  try {
    const authApi = loadAuthApi({
      port,
      checkHealth: () => true,
      onInvoke: (_command, args) => probed.push(args),
    });
    // retryNetworkErrors off: the ladder is the other suite's subject, and this one is about
    // the verdict reached once it has run out.
    const error = await authApi
      .authFetch("/api/models", undefined, { retryNetworkErrors: false })
      .then(
        () => null,
        (rejection: unknown) => rejection,
      );

    assert.ok(error instanceof Error);
    assert.equal(error.message, authApi.BACKEND_NOT_ANSWERING_MESSAGE);
    assert.notEqual(error.message, authApi.BACKEND_NOT_RUNNING_MESSAGE);
    assert.ok(!/relaunch/i.test(error.message));
    // Still tagged as a transport failure, so nothing blames the model for it.
    assert.equal(
      (error as { unslothTransportFailure?: boolean }).unslothTransportFailure,
      true,
    );
    assert.equal(
      (error as { unslothBackendStillRunning?: boolean })
        .unslothBackendStillRunning,
      true,
    );
    assert.deepEqual(probed, [{ port }]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a backend the launcher cannot see either still says to relaunch", async () => {
  const port = 61800;
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  try {
    const authApi = loadAuthApi({ port, checkHealth: () => false });
    const error = await authApi
      .authFetch("/api/models", undefined, { retryNetworkErrors: false })
      .then(
        () => null,
        (rejection: unknown) => rejection,
      );

    assert.ok(error instanceof Error);
    assert.equal(error.message, authApi.BACKEND_NOT_RUNNING_MESSAGE);
    assert.equal(
      (error as { unslothBackendStillRunning?: boolean })
        .unslothBackendStillRunning,
      undefined,
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a check_health that throws leaves the original verdict in place", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  try {
    const authApi = loadAuthApi({
      port: 61801,
      checkHealth: () => {
        throw new Error("command not registered on this build");
      },
    });
    const error = await authApi
      .authFetch("/api/models", undefined, { retryNetworkErrors: false })
      .then(
        () => null,
        (rejection: unknown) => rejection,
      );

    assert.ok(error instanceof Error);
    assert.equal(error.message, authApi.BACKEND_NOT_RUNNING_MESSAGE);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("no validated port yet means nothing is asked and nothing is claimed", async () => {
  const invoked: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  try {
    // The placeholder base the app carries before server-port arrives is port 0, which
    // never connects; probing it would answer "dead" about a backend that has not started.
    const authApi = loadAuthApi({
      port: null,
      checkHealth: () => true,
      onInvoke: (command) => invoked.push(command),
    });
    const error = await authApi
      .authFetch("/api/models", undefined, { retryNetworkErrors: false })
      .then(
        () => null,
        (rejection: unknown) => rejection,
      );

    assert.ok(error instanceof Error);
    assert.equal(error.message, authApi.BACKEND_NOT_RUNNING_MESSAGE);
    assert.deepEqual(invoked, []);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("panels that all lose the backend at once share one native probe", async () => {
  // The failure this guards is the whole hub losing its connection in the same tick. Each
  // lost poll asks the same question about the same backend, and on the reported host that
  // question is the slow one: it waits out the launcher's budget rather than being refused.
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  let probes = 0;
  let releaseProbe: (() => void) | null = null;
  const probeStarted = new Promise<void>((resolve) => {
    releaseProbe = resolve;
  });

  try {
    const authApi = loadAuthApi({
      port: 61802,
      checkHealth: async () => {
        probes += 1;
        releaseProbe?.();
        await new Promise((resolve) => setTimeout(resolve, 50));
        return true;
      },
    });
    const call = () =>
      authApi
        .authFetch("/api/models", undefined, { retryNetworkErrors: false })
        .then(
          () => null,
          (rejection: unknown) => rejection,
        );

    const first = call();
    await probeStarted;
    const errors = await Promise.all([first, call(), call()]);

    assert.equal(probes, 1, "each lost panel opened its own native health probe");
    for (const error of errors) {
      assert.ok(error instanceof Error);
      assert.equal(error.message, authApi.BACKEND_NOT_ANSWERING_MESSAGE);
    }

    // Released once it answers: the next failure is a new question about a later moment.
    const later = await call();
    assert.equal(probes, 2);
    assert.ok(later instanceof Error);
    assert.equal(later.message, authApi.BACKEND_NOT_ANSWERING_MESSAGE);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("updating an existing install does not go through the changed path", async () => {
  // An Unsloth that is already installed has to be able to reach a newer one, and the update
  // it runs deliberately stops the backend (start_backend_update in src-tauri/src/commands.rs
  // does exactly that). A flow that talked to the backend over HTTP while it was down would
  // now wait out a wider ladder and a native probe before it failed, so pin what it actually
  // uses: Tauri commands and events, nothing this change touches.
  const update = await readFile(
    new URL("../src/hooks/use-tauri-update.ts", import.meta.url),
    "utf8",
  );
  assert.ok(update.includes('invoke("start_backend_update")'));
  assert.ok(!update.includes("authFetch"), "the update flow now goes through authFetch");
  assert.ok(
    !/(?<![.\w])fetch\(/.test(update),
    "the update flow now issues its own fetch, which the transport path wraps",
  );

  // The second opinion is asked for with a command the installed launcher already registers
  // and already answers in this exact shape, so a webview that is newer than the shell it
  // runs in is not asking for anything new. A shell old enough not to have it rejects the
  // invoke, which is the case "a check_health that throws" above covers.
  const backend = await readFile(
    new URL("../src/hooks/use-tauri-backend.ts", import.meta.url),
    "utf8",
  );
  assert.ok(backend.includes('invoke<boolean>("check_health", { port })'));
  const main = await readFile(
    new URL("../../src-tauri/src/main.rs", import.meta.url),
    "utf8",
  );
  assert.ok(main.includes("commands::check_health,"));
});

test("the background chat storage filter accepts every transport verdict", async () => {
  // A background sync that could not reach a busy backend is exactly as expected as one
  // that could not reach a stopped backend. Read as source, because the module pulls in the
  // Dexie database that only a browser build can load.
  const source = await readFile(
    new URL(
      "../src/features/chat/utils/chat-history-storage.ts",
      import.meta.url,
    ),
    "utf8",
  );
  const filterStart = source.indexOf(
    "export function isExpectedBackgroundChatStorageError",
  );
  assert.ok(filterStart !== -1, "the background storage error filter moved");
  const filter = source.slice(filterStart, source.indexOf("\n}", filterStart));

  assert.ok(
    filter.includes("unslothTransportFailure"),
    "the filter does not recognise the transport marker, so a sync against a backend " +
      "that is merely busy is reported as an unexpected error",
  );
  assert.ok(
    !filter.includes("please relaunch it"),
    "the filter still matches one exact wording of the transport failure",
  );
});
