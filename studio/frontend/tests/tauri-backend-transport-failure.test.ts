// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// #10520: a firewall filter delayed the loopback handshake and Desktop said "Unsloth isn't running" about a healthy backend.

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

function loadAuthApi(options: {
  port: number | null;
  getPort?: () => number | null;
  checkHealth?: (port: number) => boolean | Promise<boolean>;
  onInvoke?: (command: string, args: Record<string, unknown>) => void;
}): AuthApi {
  const currentPort = options.getPort ?? (() => options.port);
  return loadWithStubs<AuthApi>(
    new URL("../src/features/auth/api.ts", import.meta.url),
    {
      "@/lib/api-base": {
        apiUrl: (path: string) => {
          const port = currentPort();
          return port === null
            ? `http://127.0.0.1:0${path}`
            : `http://127.0.0.1:${port}${path}`;
        },
        getApiPort: () => currentPort(),
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
          if (command !== "check_backend_present") {
            throw new Error(`unexpected command ${command}`);
          }
          return options.checkHealth?.(args.port as number) ?? false;
        },
      },
    },
  );
}

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

// The host below starts accepting at 3s: inside the new ladder, outside the old 250 + 750 + 1500ms one.
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

test("a presence probe that throws leaves the original verdict in place", async () => {
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
    // The placeholder base before server-port arrives is port 0, which never connects.
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
  // The whole hub loses its connection in the same tick and each probe waits out the launcher's budget.
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

    assert.equal(
      probes,
      1,
      "each lost panel opened its own native health probe",
    );
    for (const error of errors) {
      assert.ok(error instanceof Error);
      assert.equal(error.message, authApi.BACKEND_NOT_ANSWERING_MESSAGE);
    }

    const later = await call();
    assert.equal(probes, 2);
    assert.ok(later instanceof Error);
    assert.equal(later.message, authApi.BACKEND_NOT_ANSWERING_MESSAGE);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("updating an existing install does not go through the changed path", async () => {
  // The update deliberately stops the backend, so pin that this flow uses commands and events, not HTTP.
  const update = await readFile(
    new URL("../src/hooks/use-tauri-update.ts", import.meta.url),
    "utf8",
  );
  assert.ok(update.includes('invoke("start_backend_update")'));
  assert.ok(
    !update.includes("authFetch"),
    "the update flow now goes through authFetch",
  );
  assert.ok(
    !/(?<![.\w])fetch\(/.test(update),
    "the update flow now issues its own fetch, which the transport path wraps",
  );

  // A webview newer than its shell asks for nothing new; the rejected invoke is covered above.
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
  // check_health collapses a stalled probe onto false, the verdict this file exists to stop showing.
  assert.ok(main.includes("commands::check_backend_present,"));
  const authApiSrc = await readFile(
    new URL("../src/features/auth/api.ts", import.meta.url),
    "utf8",
  );
  assert.ok(
    authApiSrc.includes('invoke<boolean>("check_backend_present", { port })'),
    "the auth transport probe must ask for presence, not health",
  );
});

test("the background chat storage filter accepts every transport verdict", async () => {
  // Read as source: the module pulls in the Dexie database only a browser build can load.
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

test("a POST is not retried on the long ladder", async () => {
  // A network error is not an answer: the backend may have committed the request, so a retry can duplicate it.
  const port = 61797;
  const originalFetch = globalThis.fetch;
  let attempts = 0;
  globalThis.fetch = async () => {
    attempts += 1;
    throw new TypeError("fetch failed");
  };

  try {
    const authApi = loadAuthApi({ port, checkHealth: () => false });
    await authApi
      .authFetch("/api/auth/api-keys", { method: "POST", body: "{}" })
      .catch(() => undefined);
    assert.ok(
      attempts <= OLD_LADDER_ATTEMPTS,
      `a POST was sent ${attempts} times; the unsafe ladder allows ${OLD_LADDER_ATTEMPTS}`,
    );
    assert.ok(
      attempts > 1,
      "the unsafe ladder still retries, as it always did",
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("a probe pending against the old port is not the answer about the new one", async () => {
  // setApiBase can move the port inside a probe's 10s budget, so a shared pending promise reports the PREVIOUS backend.
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };

  const asked: number[] = [];
  let port = 61810;
  let releaseFirst: (() => void) | null = null;
  const firstProbeStarted = new Promise<void>((resolve) => {
    releaseFirst = resolve;
  });

  try {
    const authApi = loadAuthApi({
      port,
      getPort: () => port,
      checkHealth: async (probedPort: number) => {
        asked.push(probedPort);
        if (asked.length === 1) {
          releaseFirst?.();
          await new Promise((resolve) => setTimeout(resolve, 60));
          return false;
        }
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
    await firstProbeStarted;
    port = 61811;
    const second = await call();
    const firstError = await first;

    assert.deepEqual(asked, [61810, 61811], "the new port was never probed");
    assert.ok(second instanceof Error);
    assert.equal(second.message, authApi.BACKEND_NOT_ANSWERING_MESSAGE);
    assert.ok(firstError instanceof Error);
    assert.equal(firstError.message, authApi.BACKEND_NOT_RUNNING_MESSAGE);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
