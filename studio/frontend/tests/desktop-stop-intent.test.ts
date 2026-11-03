// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

import {
  USER_STOPPED_KEY,
  clearServerStopIntent,
  hasServerStopIntent,
  markServerStopIntent,
  stopIntentDisconnectedPort,
} from "../src/hooks/server-stop-intent.ts";

import { readSrc } from "./helpers/kit.ts";

const USE_TAURI_BACKEND = readSrc("hooks/use-tauri-backend.ts");

type Storage = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
};

function installSessionStorage(): Map<string, string> {
  const store = new Map<string, string>();
  const storage: Storage = {
    getItem: (key) => store.get(key) ?? null,
    setItem: (key, value) => {
      store.set(key, value);
    },
    removeItem: (key) => {
      store.delete(key);
    },
  };
  Object.assign(globalThis, { sessionStorage: storage });
  return store;
}

function installThrowingSessionStorage(): void {
  const boom = () => {
    throw new DOMException("The operation is insecure.", "SecurityError");
  };
  Object.assign(globalThis, {
    sessionStorage: { getItem: boom, setItem: boom, removeItem: boom },
  });
}

function uninstallSessionStorage(): void {
  Reflect.deleteProperty(globalThis, "sessionStorage");
}

test("a fresh session carries no stop intent", () => {
  installSessionStorage();
  assert.equal(hasServerStopIntent(), false);
  uninstallSessionStorage();
});

test("a disconnect keeps its port, while a stop and a clear carry none", () => {
  const store = installSessionStorage();
  markServerStopIntent(8890);
  assert.equal(hasServerStopIntent(), true);
  assert.equal(stopIntentDisconnectedPort(), 8890);

  markServerStopIntent();
  assert.equal(hasServerStopIntent(), true);
  assert.equal(stopIntentDisconnectedPort(), null);

  markServerStopIntent(8888);
  clearServerStopIntent();
  assert.equal(stopIntentDisconnectedPort(), null);

  // A value from an older build, or one edited by hand, is a plain stop.
  for (const value of ["1", "disconnected:", "disconnected:0", "disconnected:70000", "disconnected:8a"]) {
    store.set(USER_STOPPED_KEY, value);
    assert.equal(hasServerStopIntent(), true, value);
    assert.equal(stopIntentDisconnectedPort(), null, value);
  }
  uninstallSessionStorage();
});

test("an unreadable store reports no disconnected port", () => {
  installThrowingSessionStorage();
  assert.equal(stopIntentDisconnectedPort(), null);
  assert.doesNotThrow(() => markServerStopIntent(8888));
  uninstallSessionStorage();
});

test("a marked stop reads back, and clearing drops it", () => {
  const store = installSessionStorage();

  markServerStopIntent();
  assert.equal(hasServerStopIntent(), true);
  assert.deepEqual([...store.keys()], [USER_STOPPED_KEY]);

  clearServerStopIntent();
  assert.equal(hasServerStopIntent(), false);
  assert.equal(store.size, 0);
  uninstallSessionStorage();
});

test("marking twice is not two stops to clear", () => {
  installSessionStorage();
  markServerStopIntent();
  markServerStopIntent();
  clearServerStopIntent();
  assert.equal(hasServerStopIntent(), false);
  uninstallSessionStorage();
});

test("storage that throws never reaches the caller", () => {
  installThrowingSessionStorage();

  assert.equal(
    hasServerStopIntent(),
    false,
    "unreadable storage must read as no intent",
  );
  assert.doesNotThrow(() => markServerStopIntent());
  assert.doesNotThrow(() => clearServerStopIntent());

  uninstallSessionStorage();
});

test("storage missing entirely reads as a fresh session", () => {
  uninstallSessionStorage();

  assert.equal(hasServerStopIntent(), false);
  assert.doesNotThrow(() => markServerStopIntent());
  assert.doesNotThrow(() => clearServerStopIntent());
});

test("the marker key belongs to nothing else in the app", async () => {
  // Kept as URLs: a file URL pathname is "/D:/..." on Windows, which readFile misreads.
  const files: URL[] = [];
  async function walk(dir: URL) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const child = new URL(
        `${entry.name}${entry.isDirectory() ? "/" : ""}`,
        dir,
      );
      if (entry.isDirectory()) {
        await walk(child);
      } else if (/\.tsx?$/.test(entry.name)) {
        files.push(child);
      }
    }
  }
  await walk(new URL("../src/", import.meta.url));

  const owners: string[] = [];
  for (const file of files) {
    if ((await readFile(file, "utf8")).includes(`"${USER_STOPPED_KEY}"`)) {
      owners.push(file.pathname);
    }
  }

  assert.deepEqual(
    owners.map((f) => f.slice(f.indexOf("/src/"))),
    ["/src/hooks/server-stop-intent.ts"],
  );
});

test("the hook reaches storage only through the guarded helpers", async () => {
  const hook = USE_TAURI_BACKEND;

  assert.doesNotMatch(
    hook,
    /sessionStorage/,
    "the hook is back to touching sessionStorage directly",
  );
  assert.match(
    hook,
    /import \{\s*clearServerStopIntent,\s*hasServerStopIntent,\s*markServerStopIntent,\s*stopIntentDisconnectedPort,\s*\} from "\.\/server-stop-intent";/,
  );
});

test("a persisted stop is honored before preflight runs", async () => {
  const hook = USE_TAURI_BACKEND;

  const body = hook.slice(
    hook.indexOf("async function checkInstallAndStart()"),
    hook.indexOf("async function startManagedServer()"),
  );
  const guard = body.indexOf("hasServerStopIntent()");
  const preflight = body.indexOf(
    'invoke<DesktopPreflightResult>("desktop_preflight")',
  );

  assert.ok(guard > 0 && preflight > 0);
  // desktop_preflight has side effects (clears intentional_stop), so the check must come first.
  assert.ok(
    guard < preflight,
    "the stop check moved behind preflight, whose adoption side effects it exists to skip",
  );
  assert.match(
    body.slice(guard, preflight),
    /setDisconnectedServerPort\(stopIntentDisconnectedPort\(\)\);\s*setBackendStatus\("stopped"\);\s*return;/,
    "the honored stop no longer parks the screen on stopped, or forgets the server it left running",
  );
});

test("stopping records the intent before the shutdown it can outlive", async () => {
  const hook = USE_TAURI_BACKEND;
  const body = hook.slice(
    hook.indexOf("async function stopServer()"),
    hook.indexOf("async function startInstall()"),
  );

  // Reaping can block ~15s, so the marker must be written before it; order is load bearing.
  const mark = body.indexOf("markServerStopIntent();\n    try {");
  const invoke = body.indexOf('await invoke("stop_server")');
  assert.ok(
    mark > 0 && invoke > mark,
    "the marker is written after the stop it must survive",
  );

  assert.match(
    body,
    /catch \(e\) \{\s*clearServerStopIntent\(\);\s*throw e;\s*\}/,
    "a failed stop keeps a marker it did not earn",
  );

  // The tray listener is registered once, so the branch must read a ref, not the render's state.
  const external = body.slice(0, body.indexOf("const { invoke }"));
  assert.match(
    external,
    /const externalPort = externalServerPortRef\.current;\s*if \(externalPort !== null\) \{/,
    "the disconnect branch reads state the once-registered tray listener sees as its first value",
  );
  assert.match(
    external,
    /stopExternalServerPoll\(\);\s*markServerStopIntent\(externalPort\);/,
  );
});

test("a second stop cannot run while the first is in flight", async () => {
  const hook = USE_TAURI_BACKEND;

  const tray = hook.slice(hook.indexOf('register<void>("tray-toggle-server"'));
  assert.match(
    tray.slice(0, tray.indexOf("});")),
    /statusRef\.current === "running"\)\s*\{\s*stopServer\(\);/,
  );

  const guard = hook.slice(
    hook.indexOf("async function stopServer()"),
    hook.indexOf("async function runStopServer()"),
  );

  assert.match(
    guard,
    /if \(stoppingRef\.current\) return;\s*stoppingRef\.current = true;/,
    "a second stop still runs while the first is in flight",
  );
  assert.match(
    guard,
    /finally \{\s*stoppingRef\.current = false;\s*\}/,
    "a failed stop strands the guard and no later stop can run",
  );
});

test("every deliberate start drops the marker", async () => {
  const hook = USE_TAURI_BACKEND;

  const managed = hook.slice(
    hook.indexOf("async function startManagedServer()"),
    hook.indexOf("async function startRepair()"),
  );
  const clear = managed.indexOf("clearServerStopIntent()");
  const guard = managed.indexOf("if (startingRef.current)");
  assert.ok(
    clear > 0 && guard > clear,
    "a re-entrant start returns with the marker still set",
  );

  const retry = hook.slice(
    hook.indexOf("const retry = useCallback"),
    hook.indexOf("const retryInstall"),
  );
  assert.match(retry, /clearServerStopIntent\(\);/);
  assert.match(retry, /checkInstallAndStart\(\);/);
});

// The tray listener is registered once at mount, so it runs runStopServer with that render's
// state: no attached port, whatever the app is attached to now. Run the real function that way.
function stopFromMountTimeListener(attachedPort: number | null) {
  const hook = USE_TAURI_BACKEND;
  const start = hook.indexOf("  async function runStopServer()");
  const body = hook.slice(start, hook.indexOf("\n  }\n", start) + 4);
  const calls: string[] = [];
  const intents: (number | undefined)[] = [];
  const scope: Record<string, unknown> = {
    externalServerPort: null,
    isExternalServer: false,
    externalServerPortRef: { current: attachedPort },
    startingRef: { current: true },
    importTauriCore: () =>
      Promise.resolve({
        invoke: (command: string) => {
          calls.push(command);
          return Promise.resolve(null);
        },
      }),
    setExternalServerPort: (port: number | null) => calls.push(`attached:${port}`),
    setDisconnectedServerPort: (port: number | null) => calls.push(`disconnected:${port}`),
    stopExternalServerPoll: () => calls.push("stop-poll"),
    markServerStopIntent: (port?: number) => intents.push(port),
    clearServerStopIntent: () => calls.push("clear-intent"),
    setBackendStatus: (status: string) => calls.push(`status:${status}`),
  };
  const source = `${body.replace('await import("@tauri-apps/api/core")', "await importTauriCore()")}
    return runStopServer;`;
  const keys = Object.keys(scope);
  const run = new Function(...keys, source)(...keys.map((key) => scope[key])) as () => Promise<void>;
  return { run, calls, intents };
}

test("the tray's stop on an attached server disconnects and records the port it left running", async () => {
  const stop = stopFromMountTimeListener(8888);
  await stop.run();
  assert.ok(!stop.calls.includes("stop_server"), "an outside server was sent to stop_server");
  assert.deepEqual(stop.intents, [8888]);
  assert.deepEqual(stop.calls, [
    "attached:null",
    "disconnected:8888",
    "stop-poll",
    "status:stopped",
  ]);
});

test("the tray's stop on the app's own server still stops it", async () => {
  const stop = stopFromMountTimeListener(null);
  await stop.run();
  assert.deepEqual(stop.intents, [undefined]);
  assert.deepEqual(stop.calls, ["stop_server", "disconnected:null", "status:stopped"]);
});
