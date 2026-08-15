// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import test from "node:test";

// The marker module is import-free, so it needs no bundler resolver. The hook around it is
// a React hook that cannot be rendered here, so its call sites are asserted against source,
// the way the rest of the desktop startup tests do it.
import {
  USER_STOPPED_KEY,
  clearServerStopIntent,
  hasServerStopIntent,
  markServerStopIntent,
} from "../src/hooks/server-stop-intent.ts";

type Storage = {
  getItem: (key: string) => string | null;
  setItem: (key: string, value: string) => void;
  removeItem: (key: string) => void;
};

/** An in-memory sessionStorage installed under the name the module reads it by. */
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

/** Storage that throws on every access, as an opaque origin's does. */
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

function hookSource(): Promise<string> {
  return readFile(
    new URL("../src/hooks/use-tauri-backend.ts", import.meta.url),
    "utf8",
  );
}

test("a fresh session carries no stop intent", () => {
  installSessionStorage();
  assert.equal(hasServerStopIntent(), false);
  uninstallSessionStorage();
});

test("a marked stop reads back, and clearing drops it", () => {
  const store = installSessionStorage();

  markServerStopIntent();
  assert.equal(hasServerStopIntent(), true);
  // Written under the one key, so a reload of the same webview finds it.
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

  // An opaque origin throws SecurityError on every access. The read runs before the
  // startup screen has any state to fall back on, so a throw would strand it on
  // "checking"; the writes sit in front of the stop invoke, which has to happen anyway.
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

  // Not a hypothetical: bare node has no web storage, and neither does a webview with
  // storage disabled. A ReferenceError has to be absorbed the same as a SecurityError.
  assert.equal(hasServerStopIntent(), false);
  assert.doesNotThrow(() => markServerStopIntent());
  assert.doesNotThrow(() => clearServerStopIntent());
});

test("the marker key belongs to nothing else in the app", async () => {
  const files: string[] = [];
  async function walk(dir: URL) {
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const child = new URL(
        `${entry.name}${entry.isDirectory() ? "/" : ""}`,
        dir,
      );
      if (entry.isDirectory()) {
        await walk(child);
      } else if (/\.tsx?$/.test(entry.name)) {
        files.push(child.pathname);
      }
    }
  }
  await walk(new URL("../src/", import.meta.url));

  const owners: string[] = [];
  for (const file of files) {
    if ((await readFile(file, "utf8")).includes(`"${USER_STOPPED_KEY}"`)) {
      owners.push(file);
    }
  }

  // One declaration and no second reader: a key two features write would let an unrelated
  // preference reset put the desktop server on the stopped screen.
  assert.deepEqual(
    owners.map((f) => f.slice(f.indexOf("/src/"))),
    ["/src/hooks/server-stop-intent.ts"],
  );
});

test("the hook reaches storage only through the guarded helpers", async () => {
  const hook = await hookSource();

  // A raw sessionStorage call in the hook is the bug this module exists to prevent: the
  // read in checkInstallAndStart sits outside its try, so a SecurityError there rejects
  // the mount effect's floating promise and the startup screen never leaves "checking".
  assert.doesNotMatch(
    hook,
    /sessionStorage/,
    "the hook is back to touching sessionStorage directly",
  );
  assert.match(
    hook,
    /import \{\s*clearServerStopIntent,\s*hasServerStopIntent,\s*markServerStopIntent,\s*\} from "\.\/server-stop-intent";/,
  );
});

test("a persisted stop is honored before preflight runs", async () => {
  const hook = await hookSource();

  const body = hook.slice(
    hook.indexOf("async function checkInstallAndStart()"),
    hook.indexOf("async function startManagedServer()"),
  );
  const guard = body.indexOf("hasServerStopIntent()");
  const preflight = body.indexOf(
    'invoke<DesktopPreflightResult>("desktop_preflight")',
  );

  assert.ok(guard > 0 && preflight > 0);
  // desktop_preflight is not a query: adopt_backend clears intentional_stop and bumps the
  // generation, and the command then arms a health watchdog for it, which later fires
  // server-crashed over the stopped screen. So the check has to come first, not just
  // before the start.
  assert.ok(
    guard < preflight,
    "the stop check moved behind preflight, whose adoption side effects it exists to skip",
  );
  assert.match(
    body.slice(guard, preflight),
    /setBackendStatus\("stopped"\);\s*return;/,
    "the honored stop no longer parks the screen on stopped",
  );
});

test("stopping records the intent before the shutdown it can outlive", async () => {
  const hook = await hookSource();
  const body = hook.slice(
    hook.indexOf("async function stopServer()"),
    hook.indexOf("async function startInstall()"),
  );

  // Reaping the backend blocks for up to ~15s. A reload inside that window has to find
  // the marker already written, so the order here is load bearing.
  const mark = body.indexOf("markServerStopIntent();\n    try {");
  const invoke = body.indexOf('await invoke("stop_server")');
  assert.ok(
    mark > 0 && invoke > mark,
    "the marker is written after the stop it must survive",
  );

  // A stop that failed left the backend up, so the marker has to come back off or the
  // next reload shows a stopped screen over a running server.
  assert.match(
    body,
    /catch \(e\) \{\s*clearServerStopIntent\(\);\s*throw e;\s*\}/,
    "a failed stop keeps a marker it did not earn",
  );

  // The detached branch has no process to kill, but the reload still has to keep the UI
  // off the user's external server rather than re-attaching to it.
  const external = body.slice(0, body.indexOf("const { invoke }"));
  assert.match(
    external,
    /stopExternalServerPoll\(\);\s*markServerStopIntent\(\);/,
  );
});

test("a second stop cannot run while the first is in flight", async () => {
  const hook = await hookSource();

  // The tray item is never disabled and the toggle branches on statusRef, which stays
  // "running" for the whole invoke, so two Stop clicks reach stopServer concurrently.
  const tray = hook.slice(hook.indexOf('register<void>("tray-toggle-server"'));
  assert.match(
    tray.slice(0, tray.indexOf("});")),
    /statusRef\.current === "running"\)\s*\{\s*stopServer\(\);/,
  );

  const guard = hook.slice(
    hook.indexOf("async function stopServer()"),
    hook.indexOf("async function runStopServer()"),
  );

  // Two concurrent stops both mark the intent, and on an adopted backend the loser can
  // fail against the port the winner is taking down. Its rollback then drops the marker
  // the winner earned, so the next reload starts a server the user asked to stop.
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
  const hook = await hookSource();

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

  // Retry is the only way off the error screen and the tray's way back on from stopped.
  const retry = hook.slice(
    hook.indexOf("const retry = useCallback"),
    hook.indexOf("const retryInstall"),
  );
  assert.match(retry, /clearServerStopIntent\(\);/);
  assert.match(retry, /checkInstallAndStart\(\);/);
});
