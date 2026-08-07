// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";
import { fileURLToPath } from "node:url";

// lib/api-base derives `isTauri` once, at module evaluation, from globals that must
// already be in place. clipboard-resolver.mjs copies a "?bust=N" key down the import
// chain so each case gets its own evaluation of copy-to-clipboard + api-base, and
// swaps @tauri-apps/plugin-clipboard-manager for a stub.
register("./helpers/clipboard-resolver.mjs", import.meta.url);

const MODULE = fileURLToPath(
  new URL("../src/lib/copy-to-clipboard.ts", import.meta.url),
);

type StubMode = "ok" | "write-fails" | "module-missing";

type Recorder = {
  webWrites: string[];
  execCommands: string[];
  appended: number;
  removed: number;
  nativeWrites: string[];
  /** Resolves a "deferred" web write. */
  settleWeb: () => void;
};

type EnvOptions = {
  tauri: boolean;
  /** How the stubbed Tauri plugin behaves once it is reached. */
  stub?: StubMode;
  /**
   * "absent" drops navigator.clipboard entirely, as an insecure context does.
   * "deferred" leaves the write pending until `settleWeb()` is called.
   */
  clipboard?: "ok" | "reject" | "absent" | "deferred";
  execCommandResult?: boolean;
};

let generation = 0;

function define(name: string, value: unknown) {
  Object.defineProperty(globalThis, name, {
    value,
    configurable: true,
    writable: true,
  });
}

/**
 * Install the globals copy-to-clipboard and api-base read, then import a fresh copy
 * of the module under test. Returns the module plus a recorder of every writer.
 */
async function load(options: EnvOptions) {
  const {
    tauri,
    stub = "ok",
    clipboard = "ok",
    execCommandResult = true,
  } = options;

  const recorder: Recorder = {
    webWrites: [],
    execCommands: [],
    appended: 0,
    removed: 0,
    nativeWrites: [],
    settleWeb: () => {},
  };

  const windowStub: Record<string, unknown> = {
    location: { protocol: tauri ? "tauri:" : "https:" },
  };
  if (tauri) {
    windowStub.__TAURI_INTERNALS__ = {};
  }
  define("window", windowStub);

  define("document", {
    body: {
      appendChild() {
        recorder.appended += 1;
      },
      removeChild() {
        recorder.removed += 1;
      },
    },
    createElement() {
      return {
        value: "",
        readOnly: false,
        style: {} as Record<string, string>,
        setAttribute() {},
        focus() {},
        select() {},
      };
    },
    execCommand(command: string) {
      recorder.execCommands.push(command);
      return execCommandResult;
    },
  });

  define("navigator", {
    clipboard:
      clipboard === "absent"
        ? undefined
        : {
            writeText(text: string) {
              recorder.webWrites.push(text);
              if (clipboard === "reject") {
                return Promise.reject(new Error("NotAllowedError"));
              }
              if (clipboard === "deferred") {
                return new Promise<void>((resolve) => {
                  recorder.settleWeb = resolve;
                });
              }
              return Promise.resolve();
            },
          },
  });

  // A fresh array per case, not a truncated shared one: the stub resolves
  // `control.calls` at call time, so a late write cannot reach an earlier recorder.
  generation += 1;
  const control = ((globalThis as Record<string, unknown>).__TAURI_CLIPBOARD_STUB__ ??=
    {}) as { calls: string[]; mode: StubMode };
  control.calls = recorder.nativeWrites;
  control.mode = stub;

  const mod = (await import(`${MODULE}?bust=${generation}`)) as {
    copyToClipboard: (text: string) => Promise<boolean>;
  };
  const api = (await import(
    `${fileURLToPath(new URL("../src/lib/api-base.ts", import.meta.url))}?bust=${generation}`
  )) as { isTauri: boolean };

  assert.equal(api.isTauri, tauri, "isTauri did not match the staged environment");
  return { copyToClipboard: mod.copyToClipboard, recorder };
}

// Silence the module's console.warn on the deliberate-failure cases.
const realWarn = console.warn;
test.before(() => {
  console.warn = () => {};
});
test.after(() => {
  console.warn = realWarn;
});

test("web build writes through navigator.clipboard before yielding", async () => {
  const { copyToClipboard, recorder } = await load({ tauri: false });

  // Not awaited yet: an async function runs synchronously up to its first await, so
  // if the Tauri gate yielded, writeText would still be unreached at this point and
  // the browser would have dropped transient activation by the time it ran. Snapshot
  // before awaiting, and drain before asserting, so a failure cannot leave the
  // continuation writing into the next case's recorder.
  const pending = copyToClipboard("hello");
  const writtenInGesture = [...recorder.webWrites];
  const nativeInGesture = recorder.nativeWrites.length;
  const result = await pending;

  assert.deepEqual(
    writtenInGesture,
    ["hello"],
    "navigator.clipboard.writeText must be called in the same tick as the click",
  );
  assert.equal(nativeInGesture, 0, "web build must not reach Tauri IPC");
  assert.equal(result, true);
  assert.deepEqual(recorder.execCommands, []);
});

test("web build reaches execCommand in the same tick when clipboard is absent", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: false,
    clipboard: "absent",
  });

  const pending = copyToClipboard("hello");
  const ranInGesture = [...recorder.execCommands];
  const result = await pending;

  assert.deepEqual(
    ranInGesture,
    ["copy"],
    "the synchronous fallback must also run inside the gesture",
  );
  assert.equal(result, true);
  assert.equal(recorder.removed, recorder.appended, "textarea must be cleaned up");
});

test("Tauri build lets the native write decide the result", async () => {
  const { copyToClipboard, recorder } = await load({ tauri: true });

  assert.equal(await copyToClipboard("model/path.gguf"), true);
  assert.deepEqual(recorder.nativeWrites, ["model/path.gguf"]);
  assert.deepEqual(recorder.execCommands, [], "no need for the deprecated fallback");
});

test("Tauri build does not return with the armed web write still in flight", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: true,
    clipboard: "deferred",
  });

  const tick = () => new Promise((resolve) => setTimeout(resolve, 0));
  let settled = false;
  const pending = copyToClipboard("model/path.gguf").then((ok) => {
    settled = true;
    return ok;
  });

  // Drain past the plugin's dynamic import, so "not settled" below means the call is
  // genuinely waiting on the web write rather than still loading the native one.
  for (let i = 0; i < 50 && recorder.nativeWrites.length === 0; i += 1) await tick();
  assert.deepEqual(recorder.nativeWrites, ["model/path.gguf"], "native write ran");
  for (let i = 0; i < 5; i += 1) await tick();

  // Returning here would leave the armed write to land on whatever the user copies next.
  assert.equal(settled, false, "must wait for the armed write to settle");

  recorder.settleWeb();
  assert.equal(await pending, true);
});

test("Tauri build arms the web write inside the gesture", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: true,
    stub: "write-fails",
  });

  // The native attempt yields before it fails, so the only way its fallback can still
  // hold the click's activation is if writeText was already called by this point.
  const pending = copyToClipboard("model/path.gguf");
  const armedInGesture = [...recorder.webWrites];
  const result = await pending;

  assert.deepEqual(
    armedInGesture,
    ["model/path.gguf"],
    "the web write must be issued before the first await, not after the native failure",
  );
  assert.equal(result, true);
});

test("Tauri build does not resolve the native writer before the first await", async () => {
  const { copyToClipboard, recorder } = await load({ tauri: true });

  const pending = copyToClipboard("model/path.gguf");
  const nativeInGesture = [...recorder.nativeWrites];
  const result = await pending;

  assert.deepEqual(
    nativeInGesture,
    [],
    "the dynamic import necessarily yields; native IPC is exempt from the gesture rule",
  );
  assert.equal(result, true);
  assert.deepEqual(recorder.nativeWrites, ["model/path.gguf"]);
});

test("execCommand fallback runs when navigator.clipboard.writeText rejects", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: false,
    clipboard: "reject",
  });

  assert.equal(await copyToClipboard("hello"), true);
  assert.deepEqual(recorder.webWrites, ["hello"]);
  assert.deepEqual(recorder.execCommands, ["copy"]);
});

test("copyToClipboard reports failure when every writer fails", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: false,
    clipboard: "reject",
    execCommandResult: false,
  });

  assert.equal(await copyToClipboard("hello"), false);
  assert.deepEqual(recorder.execCommands, ["copy"]);
  assert.equal(recorder.removed, recorder.appended);
});

test("a failed native write still falls through to the web writers", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: true,
    stub: "write-fails",
  });

  assert.equal(await copyToClipboard("hello"), true);
  assert.deepEqual(recorder.nativeWrites, ["hello"]);
  assert.deepEqual(recorder.webWrites, ["hello"], "must degrade, not give up");
});

test("an install without the clipboard plugin falls through to the web writers", async () => {
  const { copyToClipboard, recorder } = await load({
    tauri: true,
    stub: "module-missing",
  });

  assert.equal(await copyToClipboard("hello"), true);
  assert.deepEqual(recorder.nativeWrites, [], "the import threw before writeText");
  assert.deepEqual(recorder.webWrites, ["hello"]);
});

test("empty and non-string input returns false and touches no writer", async () => {
  for (const tauri of [false, true]) {
    const { copyToClipboard, recorder } = await load({ tauri });

    assert.equal(await copyToClipboard(""), false);
    assert.equal(await copyToClipboard(undefined as unknown as string), false);
    assert.equal(await copyToClipboard(null as unknown as string), false);
    assert.equal(await copyToClipboard(42 as unknown as string), false);

    assert.deepEqual(recorder.nativeWrites, []);
    assert.deepEqual(recorder.webWrites, []);
    assert.deepEqual(recorder.execCommands, []);
    assert.equal(recorder.appended, 0);
  }
});
