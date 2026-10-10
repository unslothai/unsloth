// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, readText } from "./helpers/kit.ts";

// Neither the hook nor main.rs can run here, so both are asserted against source to hold
// that they agree on the set of statuses.

const USE_TAURI_BACKEND = readSrc("hooks/use-tauri-backend.ts");
const MAIN = readText("../../src-tauri/src/main.rs");

/** Module-scope and in-hook functions differ in indent, so match the declaration's own. */
function hookFunction(hook: string, name: string): string {
  const declaration = hook.match(
    new RegExp(`^([ ]*)function ${name}\\(`, "m"),
  );
  if (!declaration || declaration.index === undefined) {
    throw new Error(`${name} is gone from use-tauri-backend.ts`);
  }
  const close = `\n${declaration[1]}}\n`;
  return hook.slice(declaration.index, hook.indexOf(close, declaration.index));
}

function trayToggleLabel(rust: string): string {
  const start = rust.indexOf("fn tray_toggle_label(");
  if (start < 0) {
    throw new Error("tray_toggle_label is gone from main.rs");
  }
  return rust.slice(start, rust.indexOf("\n}\n", start));
}

function enabledStatuses(rust: string): string[] {
  const table = trayToggleLabel(rust);
  const enabled: string[] = [];
  for (const arm of table.matchAll(/^\s*(.+?)\s*=>\s*\([^)]*,\s*(true|false)\)/gm)) {
    if (arm[2] !== "true" || arm[1] === "_") continue;
    for (const status of arm[1].matchAll(/"([^"]*)"/g)) enabled.push(status[1]);
  }
  return enabled.sort();
}

function actionableStatuses(hook: string): string[] {
  const start = hook.indexOf('register<void>("tray-toggle-server"');
  if (start < 0) {
    throw new Error("the tray-toggle-server listener is gone");
  }
  const listener = hook.slice(start, hook.indexOf("});", start));
  return [
    ...new Set(
      [...listener.matchAll(/statusRef\.current === "([^"]*)"/g)].map(
        (match) => match[1],
      ),
    ),
  ].sort();
}

test("every status the hook commits is also pushed to the tray", async () => {
  const hook = USE_TAURI_BACKEND;

  // setStatus is reached only through these three.
  const committers = ["setBackendStatus", "setBackendError", "setAuthFailure"];
  const setStatusCalls = [...hook.matchAll(/(?<!\w)setStatus\(/g)].length;
  assert.equal(
    setStatusCalls,
    committers.length,
    "a status is now committed somewhere the tray sync does not run",
  );

  for (const name of committers) {
    assert.match(
      hookFunction(hook, name),
      /syncTrayStatus\(/,
      `${name} leaves the tray showing the previous state`,
    );
  }
});

test("a tray sync never surfaces on the web build or on a binary without the command", async () => {
  const hook = USE_TAURI_BACKEND;
  const sync = hookFunction(hook, "syncTrayStatus");

  // The browser build has no IPC, so the import must not even be attempted.
  assert.match(
    sync,
    /if \(!isTauri\) return;/,
    "syncTrayStatus reaches for the Tauri IPC outside the desktop app",
  );
  // An older binary rejects the unregistered command; that must not be an unhandled rejection.
  assert.match(
    sync,
    /\.catch\(\(\) => \{\}\)/,
    "a rejected tray sync escapes as an unhandled rejection",
  );
});

test("the tray offers a click exactly when the listener would act on it", async () => {
  const hook = USE_TAURI_BACKEND;
  const rust = MAIN;

  assert.deepEqual(
    enabledStatuses(rust),
    actionableStatuses(hook),
    "a tray toggle is clickable for a status the renderer silently drops, or greyed for one it would have handled",
  );
});

test("an unlisted status falls through rather than going unhandled", async () => {
  const rust = MAIN;
  const table = trayToggleLabel(rust);

  // A wildcard arm keeps a new status greyed instead of keeping a stale label.
  assert.match(
    table,
    /^\s*_ => \(/m,
    "tray_toggle_label has no wildcard arm for an unknown status",
  );

  const hook = USE_TAURI_BACKEND;
  const union = hook.slice(
    hook.indexOf("export type BackendStatus ="),
    hook.indexOf(";", hook.indexOf("export type BackendStatus =")),
  );
  const statuses = [...union.matchAll(/"([^"]*)"/g)].map((match) => match[1]);
  assert.ok(statuses.length >= 11, "the BackendStatus union did not parse");
  for (const status of enabledStatuses(rust)) {
    assert.ok(
      statuses.includes(status),
      `main.rs enables the tray for "${status}", which is not a BackendStatus`,
    );
  }
});

test("set_tray_server_status is registered, so the invoke can be answered", async () => {
  const rust = MAIN;
  const handler = rust.slice(
    rust.indexOf("invoke_handler(tauri::generate_handler!["),
    rust.indexOf("])", rust.indexOf("invoke_handler(tauri::generate_handler![")),
  );
  assert.match(
    handler,
    /(?<!\w)set_tray_server_status(?!\w)/,
    "the tray sync command is defined but not registered, so every invoke rejects",
  );
});

test("the tray toggle starts clickable, for a frontend older than this binary", async () => {
  const rust = MAIN;
  const built = rust.match(
    /MenuItemBuilder::with_id\("toggle", "([^"]*)"\)([\s\S]{0,40}?)\.build\(app\)/,
  );
  assert.ok(built, "the toggle item is no longer built with a literal label");
  // A bundle predating set_tray_server_status never calls it, so the tray must not seed disabled.
  assert.doesNotMatch(
    built[2],
    /\.enabled\(false\)/,
    "an old frontend bundle would leave this tray toggle permanently disabled",
  );
});
