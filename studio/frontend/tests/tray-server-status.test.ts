// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The tray label is settled across two languages: the hook pushes a BackendStatus over the
// IPC and main.rs turns it into a label and an enabled flag. Neither half can be rendered
// here, so both are asserted against source, as the rest of the desktop startup tests do.
// tray_toggle_label's own table is covered by the Rust unit tests in main.rs; only a test
// spanning both files can hold that the two halves still agree on the set of statuses.

function hookSource(): Promise<string> {
  return readFile(
    new URL("../src/hooks/use-tauri-backend.ts", import.meta.url),
    "utf8",
  );
}

function tauriSource(): Promise<string> {
  return readFile(
    new URL("../../src-tauri/src/main.rs", import.meta.url),
    "utf8",
  );
}

/**
 * The body of a `function name(...)` in the hook, to its closing brace. syncTrayStatus
 * sits at module scope and the status committers sit inside the hook, so the indent the
 * declaration and its brace share is whatever precedes the keyword.
 */
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

/** The `fn tray_toggle_label` body in main.rs. */
function trayToggleLabel(rust: string): string {
  const start = rust.indexOf("fn tray_toggle_label(");
  if (start < 0) {
    throw new Error("tray_toggle_label is gone from main.rs");
  }
  return rust.slice(start, rust.indexOf("\n}\n", start));
}

/** Every status string tray_toggle_label reports as clickable. */
function enabledStatuses(rust: string): string[] {
  const table = trayToggleLabel(rust);
  const enabled: string[] = [];
  for (const arm of table.matchAll(/^\s*(.+?)\s*=>\s*\([^)]*,\s*(true|false)\)/gm)) {
    if (arm[2] !== "true" || arm[1] === "_") continue;
    for (const status of arm[1].matchAll(/"([^"]*)"/g)) enabled.push(status[1]);
  }
  return enabled.sort();
}

/** Every status the tray-toggle-server listener branches on. */
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
  const hook = await hookSource();

  // setStatus is reached through these three and nowhere else, so covering them
  // covers every transition the tray can be told about.
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
  const hook = await hookSource();
  const sync = hookFunction(hook, "syncTrayStatus");

  // The browser build has no IPC at all, so the import must not even be attempted.
  assert.match(
    sync,
    /if \(!isTauri\) return;/,
    "syncTrayStatus reaches for the Tauri IPC outside the desktop app",
  );
  // Against an older binary the command is unregistered and the invoke rejects, leaving
  // the tray on its build-time label. That must not raise an unhandled rejection.
  assert.match(
    sync,
    /\.catch\(\(\) => \{\}\)/,
    "a rejected tray sync escapes as an unhandled rejection",
  );
});

test("the tray offers a click exactly when the listener would act on it", async () => {
  const [hook, rust] = await Promise.all([hookSource(), tauriSource()]);

  assert.deepEqual(
    enabledStatuses(rust),
    actionableStatuses(hook),
    "a tray toggle is clickable for a status the renderer silently drops, or greyed for one it would have handled",
  );
});

test("an unlisted status falls through rather than going unhandled", async () => {
  const rust = await tauriSource();
  const table = trayToggleLabel(rust);

  // BackendStatus grows; the command takes a bare String. A wildcard arm keeps a new
  // status greyed instead of leaving the label on whatever the last transition set.
  assert.match(
    table,
    /^\s*_ => \(/m,
    "tray_toggle_label has no wildcard arm for an unknown status",
  );

  const hook = await hookSource();
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
  const rust = await tauriSource();
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
  const rust = await tauriSource();
  const built = rust.match(
    /MenuItemBuilder::with_id\("toggle", "([^"]*)"\)([\s\S]{0,40}?)\.build\(app\)/,
  );
  assert.ok(built, "the toggle item is no longer built with a literal label");
  // A bundle predating set_tray_server_status never calls it, so seeding it disabled
  // would strand that user with a tray toggle they can never click.
  assert.doesNotMatch(
    built[2],
    /\.enabled\(false\)/,
    "an old frontend bundle would leave this tray toggle permanently disabled",
  );
});
