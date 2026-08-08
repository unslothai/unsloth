// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// The overlay itself is JSX, which cannot be imported here. So: drive the store the
// app-closing events write to, then assert the three call sites read and write it.
import {
  APP_CLOSING_CANCELLED_EVENT,
  APP_CLOSING_EVENT,
  clearAppClosing,
  isAppClosing,
  markAppClosing,
  subscribeAppClosing,
} from "../src/components/tauri/closing-signal.ts";

function source(path: string): Promise<string> {
  return readFile(new URL(`../src/${path}`, import.meta.url), "utf8");
}

test("app-closing raises the overlay and app-closing-cancelled clears it", () => {
  const seen: boolean[] = [];
  const unsubscribe = subscribeAppClosing((closing) => seen.push(closing));

  assert.equal(isAppClosing(), false);
  markAppClosing();
  assert.equal(isAppClosing(), true, "a requested quit left the app on screen");
  clearAppClosing();
  assert.equal(isAppClosing(), false, "a declined quit left the overlay up");

  assert.deepEqual(
    seen,
    [true, false],
    "the provider was not told to re-render",
  );
  unsubscribe();
});

test("the close button's optimistic mark does not double-fire on Rust's event", () => {
  const seen: boolean[] = [];
  const unsubscribe = subscribeAppClosing((closing) => seen.push(closing));

  markAppClosing();
  markAppClosing();
  assert.deepEqual(seen, [true]);

  clearAppClosing();
  unsubscribe();
});

test("an unsubscribed listener stops hearing about quits", () => {
  const seen: boolean[] = [];
  subscribeAppClosing((closing) => seen.push(closing))();

  markAppClosing();
  clearAppClosing();
  assert.deepEqual(seen, []);
});

test("the backend hook routes both quit events into the store", async () => {
  const hook = await source("hooks/use-tauri-backend.ts");

  assert.match(
    hook,
    /register<void>\(APP_CLOSING_EVENT,\s*\(\) => \{\s*markAppClosing\(\);/,
    "app-closing no longer raises the overlay",
  );
  assert.match(
    hook,
    /register<void>\(APP_CLOSING_CANCELLED_EVENT,\s*\(\) => \{\s*clearAppClosing\(\);/,
    "a cancelled quit would strand the overlay over a running app",
  );
  // Subscribed, not mirrored into state: the close button can raise the overlay between
  // the hook's render and the effect that a useState mirror would subscribe from.
  assert.match(
    hook,
    /const closing = useSyncExternalStore\(subscribeAppClosing, isAppClosing\);/,
  );
  assert.match(
    hook,
    /isExternalServer, closing,/,
    "the hook stopped returning the flag",
  );
});

test("the overlay covers the app instead of replacing it", async () => {
  const provider = await source("app/provider.tsx");

  // Unmounting the app subtree would cancel in-flight generations and drop debounced
  // drafts, and a declined quit has to give all of that back.
  assert.match(provider, /\{shell\}\s*\{closing && <ClosingScreen \/>\}/);
  assert.doesNotMatch(
    provider,
    /closing \? \(/,
    "the overlay is back to replacing the app it should be covering",
  );

  const screen = await source("components/tauri/startup-screen.tsx");
  const closingScreen = screen.slice(
    screen.indexOf("export function ClosingScreen()"),
  );
  assert.match(
    closingScreen,
    /className="fixed inset-0 z-\[9999\]"/,
    "a covering overlay has to outrank the titlebar and the download stack",
  );
});

test("the close button raises the overlay before awaiting the round trip", async () => {
  const titlebar = await source("components/tauri/window-titlebar.tsx");

  // Only this titlebar, and only in this order: on macOS close means hide, and marking
  // after the await would leave the window frozen for the length of the IPC hop.
  assert.match(
    titlebar,
    /markAppClosing\(\);\s*try \{\s*await appWindow\.close\(\);/,
  );
  assert.match(
    titlebar,
    /catch \(error\) \{\s*\/\/[^\n]*\n\s*clearAppClosing\(\);/,
    "a close() that never started a quit would strand the overlay",
  );
});

test("the overlay names the wait it is covering", async () => {
  const screen = await source("components/tauri/startup-screen.tsx");

  const start = screen.indexOf("function ClosingContent()");
  assert.ok(start >= 0, "ClosingContent is gone");
  const body = screen.slice(start, screen.indexOf("\nfunction ", start + 1));

  assert.match(body, /Closing Unsloth Desktop\.\.\./);
  assert.match(body, /Shutting down the backend\./);
  // A still screen reads as the freeze it is there to explain.
  assert.match(body, /<Spinner className="size-6 text-primary" \/>/);
});

test("both sides agree on the event names", async () => {
  const rust = await readFile(
    new URL("../../src-tauri/src/main.rs", import.meta.url),
    "utf8",
  );

  assert.match(
    rust,
    new RegExp(`const APP_CLOSING_EVENT: &str = "${APP_CLOSING_EVENT}";`),
  );
  assert.match(
    rust,
    new RegExp(
      `const APP_CLOSING_CANCELLED_EVENT: &str = "${APP_CLOSING_CANCELLED_EVENT}";`,
    ),
  );
});
