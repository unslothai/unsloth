// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  APP_CLOSING_CANCELLED_EVENT,
  APP_CLOSING_EVENT,
  clearAppClosing,
  isAppClosing,
  markAppClosing,
  subscribeAppClosing,
} from "../src/components/tauri/closing-signal.ts";

import { readSrc, readText } from "./helpers/kit.ts";

function closingContent(screen: string): string {
  const start = screen.indexOf("function ClosingContent()");
  if (start < 0) {
    throw new Error("ClosingContent is gone");
  }
  return screen.slice(start, screen.indexOf("\nfunction ", start + 1));
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

test("a re-emitted app-closing does not re-render the overlay", () => {
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
  const hook = await readSrc("hooks/use-tauri-backend.ts");

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
  const provider = await readSrc("app/provider.tsx");

  assert.match(provider, /\{shell\}\s*\{closing && <ClosingScreen \/>\}/);
  assert.doesNotMatch(
    provider,
    /closing \? \(/,
    "the overlay is back to replacing the app it should be covering",
  );

  const screen = await readSrc("components/tauri/startup-screen.tsx");
  const closingScreen = screen.slice(
    screen.indexOf("export function ClosingScreen()"),
  );
  assert.match(
    closingScreen,
    /className="[^"]*fixed inset-0 z-\[9999\]"/,
    "a covering overlay has to outrank the titlebar and the download stack",
  );
});

test("the overlay survives a modal's body pointer-events lockout", async () => {
  const screen = await readSrc("components/tauri/startup-screen.tsx");
  const closingScreen = screen.slice(
    screen.indexOf("export function ClosingScreen()"),
  );

  // Radix sets pointer-events:none on body while a modal is open; the overlay needs explicit auto.
  assert.match(
    closingScreen,
    /className="pointer-events-auto /,
    "an open dialog would take the clicks aimed at the overlay covering it",
  );
});

test("the close button leaves the overlay to Rust", async () => {
  const titlebar = await readSrc("components/tauri/window-titlebar.tsx");

  assert.doesNotMatch(
    titlebar,
    /markAppClosing/,
    "the close button is raising the overlay before the quit is committed",
  );
  assert.match(titlebar, /onClick=\{\(\) => runWindowAction\(\(appWindow\) =>\s*appWindow\.close\(\)\)\}/);
});

test("the overlay is presentation only, with no way out of a wedged reap", async () => {
  const signal = await readSrc("components/tauri/closing-signal.ts");
  const screen = await readSrc("components/tauri/startup-screen.tsx");
  const body = closingContent(screen);

  assert.doesNotMatch(
    signal,
    /force_quit|forceQuit/,
    "a force quit command is process management this overlay does not need",
  );
  assert.doesNotMatch(body, /Force quit/);
  assert.doesNotMatch(body, /setTimeout|useState/);
});

test("a quit with no window on screen raises no overlay", async () => {
  const rust = readText("../../src-tauri/src/main.rs");

  assert.match(
    rust,
    /fn quit_raises_the_overlay\(/,
    "the overlay is raised without asking whether anything is on screen",
  );
  assert.match(
    rust,
    /app\.get_window\("main"\)\s*\.map\(\|window\| window\.is_visible\(\)/,
  );
});

test("the overlay names the wait it is covering", async () => {
  const screen = await readSrc("components/tauri/startup-screen.tsx");
  const body = closingContent(screen);

  assert.match(body, /Closing Unsloth Desktop\.\.\./);
  assert.match(body, /Shutting down the backend\./);
  assert.match(body, /<Spinner className="size-6 text-primary" \/>/);
});

test("both sides agree on the event names", async () => {
  const rust = readText("../../src-tauri/src/main.rs");

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
