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
  FORCE_QUIT_AFTER_MS,
  isAppClosing,
  markAppClosing,
  subscribeAppClosing,
} from "../src/components/tauri/closing-signal.ts";

function source(path: string): Promise<string> {
  return readFile(new URL(`../src/${path}`, import.meta.url), "utf8");
}

/** The ClosingContent body, up to whatever component is declared after it. */
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
  // Subscribed, not mirrored into state: the listener lives inside the long event effect,
  // which has no way back to a setState from the render that registered it.
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

test("the close button leaves the overlay to Rust", async () => {
  const titlebar = await source("components/tauri/window-titlebar.tsx");

  // Raising it here would put it behind the quit confirmations, one of which asks whether
  // to keep training. Rust raises it only once those have passed.
  assert.doesNotMatch(
    titlebar,
    /markAppClosing/,
    "the close button is raising the overlay before the quit is committed",
  );
  assert.match(titlebar, /onClick=\{\(\) => runWindowAction\(\(appWindow\) =>\s*appWindow\.close\(\)\)\}/);
});

test("the overlay offers a way out of a wedged reap", async () => {
  // Past stop_backend's own worst case, or the button fires over a teardown that was
  // merely slow.
  assert.ok(
    FORCE_QUIT_AFTER_MS > 15_000,
    "force quit would race a reap that is still within its timeouts",
  );

  const signal = await source("components/tauri/closing-signal.ts");
  assert.match(signal, /await invoke\("force_quit"\)/);

  const screen = await source("components/tauri/startup-screen.tsx");
  const body = closingContent(screen);

  // Gated on the timer, and the timer is cleared on unmount: a declined quit unmounts the
  // overlay, and a stray timer would raise the button over a running app.
  assert.match(body, /const \[wedged, setWedged\] = useState\(false\);/);
  assert.match(
    body,
    /setTimeout\(\(\) => setWedged\(true\), FORCE_QUIT_AFTER_MS\)/,
  );
  assert.match(body, /return \(\) => clearTimeout\(timer\);/);
  assert.match(body, /\{wedged && \(/, "the button is not gated on the timer");
  assert.match(body, /onClick=\{\(\) => void forceQuit\(\)\}/);
  assert.match(body, /Force quit/);
});

test("the force quit command is registered with Tauri", async () => {
  const rust = await readFile(
    new URL("../../src-tauri/src/main.rs", import.meta.url),
    "utf8",
  );

  // An unregistered command fails at the invoke, which is the one moment the user has no
  // other way out.
  assert.match(rust, /^\s*force_quit,$/m);
  assert.match(rust, /#\[tauri::command\]\nfn force_quit\(\)/);
});

test("the overlay names the wait it is covering", async () => {
  const screen = await source("components/tauri/startup-screen.tsx");
  const body = closingContent(screen);

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
