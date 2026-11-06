// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * WebKit restyles the whole page (about 2-2.5 s each on a 100-turn chat) when a window or document
 * wheel / touch / pointer listener set changes, when a <style> is removed, and when a dialog exits
 * with focus inside it. These pin the guards that keep modal layers and the browser pane off those paths.
 */

import assert from "node:assert/strict";
import test from "node:test";

import { blurInsideClosingDialog } from "../src/lib/closing-dialog-focus.ts";
import {
  PINNED_POINTER_EVENTS,
  pinPointerListenerRegions,
} from "../src/lib/pointer-listener-pin.ts";
import { readSrc, readText } from "./helpers/kit.ts";

function rule(css: string, selector: string): string {
  const at = css.indexOf(`${selector} {`);
  assert.ok(at >= 0, `missing rule ${selector}`);
  return css.slice(at, css.indexOf("}", at) + 1);
}

test("react-remove-scroll resolves to the listener-free scroll lock, and only the bare specifier", () => {
  const config = readText("../vite.config.ts");
  assert.match(
    config,
    /find: \/\^react-remove-scroll\$\/,\s*replacement: path\.resolve\(__dirname, "\.\/src\/lib\/remove-scroll\.tsx"\)/,
  );
  // The stand-in imports the library's own UI half, which an unanchored alias would swallow.
  assert.match(readSrc("lib/remove-scroll.tsx"), /from "react-remove-scroll\/UI"/);
});

test("the scroll lock keeps the body counter and adds no listeners or stylesheets", () => {
  const source = readSrc("lib/remove-scroll.tsx").replace(/\/\*[\s\S]*?\*\//g, "");
  assert.match(source, /data-scroll-locked/);
  assert.doesNotMatch(source, /addEventListener|createElement\(|<style|styleSingleton/);
  assert.match(source, /export const RemoveScroll\b/);
  // JSX skips the library's defaultProps, so the defaults Radix relies on are explicit.
  assert.match(source, /enabled = true, removeScrollBar = true, inert = false/);
  assert.match(source, /classNames: RemoveScrollUI\.classNames/);
});

test("the lock's body styles are static, with what react-remove-scroll-bar injected", () => {
  const lock = rule(readSrc("index.css"), "body[data-scroll-locked]");
  for (const declaration of [
    "overflow: hidden !important;",
    "overscroll-behavior: contain;",
    "position: relative !important;",
    "margin-right: 0 !important;",
  ]) {
    assert.ok(lock.includes(declaration), `body[data-scroll-locked] lost ${declaration}`);
  }
});

test("document pointer listeners are pinned once per type, before the app mounts", () => {
  const added: [string, unknown][] = [];
  const target = {
    addEventListener(type: string, listener: unknown) {
      added.push([type, listener]);
    },
  };
  pinPointerListenerRegions(target);
  pinPointerListenerRegions(target);
  assert.deepEqual(
    [...new Set(added.map(([type]) => type))].sort(),
    [...PINNED_POINTER_EVENTS].sort(),
  );
  // One shared function, so the browser dedupes repeated calls.
  assert.equal(new Set(added.map(([, listener]) => listener)).size, 1);
  // The types react-resizable-panels adds and removes on the document.
  for (const type of ["pointermove", "pointerleave", "pointerout", "pointerdown", "pointerup"]) {
    assert.ok((PINNED_POINTER_EVENTS as readonly string[]).includes(type), type);
  }
  const main = readSrc("main.tsx");
  assert.match(main, /pinPointerListenerRegions\(document\);/);
  assert.ok(main.indexOf("pinPointerListenerRegions(document)") < main.indexOf("root.render("));
});

function dialog(state: string, role = "dialog", holds = true) {
  return {
    getAttribute: (name: string) => (name === "data-state" ? state : null),
    matches: (selector: string) => selector.includes(`[role="${role}"]`),
    contains: () => holds,
  };
}

test("a closing dialog gives up focus when its exit animation starts", () => {
  let blurred = 0;
  const active = { blur: () => void blurred++ };
  assert.equal(blurInsideClosingDialog(dialog("closed"), active), true);
  assert.equal(blurInsideClosingDialog(dialog("closed", "alertdialog"), active), true);
  assert.equal(blurred, 2);
  // Opening dialogs, other closing layers, and focus elsewhere are left alone.
  assert.equal(blurInsideClosingDialog(dialog("open"), active), false);
  assert.equal(blurInsideClosingDialog(dialog("closed", "menu"), active), false);
  assert.equal(blurInsideClosingDialog(dialog("closed", "dialog", false), active), false);
  assert.equal(blurInsideClosingDialog(null, active), false);
  assert.equal(blurInsideClosingDialog(dialog("closed"), null), false);
  assert.equal(blurred, 2);
  assert.match(readSrc("main.tsx"), /watchClosingDialogFocus\(document\);/);
});
