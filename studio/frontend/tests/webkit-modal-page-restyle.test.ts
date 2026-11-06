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

test("the side pane sits under the same modal cover as the thread", () => {
  const css = readSrc("index.css");
  const selector =
    'body[style*="pointer-events: none"] .chat-artifact-pop-surface[data-artifact-surface-visible="true"] > :is([data-slot="browser-panel"], aside)::after';
  const cover = rule(css, selector);
  // will-change: its own layer, or Chromium's compositor scrolls the pane's scroller under it.
  for (const declaration of ['content: "";', "position: absolute;", "inset: 0;", "pointer-events: auto;", "will-change: transform;"]) {
    assert.ok(cover.includes(declaration), `side pane cover lost ${declaration}`);
  }
  const z = Number(/z-index: (\d+);/.exec(cover)?.[1]);
  assert.ok(z > 0 && z < 50, `cover z-index ${z} must sit under the z-50 portaled layers`);
  // The shown pane opts out of the body lock, which is why it needs the cover.
  assert.match(rule(css, '.chat-artifact-pop-surface[data-artifact-surface-visible="true"]'), /pointer-events: auto;/);
  // The cover is positioned by the pane roots themselves.
  assert.match(readSrc("features/browser/browser-panel.tsx"), /data-slot="browser-panel"[\s\S]{0,600}"relative flex h-full min-h-0 flex-col overflow-hidden/);
  assert.match(readSrc("features/chat/components/research-activity-panel.tsx"), /<aside[\s\S]{0,200}"relative flex min-h-0 flex-col/);
});

test("pointer-events-auto restorations go inert outside the top modal layer", () => {
  const css = readSrc("index.css");
  const selector =
    ':is([role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"])[style*="pointer-events: none"] .pointer-events-auto,\n\tbody[style*="pointer-events: none"] [data-slot="sidebar-inset"] .pointer-events-auto:not(.aui-thread-root *)';
  const at = css.indexOf(`${selector} {`);
  assert.ok(at >= 0, "missing the lower-layer / page restoration rule");
  const body = css.slice(at, css.indexOf("}", at));
  assert.match(body, /pointer-events: none !important;/);
  // The Skills rows (the confirmed case) restore pointer-events with exactly this class.
  assert.match(readSrc("features/chat/components/chat-skills-dialog.tsx"), /className="pointer-events-auto absolute inset-0/);
  // The page rule needs the inset's slot, which SidebarInset renders.
  assert.match(readSrc("components/ui/sidebar.tsx"), /data-slot="sidebar-inset"/);
});
