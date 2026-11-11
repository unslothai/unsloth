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

test("on screen <body> already has the lock's overflow and position, so a lock toggle is a no-op", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /@media screen \{\s*body \{\s*overflow: hidden;\s*position: relative;\s*\}\s*\}/,
  );
  const lock = rule(css, "body[data-scroll-locked]");
  assert.ok(lock.includes("overflow: hidden !important;") && lock.includes("position: relative !important;"));
});

test("the login and change-password pages scroll in their own <main>, now that <body> never does", () => {
  const hiddenRouteMain = /<main className="([^"]*--studio-hidden-route-top-inset[^"]*)"/.exec(
    readSrc("app/routes/__root.tsx"),
  );
  assert.ok(hiddenRouteMain, "the hidden-navbar <main> is gone");
  const classes = hiddenRouteMain[1].split(" ");
  assert.ok(classes.includes("h-dvh") && classes.includes("overflow-y-auto"), hiddenRouteMain[1]);
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
    "[data-modal-lock-region] .chat-artifact-pop-surface .side-pane-lock-cover::after,\n\t[data-modal-lock-region].chat-artifact-pop-surface .side-pane-lock-cover::after";
  const cover = rule(css, selector);
  // will-change: its own layer, or Chromium's compositor scrolls the pane's scroller under it.
  for (const declaration of ['content: "";', "position: absolute;", "inset: 0;", "pointer-events: auto;", "will-change: transform;"]) {
    assert.ok(cover.includes(declaration), `side pane cover lost ${declaration}`);
  }
  const z = Number(/z-index: (\d+);/.exec(cover)?.[1]);
  assert.ok(z > 0 && z < 50, `cover z-index ${z} must sit under the z-50 portaled layers`);
  // The shown pane opts out of the body lock, which is why it needs the cover.
  assert.match(rule(css, '.chat-artifact-pop-surface[data-artifact-surface-visible="true"]'), /pointer-events: auto;/);
  // Keyed on a static class on the pane roots, which also position the cover: an :is() subject
  // here made every lock toggle restyle the thread (gate: 120 ms at 200 turns).
  assert.match(readSrc("features/browser/browser-panel.tsx"), /"side-pane-lock-cover relative flex h-full min-h-0 flex-col overflow-hidden/);
  assert.match(readSrc("features/chat/components/research-activity-panel.tsx"), /"relative flex min-h-0 flex-col[\s\S]{0,120}variant === "panel" && "side-pane-lock-cover"/);
});

test("pointer-events-auto restorations go inert in a lower modal layer, not in the page", () => {
  const css = readSrc("index.css");
  const lower =
    ':is([role="dialog"], [role="alertdialog"], [role="menu"], [role="listbox"])[data-modal-lock-lower] .pointer-events-auto:not([data-radix-popper-content-wrapper] *)';
  assert.match(rule(css, lower), /pointer-events: none !important;/);
  // The Skills rows (the confirmed case) restore pointer-events with exactly this class.
  assert.match(readSrc("features/chat/components/chat-skills-dialog.tsx"), /className="pointer-events-auto absolute inset-0/);
  // No page-wide rule: dialogs portaled into the page (Recipe Studio) host live comboboxes there.
  assert.doesNotMatch(css, /\[data-slot="sidebar-inset"\] \.pointer-events-auto/);
  // Page restorations (toolbars, the Downloads card) go inert under the body lock, with one rule:
  // inside a page region, or on a region root or a thread root's ancestor itself.
  const not =
    ':not(.aui-thread-root *):not([role="dialog"] *):not([role="alertdialog"] *):not([data-radix-popper-content-wrapper] *):not(.modal-pe-live):not(.modal-pe-live *)';
  const page = `[data-modal-lock-region] :is(.pointer-events-auto, .sidebar-row-action)${not},\n\t:is([data-modal-lock-region], [data-modal-lock-path]):is(.pointer-events-auto, .sidebar-row-action)${not}`;
  assert.match(rule(css, page), /pointer-events: none !important;/);
  assert.match(readSrc("features/hub/download-manager/download-manager-panel.tsx"), /"hub-download-panel pointer-events-auto /);
  // What is meant to work over a modal stays live.
  const live: Array<[string, RegExp]> = [
    ["components/tauri/window-titlebar.tsx", /"modal-pe-live pointer-events-none absolute inset-x-0 top-0 z-\[70\]/],
    ["components/tauri/startup-screen.tsx", /"pointer-events-auto modal-pe-live fixed inset-0 z-\[9999\]"/],
    ["components/ui/sonner.tsx", /toast: "cn-toast modal-pe-live pointer-events-auto"/],
    ["components/ui/combobox.tsx", /"isolate z-\[120\] modal-pe-live pointer-events-auto"/],
    ["features/interface-zoom/components/interface-zoom.tsx", /"modal-pe-live interface-zoom-position pointer-events-auto/],
    ["app/provider.tsx", /data-tauri-drag-region=\{true\}[\s\S]{0,80}"pointer-events-auto modal-pe-live /],
  ];
  for (const [file, re] of live) assert.match(readSrc(file), re, file);
  assert.equal(readSrc("components/tauri/window-titlebar.tsx").match(/"pointer-events-auto fixed /g), null);
});
