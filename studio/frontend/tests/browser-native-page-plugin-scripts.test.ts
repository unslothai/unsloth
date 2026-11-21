// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { runInNewContext } from "node:vm";
import { readText } from "./helpers/kit.ts";

const rust = readText("../../src-tauri/src/browser_webview.rs");
const guard = rust.split('const PAGE_GUARD_SCRIPT: &str = r#"')[1].split('"#;')[0];
const restore = rust.split('const PAGE_RESTORE_SCRIPT: &str = "')[1].split('";')[0];

// Init scripts of opener 2.5.4, dialog 2.7.2, notification 2.3.3 (Cargo.lock), wrapped as Tauri does.
const PLUGINS = [
  `window.addEventListener("click",function(e){if(e.defaultPrevented||0!==e.button||e.metaKey||e.altKey)return;const t=e.composedPath().find(e=>e instanceof Node&&"A"===e.nodeName.toUpperCase());if(!t||!t.href||"_blank"!==t.target&&!e.ctrlKey&&!e.shiftKey)return;const n=new URL(t.href);["http:","https:","mailto:","tel:"].every(e=>n.protocol!==e)||(e.preventDefault(),window.__TAURI_INTERNALS__.invoke("plugin:opener|open_url",{url:n}))})`,
  `window.alert=function(i){window.__TAURI_INTERNALS__.invoke("plugin:dialog|message",{message:i.toString()})},window.confirm=async function(i){return await window.__TAURI_INTERNALS__.invoke("plugin:dialog|confirm",{message:i.toString()})}`,
  `window.Notification=function(n,t){window.__TAURI_INTERNALS__.invoke("plugin:notification|notify",{options:{title:n}})}`,
].map((script) => `(function () { ${script} })();`);

function nativeFn(name: string) {
  return { [name]() {} }[name];
}

function loadDocument(label: string, { guarded }: { guarded: boolean }) {
  const target = new EventTarget();
  const invoked: string[] = [];
  class Node {}
  const window: Record<string, unknown> = {
    URL,
    Node,
    Event,
    alert: nativeFn("alert"),
    confirm: nativeFn("confirm"),
    Notification: nativeFn("Notification"),
    addEventListener: target.addEventListener.bind(target),
    removeEventListener: target.removeEventListener.bind(target),
    dispatchEvent: target.dispatchEvent.bind(target),
    __TAURI_INTERNALS__: {
      metadata: { currentWebview: { label } },
      invoke: async (command: string) => void invoked.push(command),
    },
  };
  window.window = window;
  const natives = { alert: window.alert, confirm: window.confirm, Notification: window.Notification };
  const run = (script: string) => runInNewContext(script, window);
  if (guarded) run(guard);
  PLUGINS.forEach(run);
  if (guarded && label.startsWith("unsloth-browser-")) run(restore);
  const click = (href: string, targetAttr: string) =>
    run(`(() => {
      const link = Object.assign(new Node(), { nodeName: "A", href: ${JSON.stringify(href)}, target: ${JSON.stringify(targetAttr)} });
      const event = new Event("click", { cancelable: true });
      Object.defineProperties(event, { button: { value: 0 }, composedPath: { value: () => [link] } });
      window.dispatchEvent(event);
      return event.defaultPrevented;
    })()`) as boolean;
  return { window, natives, invoked, click, run };
}

test("unguarded, a page's target=_blank clicks are cancelled and alert is a refused command", () => {
  const page = loadDocument("unsloth-browser-tab1", { guarded: false });
  assert.equal(page.click("https://example.org/", "_blank"), true);
  assert.notEqual(page.window.alert, page.natives.alert);
  (page.window.alert as (message: string) => void)("hi");
  assert.deepEqual(page.invoked, ["plugin:opener|open_url", "plugin:dialog|message"]);
});

test("a browser page keeps its own target=_blank clicks, alert, confirm and Notification", () => {
  const page = loadDocument("unsloth-browser-tab1", { guarded: true });
  assert.equal(page.click("https://example.org/", "_blank"), false);
  assert.equal(page.window.alert, page.natives.alert);
  assert.equal(page.window.confirm, page.natives.confirm);
  assert.equal(page.window.Notification, page.natives.Notification);
  assert.equal("__unslothPageGuard" in page.window, false);
  assert.deepEqual(page.invoked, []);
  page.run(`window.alert = () => { window.ownAlert = true; }; alert("x");
    window.addEventListener("click", () => { window.ownClick = true; });`);
  page.click("https://example.org/", "");
  assert.equal(page.window.ownAlert, true);
  assert.equal(page.window.ownClick, true);
});

test("the app's own webview keeps the plugins", () => {
  const main = loadDocument("main", { guarded: true });
  assert.equal(main.click("https://example.org/", "_blank"), true);
  assert.notEqual(main.window.alert, main.natives.alert);
  assert.equal("__unslothPageGuard" in main.window, false);
  assert.deepEqual(main.invoked, ["plugin:opener|open_url"]);
});

test("the guard plugin is registered before every plugin that injects page scripts", () => {
  const main = readText("../../src-tauri/src/main.rs");
  const at = (needle: string) => {
    const index = main.indexOf(needle);
    assert.ok(index >= 0, needle);
    return index;
  };
  const guardAt = at(".plugin(browser_webview::page_guard_plugin())");
  for (const plugin of ["tauri_plugin_opener::", "tauri_plugin_dialog::", "tauri_plugin_notification::"]) {
    assert.ok(guardAt < at(`.plugin(${plugin}`), plugin);
  }
  assert.match(rust, /\.initialization_script\(PAGE_RESTORE_SCRIPT\)/);
});
