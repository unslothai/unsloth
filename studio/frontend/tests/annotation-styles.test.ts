// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const backend = readFileSync(new URL("../../backend/routes/browser.py", import.meta.url), "utf8");
const annotation = backend.slice(backend.indexOf('_ANNOTATE_JS = r"""'));
const helper = annotation.slice(annotation.indexOf("const addStyle ="), annotation.indexOf("const build ="));

test("annotation sheets work without inserting inline styles and preserve existing sheets", () => {
  class Sheet {
    css = "";
    replaceSync(css: string) { this.css = css; }
  }
  const addStyle = vm.runInNewContext(`${helper}; addStyle`, {
    CSSStyleSheet: Sheet,
    document: { createElement: () => { throw new Error("Inline styles blocked by CSP"); } },
  });
  const existing = new Sheet();
  const target = { adoptedStyleSheets: [existing] };
  const remove = addStyle(target, ".mark { position: fixed; border: 1px dashed blue; }");
  assert.equal(target.adoptedStyleSheets[0], existing);
  assert.equal(target.adoptedStyleSheets[1]!.css, ".mark { position: fixed; border: 1px dashed blue; }");
  const other = new Sheet();
  target.adoptedStyleSheets.push(other);
  remove();
  assert.deepEqual([...target.adoptedStyleSheets], [existing, other]);
});

test("older webviews retain their inline-style path", () => {
  const inserted: unknown[] = [];
  let removed = false;
  const style = { textContent: "", remove: () => { removed = true; } };
  const addStyle = vm.runInNewContext(`${helper}; addStyle`, { document: { createElement: () => style } });
  const remove = addStyle({ head: { appendChild: (value: unknown) => inserted.push(value) } }, ".pin { display: flex; }");
  assert.equal(style.textContent, ".pin { display: flex; }");
  assert.deepEqual(inserted, [style]);
  remove();
  assert.equal(removed, true);
});


test("annotation controls can be built when Trusted Types rejects HTML sinks", () => {
  const children: Array<{ className: string }> = [];
  const root = {
    set innerHTML(_value: string) { throw new TypeError("TrustedHTML required"); },
    appendChild: (element: { className: string }) => children.push(element),
    querySelector: (selector: string) => children.find((element) => `.${element.className}` === selector),
  };
  const overlay = { style: {}, attachShadow: () => root };
  const buildSource = annotation.slice(annotation.indexOf("const build ="), annotation.indexOf("const pinBox ="));
  const build = vm.runInNewContext(`${buildSource}; build`, {
    root: null, overlay: null, outline: null, area: null,
    document: { createElement: (tag: string) => tag === "unsloth-annotate" ? overlay : { className: "" } },
    color: "#123456", PIN: 28, addStyle: () => {},
  });
  build();
  assert.deepEqual(children.map((element) => element.className), ["hover", "area"]);
});
