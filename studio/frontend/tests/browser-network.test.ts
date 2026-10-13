// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

// The page shell's fetch and XHR (routes/browser_network.py), run as shipped.
const source = readFileSync(new URL("../../backend/routes/browser_network.py", import.meta.url), "utf8")
  .split('r"""')[1]
  .split('"""')[0];

/** A stand-in for the browser's own XHR: a request ends when the test fires its loadend. */
class NativeXHR extends EventTarget {
  open() {}
  send() {}
}

/** As a browser's: send() on a request already in flight throws InvalidStateError. */
class StrictXHR extends NativeXHR {
  inFlight = false;
  constructor() {
    super();
    this.addEventListener("loadend", () => (this.inFlight = false));
  }
  open() {
    this.inFlight = false;
  }
  send() {
    if (this.inFlight) throw new Error("InvalidStateError");
    this.inFlight = true;
  }
}

function setup(XHR: typeof NativeXHR = NativeXHR) {
  let now = 0;
  // The script patches its Response's body readers: a class of its own keeps that out of other tests.
  class PageResponse extends Response {}
  const answers: Array<(response: Response) => void> = [];
  const window = Object.assign(new EventTarget(), {
    fetch: () => new Promise<Response>((resolve) => answers.push(resolve)),
    XMLHttpRequest: XHR,
  });
  runInNewContext(source, { window, Date: { now: () => now }, Response: PageResponse, Symbol, Object, Set });
  const pending = (since: number) => (window as unknown as Record<symbol, (since: number) => number>)[Symbol.for("unsloth.browser.pending")](since);
  const page = window as unknown as { fetch: (url: string) => Promise<Response>; XMLHttpRequest: new () => NativeXHR };
  return {
    window: page, pending, advance: (ms: number) => (now += ms),
    answer: (i: number, body = "{}") => answers[i](new PageResponse(body)),
    Response: PageResponse,
  };
}

test("only requests started since the action count, however long they stay open", async () => {
  const t = setup();
  const poll = t.window.fetch("/stream");
  // A long poll the page already had open isn't the action's.
  t.advance(1000);
  assert.equal(t.pending(1000), 0);
  // Cross-origin requests count too: a list fetched from another API is still content on its way.
  const api = t.window.fetch("https://api.github.com/x");
  t.advance(60_000);
  assert.equal(t.pending(1000), 1);
  assert.equal(t.pending(0), 2);
  t.answer(0);
  t.answer(1);
  await Promise.all([poll, api]);
  assert.equal(t.pending(0), 0);
});

test("reading a response's body still counts, since the page draws only once it has it", async () => {
  const t = setup();
  const request = t.window.fetch("https://api.github.com/repos/x/git/trees/y");
  t.answer(0, "tree");
  const response = await request;
  assert.equal(t.pending(0), 0);
  const body = response.text();
  assert.equal(t.pending(0), 1);
  assert.equal(await body, "tree");
  assert.equal(t.pending(0), 0);
});

test("an XHR counts from send until loadend", () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.open();
  xhr.send();
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

test("reopening an XHR in flight ends its count, since that request never fires loadend", () => {
  const t = setup();
  const xhr = new t.window.XMLHttpRequest();
  xhr.open();
  xhr.send();
  xhr.open();
  assert.equal(t.pending(0), 0);
  xhr.send();
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

test("a body read that fails leaves no count behind", async () => {
  const t = setup();
  const text = t.Response.prototype.text;
  await assert.rejects(text.call({}));
  assert.equal(t.pending(0), 0);
});

test("a second send the browser refuses keeps counting the request in flight", () => {
  const t = setup(StrictXHR);
  const xhr = new t.window.XMLHttpRequest();
  xhr.open();
  xhr.send();
  assert.throws(() => xhr.send());
  assert.equal(t.pending(0), 1);
  xhr.dispatchEvent(new Event("loadend"));
  assert.equal(t.pending(0), 0);
});

test("the shell reports how long the DOM has been still since the document began", () => {
  let now = 1000;
  let notify: () => void = () => {};
  class FakeObserver {
    constructor(callback: () => void) {
      notify = callback;
    }
    observe() {}
  }
  class PageResponse extends Response {}
  const window = Object.assign(new EventTarget(), { fetch: async () => new PageResponse(""), XMLHttpRequest: NativeXHR });
  runInNewContext(source, {
    window, document: {}, MutationObserver: FakeObserver, Date: { now: () => now }, Response: PageResponse, Symbol, Object, Set,
  });
  const quiet = (window as unknown as Record<symbol, () => number>)[Symbol.for("unsloth.browser.quietMs")];
  now += 400;
  assert.equal(quiet(), 400);
  notify();
  now += 50;
  assert.equal(quiet(), 50);
});
