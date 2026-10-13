# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The page shell's fetch and XMLHttpRequest, wrapped to count the page's requests in flight, so the
browser agent can wait for content a page fetches after load before it reads the page.
Runs inside ``boot`` in routes/browser.py."""

BROWSER_NETWORK_JS = r"""
        const nativeFetch = window.fetch.bind(window);
        const NativeXHR = window.XMLHttpRequest;
        // Start times of the page's requests in flight, so the agent can wait for content still coming.
        // The wall clock, which the panel shares, captured before a page script can replace it.
        const clock = Date.now.bind(Date);
        const inflight = new Set();
        const track = () => { const entry = { at: clock() }; inflight.add(entry); return () => inflight.delete(entry); };
        // Only those started since the agent's action: a long poll the page already had open isn't its.
        Object.defineProperty(window, Symbol.for("unsloth.browser.pending"), {
          value: (since) => { let n = 0; for (const r of inflight) if (r.at >= since) n++; return n; },
        });
        // fetch() resolves at the headers; a list is drawn only once its body is read, so a read counts too.
        // The page calls it in the same microtask run as fetch resolves, before any agent command arrives.
        // A body streamed through response.body.getReader() is not counted.
        for (const name of ["arrayBuffer", "blob", "bytes", "formData", "json", "text"]) {
          const read = Response.prototype[name];
          if (typeof read !== "function") continue;
          Response.prototype[name] = function (...args) {
            const reading = read.apply(this, args);
            return reading.finally(track());
          };
        }
        window.fetch = (input, init) => {
          const done = track();
          return nativeFetch(input, init).finally(done);
        };
        window.XMLHttpRequest = class extends NativeXHR {
          // open() ends a request in flight without a loadend, so its count ends here too.
          open(...args) {
            this._done?.();
            super.open(...args);
          }
          send(body) {
            this._done?.();
            const done = this._done = track();
            this.addEventListener("loadend", done, { once: true });
            try { super.send(body); } catch (error) { done(); throw error; }
          }
        };
"""
