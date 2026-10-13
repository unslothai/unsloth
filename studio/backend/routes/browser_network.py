# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The page shell's fetch and XMLHttpRequest. Wrapped to count the page's requests in flight, so the
browser agent can wait for content a page fetches after load before it reads the page; and to send
the page's requests to its own site through the panel, as the sandbox's opaque origin makes them
cross-origin. Runs inside ``boot`` in routes/browser.py."""

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

        // To the browser the sandbox's origin is opaque, so even the page's own site is another origin,
        // and an API of it that sends no CORS headers can't be read. Requests to the page's origin go
        // through the panel to Studio's backend (POST /api/browser/request), which makes them as the
        // same-origin requests they are. Requests to other sites stay the browser's.
        const pageOrigin = (() => { try { return cfg.url ? new URL(cfg.url).origin : null; } catch { return null; } })();
        const ownSite = (url) => { try { return pageOrigin !== null && new URL(url).origin === pageOrigin; } catch { return false; } };
        const replies = new Map();
        let lastRequest = 0;
        window.addEventListener("message", (event) => {
          const data = event.data;
          if (event.source !== parent || !data || data.type !== "unsloth:browser-command" || data.command !== "response") return;
          replies.get(data.id)?.(data);
        }, true);
        // The panel's { status, statusText, url, headers, body }. A failure rejects as fetch's network
        // error does; an abort rejects with the signal's reason, and the panel stops the request.
        const relay = (request, signal) => new Promise((resolve, reject) => {
          if (signal.aborted) { reject(signal.reason); return; }
          const id = "q" + ++lastRequest;
          const settle = () => { replies.delete(id); signal.removeEventListener("abort", onAbort); };
          const onAbort = () => { settle(); post({ type: "requestAbort", id }); reject(signal.reason); };
          signal.addEventListener("abort", onAbort);
          replies.set(id, (reply) => {
            settle();
            if (reply.ok === true) resolve(reply);
            else reject(new TypeError("Failed to fetch"));
          });
          post({ type: "request", id, ...request });
        });
        const bodiless = (method) => method === "GET" || method === "HEAD";
        const NULL_BODY_STATUSES = new Set([101, 103, 204, 205, 304]);
        const withoutHash = (url) => url.split("#")[0];
        // What a fetched response has and a constructed one lacks, kept through clone(): "basic" on
        // the page's origin, "cors" when a redirect led to another site.
        const asFetched = (response, url, redirected, type) => {
          const clone = response.clone;
          return Object.defineProperties(response, {
            url: { value: url },
            redirected: { value: redirected },
            type: { value: type },
            clone: { value: () => asFetched(clone.call(response), url, redirected, type) },
          });
        };
        const relayFetch = async (request) => {
          const body = bodiless(request.method) ? null : await request.arrayBuffer();
          const reply = await relay(
            { url: request.url, method: request.method, headers: [...request.headers], body, redirect: request.redirect },
            request.signal,
          );
          // redirect: "manual" stopped at a redirect, which a page sees only as an opaque one.
          if (reply.opaqueRedirect === true) return asFetched(Response.error(), withoutHash(request.url), false, "opaqueredirect");
          const headers = new Headers();
          for (const [name, value] of reply.headers) try { headers.append(name, value); } catch {}
          const content = request.method === "HEAD" || NULL_BODY_STATUSES.has(reply.status) ? null : reply.body;
          let response;
          // A reason phrase a Response refuses is left out; HTTP/2 has none anyway.
          try { response = new Response(content, { status: reply.status, statusText: reply.statusText, headers }); }
          catch { response = new Response(content, { status: reply.status, headers }); }
          const url = withoutHash(reply.url);
          return asFetched(response, url, url !== withoutHash(request.url), ownSite(url) ? "basic" : "cors");
        };
        window.fetch = (input, init) => {
          const done = track();
          // Relative addresses resolve against the page's <base>, as the browser's would. Other inputs
          // (a Request from another frame) are left to the browser.
          let url = null;
          try {
            if (input instanceof Request) url = input.url;
            else if (typeof input === "string" || input instanceof URL) url = new URL(input, document.baseURI).href;
          } catch {}
          if (!ownSite(url)) return nativeFetch(input, init).finally(done);
          let request;
          try { request = new Request(input instanceof Request ? input : url, init); } catch (error) { done(); return Promise.reject(error); }
          return relayFetch(request).finally(done);
        };

        // An XHR to the page's site stays OPENED while the panel makes it, then the browser loads the
        // answer from a blob: URL, so events, responseType and decoding are the browser's own; status,
        // headers and URL read as the site's. A synchronous XHR can't wait, so it stays the browser's.
        const relayed = new WeakMap();
        const headerMap = (list) => {
          const merged = new Map();
          for (const [name, value] of list) {
            const key = name.toLowerCase();
            merged.set(key, merged.has(key) ? merged.get(key) + ", " + value : value);
          }
          return merged;
        };
        const nativeReadyState = (xhr) => Reflect.get(NativeXHR.prototype, "readyState", xhr);
        // The site's answer, once the browser has loaded its headers.
        const answered = (xhr) => {
          const state = relayed.get(xhr);
          return state?.reply && nativeReadyState(xhr) >= 2 ? state.reply : null;
        };
        window.XMLHttpRequest = class extends NativeXHR {
          constructor() {
            super();
            // send() fires loadstart; the blob: load's own would be a second. Added before any of the
            // page's listeners, so stopping it here keeps it from all of them.
            this.addEventListener("loadstart", (event) => {
              if (event.isTrusted && relayed.get(this)?.reply) event.stopImmediatePropagation();
            }, true);
          }
          // open() ends a request in flight without a loadend, so its count ends here too.
          open(...args) {
            this._done?.();
            relayed.get(this)?.controller?.abort();
            relayed.delete(this);
            super.open(...args);
            const url = args.length > 1 ? resolve(String(args[1])) : null;
            const async = args.length < 3 || Boolean(args[2]);
            if (async && ownSite(url)) relayed.set(this, { method: String(args[0]).toUpperCase(), url, headers: [] });
          }
          setRequestHeader(name, value) {
            super.setRequestHeader(name, value);
            relayed.get(this)?.headers.push([String(name), String(value)]);
          }
          send(body) {
            const state = relayed.get(this);
            // As the browser's: one already sent, or ended, can't be sent again before open().
            if (state && (state.controller || state.readyState !== undefined)) {
              throw new DOMException("The object's state must be OPENED.", "InvalidStateError");
            }
            const done = track();
            // Listening first: a synchronous request fires loadend inside send().
            this.addEventListener("loadend", done, { once: true });
            if (state) this.#relay(state, body);
            else {
              // A send the browser refuses (one already in flight) leaves that one counted.
              try { super.send(body); } catch (error) { this.removeEventListener("loadend", done); done(); throw error; }
            }
            if (this._done !== done) this._done?.();
            this._done = done;
          }
          #relay(state, body) {
            const controller = (state.controller = new AbortController());
            // XHR's request error steps: DONE, then readystatechange, the error's event and loadend; an
            // abort leaves it UNSENT.
            const fail = (type) => {
              if (relayed.get(this) !== state || state.reply || state.readyState !== undefined) return;
              clearTimeout(timer);
              controller.abort();
              state.readyState = 4;
              this.dispatchEvent(new Event("readystatechange"));
              this.dispatchEvent(new ProgressEvent(type));
              this.dispatchEvent(new ProgressEvent("loadend"));
              if (type === "abort") state.readyState = 0;
            };
            const timer = this.timeout > 0 ? setTimeout(() => fail("timeout"), this.timeout) : 0;
            state.fail = fail;
            this.dispatchEvent(new ProgressEvent("loadstart"));
            // The body as bytes, with the Content-Type the browser would give it.
            const bytes = bodiless(state.method) || body == null ? Promise.resolve(null) : (async () => {
              // XHR serializes a Document, which Fetch's bodies don't take, as UTF-8 HTML or XML.
              const html = body instanceof Document && body.contentType === "text/html";
              const content = !(body instanceof Document) ? body
                : html ? (body.doctype ? "<!DOCTYPE " + body.doctype.name + ">" : "") + (body.documentElement?.outerHTML ?? "")
                : new XMLSerializer().serializeToString(body);
              const request = new Request(state.url, { method: "POST", body: content });
              const type = !(body instanceof Document) ? request.headers.get("content-type")
                : (html ? "text/html" : "application/xml") + ";charset=UTF-8";
              const own = state.headers.filter(([name]) => name.toLowerCase() === "content-type");
              if (!own.length) { if (type) state.headers.push(["Content-Type", type]); }
              // Sent as UTF-8, so the browser relabels the page's charset as UTF-8, other parameters kept.
              else if (typeof content === "string" || content instanceof URLSearchParams) {
                for (const header of own) header[1] = header[1].replace(/(;\s*charset=)("?)[^;"]*/i, "$1$2UTF-8");
              }
              return request.arrayBuffer();
            })();
            bytes
              .then((content) => relay({ url: state.url, method: state.method, headers: state.headers, body: content, redirect: "follow" }, controller.signal))
              .then((reply) => {
                if (relayed.get(this) !== state || state.readyState !== undefined) return;
                clearTimeout(timer);
                const type = headerMap(reply.headers).get("content-type");
                const link = URL.createObjectURL(new Blob([reply.body], type ? { type } : {}));
                this.addEventListener("loadend", () => URL.revokeObjectURL(link), { once: true });
                state.reply = reply;
                super.open("GET", link);
                super.send();
              }, () => fail("error"));
          }
          abort() {
            const state = relayed.get(this);
            if (state?.fail && !state.reply) state.fail("abort");
            else super.abort();
          }
          get readyState() { return relayed.get(this)?.readyState ?? super.readyState; }
          get status() { return answered(this)?.status ?? super.status; }
          get statusText() { return answered(this)?.statusText ?? super.statusText; }
          get responseURL() { const reply = answered(this); return reply ? withoutHash(reply.url) : super.responseURL; }
          getResponseHeader(name) {
            const reply = answered(this);
            return reply ? (headerMap(reply.headers).get(String(name).toLowerCase()) ?? null) : super.getResponseHeader(name);
          }
          getAllResponseHeaders() {
            const reply = answered(this);
            if (!reply) return super.getAllResponseHeaders();
            return [...headerMap(reply.headers)]
              .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
              .map(([name, value]) => name + ": " + value + "\r\n")
              .join("");
          }
        };
"""
