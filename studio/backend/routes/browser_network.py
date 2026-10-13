# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The page shell's fetch and XMLHttpRequest. A page in the sandbox has an opaque origin, so CORS refuses
its own site's API (a search box, a filter, "load more"). Public same-origin GETs go to the panel, which
reads them through ``/api/browser/resource`` without Studio's credentials ever entering the sandbox.
Runs inside ``boot`` in routes/browser.py, which supplies ``cfg`` and ``post``."""

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
        for (const name of ["arrayBuffer", "blob", "bytes", "formData", "json", "text"]) {
          const read = Response.prototype[name];
          if (typeof read !== "function") continue;
          Response.prototype[name] = function (...args) {
            const done = track();
            return read.apply(this, args).finally(done);
          };
        }
        const resources = new Map();
        let resourceId = 0;
        const readResource = (url, accept, signal) => new Promise((resolve, reject) => {
          if (signal?.aborted) { reject(new DOMException("Aborted", "AbortError")); return; }
          const id = `resource-${++resourceId}`;
          const settle = () => { clearTimeout(timer); signal?.removeEventListener("abort", abort); resources.delete(id); };
          const abort = () => { post({ type: "resourceCancel", id }); settle(); reject(new DOMException("Aborted", "AbortError")); };
          const timer = setTimeout(() => { post({ type: "resourceCancel", id }); settle(); reject(new TypeError("Failed to fetch")); }, 60000);
          resources.set(id, (result) => { settle(); result && !result.error ? resolve(result) : reject(new TypeError("Failed to fetch")); });
          signal?.addEventListener("abort", abort, { once: true });
          post({ type: "resource", id, url, accept });
        });
        window.addEventListener("message", (event) => {
          if (event.source !== parent) return;
          const data = event.data;
          if (data?.type === "unsloth:browser-command" && data.command === "resourceResult") resources.get(data.id)?.(data.result);
        });
        const resourceUrl = (value) => new URL(String(value), document.baseURI).href;
        const relayed = (url, method) => {
          try {
            const target = new URL(url);
            return method === "GET" && /^https?:$/.test(target.protocol) && target.origin === new URL(cfg.url).origin;
          } catch { return false; }
        };
        const bytesOf = (body) => {
          const text = atob(body);
          const bytes = new Uint8Array(text.length);
          for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
          return bytes;
        };
        window.fetch = (input, init) => {
          const done = track();
          return (async () => {
            const request = new Request(input instanceof Request ? input : resourceUrl(input), init);
            if (!relayed(request.url, request.method) || request.headers.has("authorization") || request.credentials === "include") {
              return nativeFetch(request);
            }
            const result = await readResource(request.url, request.headers.get("accept") || "*/*", request.signal);
            const response = new Response([101, 204, 205, 304].includes(result.status) ? null : bytesOf(result.body), {
              status: result.status, headers: result.headers,
            });
            Object.defineProperty(response, "url", { value: result.url });
            return response;
          })().finally(done);
        };
        // Writes, credentials and other origins keep the native request; public same-origin reads go
        // through the panel like fetch, with XHR's events in their usual order.
        window.XMLHttpRequest = class extends NativeXHR {
          open(method, url, async = true, user, password) {
            // open() ends a request in flight without events; native open() does that for its own.
            this._controller?.abort();
            this._controller = null;
            // responseType may be set before open(); a relayed request keeps it, as a native one does.
            const type = this._relay ? this._type : super.responseType;
            this._url = resourceUrl(url);
            this._relay = async !== false && !user && !password && relayed(this._url, String(method).toUpperCase());
            this._headers = new Headers();
            this._state = this._relay ? 1 : 0;
            this._result = null;
            this._bytes = null;
            this._text = null;
            this._type = type;
            super.open(method, this._url, async, user, password);
          }
          get readyState() { return this._relay ? this._state : super.readyState; }
          get status() { return this._relay ? this._result?.status || 0 : super.status; }
          get statusText() { return this._relay ? "" : super.statusText; }
          get responseURL() { return this._relay ? this._result?.url || "" : super.responseURL; }
          get responseType() { return this._relay ? this._type : super.responseType; }
          set responseType(value) { if (this._relay) this._type = value; else super.responseType = value; }
          get _decoded() { return (this._text ??= this._bytes ? new TextDecoder().decode(this._bytes) : ""); }
          get responseText() {
            if (!this._relay) return super.responseText;
            if (this._type && this._type !== "text") throw new DOMException("Not a text response", "InvalidStateError");
            return this._decoded;
          }
          get responseXML() {
            if (!this._relay) return super.responseXML;
            const type = this.getResponseHeader("content-type") || "";
            // HTML is a document only when asked for one, as in a browser.
            const html = /html/.test(type);
            if (!this._bytes || !/xml|html/.test(type) || (html && this._type !== "document")) return null;
            return new DOMParser().parseFromString(this._decoded, html ? "text/html" : "text/xml");
          }
          get response() {
            if (!this._relay) return super.response;
            if (this._state !== 4) return this._type && this._type !== "text" ? null : "";
            if (this._type === "arraybuffer") return this._bytes?.buffer || null;
            if (this._type === "blob") return this._bytes ? new Blob([this._bytes], { type: this.getResponseHeader("content-type") || "" }) : null;
            if (this._type === "document") return this.responseXML;
            if (this._type === "json") { try { return JSON.parse(this._decoded); } catch { return null; } }
            return this.responseText;
          }
          setRequestHeader(name, value) { super.setRequestHeader(name, value); this._headers?.append(name, value); }
          getResponseHeader(name) { return this._relay ? this._result?.headers?.[String(name).toLowerCase()] ?? null : super.getResponseHeader(name); }
          getAllResponseHeaders() {
            return this._relay ? Object.entries(this._result?.headers || {}).map(([k, v]) => `${k}: ${v}\r\n`).join("") : super.getAllResponseHeaders();
          }
          abort() {
            const active = this._controller;
            this._controller = null;
            if (active) {
              active.abort();
              this._state = 4; this._result = null; this._bytes = null;
              this.dispatchEvent(new Event("readystatechange"));
              this.dispatchEvent(new ProgressEvent("abort"));
              this.dispatchEvent(new ProgressEvent("loadend"));
              this._state = 0;
            }
            super.abort();
          }
          send(body) {
            if (!this._relay || this.withCredentials || this._headers.has("authorization")) {
              if (this._relay) { this._relay = false; super.responseType = this._type; }
              const done = track();
              this.addEventListener("loadend", done, { once: true });
              try { super.send(body); } catch (error) { done(); throw error; }
              return;
            }
            if (this._state !== 1 || this._controller) throw new DOMException("Request already sent", "InvalidStateError");
            const controller = this._controller = new AbortController();
            let timedOut = false;
            const timer = this.timeout ? setTimeout(() => { timedOut = true; controller.abort(); }, this.timeout) : null;
            const done = track();
            this.dispatchEvent(new ProgressEvent("loadstart"));
            void (async () => {
              let outcome = "load";
              try {
                const result = await readResource(this._url, this._headers.get("accept") || "*/*", controller.signal);
                if (this._controller !== controller) return;
                this._result = result;
                this._state = 2; this.dispatchEvent(new Event("readystatechange"));
                if (this._controller !== controller) return;
                this._bytes = bytesOf(result.body);
                this._state = 3; this.dispatchEvent(new Event("readystatechange"));
                if (this._controller !== controller) return;
                this.dispatchEvent(new ProgressEvent("progress", { loaded: this._bytes.length }));
              } catch { outcome = timedOut ? "timeout" : "error"; }
              finally { if (timer) clearTimeout(timer); }
              if (this._controller !== controller) return;
              this._controller = null;
              this._state = 4; this.dispatchEvent(new Event("readystatechange"));
              this.dispatchEvent(new ProgressEvent(outcome));
              this.dispatchEvent(new ProgressEvent("loadend"));
            })().finally(done);
          }
        };
"""
