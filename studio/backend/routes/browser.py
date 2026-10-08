# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app browser: pages are fetched here (SSRF-guarded, since most sites refuse framing) and
rendered in ``/frame``, an opaque-origin sandbox; an injected script routes navigations back here.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import codecs
import hashlib
import hmac
import html as _html
import json
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional
from urllib.parse import quote, urljoin, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.inference.tools import (
    _USER_AGENTS,
    _WHATWG_CHARSET_CODECS,
    _fetch_url_raw,
    _normalize_url_scheme,
    _sniff_meta_charset,
)
from loggers import get_logger

# same embedders as the canvas shell.
from routes.inference import _ARTIFACT_PREVIEW_FRAME_ANCESTORS as _FRAME_ANCESTORS

logger = get_logger(__name__)

router = APIRouter()

BROWSER_FRAME_PATH = "/api/browser/frame"
BROWSER_PRINT_PATH = "/api/browser/print"
KIND_HEADER = "X-Unsloth-Browser-Kind"
URL_HEADER = "X-Unsloth-Browser-Url"
NAME_HEADER = "X-Unsloth-Browser-Filename"
EXPOSED_HEADERS = (KIND_HEADER, URL_HEADER, NAME_HEADER)

_MAX_BROWSER_FETCH_BYTES = 50 * 1024 * 1024
# HTML goes into srcdoc and the page cache; more would stall the renderer.
_MAX_BROWSER_HTML_BYTES = 10 * 1024 * 1024
_FETCH_TIMEOUT_S = 25
_MAX_REFRESH_DELAY_S = 10
_BROWSER_UA = _USER_AGENTS[1]
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
# Own pool, so slow sites can't starve chat inference's default executor.
_FETCH_POOL = ThreadPoolExecutor(max_workers = 8, thread_name_prefix = "browser-fetch")
_DISCONNECT_POLL_S = 0.25

# Quoted values are bounded and a bare "<" ends a tag, so stripping stays linear on hostile pages.
_TAG_BODY = r"""(?:[^<>"']|"[^"]{0,4096}"|'[^']{0,4096}')*>"""
_BASE_TAG_RE = re.compile(r"<base\b" + _TAG_BODY, re.IGNORECASE)
_ATTR_HREF_RE = re.compile(r"""\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_META_TAG_RE = re.compile(r"<meta\b" + _TAG_BODY, re.IGNORECASE)
_HTTP_EQUIV_RE = re.compile(r"""\bhttp-equiv\s*=\s*["']?([\w-]+)""", re.IGNORECASE)
_CONTENT_ATTR_RE = re.compile(
    r"""\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>]{1,4096}))""", re.IGNORECASE
)
_REFRESH_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:[;,]\s*(?:url\s*=\s*)?['\"]?([^'\"]*)['\"]?)?", re.IGNORECASE
)
# The sandbox's opaque origin fails CORS module loads: self-contained modules are inlined, ones with imports keep src.
_MODULE_SCRIPT_RE = re.compile(r"<script\b" + _TAG_BODY + r"\s*</script\s*>", re.IGNORECASE)
# One start-tag attribute; quoted values are skipped whole so a name inside a value isn't matched.
_TAG_ATTR_RE = re.compile(r"""([^\s/>][^\s/>=]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]*)))?|/""")
_FETCH_ATTRS = frozenset({"src", "integrity", "crossorigin"})
# Imports resolve against the module URL. "/" after import may start a comment; after from it may be division (keeping the tag is the safe side).
_MODULE_IMPORT_RE = re.compile(r"""(?<![\w$.])(?:import\b\s*[{*("'`\w$./]|from\s*["'`/])""")
_MAX_AGE_RE = re.compile(r"(?:^|[\s,])(s-maxage|max-age)\s*=\s*\"?(\d+)", re.IGNORECASE)
_SCRIPT_OPEN_RE = re.compile(r"<script", re.IGNORECASE)
# Text, not markup: comments, raw-text/RCDATA/noscript/template content, and other tags' attribute values.
_INERT_START_RE = re.compile(
    r"<!--|<(script|style|textarea|title|xmp|iframe|noembed|noframes|noscript|plaintext|template)\b"
    + _TAG_BODY
    + r"|<[a-z][^\s/>]*"
    + _TAG_BODY,
    re.IGNORECASE,
)
_TEMPLATE_TAG_RE = re.compile(r"<(/)?template(?=[\s/>])", re.IGNORECASE)
_CLOSING_TAG_RES = {
    name: re.compile(rf"</{name}(?=[\s/>])", re.IGNORECASE)
    for name in (
        "script",
        "style",
        "textarea",
        "title",
        "xmp",
        "iframe",
        "noembed",
        "noframes",
        "noscript",
    )
}
# The JavaScript MIME types a module script may be served as (WHATWG MIME Sniffing).
_JS_TYPES = frozenset(
    "application/ecmascript application/javascript application/x-ecmascript "
    "application/x-javascript text/ecmascript text/javascript text/javascript1.0 "
    "text/javascript1.1 text/javascript1.2 text/javascript1.3 text/javascript1.4 "
    "text/javascript1.5 text/jscript text/livescript text/x-ecmascript text/x-javascript".split()
)
# Strongest last, for Subresource Integrity's "strongest algorithm wins".
_SRI_ALGORITHMS = ("sha256", "sha384", "sha512")
_MAX_INLINED_MODULES = 6
_MAX_MODULE_BYTES = 2 * 1024 * 1024
_MODULE_TIMEOUT_S = 8
# Separate from _FETCH_POOL, whose workers wait on these.
_MODULE_POOL = ThreadPoolExecutor(max_workers = 4, thread_name_prefix = "browser-module")
# (url, integrity, credentials) -> (expiry, code or None); kept only while the response says fresh, failures never.
_MODULE_CACHE: "OrderedDict[tuple[str, str, bool], tuple[float, Optional[str]]]" = OrderedDict()
_MODULE_CACHE_LOCK = threading.Lock()
_MODULE_CACHE_TTL_S = 600
_MODULE_CACHE_ENTRIES = 128
_MODULE_CACHE_CHARS = 16 * 1024 * 1024
_module_cache_chars = 0

# https only (http could hit local services; WebKit lacks local network protection). The sandbox
# (no allow-same-origin) isolates pages; the injected script submits forms.
_FRAME_CSP = (
    "default-src https: data: blob:; "
    "script-src 'unsafe-inline' 'unsafe-eval' https: data: blob:; "
    "style-src 'unsafe-inline' https: data: blob:; "
    "img-src https: data: blob:; "
    "font-src https: data: blob:; "
    "media-src https: data: blob:; "
    "connect-src https: wss: data: blob:; "
    "worker-src https: blob:; "
    "frame-src https: data: blob:; "
    "object-src 'none'; "
    "base-uri http: https:; "
    "form-action 'none'; "
    f"frame-ancestors {_FRAME_ANCESTORS}; "
    "upgrade-insecure-requests; "
    "sandbox allow-scripts allow-forms; "
    # The shell is served from loopback; don't let that exempt pages from local network checks.
    "treat-as-public-address"
)

# Page shell: injects <base> and a script turning navigations into panel messages. The page is a
# srcdoc child; the shell then sets frame-src 'none' on itself so the child can't be navigated
# (it keeps its own policy copy, so embeds still load), independent of the embedder's policy.
_FRAME_HTML = r"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      html, body { margin: 0; height: 100%; overflow: hidden; }
      iframe { display: block; width: 100%; height: 100%; border: 0; }
    </style>
  </head>
  <body>
    <script>
      const boot = (cfg) => {
        // Captured before the page's scripts run, so they can't swap it.
        const compile = Function;
        // about:srcdoc has an opaque origin, and data: and blob: URLs share it.
        const shellOrigin = location.origin === "null" ? null : location.origin;
        let pageUrl = cfg.url || location.href;
        const post = (message) => parent.postMessage({ source: "unsloth-browser", ...message }, "*");
        // URLs built from location point at this shell; map them to the page's site.
        const toPage = (url) => {
          if (!cfg.url || url.origin !== shellOrigin) return url.href;
          return new URL(url.pathname + url.search + url.hash, pageUrl).href;
        };
        const resolve = (href) => {
          try { return toPage(new URL(href, document.baseURI)); } catch { return null; }
        };
        const samePage = (url) => {
          const a = new URL(url), b = new URL(pageUrl);
          return a.origin === b.origin && a.pathname === b.pathname && a.search === b.search;
        };
        // Best effort: give routers the page's path. Chromium refuses this for opaque origins.
        const originalPush = history.pushState;
        const originalReplace = history.replaceState;
        const shellPath = (url) => {
          if (!cfg.url) return undefined;
          try {
            const target = new URL(url);
            if (target.origin !== new URL(cfg.url).origin) return undefined;
            return target.pathname + target.search + target.hash;
          } catch { return undefined; }
        };
        try { originalReplace.call(history, history.state, "", shellPath(pageUrl)); } catch {}
        const scrollToHash = (hash) => {
          const id = decodeURIComponent(hash.replace(/^#/, ""));
          if (!id) { window.scrollTo(0, 0); return; }
          const target = document.getElementById(id) || document.getElementsByName(id)[0];
          target?.scrollIntoView();
        };
        // An in-page jump: scroll, and show its fragment in the address bar.
        const followHash = (hash) => {
          scrollToHash(hash);
          try {
            const next = new URL(pageUrl);
            next.hash = hash;
            pageUrl = next.href;
            reportUrl();
          } catch {}
        };
        const navigate = (raw, newTab, background) => {
          const url = resolve(raw);
          if (!url) return;
          if (!/^https?:/i.test(url)) { post({ type: "external", url }); return; }
          if (!newTab && samePage(url) && new URL(url).hash) { followHash(new URL(url).hash); return; }
          // Relative links in a local file can't be reached.
          if (!cfg.url && new URL(url).origin === shellOrigin) return;
          post({ type: "navigate", url, newTab: Boolean(newTab), background: Boolean(background) });
        };
        const fallbackStorage = () => {
          const data = new Map();
          return {
            get length() { return data.size; },
            key: (i) => Array.from(data.keys())[i] ?? null,
            getItem: (k) => data.has(String(k)) ? data.get(String(k)) : null,
            setItem: (k, v) => void data.set(String(k), String(v)),
            removeItem: (k) => void data.delete(String(k)),
            clear: () => data.clear(),
          };
        };
        // Opaque origins throw on storage and cookies; stub them so pages don't crash.
        for (const name of ["localStorage", "sessionStorage"]) {
          try { void window[name]; } catch {
            try { Object.defineProperty(window, name, { value: fallbackStorage(), configurable: true }); } catch {}
          }
        }
        try {
          let jar = "";
          Object.defineProperty(document, "cookie", { get: () => jar, set: () => {}, configurable: true });
        } catch {}
        const shellPathname = location.pathname;
        // Throttle to 200 ms: some sites replaceState per scroll.
        let reportedUrl = pageUrl;
        let urlTimer = 0;
        const reportUrl = () => {
          if (urlTimer) return;
          urlTimer = setTimeout(() => {
            urlTimer = 0;
            if (pageUrl === reportedUrl) return;
            reportedUrl = pageUrl;
            post({ type: "url", url: pageUrl });
          }, 200);
        };
        for (const [method, original] of [["pushState", originalPush], ["replaceState", originalReplace]]) {
          history[method] = function (state, title, url) {
            let next = url != null ? resolve(String(url)) : null;
            // A router re-asserting the shell's own path is not a navigation.
            if (next && new URL(next).pathname === shellPathname) next = null;
            try { original.call(this, state, title, next ? shellPath(next) : undefined); }
            catch { original.call(this, state, title); }
            if (next) {
              pageUrl = next;
              reportUrl();
            }
          };
        }
        // Catch navigations the click and submit handlers miss (location.href, reload).
        if (window.navigation) {
          window.navigation.addEventListener("navigate", (event) => {
            if (event.destination.sameDocument || event.hashChange || !event.cancelable) return;
            event.preventDefault();
            if (event.navigationType === "reload") post({ type: "reload" });
            else navigate(event.destination.url, false);
          });
        }
        // Popup blocker: only open tabs on user activation.
        window.open = (url) => {
          if (url && navigator.userActivation?.isActive !== false) navigate(String(url), true);
          return null;
        };

        const linkFrom = (event) => {
          for (const node of event.composedPath()) {
            if (node instanceof Element && node.matches("a[href], area[href]")) return node;
          }
          return null;
        };
        // Act once the page's own handlers have run: one that cancels the event (a router's link,
        // a form sent with fetch) keeps it in the page, as in a browser.
        const unlessCancelled = (event, action) => {
          let cancelled = event.defaultPrevented;
          Event.prototype.preventDefault.call(event);
          Object.defineProperties(event, {
            preventDefault: { value: () => { cancelled = true; } },
            defaultPrevented: { get: () => cancelled },
            returnValue: { get: () => !cancelled, set: (value) => { if (value === false) cancelled = true; } },
          });
          // An inline handler's "return false" cancels inside the engine: wrap them for this event.
          const slot = "on" + event.type;
          const inline = [];
          for (const node of event.composedPath()) {
            const handler = node[slot];
            if (typeof handler !== "function") continue;
            inline.push([node, handler]);
            node[slot] = function (...args) {
              const result = handler.apply(this, args);
              if (result === false) cancelled = true;
              return result;
            };
          }
          setTimeout(() => {
            for (const [node, handler] of inline) node[slot] = handler;
            if (!cancelled) action();
          });
        };
        const onLinkClick = (event) => {
          const link = linkFrom(event);
          if (!link) return;
          const raw = (link.getAttribute("href") || "").trim();
          if (/^javascript:/i.test(raw)) return;
          const modified = event.button === 1 || event.metaKey || event.ctrlKey || event.shiftKey;
          unlessCancelled(event, () => {
            // Keep "#" links on the page; against <base> they would navigate.
            if (raw.startsWith("#")) {
              if (raw.length > 1) followHash(raw);
              return;
            }
            navigate(raw, modified || /^_blank$/i.test(link.target || ""), modified);
          });
        };
        document.addEventListener("click", onLinkClick, true);
        document.addEventListener("auxclick", (event) => { if (event.button === 1) onLinkClick(event); }, true);

        const submitForm = (form, submitter) => {
          const method = ((submitter && submitter.getAttribute("formmethod")) || form.getAttribute("method") || "get").toLowerCase();
          const action = resolve((submitter && submitter.getAttribute("formaction")) || form.getAttribute("action") || pageUrl);
          if (!action || method === "dialog") return;
          const params = new URLSearchParams();
          let data;
          try { data = new FormData(form, submitter || undefined); } catch { data = new FormData(form); }
          // Files can't go through the proxy: say so rather than send the form without them.
          for (const value of data.values()) {
            if (typeof value !== "string" && (value.name || value.size)) { post({ type: "upload" }); return; }
          }
          for (const [key, value] of data) if (typeof value === "string") params.append(key, value);
          const target = (submitter && submitter.getAttribute("formtarget")) || form.target || "";
          if (method === "post") {
            post({ type: "navigate", url: action, method: "POST", body: params.toString(), newTab: /^_blank$/i.test(target) });
          } else {
            const url = new URL(action);
            url.search = params.toString();
            navigate(url.href, /^_blank$/i.test(target));
          }
        };
        // requestSubmit() stays native: it validates the form, then fires this event.
        document.addEventListener("submit", (event) => {
          const { target, submitter } = event;
          unlessCancelled(event, () => submitForm(target, submitter));
        }, true);
        HTMLFormElement.prototype.submit = function () { submitForm(this, null); };

        const report = () => {
          const icon = document.querySelector("link[rel~='icon'][href], link[rel='shortcut icon'][href], link[rel='apple-touch-icon'][href]");
          post({
            type: "loaded",
            title: document.title || "",
            favicon: icon ? resolve(icon.getAttribute("href")) : resolve("/favicon.ico"),
          });
          if (new URL(pageUrl).hash) scrollToHash(new URL(pageUrl).hash);
        };
        if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", report, { once: true });
        else report();
        // Watch <head> only; the whole document fires on every DOM change.
        let lastTitle = null;
        const titleObserver = new MutationObserver(() => {
          const title = document.title || "";
          if (title === lastTitle) return;
          lastTitle = title;
          post({ type: "title", title });
        });
        const observeTitle = () => {
          if (document.head) titleObserver.observe(document.head, { childList: true, characterData: true, subtree: true });
        };
        if (document.head) observeTitle();
        else document.addEventListener("DOMContentLoaded", observeTitle, { once: true });
        if (cfg.refresh) {
          setTimeout(() => post({ type: "navigate", url: cfg.refresh.url, replace: true }), cfg.refresh.delay * 1000);
        }
        window.addEventListener("keydown", (event) => {
          if ((event.metaKey || event.ctrlKey) && ["l", "t", "w", "r", "f", "d"].includes(event.key.toLowerCase())) {
            event.preventDefault();
            post({ type: "shortcut", key: event.key.toLowerCase(), shift: event.shiftKey });
          }
        }, true);
        // Zoom keys and Ctrl+wheel (a pinch, too) zoom this page through the panel, not the app
        // around it: Cmd on macOS, Ctrl elsewhere, as the app's own zoom keys.
        const mac = /Mac|iPhone|iPad/.test(navigator.platform);
        window.addEventListener("keydown", (event) => {
          if (event.isComposing || event.altKey || !(mac ? event.metaKey && !event.ctrlKey : event.ctrlKey && !event.metaKey)) return;
          const key = event.key;
          const code = event.code;
          const direction = key === "+" || key === "=" || code === "NumpadAdd" || code === "Equal" ? 1
            : key === "-" || key === "_" || code === "NumpadSubtract" || code === "Minus" ? -1
            : key === "0" || code === "Digit0" || code === "Numpad0" ? 0 : null;
          if (direction === null) return;
          event.preventDefault();
          post({ type: "zoom", direction, wheel: false });
        }, true);
        let wheelTravel = 0;
        let wheelAt = 0;
        window.addEventListener("wheel", (event) => {
          if (!event.ctrlKey || event.metaKey || event.altKey || event.deltaY === 0) return;
          event.preventDefault();
          const delta = event.deltaY * (event.deltaMode === 1 ? 40 : event.deltaMode === 2 ? 800 : 1);
          if (event.timeStamp - wheelAt > 250 || Math.sign(delta) !== Math.sign(wheelTravel)) wheelTravel = 0;
          wheelAt = event.timeStamp;
          wheelTravel += delta;
          if (Math.abs(wheelTravel) < 50) return;
          post({ type: "zoom", direction: wheelTravel < 0 ? 1 : -1, wheel: true });
          wheelTravel = 0;
        }, { passive: false, capture: true });
        const applyZoom = (zoom) => {
          const value = Number(zoom);
          if (!(value >= 0.25 && value <= 5)) return;
          document.documentElement.style.zoom = value === 1 ? "" : String(value);
        };
        if (cfg.zoom) {
          if (document.documentElement) applyZoom(cfg.zoom);
          document.addEventListener("DOMContentLoaded", () => applyZoom(cfg.zoom), { once: true });
        }
        // Mute tab: media elements (in the page or not, as new Audio() makes) play muted, and
        // Web Audio is suspended. Set before the page's scripts run, so it sees every element.
        const setMuted = (() => {
          let muted = false;
          const silenced = new Set();
          const contexts = new Set();
          // Contexts this suspended, so unmuting resumes only those and not one the page paused.
          const paused = new Set();
          const pause = (context) => {
            if (context.state !== "running") return;
            paused.add(context);
            context.suspend().catch(() => {});
          };
          const silence = (media) => {
            if (!muted || media.muted) return;
            media.muted = true;
            silenced.add(media);
          };
          const play = HTMLMediaElement.prototype.play;
          HTMLMediaElement.prototype.play = function (...args) {
            silence(this);
            return play.apply(this, args);
          };
          document.addEventListener("play", (event) => event.target instanceof HTMLMediaElement && silence(event.target), true);
          // A page unmuting its own player while the tab is muted is muted again.
          document.addEventListener("volumechange", (event) => event.target instanceof HTMLMediaElement && silence(event.target), true);
          for (const name of ["AudioContext", "webkitAudioContext"]) {
            const Base = window[name];
            if (typeof Base !== "function") continue;
            window[name] = class extends Base {
              constructor(...args) {
                super(...args);
                contexts.add(this);
                if (muted) pause(this);
              }
              resume() {
                if (!muted) return super.resume();
                // Held until the tab is unmuted, which resumes it then.
                paused.add(this);
                return Promise.resolve();
              }
            };
          }
          return (value) => {
            muted = value === true;
            if (muted) {
              for (const media of document.querySelectorAll("audio, video")) silence(media);
              for (const context of contexts) pause(context);
              return;
            }
            for (const media of silenced) media.muted = false;
            silenced.clear();
            for (const context of paused) context.resume().catch(() => {});
            paused.clear();
          };
        })();
        if (cfg.muted) setMuted(true);
        // Annotate: the panel marks parts of the page to ask about (_ANNOTATE_JS). Its code comes from
        // the panel when annotate mode turns on, so pages never annotated don't parse it. Until then
        // only the latest on/off and numbering are kept, to apply once it is installed.
        let annotation = null, wanted = null, numbers = null;
        const install = (code) => {
          if (annotation || typeof code !== "string" || code.length > 262144) return;
          try { annotation = compile("post", code)(post); } catch { return; }
          if (wanted?.on) annotation.start(wanted.color);
          if (numbers) annotation.number(numbers);
          wanted = numbers = null;
        };
        // The page as it is now, for printing: scripts' DOM, typed values and canvases included.
        // Printed from a copy in a separate frame, as this sandbox can't open the print dialog.
        const snapshot = () => {
          try {
            const source = document.documentElement;
            const copy = source.cloneNode(true);
            const twins = (selector) => [source.querySelectorAll(selector), copy.querySelectorAll(selector)];
            const [fields, fieldCopies] = twins("input, textarea, select");
            fields.forEach((field, index) => {
              const twin = fieldCopies[index];
              if (!twin) return;
              if (field.tagName === "TEXTAREA") twin.textContent = field.value;
              else if (field.tagName === "SELECT") [...twin.options].forEach((option, at) => option.toggleAttribute("selected", Boolean(field.options[at]?.selected)));
              else if (field.type === "checkbox" || field.type === "radio") twin.toggleAttribute("checked", field.checked);
              else if (field.type !== "password" && field.type !== "file") twin.setAttribute("value", field.value);
            });
            const [canvases, canvasCopies] = twins("canvas");
            canvases.forEach((canvas, index) => {
              const twin = canvasCopies[index];
              if (!twin) return;
              try {
                const image = document.createElement("img");
                for (const name of ["class", "style", "width", "height"]) if (twin.hasAttribute(name)) image.setAttribute(name, twin.getAttribute(name));
                image.src = canvas.toDataURL();
                twin.replaceWith(image);
              } catch {}
            });
            // Styles added through the CSSOM (CSS-in-JS) aren't in the markup.
            const [styles, styleCopies] = twins("style");
            styles.forEach((style, index) => {
              try {
                const rules = style.sheet ? [...style.sheet.cssRules].map((rule) => rule.cssText).join("\n") : null;
                if (rules && styleCopies[index]) styleCopies[index].textContent = rules;
              } catch {}
            });
            try {
              const adopted = (document.adoptedStyleSheets || []).flatMap((sheet) => [...sheet.cssRules].map((rule) => rule.cssText));
              if (adopted.length) {
                const extra = document.createElement("style");
                extra.textContent = adopted.join("\n");
                (copy.querySelector("head") || copy).appendChild(extra);
              }
            } catch {}
            for (const node of copy.querySelectorAll("script, [data-unsloth-annotate]")) node.remove();
            const html = "<!doctype html>" + copy.outerHTML;
            return html.length <= 8 * 1024 * 1024 ? html : null;
          } catch {
            return null;
          }
        };

        // Find in page for Studio's find bar: every visible match of the query, case-insensitive,
        // painted with the same two highlights as the chat's (all matches, then the active one) and
        // counted back to the bar. Highlights sit over the text, so the page itself is not changed.
        const finder = (() => {
          const MAX = 1000;
          const SKIP = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "SELECT", "OPTION", "TEXTAREA"]);
          const painted = typeof Highlight === "function" && typeof CSS !== "undefined" && CSS.highlights;
          let ranges = [];
          let active = -1;
          let styled = false;
          const style = () => {
            if (styled || !painted) return;
            styled = true;
            const sheet = document.createElement("style");
            sheet.textContent = "::highlight(unsloth-find){background-color:#ffd84d;color:#1a1a1a}::highlight(unsloth-find-active){background-color:#ff8c1a;color:#1a1a1a}";
            (document.head || document.documentElement).appendChild(sheet);
          };
          const shown = (element) => {
            if (typeof element.checkVisibility === "function") return element.checkVisibility({ visibilityProperty: true });
            return element.getClientRects().length > 0;
          };
          const collect = (query) => {
            const needle = query.toLowerCase();
            const found = [];
            if (!document.body) return found;
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
              acceptNode: (node) => {
                const element = node.parentElement;
                if (!element || SKIP.has(element.tagName) || element.closest("[data-unsloth-annotate]")) return NodeFilter.FILTER_REJECT;
                return NodeFilter.FILTER_ACCEPT;
              },
            });
            for (let node = walker.nextNode(); node && found.length < MAX; node = walker.nextNode()) {
              const text = node.nodeValue || "";
              const lower = text.toLowerCase();
              // Lowercasing can change a few characters' lengths; those nodes would map wrongly.
              if (lower.length !== text.length) continue;
              let at = lower.indexOf(needle);
              if (at < 0 || !shown(node.parentElement)) continue;
              while (at >= 0 && found.length < MAX) {
                const range = document.createRange();
                range.setStart(node, at);
                range.setEnd(node, at + needle.length);
                found.push(range);
                at = lower.indexOf(needle, at + needle.length);
              }
            }
            return found;
          };
          const paint = () => {
            if (!painted) return;
            style();
            if (ranges.length) CSS.highlights.set("unsloth-find", new Highlight(...ranges));
            else CSS.highlights.delete("unsloth-find");
            if (ranges[active]) CSS.highlights.set("unsloth-find-active", new Highlight(ranges[active]));
            else CSS.highlights.delete("unsloth-find-active");
          };
          const reveal = () => {
            const range = ranges[active];
            if (!range) return;
            const rect = range.getBoundingClientRect();
            if (rect.top < 48 || rect.bottom > innerHeight - 24 || rect.left < 0 || rect.right > innerWidth) {
              range.startContainer.parentElement?.scrollIntoView({ block: "center", inline: "nearest" });
            }
            // Without highlights, the selection marks the match instead.
            if (!painted) {
              const selection = getSelection();
              selection?.removeAllRanges();
              selection?.addRange(range);
            }
          };
          const report = () => post({ type: "findResult", count: ranges.length, active });
          return {
            search: (query) => {
              ranges = query ? collect(query) : [];
              // From where the reader is: the first match at or below the top of the view.
              const below = ranges.findIndex((range) => range.getBoundingClientRect().bottom >= 0);
              active = ranges.length ? Math.max(below, 0) : -1;
              paint();
              reveal();
              report();
            },
            step: (delta) => {
              if (ranges.length) {
                active = (active + delta + ranges.length) % ranges.length;
                paint();
                reveal();
              }
              report();
            },
          };
        })();
        // Commands from the panel, relayed by the shell (the only parent this page has).
        window.addEventListener("message", (event) => {
          if (event.source !== parent) return;
          const data = event.data;
          if (!data || data.type !== "unsloth:browser-command") return;
          if (data.command === "annotateInstall") install(data.code);
          else if (data.command === "annotate") {
            if (!annotation) wanted = { on: data.on === true, color: data.color };
            else if (data.on === true) annotation.start(data.color);
            else annotation.stop();
          }
          else if (data.command === "annotateForget") annotation?.forget(Number(data.id));
          else if (data.command === "annotateNumbers") annotation ? annotation.number(data.numbers) : (numbers = data.numbers);
          else if (data.command === "zoom") applyZoom(data.value);
          else if (data.command === "mute") setMuted(data.on === true);
          else if (data.command === "find" && typeof data.query === "string" && data.query.length <= 1000) finder.search(data.query);
          else if (data.command === "findStep") finder.step(data.delta === -1 ? -1 : 1);
          else if (data.command === "snapshot") post({ type: "snapshot", html: snapshot() });
        });
        // The panel may have asked for annotate before this page could hear it.
        post({ type: "annotate", event: "ready" });
      };
      const escapeAttr = (value) => value.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
      const inject = (html, tags) => {
        const at = (match) => match.index + match[0].length;
        const head = /<head\b[^<>]*>/i.exec(html);
        if (head) return html.slice(0, at(head)) + tags + html.slice(at(head));
        const root = /<html\b[^<>]*>/i.exec(html) || /^\s*<!doctype[^>]*>/i.exec(html);
        if (root) return html.slice(0, at(root)) + "<head>" + tags + "</head>" + html.slice(at(root));
        return "<head>" + tags + "</head>" + html;
      };
      let page = null;
      window.addEventListener("message", (event) => {
        // Relay the page's messages; the panel only trusts this window.
        if (page && event.source === page.contentWindow) {
          if (event.data && event.data.source === "unsloth-browser") parent.postMessage(event.data, "*");
          return;
        }
        if (event.source !== parent) return;
        const data = event.data;
        if (page) {
          if (data && data.type === "unsloth:browser-command") page.contentWindow.postMessage(data, "*");
          return;
        }
        // Only the parent may drive the shell, once.
        if (!data || data.type !== "unsloth:browser-html" || typeof data.html !== "string") return;
        const cfg = { url: data.url || null, refresh: data.refresh || null, zoom: Number(data.zoom) || 1, muted: data.muted === true };
        const base = data.base ? `<base href="${escapeAttr(data.base)}">` : "";
        const script = `<script>(${boot.toString()})(${JSON.stringify(cfg).replace(/</g, "\\u003c")});<\/script>`;
        page = document.createElement("iframe");
        page.setAttribute("sandbox", "allow-scripts allow-forms");
        page.srcdoc = inject(data.html, base + script);
        document.body.appendChild(page);
        // A script navigation the page's hooks missed (no Navigation API) hits the lock below; the
        // report only names the origin, so the panel can say what happened but not follow it.
        let blocked = false;
        document.addEventListener("securitypolicyviolation", (event) => {
          if (blocked || event.effectiveDirective !== "frame-src") return;
          blocked = true;
          parent.postMessage({ source: "unsloth-browser", type: "scriptNavigation" }, "*");
        });
        const lock = document.createElement("meta");
        lock.httpEquiv = "Content-Security-Policy";
        lock.content = "frame-src 'none'";
        document.head.appendChild(lock);
      });
    </script>
  </body>
</html>"""

# Annotate, the body of a function of `post` that the page shell's `install` runs on the panel's
# request: while it is on, pointer input is the panel's, and the page only reports the block under
# the pointer, what a click or drag marks, and where the marks sit as it scrolls.
_ANNOTATE_JS = r"""
          // Clicks mark these whole; anything else marks the nearest element that holds text itself.
          const BLOCK = "p, li, h1, h2, h3, h4, h5, h6, pre, blockquote, td, th, dt, dd, figcaption, caption, img, picture, video, svg, button, label, a[href], input, textarea, select";
          // A list item marks its own line, not the lists nested under it.
          const NESTED = "ul, ol, table, pre, blockquote, div";
          const MEDIA = /^(img|picture|video|svg)$/i;
          const PAD = 5, DRAG = 4, MAX_QUOTE = 280, PIN = 28;
          // An element covering most of the view is a layout box, not something to ask about.
          const MAX_SHARE = 0.6;
          const marks = new Map();
          let on = false, press = null, hover = null, nextId = 1, color = "#3b82f6";
          let frame = 0, pending = null, lastMove = null, overlay = null;
          const send = (event, data) => post({ type: "annotate", event, ...data });
          const hasOwnText = (element) =>
            [...element.childNodes].some((node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim());
          const tooBig = (element) => {
            const rect = element.getBoundingClientRect();
            return rect.width * rect.height > innerWidth * innerHeight * MAX_SHARE;
          };
          const rangeOf = (element) => {
            const range = document.createRange();
            if (MEDIA.test(element.tagName) || /^(input|textarea|select)$/i.test(element.tagName) || !element.textContent.trim()) {
              range.selectNode(element);
              return range;
            }
            range.selectNodeContents(element);
            if (element.tagName === "LI") {
              const nested = [...element.children].find((child) => child.matches(NESTED));
              if (nested) range.setEndBefore(nested);
            }
            return range;
          };
          const blockAt = (target) => {
            if (!(target instanceof Element) || target === overlay) return null;
            let element = target.closest(BLOCK);
            if (!element) {
              element = target;
              while (element && element !== document.body && !hasOwnText(element)) element = element.parentElement;
            }
            if (!element || element === document.body || element === document.documentElement || tooBig(element)) return null;
            return [rangeOf(element)];
          };
          const overlaps = (a, b) =>
            a.width > 0 && a.height > 0 && a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
          // Everything a dragged area touches, block by block, leaving out blocks around other hits.
          const blocksIn = (area) => {
            // The blocks, and the plain elements a page puts its text in directly (a snippet's div).
            const found = [...document.body.querySelectorAll("*")]
              .filter((element) => element !== overlay && (element.matches(BLOCK) || hasOwnText(element)))
              .filter((element) => overlaps(element.getBoundingClientRect(), area) && !tooBig(element))
              .map((element) => ({ element, range: rangeOf(element) }))
              .filter(({ range }) => [...range.getClientRects()].some((rect) => overlaps(rect, area)));
            // The outermost hit wins, so a snippet keeps its bold words; but an element mostly
            // outside the area gives way to the hits inside it.
            const mostlyInside = (element) => {
              const rect = element.getBoundingClientRect();
              const width = Math.max(0, Math.min(rect.right, area.right) - Math.max(rect.left, area.left));
              const height = Math.max(0, Math.min(rect.bottom, area.bottom) - Math.max(rect.top, area.top));
              return width * height >= rect.width * rect.height * 0.5;
            };
            const kept = found.filter(({ element }) =>
              mostlyInside(element) || !found.some((other) => other.element !== element && element.contains(other.element)));
            const hits = kept.filter(({ element }) => !kept.some((other) => other.element !== element && other.element.contains(element)));
            return hits.map(({ range }) => range);
          };
          // What a blank area is pinned to: a bar fixed on screen it sits on, else the content of
          // whatever scrolls under it, so the area keeps to its place on the page as it scrolls.
          const anchorOf = (area) => {
            const at = document.elementFromPoint(area.left + area.width / 2, area.top + area.height / 2);
            let element = at instanceof Element ? at : null;
            for (; element && element !== document.body && element !== document.documentElement; element = element.parentElement) {
              const style = getComputedStyle(element);
              // A layer over the whole view is no bar; the page scrolls under it.
              if (style.position === "fixed") return tooBig(element) ? document.documentElement : element;
              const scrolls = /auto|scroll/.test(style.overflowX + style.overflowY) &&
                (element.scrollHeight > element.clientHeight || element.scrollWidth > element.clientWidth);
              if (scrolls) return element.firstElementChild || element;
            }
            return document.documentElement;
          };
          // Where a mark is now: around what it marks, or its blank area, moved with its anchor.
          const markBox = (mark) => {
            if (mark.ranges) return boxOf(mark.ranges);
            const { element, dx, dy, width, height } = mark.anchor;
            if (!element.isConnected) return null;
            const rect = element.getBoundingClientRect();
            return { left: rect.left + dx, top: rect.top + dy, width, height };
          };
          const boxOf = (ranges) => {
            let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
            for (const range of ranges) {
              for (const rect of range.getClientRects()) {
                if (!rect.width && !rect.height) continue;
                left = Math.min(left, rect.left); top = Math.min(top, rect.top);
                right = Math.max(right, rect.right); bottom = Math.max(bottom, rect.bottom);
              }
            }
            if (!Number.isFinite(left)) return null;
            return { left: left - PAD, top: top - PAD, width: right - left + PAD * 2, height: bottom - top + PAD * 2 };
          };
          const quoteOf = (ranges) => {
            const text = ranges.map((range) => {
              const node = range.startContainer.childNodes?.[range.startOffset];
              // A field says what it holds, or what it is for; never a password, which the chat would keep.
              if (node instanceof Element && /^(input|textarea|select)$/i.test(node.tagName)) {
                const value = node.type === "password" ? "" : node.value;
                return value || node.getAttribute("placeholder") || node.getAttribute("aria-label") || "";
              }
              return range.toString();
            }).join(" ").replace(/\s+/g, " ").trim();
            return text.length > MAX_QUOTE ? text.slice(0, MAX_QUOTE - 1) + "…" : text;
          };
          // The first picture marked, by its alt text; null when none is.
          const pictureOf = (ranges) => {
            for (const range of ranges) {
              const picture = range.cloneContents().querySelector("img, picture, video, svg");
              if (picture) return (picture.getAttribute("alt") || picture.getAttribute("aria-label") || "").slice(0, 300);
            }
            return null;
          };
          const same = (a, b) =>
            a === b || (a && b && a.length === b.length && a.every((range, index) =>
              range.startContainer === b[index].startContainer && range.startOffset === b[index].startOffset &&
              range.endContainer === b[index].endContainer && range.endOffset === b[index].endOffset));

          // Drawn here rather than in the panel, so outlines follow the pointer and the page with no
          // round trip. A closed shadow root keeps the page's styles off them.
          let root = null, outline = null, area = null;
          const zoomOf = () => parseFloat(document.documentElement.style.zoom) || 1;
          const place = (element, box, display = "block") => {
            if (!box) { element.style.display = "none"; return; }
            Object.assign(element.style, {
              display, left: box.left + "px", top: box.top + "px", width: box.width + "px", height: box.height + "px",
            });
          };
          const build = () => {
            overlay = document.createElement("unsloth-annotate");
            overlay.style.cssText = "all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483647;";
            root = overlay.attachShadow({ mode: "closed" });
            root.innerHTML = `<style>
              :host { --c: ${color}; }
              div { position: fixed; box-sizing: border-box; border-radius: 6px; display: none; }
              .hover { border: 1.5px solid color-mix(in srgb, var(--c) 75%, transparent); transition: left .08s, top .08s, width .08s, height .08s; }
              .area, .mark { border: 1.5px dashed color-mix(in srgb, var(--c) 70%, transparent); background: color-mix(in srgb, var(--c) 6%, transparent); }
              .pin { width: ${PIN}px; height: ${PIN}px; border-radius: 50% 50% 50% 4px; background: var(--c); box-shadow: 0 0 0 2px #fff, 0 1px 4px rgb(0 0 0 / .3);
                align-items: center; justify-content: center; color: #fff; font: 600 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
            </style><div class="hover"></div><div class="area"></div>`;
            outline = root.querySelector(".hover");
            area = root.querySelector(".area");
          };
          const pinBox = (box) => ({ left: box.left + box.width - 14, top: box.top - 18, width: PIN, height: PIN });
          const draw = () => {
            frame = 0;
            if (!on) return;
            const zoom = zoomOf();
            overlay.style.zoom = zoom === 1 ? "" : String(1 / zoom);
            place(outline, hover ? boxOf(hover) : null);
            const rects = [];
            for (const [id, mark] of marks) {
              const box = markBox(mark);
              place(mark.box, box);
              place(mark.pin, box && pinBox(box), "flex");
              mark.at = box;
              rects.push([id, box]);
            }
            send("rects", { rects });
          };
          const redraw = () => { if (on && !frame) frame = requestAnimationFrame(draw); };
          const setHover = (next) => {
            if (same(next, hover)) return;
            hover = next;
            place(outline, next ? boxOf(next) : null);
          };
          const createMark = (target, details) => {
            const id = nextId++;
            const box = document.createElement("div");
            box.className = "mark";
            const pin = document.createElement("div");
            pin.className = "pin";
            root.append(box, pin);
            const mark = { ...target, box, pin, at: null };
            mark.at = markBox(mark);
            place(box, mark.at);
            place(pin, mark.at && pinBox(mark.at), "flex");
            marks.set(id, mark);
            send("mark", { id, rect: mark.at, ...details });
          };
          const addMark = (ranges) => {
            if (!ranges || !ranges.length) return;
            const quote = quoteOf(ranges);
            const picture = pictureOf(ranges);
            if (!quote && picture === null) return;
            createMark({ ranges }, { quote, image: picture !== null, alt: picture || "", area: false });
          };
          // An area with nothing in it marks that part of the page itself.
          const addArea = (area) => {
            const element = anchorOf(area);
            const rect = element.getBoundingClientRect();
            const anchor = { element, dx: area.left - rect.left, dy: area.top - rect.top, width: area.width, height: area.height };
            createMark({ anchor }, { quote: "", image: false, alt: "", area: true });
          };
          // The panel numbers the marks, matching the order they take in the chat.
          const number = (numbers) => {
            if (!Array.isArray(numbers)) return;
            for (const entry of numbers) {
              const mark = Array.isArray(entry) ? marks.get(Number(entry[0])) : null;
              const value = Array.isArray(entry) ? Number(entry[1]) : NaN;
              if (mark && Number.isInteger(value) && value > 0 && value < 1000) mark.pin.textContent = String(value);
            }
          };
          const forget = (id) => {
            const mark = marks.get(id);
            if (!mark) return;
            mark.box.remove();
            mark.pin.remove();
            marks.delete(id);
          };
          // A press on a mark's pin reopens its comment.
          const pinAt = (x, y) => {
            for (const [id, mark] of marks) {
              const pin = mark.at && pinBox(mark.at);
              if (pin && x >= pin.left && x <= pin.left + pin.width && y >= pin.top && y <= pin.top + pin.height) return id;
            }
            return null;
          };
          const areaFrom = (event) => ({
            left: Math.min(press.x, event.clientX), top: Math.min(press.y, event.clientY),
            width: Math.abs(event.clientX - press.x), height: Math.abs(event.clientY - press.y),
          });
          const swallow = (event) => { event.preventDefault(); event.stopPropagation(); };
          const handleMove = (event) => {
            if (press) {
              if (!press.dragging && Math.hypot(event.clientX - press.x, event.clientY - press.y) < DRAG) return;
              press.dragging = true;
              setHover(null);
              place(area, areaFrom(event));
              return;
            }
            setHover(pinAt(event.clientX, event.clientY) === null ? blockAt(event.target) : null);
          };
          const onMove = (event) => {
            event.stopPropagation();
            lastMove = event;
            if (pending) return;
            pending = requestAnimationFrame(() => {
              pending = null;
              if (lastMove && on) handleMove(lastMove);
            });
          };
          const onDown = (event) => {
            swallow(event);
            if (event.button !== 0) return;
            press = { x: event.clientX, y: event.clientY, dragging: false, pin: pinAt(event.clientX, event.clientY) };
            // Keep the drag ours when it leaves the page.
            try { document.documentElement.setPointerCapture(event.pointerId); } catch {}
          };
          const onUp = (event) => {
            swallow(event);
            if (!press) return;
            const { dragging, pin } = press;
            const box = dragging ? areaFrom(event) : null;
            press = null;
            place(area, null);
            try { document.documentElement.releasePointerCapture(event.pointerId); } catch {}
            // Before the mark, so the panel keeps the comment it had open.
            send("up");
            if (box) {
              const ranges = blocksIn(new DOMRect(box.left, box.top, box.width, box.height));
              ranges.length ? addMark(ranges) : addArea(box);
            }
            else if (pin !== null) send("open", { id: pin });
            else addMark(blockAt(document.elementFromPoint(event.clientX, event.clientY) || event.target));
          };
          const onLeave = (event) => { if (!event.relatedTarget && !press) setHover(null); };
          const onKey = (event) => {
            if (event.key !== "Escape") return;
            swallow(event);
            send("escape");
          };
          const style = document.createElement("style");
          const cursor = () => {
            const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="30" height="30"><path d="M3 27V15A12 12 0 1 1 15 27Z" fill="${color}" stroke="white" stroke-width="2"/></svg>`;
            return `url("data:image/svg+xml,${encodeURIComponent(svg)}") 3 27, crosshair`;
          };
          const listeners = [
            ["pointerdown", onDown], ["pointermove", onMove], ["pointerup", onUp], ["click", swallow],
            ["dblclick", swallow], ["auxclick", swallow], ["mousedown", swallow], ["mouseup", swallow],
            ["contextmenu", swallow], ["dragstart", swallow], ["selectstart", swallow], ["keydown", onKey],
          ];
          const start = (accent) => {
            // Only a plain colour: it goes into a style and an SVG.
            if (typeof accent === "string" && /^(#[0-9a-f]{3,8}|rgba?\([\d.,\s%]+\))$/i.test(accent)) color = accent;
            if (on) return;
            on = true;
            build();
            style.textContent = `*, *::before, *::after { cursor: ${cursor()} !important; user-select: none !important; -webkit-user-select: none !important; }`;
            for (const [type, listener] of listeners) window.addEventListener(type, listener, true);
            document.addEventListener("mouseout", onLeave, true);
            window.addEventListener("scroll", redraw, true);
            window.addEventListener("resize", redraw);
            (document.head || document.documentElement).appendChild(style);
            document.documentElement.appendChild(overlay);
            redraw();
          };
          const stop = () => {
            if (!on) return;
            on = false;
            for (const [type, listener] of listeners) window.removeEventListener(type, listener, true);
            document.removeEventListener("mouseout", onLeave, true);
            window.removeEventListener("scroll", redraw, true);
            window.removeEventListener("resize", redraw);
            style.remove();
            overlay?.remove();
            cancelAnimationFrame(frame);
            if (pending) cancelAnimationFrame(pending);
            frame = 0;
            pending = press = hover = lastMove = overlay = root = null;
            marks.clear();
          };
          return { start, stop, forget, number };
"""


# Print shell: the panel posts it a copy of the page (`snapshot` above), which it shows without
# scripts and prints. Its own sandbox allows the print dialog, which the page's does not; its CSP
# runs only this script, so nothing in the copy can.
_PRINT_SCRIPT = r"""
(() => {
  let started = false;
  const tell = (type) => parent.postMessage({ source: "unsloth-print", type }, "*");
  addEventListener("message", (event) => {
    const data = event.data;
    if (started || event.source !== parent || !data || data.type !== "unsloth:browser-print" || typeof data.html !== "string") return;
    started = true;
    const copy = new DOMParser().parseFromString(data.html, "text/html");
    for (const node of copy.querySelectorAll("script, iframe, frame, frameset, object, embed, meta[http-equiv]")) node.remove();
    for (const node of copy.querySelectorAll("*")) {
      for (const attribute of [...node.attributes]) if (/^on/i.test(attribute.name)) node.removeAttribute(attribute.name);
    }
    // Lazy images below the fold would print empty.
    for (const image of copy.querySelectorAll("img[loading]")) image.removeAttribute("loading");
    document.replaceChild(document.adoptNode(copy.documentElement), document.documentElement);
    const settled = (element) => new Promise((resolve) => {
      element.addEventListener("load", resolve, { once: true });
      element.addEventListener("error", resolve, { once: true });
    });
    const waits = [
      ...[...document.querySelectorAll('link[rel~="stylesheet"]')].filter((link) => !link.sheet).map(settled),
      ...[...document.images].filter((image) => !image.complete).map(settled),
      document.fonts ? document.fonts.ready : null,
    ];
    Promise.race([Promise.all(waits), new Promise((resolve) => setTimeout(resolve, 5000))]).then(() => {
      setTimeout(() => {
        try {
          print();
        } finally {
          tell("printed");
        }
      }, 50);
    });
  });
  tell("ready");
})();
"""
_PRINT_HTML = (
    '<!doctype html><html><head><meta charset="utf-8"></head><body><script>'
    + _PRINT_SCRIPT
    + "</script></body></html>"
)
_PRINT_CSP = (
    "default-src 'none'; "
    f"script-src 'sha256-{base64.b64encode(hashlib.sha256(_PRINT_SCRIPT.encode()).digest()).decode()}'; "
    "style-src 'unsafe-inline' https: data: blob:; "
    "img-src https: data: blob:; "
    "font-src https: data: blob:; "
    "media-src https: data: blob:; "
    "base-uri http: https:; "
    "form-action 'none'; "
    f"frame-ancestors {_FRAME_ANCESTORS}; "
    "upgrade-insecure-requests; "
    "sandbox allow-scripts allow-modals; "
    "treat-as-public-address"
)


class BrowserFetchRequest(BaseModel):
    url: str = Field(..., min_length = 1, max_length = 8192)
    method: Literal["GET", "POST"] = "GET"
    body: Optional[str] = Field(default = None, max_length = 1024 * 1024)
    # favicon callers can lower the 50 MB fetch cap.
    max_bytes: Optional[int] = Field(default = None, ge = 1, le = _MAX_BROWSER_FETCH_BYTES)


_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)


# headers may name UTF-16; meta labels map it to UTF-8 through the shared table.
_HEADER_CODECS = {
    **_WHATWG_CHARSET_CODECS,
    **dict.fromkeys(
        "csunicode iso-10646-ucs-2 ucs-2 unicode unicodefeff utf-16 utf-16le".split(), "utf-16"
    ),
    "unicodefffe": "utf-16-be",
    "utf-16be": "utf-16-be",
    "latin-1": "cp1252",
}


def _codec(label: Optional[str], table: dict = _HEADER_CODECS) -> Optional[str]:
    return table.get(label.strip().lower()) if label else None


def _raw_codec(label: Optional[str]) -> Optional[str]:
    codec = _codec(label)
    if codec or not label:
        return codec
    try:
        return codecs.lookup(label).name
    except (LookupError, ValueError):
        return None


def _decode_html(raw: bytes, charset: Optional[str]) -> str:
    # byte order marks override declared charsets, matching browsers.
    for bom, codec in _BOMS:
        if raw.startswith(bom):
            return raw.decode(codec, errors = "replace")
    labelled = _codec(charset)
    if labelled is None:
        labelled = _sniff_meta_charset(raw[:4096], "text/html")
    if labelled:
        return raw.decode(labelled, errors = "replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    # browsers use U+FFFD for bad bytes in labelled pages and windows-1252 for unlabelled pages.
    return raw.decode("cp1252", errors = "replace")


def _attr(match: "re.Match[str] | None") -> Optional[str]:
    if not match:
        return None
    return _html.unescape(next(group for group in match.groups() if group is not None))


def _join(base: str, href: str) -> Optional[str]:
    """return None when ``urljoin`` cannot parse an address such as ``http://[bad``."""
    try:
        return urljoin(base, href.strip())
    except ValueError:
        return None


def _prepare_page(page: str, url: str) -> tuple[str, str, Optional[dict]]:
    """Return ``(page, base_url, refresh)`` with <base>, CSP and refresh meta tags removed."""
    base_url = url
    for tag in _BASE_TAG_RE.findall(page):
        href = _attr(_ATTR_HREF_RE.search(tag))
        if href:
            base_url = _join(url, href) or url
            break
    page = _BASE_TAG_RE.sub("", page)

    refresh = None

    def strip_meta(match: "re.Match[str]") -> str:
        nonlocal refresh
        tag = match.group(0)
        equiv = _HTTP_EQUIV_RE.search(tag)
        if not equiv:
            return tag
        name = equiv.group(1).lower()
        # The page's CSP doesn't fit the sandbox; a refresh would navigate the frame.
        if name == "content-security-policy":
            return ""
        if name == "refresh":
            parsed = _REFRESH_RE.match(_attr(_CONTENT_ATTR_RE.search(tag)) or "")
            if parsed and parsed.group(2) and refresh is None:
                delay = float(parsed.group(1))
                target = _join(base_url, parsed.group(2))
                if delay <= _MAX_REFRESH_DELAY_S and target:
                    refresh = {"delay": delay, "url": target}
            return ""
        return tag

    return _META_TAG_RE.sub(strip_meta, page), base_url, refresh


# unsloth.ai's Cloudflare skips its bot challenge for requests carrying this. Only its own hosts get
# it: the challenge can't be solved from the proxied frame, and other sites have no use for it.
_STUDIO_HEADERS = {"X-Unsloth-Studio": "1"}


def _studio_headers(host: str) -> dict:
    host = host.lower().rstrip(".")
    return _STUDIO_HEADERS if host == "unsloth.ai" or host.endswith(".unsloth.ai") else {}


def _integrity_ok(body: bytes, integrity: str) -> bool:
    """Whether ``body`` matches the tag's SRI ``integrity``; inlining drops the attribute, so check here."""
    hashes: dict[str, list[str]] = {}
    for token in integrity.split():
        algorithm, _, value = token.partition("-")
        algorithm = algorithm.lower()
        if algorithm in _SRI_ALGORITHMS and value:
            hashes.setdefault(algorithm, []).append(value.split("?", 1)[0])
    if not hashes:
        # No metadata the browser understands: it loads the script unchecked.
        return True
    algorithm = max(hashes, key = _SRI_ALGORITHMS.index)
    digest = base64.b64encode(hashlib.new(algorithm, body).digest()).decode("ascii")
    return any(hmac.compare_digest(digest, value) for value in hashes[algorithm])


def _fresh_for(cache_control: Optional[str], age: Optional[str]) -> float:
    """Seconds a response may be reused by this shared cache; ``private``/``no-cache`` rule it out."""
    directives = (cache_control or "").lower()
    if any(word in directives for word in ("no-store", "no-cache", "private")):
        return 0
    lifetimes = dict((name.lower(), int(value)) for name, value in _MAX_AGE_RE.findall(directives))
    lifetime = lifetimes.get("s-maxage", lifetimes.get("max-age", 0))
    try:
        lifetime -= max(int(age or 0), 0)
    except ValueError:
        pass
    return float(min(max(lifetime, 0), _MODULE_CACHE_TTL_S))


def _module_cached(key: tuple[str, str, bool]) -> tuple[bool, Optional[str]]:
    with _MODULE_CACHE_LOCK:
        hit = _MODULE_CACHE.get(key)
        if hit is None or time.monotonic() >= hit[0]:
            return False, None
        _MODULE_CACHE.move_to_end(key)
        return True, hit[1]


def _cache_module(key: tuple[str, str, bool], code: Optional[str], fresh_for: float) -> None:
    global _module_cache_chars
    size = len(code or "")
    if fresh_for <= 0 or size > _MODULE_CACHE_CHARS // 4:
        return
    with _MODULE_CACHE_LOCK:
        old = _MODULE_CACHE.pop(key, None)
        if old is not None:
            _module_cache_chars -= len(old[1] or "")
        _MODULE_CACHE[key] = (time.monotonic() + fresh_for, code)
        _module_cache_chars += size
        while (
            len(_MODULE_CACHE) > _MODULE_CACHE_ENTRIES or _module_cache_chars > _MODULE_CACHE_CHARS
        ):
            _, (_, dropped) = _MODULE_CACHE.popitem(last = False)
            _module_cache_chars -= len(dropped or "")


def _fetch_module(
    url: str,
    integrity: str,
    credentials: bool,
    deadline: float,
    cancel_event: Optional[threading.Event],
) -> Optional[str]:
    """A self-contained module's code, safe to inline; None to leave its tag alone."""
    key = (url, integrity, credentials)
    found, code = _module_cached(key)
    if found:
        return code
    meta: dict = {}
    try:
        error, body, content_type = _fetch_url_raw(
            url,
            timeout = _MODULE_TIMEOUT_S,
            extra_headers = {"User-Agent": _BROWSER_UA, "Accept": "*/*"},
            deadline = deadline,
            raw_bytes_max = _MAX_MODULE_BYTES,
            meta_out = meta,
            cancel_event = cancel_event,
            host_headers = _studio_headers,
        )
    except Exception as exc:
        logger.warning("browser_module_fetch_failed", error = type(exc).__name__)
        return None
    if error is not None or not isinstance(body, bytes):
        return None
    # Redirected to http: tamperable, and the frame's upgrade-insecure-requests wouldn't run it either.
    if not str(meta.get("url") or url).lower().startswith("https://"):
        return None
    code = None
    allow_origin = (meta.get("allow_origin") or "").strip()
    # ACAO * or null already loads from the sandbox (where SRI is enforced too).
    loads_itself = allow_origin in ("*", "null") and not credentials
    if content_type in _JS_TYPES and not loads_itself and _integrity_ok(body, integrity):
        # Browsers decode module scripts as UTF-8 whatever the header says.
        text = body.decode("utf-8", errors = "replace")
        # Imports would resolve against the page; "<!--" then "<script" keeps an inline tag open.
        if not _MODULE_IMPORT_RE.search(text) and not (
            "<!--" in text and _SCRIPT_OPEN_RE.search(text)
        ):
            code = re.sub(r"</(script)", r"<\\/\1", text, flags = re.IGNORECASE)
    # Only the final hop's headers are known, so a redirected answer isn't cached.
    redirected = str(meta.get("url") or url) != url
    fresh_for = 0 if redirected else _fresh_for(meta.get("cache_control"), meta.get("age"))
    _cache_module(key, code, fresh_for)
    return code


def _inert_spans(page: str) -> list[tuple[int, int]]:
    """Spans of ``page`` that are text, not markup, in order, read front to back as the parser does."""
    spans: list[tuple[int, int]] = []
    at = 0
    while match := _INERT_START_RE.search(page, at):
        if match.group(0).startswith("<!--"):
            close = page.find("-->", match.end())
            end = len(page) if close < 0 else close + 3
            spans.append((match.start(), end))
        elif match.group(1) is None:
            end = match.end()
            spans.append((match.start(), end))
        elif match.group(1).lower() == "template":
            depth, close = 1, None
            for tag in _TEMPLATE_TAG_RE.finditer(page, match.end()):
                depth += -1 if tag.group(1) else 1
                if depth == 0:
                    close = tag
                    break
            end = close.end() if close else len(page)
            spans.append((match.end(), close.start() if close else len(page)))
        else:
            name = match.group(1).lower()
            close = (
                None if name == "plaintext" else _CLOSING_TAG_RES[name].search(page, match.end())
            )
            end = close.end() if close else len(page)
            spans.append((match.end(), close.start() if close else len(page)))
        at = max(end, match.end())
    return spans


def _open_tag(script: str) -> str:
    """A script element's start tag, from a match of _MODULE_SCRIPT_RE."""
    return script[: script.lower().rindex("</script")].rstrip()


def _script_attrs(open_tag: str) -> list[tuple[str, str, str]]:
    """A ``<script ...>`` start tag's attributes as ``(lowercase name, value, source text)``."""
    attrs = []
    for match in _TAG_ATTR_RE.finditer(open_tag, len("<script"), len(open_tag) - 1):
        if match.group(1) is None:
            continue
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attrs.append((match.group(1).lower(), _html.unescape(value), match.group(0)))
    return attrs


def _inline_module_scripts(
    page: str,
    base_url: str,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Inline the page's self-contained module scripts, which the sandbox can't load itself."""
    tags: list[tuple[re.Match[str], str, str, bool]] = []
    inert: Optional[list[tuple[int, int]]] = None
    for match in _MODULE_SCRIPT_RE.finditer(page):
        attrs: dict[str, str] = {}
        for name, value, _text in _script_attrs(_open_tag(match.group(0))):
            # The first of a repeated attribute is the one that counts.
            attrs.setdefault(name, value)
        if attrs.get("type", "").strip().lower() != "module":
            continue
        src = attrs.get("src")
        url = _join(base_url, src) if src else None
        if not url or not url.lower().startswith("https://"):
            continue
        # A tag inside text (comment, textarea, style...) stays; inlined code could end that element.
        if inert is None:
            inert = _inert_spans(page)
        at = bisect.bisect_right(inert, (match.start(), len(page))) - 1
        if at >= 0 and inert[at][0] <= match.start() < inert[at][1]:
            continue
        credentials = attrs.get("crossorigin", "").strip().lower() == "use-credentials"
        tags.append((match, url, attrs.get("integrity", ""), credentials))
        if len(tags) == _MAX_INLINED_MODULES:
            break
    if not tags:
        return page
    deadline = time.monotonic() + _MODULE_TIMEOUT_S
    codes = list(
        _MODULE_POOL.map(lambda tag: _fetch_module(*tag[1:], deadline, cancel_event), tags)
    )
    parts: list[str] = []
    end = 0
    room = _MAX_BROWSER_HTML_BYTES - len(page.encode("utf-8"))
    for (match, *_), code in zip(tags, codes):
        size = len(code.encode("utf-8")) if code is not None else 0
        if code is None or size > room:
            continue
        room -= size
        kept = (
            text
            for name, _value, text in _script_attrs(_open_tag(match.group(0)))
            if name not in _FETCH_ATTRS
        )
        open_tag = "".join(["<script", *(" " + text for text in kept), ">"])
        parts += [page[end : match.start()], open_tag, code, "</script>"]
        end = match.end()
    parts.append(page[end:])
    return "".join(parts)


def _fetch(
    request: BrowserFetchRequest, cancel_event: threading.Event
) -> tuple[Optional[str], bytes, str, dict]:
    meta: dict = {}
    error, body, content_type = _fetch_url_raw(
        request.url,
        timeout = _FETCH_TIMEOUT_S,
        extra_headers = {
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        deadline = time.monotonic() + _FETCH_TIMEOUT_S,
        raw_bytes_max = request.max_bytes or _MAX_BROWSER_FETCH_BYTES,
        post_data = (request.body or "").encode() if request.method == "POST" else None,
        meta_out = meta,
        cancel_event = cancel_event,
        host_headers = _studio_headers,
    )
    return error, body if isinstance(body, bytes) else b"", content_type, meta


def _attachment_name(meta: dict) -> Optional[str]:
    """The server's name for a download, as a bare file name."""
    name = meta.get("filename")
    if not isinstance(name, str):
        return None
    name = re.sub(r"[\x00-\x1f\x7f]", "", name.replace("\\", "/").rsplit("/", 1)[-1]).strip()[:255]
    return name if name.strip(".") else None


def _build_response(
    url: str,
    error: Optional[str],
    body: bytes,
    content_type: str,
    meta: dict,
    cancel_event: Optional[threading.Event] = None,
) -> Response:
    """Build the panel's response. Runs in the fetch pool to keep large pages off the event loop."""
    if error is not None:
        # The host only: a page address can carry a sign-in token.
        try:
            host = urlsplit(url).hostname
        except ValueError:
            host = None
        logger.info("browser_fetch_failed", host = host, error = error)
        if meta.get("bot_check"):
            raise HTTPException(status_code = 502, detail = {"message": error, "botCheck": True})
        raise HTTPException(status_code = 502, detail = error)

    final_url = meta.get("url") or url
    looks_html = not content_type and body[:512].lstrip().lower().startswith(
        (b"<!doctype html", b"<html")
    )
    if content_type in _HTML_TYPES or looks_html:
        if len(body) > _MAX_BROWSER_HTML_BYTES:
            raise HTTPException(
                status_code = 502,
                detail = f"(page exceeds the {_MAX_BROWSER_HTML_BYTES} byte limit for the panel)",
            )
        page, base_url, refresh = _prepare_page(_decode_html(body, meta.get("charset")), final_url)
        page = _inline_module_scripts(page, base_url, cancel_event)
        payload = {"url": final_url, "base": base_url, "refresh": refresh, "html": page}
        return Response(
            content = json.dumps(payload, ensure_ascii = False).encode("utf-8"),
            media_type = "application/json",
            headers = {KIND_HEADER: "html"},
        )
    codec = _raw_codec(meta.get("charset"))
    textual = content_type.startswith("text/") or content_type.endswith(
        ("json", "xml", "javascript")
    )
    if textual and codec and codec != "utf-8":
        # transcode text to UTF-8 to match the response label.
        try:
            body = body.decode(codec, errors = "replace").encode("utf-8")
        except (LookupError, UnicodeError, ValueError):
            pass
    return Response(
        content = body,
        media_type = content_type or "application/octet-stream",
        headers = {
            KIND_HEADER: "raw",
            URL_HEADER: quote(final_url, safe = ":/?#[]@!$&'()*+,;=%~"),
            # never rendered on Studio's origin
            "Content-Security-Policy": "sandbox",
            **({NAME_HEADER: quote(name, safe = "")} if (name := _attachment_name(meta)) else {}),
        },
    )


def _fetch_and_build(request: BrowserFetchRequest, cancel_event: threading.Event) -> Response:
    error, body, content_type, meta = _fetch(request, cancel_event)
    return _build_response(request.url, error, body, content_type, meta, cancel_event)


@router.post("/fetch")
async def browser_fetch(
    request: BrowserFetchRequest,
    http_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Fetch a page: HTML as JSON for the shell, anything else raw with its content type."""
    request.url = _normalize_url_scheme(request.url.strip())
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(_FETCH_POOL, _fetch_and_build, request, cancel_event)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout = _DISCONNECT_POLL_S)
            if done:
                return task.result()
            if await http_request.is_disconnected():
                cancel_event.set()
                raise HTTPException(status_code = 499, detail = "Client closed request")
    finally:
        cancel_event.set()


@router.get("/print", include_in_schema = False)
async def browser_print():
    """Print shell; static and unauthenticated like ``/frame``."""
    return Response(
        content = _PRINT_HTML,
        media_type = "text/html; charset=utf-8",
        headers = {
            "Cache-Control": "no-store",
            "Content-Security-Policy": _PRINT_CSP,
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/frame", include_in_schema = False)
async def browser_frame():
    """Sandbox shell. No auth (like the artifact frame): it is static and the page can see its URL."""
    return Response(
        content = _FRAME_HTML,
        media_type = "text/html; charset=utf-8",
        headers = {
            "Cache-Control": "no-store",
            "Content-Security-Policy": _FRAME_CSP,
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/annotate.js", include_in_schema = False)
async def browser_annotate_script():
    """The page shell's annotate code, sent into a page when annotate mode turns on; static like ``/frame``."""
    return Response(
        content = _ANNOTATE_JS,
        media_type = "text/javascript; charset=utf-8",
        headers = {
            "Cache-Control": "no-cache",
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )
