# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app browser for the chat panel: a page fetcher and the sandbox shell pages render in.

Most sites refuse framing, so pages are fetched here with the SSRF-guarded web tool fetcher and
written into ``/frame``, an opaque-origin sandbox with no access to Studio's storage or API. An
injected script turns navigations into messages, so they come back through this fetcher.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Optional
from urllib.parse import quote, urljoin

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from core.inference.tools import _USER_AGENTS, _fetch_url_raw, _normalize_url_scheme
from loggers import get_logger

# Same embedders as the canvas shell.
from routes.inference import _ARTIFACT_PREVIEW_FRAME_ANCESTORS as _FRAME_ANCESTORS

logger = get_logger(__name__)

router = APIRouter()

BROWSER_FRAME_PATH = "/api/browser/frame"
KIND_HEADER = "X-Unsloth-Browser-Kind"
URL_HEADER = "X-Unsloth-Browser-Url"
EXPOSED_HEADERS = (KIND_HEADER, URL_HEADER)

# Cap for documents (PDFs, images), which come back whole.
_MAX_BROWSER_FETCH_BYTES = 50 * 1024 * 1024
_FETCH_TIMEOUT_S = 25
_MAX_REFRESH_DELAY_S = 10
# Fixed UA so a site's layout doesn't change between requests.
_BROWSER_UA = _USER_AGENTS[1]
_HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})
# Own pool, so slow sites can't starve the default executor that chat inference uses.
_FETCH_POOL = ThreadPoolExecutor(max_workers = 4, thread_name_prefix = "browser-fetch")
_DISCONNECT_POLL_S = 0.25

_BASE_TAG_RE = re.compile(r"<base\b[^>]*>", re.IGNORECASE)
_ATTR_HREF_RE = re.compile(r"""\bhref\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_META_TAG_RE = re.compile(r"<meta\b[^>]*>", re.IGNORECASE)
_HTTP_EQUIV_RE = re.compile(r"""\bhttp-equiv\s*=\s*["']?([\w-]+)""", re.IGNORECASE)
_CONTENT_ATTR_RE = re.compile(r"""\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
_REFRESH_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:[;,]\s*(?:url\s*=\s*)?['\"]?([^'\"]*)['\"]?)?", re.IGNORECASE
)
_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset\s*=\s*["']?([\w:.-]+)""", re.IGNORECASE)

# Pages load resources from anywhere; the sandbox (no allow-same-origin) isolates them from Studio.
# Forms are submitted by the injected script.
_FRAME_CSP = (
    "default-src http: https: data: blob:; "
    "script-src 'unsafe-inline' 'unsafe-eval' http: https: data: blob:; "
    "style-src 'unsafe-inline' http: https: data: blob:; "
    "img-src http: https: data: blob:; "
    "font-src http: https: data: blob:; "
    "media-src http: https: data: blob:; "
    # No http: or ws:, so scripts can't hit local services (WebKit has no local network protection).
    "connect-src https: wss: data: blob:; "
    "worker-src http: https: blob:; "
    "frame-src http: https: data: blob:; "
    "object-src 'none'; "
    "base-uri http: https:; "
    "form-action 'none'; "
    f"frame-ancestors {_FRAME_ANCESTORS}; "
    "sandbox allow-scripts allow-forms; "
    # The shell is served from loopback; don't let that exempt pages from local network checks.
    "treat-as-public-address"
)

# Shell a page is written into. It injects <base> and a script that turns navigations into messages
# to the panel, since a real navigation would leave the proxy.
_FRAME_HTML = r"""<!doctype html>
<html>
  <head><meta charset="utf-8" /></head>
  <body>
    <script>
      const boot = (cfg) => {
        const shellOrigin = location.origin;
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
        const navigate = (raw, newTab, background) => {
          const url = resolve(raw);
          if (!url) return;
          if (!/^https?:/i.test(url)) { post({ type: "external", url }); return; }
          if (!newTab && samePage(url) && new URL(url).hash) { scrollToHash(new URL(url).hash); return; }
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
        for (const [method, original] of [["pushState", originalPush], ["replaceState", originalReplace]]) {
          history[method] = function (state, title, url) {
            let next = url != null ? resolve(String(url)) : null;
            // A router re-asserting the shell's own path is not a navigation.
            if (next && new URL(next).pathname === shellPathname) next = null;
            try { original.call(this, state, title, next ? shellPath(next) : undefined); }
            catch { original.call(this, state, title); }
            if (next) {
              pageUrl = next;
              post({ type: "url", url: next });
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
        const onLinkClick = (event) => {
          const link = linkFrom(event);
          if (!link) return;
          const raw = (link.getAttribute("href") || "").trim();
          if (/^javascript:/i.test(raw)) return;
          event.preventDefault();
          // Keep "#" links on the page; against <base> they would navigate.
          if (raw.startsWith("#")) {
            if (raw.length > 1) scrollToHash(raw);
            return;
          }
          const modified = event.button === 1 || event.metaKey || event.ctrlKey || event.shiftKey;
          navigate(raw, modified || /^_blank$/i.test(link.target || ""), modified);
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
        document.addEventListener("submit", (event) => {
          event.preventDefault();
          submitForm(event.target, event.submitter);
        }, true);
        HTMLFormElement.prototype.submit = function () { submitForm(this, null); };
        HTMLFormElement.prototype.requestSubmit = function (submitter) {
          const event = new SubmitEvent("submit", { bubbles: true, cancelable: true, submitter: submitter || null });
          if (this.dispatchEvent(event)) submitForm(this, submitter || null);
        };

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
          if ((event.metaKey || event.ctrlKey) && ["l", "t", "w", "r"].includes(event.key.toLowerCase())) {
            event.preventDefault();
            post({ type: "shortcut", key: event.key.toLowerCase(), shift: event.shiftKey });
          }
        }, true);
      };
      const escapeAttr = (value) => value.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
      const inject = (html, tags) => {
        const at = (match) => match.index + match[0].length;
        const head = /<head\b[^>]*>/i.exec(html);
        if (head) return html.slice(0, at(head)) + tags + html.slice(at(head));
        const root = /<html\b[^>]*>/i.exec(html) || /^\s*<!doctype[^>]*>/i.exec(html);
        if (root) return html.slice(0, at(root)) + "<head>" + tags + "</head>" + html.slice(at(root));
        return "<head>" + tags + "</head>" + html;
      };
      window.addEventListener("message", (event) => {
        // Only the parent may drive the shell.
        if (event.source !== parent) return;
        const data = event.data;
        if (!data || data.type !== "unsloth:browser-html" || typeof data.html !== "string") return;
        const cfg = { url: data.url || null, refresh: data.refresh || null };
        const base = data.base ? `<base href="${escapeAttr(data.base)}">` : "";
        const script = `<script>(${boot.toString()})(${JSON.stringify(cfg).replace(/</g, "\\u003c")});<\/script>`;
        document.open();
        document.write(inject(data.html, base + script));
        document.close();
      });
    </script>
  </body>
</html>"""


class BrowserFetchRequest(BaseModel):
    url: str = Field(..., min_length = 1, max_length = 8192)
    method: Literal["GET", "POST"] = "GET"
    # Urlencoded form body from the injected script.
    body: Optional[str] = Field(default = None, max_length = 1024 * 1024)


def _decode_html(raw: bytes, charset: Optional[str]) -> str:
    candidates = [charset] if charset else []
    sniffed = _META_CHARSET_RE.search(raw[:4096])
    if sniffed:
        candidates.append(sniffed.group(1).decode("ascii", "ignore"))
    candidates.append("utf-8")
    for candidate in candidates:
        try:
            return raw.decode(candidate)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors = "replace")


def _attr(match: "re.Match[str] | None") -> Optional[str]:
    if not match:
        return None
    return _html.unescape(next(group for group in match.groups() if group is not None))


def _prepare_page(page: str, url: str) -> tuple[str, str, Optional[dict]]:
    """Return ``(page, base_url, refresh)`` with <base>, CSP and refresh meta tags removed."""
    base_url = url
    for tag in _BASE_TAG_RE.findall(page):
        href = _attr(_ATTR_HREF_RE.search(tag))
        if href:
            base_url = urljoin(url, href.strip())
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
                if delay <= _MAX_REFRESH_DELAY_S:
                    refresh = {"delay": delay, "url": urljoin(base_url, parsed.group(2).strip())}
            return ""
        return tag

    return _META_TAG_RE.sub(strip_meta, page), base_url, refresh


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
        raw_bytes_max = _MAX_BROWSER_FETCH_BYTES,
        post_data = (request.body or "").encode() if request.method == "POST" else None,
        meta_out = meta,
        cancel_event = cancel_event,
    )
    return error, body if isinstance(body, bytes) else b"", content_type, meta


def _build_response(
    url: str, error: Optional[str], body: bytes, content_type: str, meta: dict
) -> Response:
    """Build the panel's response. Runs in the fetch pool to keep large pages off the event loop."""
    if error is not None:
        logger.info("browser_fetch_failed", url = url, error = error)
        raise HTTPException(status_code = 502, detail = error)

    final_url = meta.get("url") or url
    looks_html = not content_type and body[:512].lstrip().lower().startswith(
        (b"<!doctype html", b"<html")
    )
    if content_type in _HTML_TYPES or looks_html:
        page, base_url, refresh = _prepare_page(_decode_html(body, meta.get("charset")), final_url)
        payload = {"url": final_url, "base": base_url, "refresh": refresh, "html": page}
        return Response(
            content = json.dumps(payload, ensure_ascii = False).encode("utf-8"),
            media_type = "application/json",
            headers = {KIND_HEADER: "html"},
        )
    return Response(
        content = body,
        media_type = content_type or "application/octet-stream",
        headers = {
            KIND_HEADER: "raw",
            URL_HEADER: quote(final_url, safe = ":/?#[]@!$&'()*+,;=%~"),
            # Never rendered on Studio's origin.
            "Content-Security-Policy": "sandbox",
        },
    )


def _fetch_and_build(request: BrowserFetchRequest, cancel_event: threading.Event) -> Response:
    error, body, content_type, meta = _fetch(request, cancel_event)
    return _build_response(request.url, error, body, content_type, meta)


@router.post("/fetch")
async def browser_fetch(
    request: BrowserFetchRequest,
    http_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Fetch a page for the browser panel. HTML returns as JSON for the sandbox shell; anything
    else (PDF, images, text) returns raw with its content type."""
    request.url = _normalize_url_scheme(request.url.strip())
    cancel_event = threading.Event()
    loop = asyncio.get_running_loop()
    task = loop.run_in_executor(_FETCH_POOL, _fetch_and_build, request, cancel_event)
    try:
        # Stop the fetch when the panel aborts the load.
        while True:
            done, _ = await asyncio.wait({task}, timeout = _DISCONNECT_POLL_S)
            if done:
                return task.result()
            if await http_request.is_disconnected():
                cancel_event.set()
                raise HTTPException(status_code = 499, detail = "Client closed request")
    finally:
        cancel_event.set()


@router.get("/frame", include_in_schema = False)
async def browser_frame():
    """Sandbox shell for pages. No auth, like the artifact frame: it is static, and the hosted page
    can read its URL."""
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
