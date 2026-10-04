# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-app browser: pages are fetched here (SSRF-guarded, since most sites refuse framing) and
rendered in ``/frame``, an opaque-origin sandbox; an injected script routes navigations back here.
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
NAME_HEADER = "X-Unsloth-Browser-Filename"
EXPOSED_HEADERS = (KIND_HEADER, URL_HEADER, NAME_HEADER)

_MAX_BROWSER_FETCH_BYTES = 50 * 1024 * 1024
# HTML goes into srcdoc and the page cache; more would stall the renderer.
_MAX_BROWSER_HTML_BYTES = 10 * 1024 * 1024
_FETCH_TIMEOUT_S = 25
_MAX_REFRESH_DELAY_S = 10
# Fixed UA so a site's layout doesn't change between requests.
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
_CONTENT_ATTR_RE = re.compile(r"""\bcontent\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE)
_REFRESH_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(?:[;,]\s*(?:url\s*=\s*)?['\"]?([^'\"]*)['\"]?)?", re.IGNORECASE
)
_META_CHARSET_RE = re.compile(rb"""<meta[^<>]+charset\s*=\s*["']?([\w:.-]+)""", re.IGNORECASE)

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
        const onLinkClick = (event) => {
          const link = linkFrom(event);
          if (!link) return;
          const raw = (link.getAttribute("href") || "").trim();
          if (/^javascript:/i.test(raw)) return;
          event.preventDefault();
          // Keep "#" links on the page; against <base> they would navigate.
          if (raw.startsWith("#")) {
            if (raw.length > 1) followHash(raw);
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
          if ((event.metaKey || event.ctrlKey) && ["l", "t", "w", "r", "f"].includes(event.key.toLowerCase())) {
            event.preventDefault();
            post({ type: "shortcut", key: event.key.toLowerCase(), shift: event.shiftKey });
          }
        }, true);
        const applyZoom = (zoom) => {
          const value = Number(zoom);
          if (!(value >= 0.25 && value <= 5)) return;
          document.documentElement.style.zoom = value === 1 ? "" : String(value);
        };
        if (cfg.zoom) {
          if (document.documentElement) applyZoom(cfg.zoom);
          document.addEventListener("DOMContentLoaded", () => applyZoom(cfg.zoom), { once: true });
        }
        // Annotate: the panel marks parts of the page to ask about. While it is on, pointer input
        // is the panel's: the page only reports the block under the pointer, what a click or drag
        // marks, and where the marks sit as it scrolls.
        const annotation = (() => {
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
            if (hits.length) return hits.map(({ range }) => range);
            return blockAt(document.elementFromPoint(area.left + area.width / 2, area.top + area.height / 2)) || [];
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
              // A field says what it holds, or what it is for.
              if (node instanceof Element && /^(input|textarea|select)$/i.test(node.tagName)) {
                return node.value || node.getAttribute("placeholder") || node.getAttribute("aria-label") || "";
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
              const box = boxOf(mark.ranges);
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
          const addMark = (ranges) => {
            if (!ranges || !ranges.length) return;
            const quote = quoteOf(ranges);
            const picture = pictureOf(ranges);
            if (!quote && picture === null) return;
            const id = nextId++;
            const box = document.createElement("div");
            box.className = "mark";
            const pin = document.createElement("div");
            pin.className = "pin";
            root.append(box, pin);
            const at = boxOf(ranges);
            place(box, at);
            place(pin, at && pinBox(at), "flex");
            marks.set(id, { ranges, box, pin, at });
            send("mark", { id, rect: at, quote, image: picture !== null, alt: picture || "" });
          };
          // The panel numbers the marks, as they will be read in the chat.
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
            if (box) addMark(blocksIn(new DOMRect(box.left, box.top, box.width, box.height)));
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
        })();
        // Commands from the panel, relayed by the shell (the only parent this page has).
        window.addEventListener("message", (event) => {
          if (event.source !== parent) return;
          const data = event.data;
          if (!data || data.type !== "unsloth:browser-command") return;
          if (data.command === "annotate") data.on === true ? annotation.start(data.color) : annotation.stop();
          else if (data.command === "annotateForget") annotation.forget(Number(data.id));
          else if (data.command === "annotateNumbers") annotation.number(data.numbers);
          else if (data.command === "zoom") applyZoom(data.value);
          else if (data.command === "find" && typeof data.query === "string" && data.query) {
            let found = false;
            try { found = window.find(data.query, false, Boolean(data.backwards), true, false, false, false); } catch {}
            post({ type: "found", found });
          }
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
        const cfg = { url: data.url || null, refresh: data.refresh || null, zoom: Number(data.zoom) || 1 };
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


class BrowserFetchRequest(BaseModel):
    url: str = Field(..., min_length = 1, max_length = 8192)
    method: Literal["GET", "POST"] = "GET"
    body: Optional[str] = Field(default = None, max_length = 1024 * 1024)
    # Smaller cap for favicons, so an icon can't be 50 MB.
    max_bytes: Optional[int] = Field(default = None, ge = 1, le = _MAX_BROWSER_FETCH_BYTES)


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
        raw_bytes_max = request.max_bytes or _MAX_BROWSER_FETCH_BYTES,
        post_data = (request.body or "").encode() if request.method == "POST" else None,
        meta_out = meta,
        cancel_event = cancel_event,
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
    url: str, error: Optional[str], body: bytes, content_type: str, meta: dict
) -> Response:
    """Build the panel's response. Runs in the fetch pool to keep large pages off the event loop."""
    if error is not None:
        logger.info("browser_fetch_failed", url = url, error = error)
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
            **({NAME_HEADER: quote(name, safe = "")} if (name := _attachment_name(meta)) else {}),
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
