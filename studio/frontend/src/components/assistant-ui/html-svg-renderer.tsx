// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import DOMPurify from "dompurify";
import { CodeIcon, EyeIcon, Maximize2Icon, Minimize2Icon } from "lucide-react";
import {
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

export type HtmlSvgLanguage = "html" | "svg";

export type HtmlSvgRendererProps = {
  language: HtmlSvgLanguage;
  source: string;
  // Pre-rendered syntax-highlighted code view. Optional; if omitted the
  // renderer falls back to a plain <pre><code> block.
  codeView?: ReactNode;
  // When the markdown stream is still arriving the fence may be partial.
  // In that case we force the Code tab and disable the toggle controls.
  isIncomplete?: boolean;
};

const DEFAULT_PREVIEW_HEIGHT = 500;
const POPOUT_HEIGHT_VH = 80;
const COPY_RESET_MS = 2000;

// Conservative regex covering the same vectors we previously stripped before
// DOMPurify was wired in. DOMPurify itself does the heavy lifting; this is
// belt-and-braces so a quick visual inspection of the source still catches the
// usual XSS hot spots.
const HEURISTIC_UNSAFE_SVG_RE =
  /<script[\s>]|<foreignObject[\s>]|<iframe[\s>]|<embed[\s>]|<object[\s>]/i;

// SVG previews used to live inside a `<img src="data:image/svg+xml,...">` tag
// where the browser treats the SVG as an image and disables scripts and
// external resource loads. Mounting sanitized SVG directly into the host
// Studio document loses those guarantees, so we now (a) strip every node that
// can leak into the host page (`<style>`, `<image>`, `<use>`, scripts, etc.)
// and (b) render the surviving markup in a fully sandboxed iframe at
// `SvgPreview` below for defence in depth. See:
// https://developer.mozilla.org/en-US/docs/Web/SVG/Guides/SVG_as_an_image
const SVG_PURIFY_CONFIG = {
  USE_PROFILES: { svg: true, svgFilters: true },
  // ``style`` -- inline CSS would otherwise leak to the host page selectors.
  // ``image`` / ``use`` -- carry ``href``/``xlink:href`` and would let an
  //   assistant fetch attacker-controlled URLs from the user's browser.
  // ``foreignObject`` -- can embed HTML inside the SVG and re-introduce XSS.
  FORBID_TAGS: [
    "script",
    "style",
    "foreignObject",
    "iframe",
    "embed",
    "object",
    "image",
    "use",
    "link",
    "meta",
  ],
  // Drop attributes that fetch external resources or otherwise interpret an
  // attacker-controlled URL even after the tag-level filter above:
  //   ``filter`` / ``mask`` / ``clip-path`` -- accept ``url(https://...)``
  //     and the CSS engine fetches that URL when the SVG renders.
  //   ``style`` -- inline CSS ``@import`` / ``url(...)`` does the same.
  // ``href`` / ``xlink:href`` are NOT forbidden here so safe same-document
  // fragment references survive (``<textPath href="#labelPath">``, gradient
  // ``href="#g1"``, etc.); external-scheme values are pruned in the hook
  // below so a beacon ``href`` cannot make it through.
  FORBID_ATTR: ["style", "filter", "mask", "clip-path"],
  ALLOW_DATA_ATTR: false,
};

// Pin every URI-bearing attribute (href, xlink:href, the rare animate
// attributeName, etc.) to same-document fragments. Done as a hook rather
// than DOMPurify's ALLOWED_URI_REGEXP because the regex option also
// filters non-URI presentation attributes (cx/cy/r/fill/width/height)
// and ends up rendering circles with r=0. Hook is global so we install
// it exactly once at module load.
const FRAGMENT_HREF_HOOK_TAG = "__unsloth_svg_frag_href__";
const URI_ATTRS = new Set(["href", "xlink:href"]);

if (!(DOMPurify as unknown as { [k: string]: unknown })[FRAGMENT_HREF_HOOK_TAG]) {
  DOMPurify.addHook("uponSanitizeAttribute", (_node, data) => {
    if (!URI_ATTRS.has(data.attrName)) return;
    const value = (data.attrValue ?? "").trim();
    if (!value.startsWith("#")) {
      data.keepAttr = false;
    }
  });
  (DOMPurify as unknown as { [k: string]: unknown })[FRAGMENT_HREF_HOOK_TAG] =
    true;
}

/** Strip every XML processing instruction and disallowed node from an SVG. */
export function sanitizeSvgSource(source: string): string {
  // Drop XML declarations -- DOMPurify keeps them but some renderers choke.
  const stripped = source.replace(/^\s*<\?xml[^?]*\?>\s*/i, "");
  // First pass: regex screen. We do NOT bail out -- DOMPurify will still
  // produce a safe string -- but logging here helps debugging.
  if (HEURISTIC_UNSAFE_SVG_RE.test(stripped)) {
    // eslint-disable-next-line no-console
    console.debug("SVG renderer: stripping unsafe nodes before sanitize");
  }
  return DOMPurify.sanitize(stripped, SVG_PURIFY_CONFIG);
}

function useCopyState() {
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const flash = useCallback(() => {
    setCopied(true);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setCopied(false);
      timeoutRef.current = null;
    }, COPY_RESET_MS);
  }, []);

  return { copied, flash };
}

function CopyButton({ source }: { source: string }) {
  const { copied, flash } = useCopyState();
  return (
    <button
      type="button"
      title="Copy code"
      aria-label="Copy code"
      className={cn(
        "flex size-8 cursor-pointer items-center justify-center rounded-[10px]",
        "text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
      )}
      onClick={async () => {
        if (await copyToClipboard(source)) flash();
      }}
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        strokeWidth={1.75}
        className="size-icon"
      />
    </button>
  );
}

type TabKey = "preview" | "code";

function TabButton({
  active,
  disabled,
  icon,
  id,
  controls,
  label,
  onSelect,
}: {
  active: boolean;
  disabled?: boolean;
  icon: ReactNode;
  id: string;
  controls: string;
  label: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      role="tab"
      id={id}
      aria-controls={controls}
      aria-selected={active}
      tabIndex={active ? 0 : -1}
      disabled={disabled}
      onClick={onSelect}
      className={cn(
        "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-colors",
        active
          ? "bg-background text-foreground shadow-sm"
          : "text-muted-foreground hover:text-foreground",
        disabled && "cursor-not-allowed opacity-50",
      )}
    >
      {icon}
      {label}
    </button>
  );
}

// SVG preview goes inside a sandboxed iframe (no allow-scripts, no
// allow-same-origin) plus a `default-src 'none'` CSP so the sanitizer is
// not the only line of defence -- even if a future DOMPurify regression
// leaks a URL-bearing attribute, the browser blocks the request.
const SVG_IFRAME_CSP =
  "default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:;";

function buildSvgSrcDoc(safeSvg: string): string {
  return [
    "<!doctype html>",
    `<meta http-equiv="Content-Security-Policy" content="${SVG_IFRAME_CSP}">`,
    // Fit the SVG within the iframe viewport in both dimensions so a square
    // viewBox (e.g. 200x200) scaled to the container width does not overflow
    // vertically and clip. width/height auto keeps aspect ratio intact.
    "<style>html,body{margin:0;padding:0;height:100%;background:white;}",
    "body{display:flex;align-items:center;justify-content:center;padding:16px;box-sizing:border-box;}",
    "svg{max-width:100%;max-height:100%;width:auto;height:auto;}</style>",
    safeSvg,
  ].join("");
}

function SvgPreview({ source }: { source: string }) {
  const srcDoc = useMemo(() => buildSvgSrcDoc(sanitizeSvgSource(source)), [
    source,
  ]);
  return (
    <iframe
      data-testid="html-svg-renderer-svg-preview"
      title="SVG preview"
      srcDoc={srcDoc}
      // SECURITY: sandbox="" forbids scripts AND blocks the iframe from
      // inheriting the host origin, so even sanitized SVG cannot reach the
      // Studio document or run network requests against host-cookied URLs.
      sandbox=""
      style={{
        width: "100%",
        height: 360,
        border: "none",
        display: "block",
        background: "white",
      }}
    />
  );
}

// Tiny helper script that posts the document height back to the parent so
// we can right-size the iframe. Communication is one-way and the iframe
// cannot read parent.document because we never grant allow-same-origin.
const HTML_PREVIEW_HEIGHT_REPORTER =
  '<script>(()=>{const post=()=>parent.postMessage({htmlPreviewHeight:document.documentElement.scrollHeight},"*");window.addEventListener("load",post);new ResizeObserver(post).observe(document.documentElement);})();</script>';

// Meta-CSP enforced INSIDE the srcdoc iframe. Chromium inherits the
// embedder CSP into srcdoc, data:, AND blob: iframes per HTML / CSP3
// § initialize-document-csp, so the host Studio ``script-src 'self'``
// already blocks assistant inline scripts and on* handlers here --
// confirmed empirically on the live Studio with a click-to-alert demo.
// Until a same-origin backend route is added (response-header CSPs
// do NOT inherit), the preview deliberately ships as a static-render
// surface. The meta-CSP below is defense in depth: it adds
// ``connect-src 'none'`` + ``frame-src 'none'`` so even if the host
// CSP ever loosens enough to let inline scripts run, the preview
// still cannot beacon out or nest tracking iframes.
const HTML_IFRAME_CSP = [
  "default-src 'none'",
  "script-src 'self' 'unsafe-inline'",
  "style-src 'self' 'unsafe-inline'",
  "img-src data: blob:",
  "media-src data: blob:",
  "font-src data:",
  "connect-src 'none'",
  "worker-src 'none'",
  "frame-src 'none'",
  "object-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

function buildHtmlSrcDoc(source: string): string {
  return [
    "<!doctype html>",
    `<meta http-equiv="Content-Security-Policy" content="${HTML_IFRAME_CSP}">`,
    // Outbound links open in a new tab rather than navigating the
    // sandboxed frame itself away from the preview.
    '<base target="_blank">',
    source,
    HTML_PREVIEW_HEIGHT_REPORTER,
  ].join("");
}

function HtmlPreview({
  source,
  popped,
  onHeightChange,
}: {
  source: string;
  popped: boolean;
  onHeightChange?: (h: number | null) => void;
}) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  // srcdoc, blob:, and data: all inherit the host CSP in Chromium, so
  // the choice between them does not affect script execution today.
  // srcdoc is the simplest and avoids URL.createObjectURL churn, so
  // that is what we use. Inline <script> / on* handlers in the
  // assistant HTML do NOT execute under the current host CSP; the
  // preview is for layout, images, and styles. The auto-height
  // postMessage reporter is appended for the future state where the
  // host CSP is relaxed via a backend-served preview route.
  const srcDoc = useMemo(() => buildHtmlSrcDoc(source), [source]);

  const [autoHeight, setAutoHeight] = useState<number | null>(null);
  // Reset auto-sizing whenever the source changes so we never show the
  // previous message's iframe size during the gap before the new doc
  // loads and posts its first height.
  useEffect(() => {
    setAutoHeight(null);
    onHeightChange?.(null);
  }, [source, onHeightChange]);

  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.source !== iframeRef.current?.contentWindow) return;
      const raw = (e.data as { htmlPreviewHeight?: unknown })
        ?.htmlPreviewHeight;
      if (typeof raw === "number" && Number.isFinite(raw)) {
        const next = Math.max(100, raw);
        setAutoHeight(next);
        onHeightChange?.(next);
      }
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [onHeightChange]);

  // In the docked view we cap at DEFAULT_PREVIEW_HEIGHT; in the popout we
  // let the iframe fill the modal panel.
  const iframeHeight = popped
    ? "100%"
    : Math.min(autoHeight ?? DEFAULT_PREVIEW_HEIGHT, DEFAULT_PREVIEW_HEIGHT);

  return (
    <iframe
      ref={iframeRef}
      data-testid="html-svg-renderer-iframe"
      title="HTML preview"
      srcDoc={srcDoc}
      // SECURITY:
      //   allow-scripts        -- ready for the day the host CSP
      //                           gives the preview a script-src
      //                           that includes 'unsafe-inline'
      //   allow-modals         -- alert/confirm/prompt are not no-ops
      //                           when scripts do fire
      //   allow-popups         -- the ``<base target="_blank">`` link
      //                           rule can open a new tab instead of
      //                           silently dropping the click
      // We do NOT grant:
      //   allow-same-origin / allow-top-navigation -- the iframe
      //     cannot read parent.document or navigate the host page
      //   allow-popups-to-escape-sandbox -- popups INHERIT the sandbox
      //     so an opened tab cannot use ``window.opener.top.location``
      //     to tabnab the Studio tab. The opened tab loads with an
      //     opaque origin (some sites will render degraded) which is
      //     the deliberate trade-off for tabnabbing safety.
      sandbox="allow-scripts allow-modals allow-popups"
      style={{
        width: "100%",
        height: iframeHeight,
        border: "none",
        display: "block",
        background: "white",
      }}
    />
  );
}

export function HtmlSvgRenderer({
  language,
  source,
  codeView,
  isIncomplete,
}: HtmlSvgRendererProps) {
  // Stream-in: while the fence is still being filled in we lock to the Code
  // tab so users see the in-flight tokens rather than a flashing preview.
  const lockedToCode = Boolean(isIncomplete);
  const [tab, setTab] = useState<TabKey>("preview");
  const [popped, setPopped] = useState(false);
  // Live HTML iframe height, lifted out of HtmlPreview so the pop-out spacer
  // (rendered here, not inside HtmlPreview) can match the current preview
  // size and avoid a layout jump when entering pop-out mode.
  const [htmlHeight, setHtmlHeight] = useState<number | null>(null);

  const activeTab: TabKey = lockedToCode ? "code" : tab;

  const reactId = useId();
  const previewTabId = `${reactId}-tab-preview`;
  const codeTabId = `${reactId}-tab-code`;
  const previewPanelId = `${reactId}-panel-preview`;
  const codePanelId = `${reactId}-panel-code`;

  // Escape key exits the popout view.
  useEffect(() => {
    if (!popped) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setPopped(false);
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [popped]);

  const codeFallback = useMemo(
    () =>
      codeView ?? (
        <pre className="overflow-x-auto rounded-md bg-muted/40 p-3 text-xs">
          <code>{source}</code>
        </pre>
      ),
    [codeView, source],
  );

  const onHtmlHeight = useCallback((h: number | null) => setHtmlHeight(h), []);

  const preview =
    language === "svg" ? (
      <SvgPreview source={source} />
    ) : (
      <HtmlPreview
        source={source}
        popped={popped}
        onHeightChange={onHtmlHeight}
      />
    );

  const previewLabel = language === "svg" ? "SVG preview" : "HTML preview";
  // Use the live HTML iframe height for the pop-out placeholder so swapping
  // a short preview into pop-out mode does not leave a 500px hole in the
  // chat bubble. Falls back to DEFAULT_PREVIEW_HEIGHT before the first
  // height post arrives, and is always capped to DEFAULT_PREVIEW_HEIGHT.
  const popoutSpacerHeight = Math.min(
    htmlHeight ?? DEFAULT_PREVIEW_HEIGHT,
    DEFAULT_PREVIEW_HEIGHT,
  );

  return (
    <div
      data-testid="html-svg-renderer"
      data-language={language}
      data-active-tab={activeTab}
      className="my-4 overflow-hidden rounded-xl border border-border"
    >
      <div
        role="tablist"
        aria-label={`${language.toUpperCase()} fence view`}
        className="flex items-center justify-between gap-2 border-b border-border bg-muted/40 px-2 py-1.5"
      >
        <div className="flex items-center gap-1 rounded-lg bg-muted p-0.5">
          <TabButton
            active={activeTab === "preview"}
            disabled={lockedToCode}
            icon={<EyeIcon className="size-3.5" />}
            id={previewTabId}
            controls={previewPanelId}
            label="Preview"
            onSelect={() => setTab("preview")}
          />
          <TabButton
            active={activeTab === "code"}
            icon={<CodeIcon className="size-3.5" />}
            id={codeTabId}
            controls={codePanelId}
            label="Code"
            onSelect={() => setTab("code")}
          />
        </div>
        <div className="flex items-center gap-1">
          {activeTab === "code" && !codeView ? (
            // When a custom code view is supplied it typically renders its
            // own copy/download chrome (see CodeBlockActions in
            // markdown-text.tsx). We only surface the fallback Copy button
            // when we are using the plain <pre> fallback below.
            <CopyButton source={source} />
          ) : null}
          {activeTab === "preview" && language === "html" ? (
            <button
              type="button"
              title={popped ? "Exit pop out" : "Pop out preview"}
              aria-label={popped ? "Exit pop out" : "Pop out preview"}
              className={cn(
                "flex size-8 cursor-pointer items-center justify-center rounded-[10px]",
                "text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover",
              )}
              onClick={() => setPopped((p) => !p)}
            >
              {popped ? (
                <Minimize2Icon className="size-4" />
              ) : (
                <Maximize2Icon className="size-4" />
              )}
            </button>
          ) : null}
        </div>
      </div>

      <div data-testid="html-svg-renderer-body" className="bg-background">
        {activeTab === "preview" ? (
          popped && language === "html" ? (
            <>
              {/* Keep layout stable behind the modal. */}
              <div
                style={{ height: popoutSpacerHeight }}
                aria-hidden={true}
              />
              <div
                role="dialog"
                aria-modal="true"
                aria-label="HTML preview pop out"
                className="fixed inset-0 z-50 flex flex-col bg-background/80 backdrop-blur-sm"
                onClick={(e) => {
                  if (e.target === e.currentTarget) setPopped(false);
                }}
              >
                <div className="flex items-center justify-end px-4 py-2">
                  <button
                    type="button"
                    className="flex items-center gap-1.5 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
                    onClick={() => setPopped(false)}
                    title="Exit pop out (Esc)"
                  >
                    <Minimize2Icon className="size-4" />
                    Exit pop out
                  </button>
                </div>
                <div
                  className="mx-4 mb-4 flex-1 overflow-hidden rounded-lg border border-border bg-background"
                  style={{ maxHeight: `${POPOUT_HEIGHT_VH}vh` }}
                >
                  {preview}
                </div>
              </div>
            </>
          ) : (
            <div
              role="tabpanel"
              id={previewPanelId}
              aria-labelledby={previewTabId}
              aria-label={previewLabel}
            >
              {preview}
            </div>
          )
        ) : (
          <div
            role="tabpanel"
            id={codePanelId}
            aria-labelledby={codeTabId}
            data-testid="html-svg-renderer-code"
            className="min-w-0"
          >
            {codeFallback}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- Fence helpers (exported for tests + reuse from markdown-text.tsx) ----

export type CodeFenceInfo = {
  language: string | null;
  source: string;
};

const CODE_FENCE_RE = /^```([^\r\n`]*)\r?\n([\s\S]*?)\r?\n?```$/;
// Open fence: opening backticks + lang + body but no closing fence yet. Used
// while a fenced block is still streaming in -- without this the markdown
// pipeline falls through to the generic code block until the closing fence
// arrives, so HtmlSvgRenderer's isIncomplete (lock-Code) path is dead code.
const OPEN_CODE_FENCE_RE = /^```([^\r\n`]*)\r?\n([\s\S]*)$/;

export function parseCodeFence(blockContent: string): CodeFenceInfo | null {
  const match = blockContent.trimEnd().match(CODE_FENCE_RE);
  if (!match) return null;
  return {
    language: match[1]?.trim() || null,
    source: match[2],
  };
}

/** Parse a code fence that may still be streaming (no closing ``` yet). */
export function parseIncompleteCodeFence(
  blockContent: string,
): CodeFenceInfo | null {
  const match = blockContent.match(OPEN_CODE_FENCE_RE);
  if (!match) return null;
  // Strip an in-flight trailing ``` line so a fence captured mid-close does
  // not render a stray "```" in the preview.
  const body = match[2].replace(/\r?\n?```\s*$/, "");
  return {
    language: match[1]?.trim() || null,
    source: body,
  };
}

export function isSvgFence(fence: CodeFenceInfo): boolean {
  const lang = fence.language?.toLowerCase() ?? "";
  if (lang === "svg") return true;
  if (lang === "xml" || lang === "html") {
    const trimmed = fence.source.trimStart();
    if (trimmed.startsWith("<svg")) return true;
    if (trimmed.startsWith("<?xml") && trimmed.includes("<svg")) return true;
  }
  return false;
}

export function isHtmlFence(fence: CodeFenceInfo): boolean {
  const lang = fence.language?.toLowerCase() ?? "";
  return lang === "html" && !isSvgFence(fence);
}
