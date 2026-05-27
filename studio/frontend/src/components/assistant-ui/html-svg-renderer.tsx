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
  // Optional syntax-highlighted code view; falls back to plain <pre><code>.
  codeView?: ReactNode;
  // Partial fence: lock to Code tab and disable toggle.
  isIncomplete?: boolean;
};

const DEFAULT_PREVIEW_HEIGHT = 500;
const POPOUT_HEIGHT_VH = 80;
const COPY_RESET_MS = 2000;

// Belt-and-braces pre-screen; DOMPurify does the real work below.
const HEURISTIC_UNSAFE_SVG_RE =
  /<script[\s>]|<foreignObject[\s>]|<iframe[\s>]|<embed[\s>]|<object[\s>]/i;

// SVG sanitizer config. Surviving markup is rendered in the sandboxed
// SvgPreview iframe below as a second layer.
const SVG_PURIFY_CONFIG = {
  USE_PROFILES: { svg: true, svgFilters: true },
  // image/use carry href, foreignObject embeds HTML, script/link/meta/iframe/
  // embed/object are XSS or network surfaces. <style> stays -- the iframe
  // sandbox + default-src 'none' CSP cap its blast radius.
  FORBID_TAGS: [
    "script",
    "foreignObject",
    "iframe",
    "embed",
    "object",
    "image",
    "use",
    "link",
    "meta",
  ],
  // filter / mask / clip-path accept url(...) which fetches at render time;
  // style allows inline @import / url(...). href / xlink:href stay so safe
  // same-doc fragments survive -- external schemes pruned in the hook below.
  FORBID_ATTR: ["style", "filter", "mask", "clip-path"],
  ALLOW_DATA_ATTR: false,
};

// Pin href / xlink:href to same-doc fragments. Done as a hook because
// DOMPurify's ALLOWED_URI_REGEXP also rejects presentation attrs (cx, r, ...).
// Installed once at module load.
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

/** Strip XML PIs and disallowed nodes from an SVG. */
export function sanitizeSvgSource(source: string): string {
  // Drop XML declarations -- DOMPurify keeps them, some renderers choke.
  const stripped = source.replace(/^\s*<\?xml[^?]*\?>\s*/i, "");
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

// Defence in depth on top of the sanitizer.
const SVG_IFRAME_CSP =
  "default-src 'none'; img-src data:; style-src 'unsafe-inline'; font-src data:;";

function buildSvgSrcDoc(safeSvg: string): string {
  return [
    "<!doctype html>",
    `<meta http-equiv="Content-Security-Policy" content="${SVG_IFRAME_CSP}">`,
    // Cap both dimensions so a square viewBox does not clip vertically.
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
      // No scripts, no same-origin: SVG cannot touch host document or network.
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

// Iframe posts its scrollHeight to the parent for auto-sizing. One-way:
// no allow-same-origin, so iframe cannot read parent.document.
const HTML_PREVIEW_HEIGHT_REPORTER =
  '<script>(()=>{const post=()=>parent.postMessage({htmlPreviewHeight:document.documentElement.scrollHeight},"*");window.addEventListener("load",post);new ResizeObserver(post).observe(document.documentElement);})();</script>';

// Fallback for when the preview route is unreachable. srcdoc inherits the
// host script-src 'self' per CSP3 so inline scripts will NOT run on this
// path -- it is layout-only. Interactive surface is the API route below.
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
    '<base target="_blank">',
    source,
    HTML_PREVIEW_HEIGHT_REPORTER,
  ].join("");
}

// Same-origin route whose response CSP permits inline scripts. POST source,
// use returned random-token URL as iframe src.
const HTML_PREVIEW_API = "/api/preview/html";

// Same key as features/auth/session.ts:AUTH_TOKEN_KEY; inlined to avoid cycle.
function getStoredAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem("unsloth_auth_token");
  } catch {
    return null;
  }
}

type PreviewState =
  | { kind: "loading" }
  | { kind: "ready"; url: string }
  | { kind: "error" };

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
  // POST source -> token URL. srcdoc / data: / blob: inherit the host CSP
  // per CSP3, so a same-origin route is the only way to unlock inline scripts.
  const [previewState, setPreviewState] = useState<PreviewState>({
    kind: "loading",
  });

  useEffect(() => {
    let cancelled = false;
    setPreviewState({ kind: "loading" });
    onHeightChange?.(null);

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    const token = getStoredAccessToken();
    if (token) headers.Authorization = `Bearer ${token}`;

    void fetch(HTML_PREVIEW_API, {
      method: "POST",
      credentials: "same-origin",
      headers,
      body: JSON.stringify({ source }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(`HTML preview HTTP ${r.status}`);
        return (await r.json()) as { url: string };
      })
      .then(({ url }) => {
        if (cancelled) return;
        if (typeof url !== "string" || !url.startsWith("/api/preview/html/")) {
          throw new Error("HTML preview returned an unexpected URL shape");
        }
        setPreviewState({ kind: "ready", url });
      })
      .catch(() => {
        if (cancelled) return;
        setPreviewState({ kind: "error" });
      });

    return () => {
      cancelled = true;
    };
  }, [source, onHeightChange]);

  const [autoHeight, setAutoHeight] = useState<number | null>(null);
  useEffect(() => {
    setAutoHeight(null);
  }, [previewState]);

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

  const iframeHeight = popped
    ? "100%"
    : Math.min(autoHeight ?? DEFAULT_PREVIEW_HEIGHT, DEFAULT_PREVIEW_HEIGHT);

  // Backend unreachable: fall back to srcdoc (static layout, no scripts)
  // instead of going blank.
  const errorSrcDoc = useMemo(
    () => (previewState.kind === "error" ? buildHtmlSrcDoc(source) : null),
    [previewState.kind, source],
  );

  return (
    <iframe
      ref={iframeRef}
      data-testid="html-svg-renderer-iframe"
      data-preview-state={previewState.kind}
      title="HTML preview"
      src={previewState.kind === "ready" ? previewState.url : "about:blank"}
      srcDoc={errorSrcDoc ?? undefined}
      // allow-scripts: inline JS runs (route CSP permits it).
      // allow-modals: alert / confirm / prompt usable.
      // allow-popups: <base target="_blank"> links open a tab.
      // NOT granted:
      //   allow-same-origin / allow-top-navigation -- preview JS cannot
      //   reach parent.document or navigate the host (URL is same-origin
      //   but iframe is treated as opaque).
      //   allow-popups-to-escape-sandbox -- opened tabs inherit sandbox so
      //   window.opener.top.location.* tabnabbing is blocked.
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
  // Lock to Code while streaming so users see tokens, not a flashing preview.
  const lockedToCode = Boolean(isIncomplete);
  const [tab, setTab] = useState<TabKey>("preview");
  const [popped, setPopped] = useState(false);
  // Lifted from HtmlPreview so the pop-out spacer can match current height.
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
  // Match live iframe height so popping out a short preview does not leave
  // a 500px hole. Falls back to DEFAULT before the first height post.
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
            // Custom code views ship their own copy/download chrome
            // (CodeBlockActions in markdown-text.tsx); only show Copy
            // when we are using the plain <pre> fallback.
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
// Open fence: no closing ``` yet. Lets HtmlSvgRenderer mount mid-stream
// (otherwise the markdown pipeline falls through until the close arrives).
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
  // Strip a trailing partial ``` so mid-close captures do not leak it.
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
