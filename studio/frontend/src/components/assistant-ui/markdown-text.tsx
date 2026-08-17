// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { ArtifactCard, useChatRuntimeStore } from "@/features/chat";
import {
  getCodeFence,
  isFullHtmlDocument,
  isHtmlFence,
  isRenderableRenderHtmlToolPart,
  isSvgFence,
} from "@/features/chat/artifacts/html-fences";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { preprocessLaTeX } from "@/lib/latex";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { openLink } from "@/lib/open-link";
import { safeMarkdownUrl } from "@/lib/safe-markdown-url";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { INTERNAL, useAuiState, useMessagePartText } from "@assistant-ui/react";

import type { HighlightResult } from "@streamdown/code";
import { Copy01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { createMathPlugin } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";

import { useVirtualizer } from "@tanstack/react-virtual";
import {
  type ComponentProps,
  type CSSProperties,

  createContext,
  isValidElement,
  memo,
  type ReactNode,
  useCallback,

  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Block,
  type BlockProps,
  type ExtraProps,
  parseMarkdownIntoBlocks,
  Streamdown,
  type StreamdownProps,
  useIsCodeFenceIncomplete,
} from "streamdown";
import { createCodePlugin } from "./code-plugin";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import { stabilizeStreamingMarkdown } from "./streaming-markdown";
import {
  IncrementalMarkdownCache,
  type MarkdownBlockSnapshot,
  withoutStreamdownAnimationPlugin,
} from "./streaming-render-schedule";

const math = createMathPlugin({ singleDollarTextMath: true });
const code = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});
const STREAMDOWN_PLUGINS = { code, math, mermaid } satisfies NonNullable<
  StreamdownProps["plugins"]
>;

// Syntax highlighting is finalized once generation stops. Even completed fences
// can be remounted repeatedly while the virtualized live edge is moving.
const STREAMDOWN_STREAMING_PLUGINS = { math, mermaid } satisfies NonNullable<
  StreamdownProps["plugins"]
>;
const STREAMDOWN_CONTROLS = {
  code: false,
  mermaid: {
    fullscreen: true,
    download: true,
    copy: false,
    panZoom: true,
  },
} satisfies NonNullable<StreamdownProps["controls"]>;
const STREAMDOWN_SHIKI_THEME = [
  unslothLightTheme,
  unslothDarkTheme,
] satisfies NonNullable<StreamdownProps["shikiTheme"]>;
const { withSmoothContextProvider } = INTERNAL;

// Streamdown 2.5 schedules ordinary streaming blocks in an interruptible React
// transition. A continuous token stream can starve that transition for seconds.
// Its animated path commits every block update directly. StreamdownBlock removes
// the animation transformer while retaining this direct scheduling path.
const STREAMDOWN_IMMEDIATE_UPDATES = {
  duration: 0,
  stagger: 0,
} satisfies NonNullable<StreamdownProps["animated"]>;

const STREAMDOWN_COMPONENTS = {
  a: ({ href, children, ...props }: ComponentProps<"a">) => (
    <a
      href={href}
      rel="noopener noreferrer"
      className="text-primary underline underline-offset-2 decoration-primary/40 hover:decoration-primary transition-colors cursor-pointer"
      onClick={(e) => {
        if (href && openLink(href)) {
          e.preventDefault();
        }
      }}
      {...props}
    >
      {children}
    </a>
  ),
};

const ActiveStreamingBlockContext = createContext(false);

export const PlainStreamingMarkdownContext = createContext(false);
const COPY_RESET_MS = 2000;
const MERMAID_SOURCE_RE = /```mermaid\s*([\s\S]*?)```/i;
const ACTION_PANEL_CLASS =
  "pointer-events-auto flex shrink-0 items-center gap-1";
const ACTION_BUTTON_CLASS =
  "flex size-8 cursor-pointer items-center justify-center rounded-[10px] text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover disabled:cursor-not-allowed disabled:opacity-50";

function getMermaidSource(blockContent: string): string | null {
  const source = blockContent.match(MERMAID_SOURCE_RE)?.[1]?.trim();
  return source && source.length > 0 ? source : null;
}

function getCodeFilename(language: string | null) {
  const extByLanguage: Record<string, string> = {
    bash: "sh",
    "c++": "cpp",
    csharp: "cs",
    javascript: "js",
    js: "js",
    json: "json",
    jsx: "jsx",
    markdown: "md",
    md: "md",
    python: "py",
    py: "py",
    ruby: "rb",
    rust: "rs",
    shell: "sh",
    sh: "sh",
    sql: "sql",
    ts: "ts",
    tsx: "tsx",
    typescript: "ts",
    svg: "svg",
    yaml: "yml",
    yml: "yml",
  };

  const normalized = language?.toLowerCase();
  const fallbackExt = normalized?.replace(/[^a-z0-9]+/g, "-");
  const ext = normalized
    ? extByLanguage[normalized] || fallbackExt || "txt"
    : "txt";
  return `snippet.${ext}`;
}

const UNSAFE_SVG_RE =
  /<script[\s>]|on\w+\s*=|javascript:|<foreignObject[\s>]|<iframe[\s>]|<embed[\s>]|<object[\s>]/i;

function sanitizeSvg(source: string): string | null {
  if (UNSAFE_SVG_RE.test(source)) return null;
  // Strip XML declaration: unneeded for data URIs and breaks some renderers.
  return source.replace(/^\s*<\?xml[^?]*\?>\s*/i, "");
}

function SvgPreview({ source }: { source: string }) {
  const dataUri = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(source)}`;
  return (
    <div className="mt-2 flex justify-center rounded-lg border border-border bg-white p-4 dark:bg-neutral-100">
      <img
        src={dataUri}
        alt="SVG preview"
        style={{ maxWidth: "100%", maxHeight: 512 }}
      />
    </div>
  );
}

function downloadTextFile(filename: string, text: string): void {
  void downloadFile(text, filename, "text/plain;charset=utf-8").catch(
    (error) => {
      if (!isDownloadCancelled(error)) {
        toast.error("Could not save file.");
      }
    },
  );
}

function useCopiedState() {
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimeoutRef.current) {
        clearTimeout(resetTimeoutRef.current);
      }
    };
  }, []);

  const showCopied = () => {
    setCopied(true);
    if (resetTimeoutRef.current) {
      clearTimeout(resetTimeoutRef.current);
    }
    resetTimeoutRef.current = setTimeout(() => {
      setCopied(false);
      resetTimeoutRef.current = null;
    }, COPY_RESET_MS);
  };

  return { copied, showCopied };
}

function MermaidCopyButton({ source }: { source: string }) {
  const { copied, showCopied } = useCopiedState();

  return (
    <button
      type="button"
      className="absolute top-3.5 right-20 z-20 cursor-pointer text-muted-foreground transition-all hover:text-foreground"
      title="Copy Mermaid source"
      onClick={async () => {
        if (!(await copyToClipboard(source))) {
          return;
        }
        showCopied();
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

function CodeBlockActions({
  disabled,
  language,
  source,
}: {
  disabled: boolean;
  language: string | null;
  source: string;
}) {
  const { copied, showCopied } = useCopiedState();

  return (
    <div className="pointer-events-none absolute top-3 right-3 z-20 flex items-center justify-end">
      <div className={ACTION_PANEL_CLASS}>
        <button
          type="button"
          className={ACTION_BUTTON_CLASS}
          title="Copy code"
          disabled={disabled}
          onClick={async () => {
            if (!(await copyToClipboard(source))) {
              return;
            }
            showCopied();
          }}
        >
          <HugeiconsIcon
            icon={copied ? Tick02Icon : Copy01Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
        </button>
        <button
          type="button"
          className={ACTION_BUTTON_CLASS}
          title="Download file"
          disabled={disabled}
          onClick={() => {
            downloadTextFile(getCodeFilename(language), source);
          }}
        >
          <HugeiconsIcon icon={Download01Icon} className="size-icon" />
        </button>
      </div>
    </div>
  );
}

function useAnimationFreeBlockProps(props: BlockProps): BlockProps {
  // `animated` is needed only to bypass Streamdown's starvable React transition.
  // Its rehype plugin still wraps every word even with duration and stagger set
  // to zero. Remove that one plugin before parsing so long streams do not create
  // thousands of animation spans. Keep the filtered array stable so completed
  // blocks remain memoised while the final block continues streaming.
  const rehypePlugins = useMemo(
    () =>
      withoutStreamdownAnimationPlugin(
        props.rehypePlugins,
        props.animatePlugin,
      ),
    [props.animatePlugin, props.rehypePlugins],
  );
  return {
    ...props,
    animatePlugin: null,
    rehypePlugins,
  } satisfies BlockProps;
}

// Active fences intentionally remain plain: one text node is substantially
// cheaper and more stable than a Shiki token tree that changes ten times/second.
function StreamingPlainCodeBlock({
  language,
  source,
}: {
  language: string | null;
  source: string;
}) {
  const label = language?.trim() || "text";
  return (
    <div
      className="my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2"
      data-incomplete="true"
      data-language={label}
      data-streamdown="code-block"
      data-streaming-code="true"
    >
      <div
        className="flex h-8 items-center text-xs text-muted-foreground"
        data-language={label}
        data-streamdown="code-block-header"
      >
        <span className="ml-1 font-mono lowercase">{label}</span>
      </div>
      <div
        className="overflow-x-auto rounded-md border border-border bg-background p-4 text-sm"
        data-language={label}
        data-streamdown="code-block-body"
      >
        <pre className="m-0 min-w-max bg-transparent p-0 font-mono text-sm leading-5">
          <code>{source || "\n"}</code>
        </pre>
      </div>
    </div>
  );
}

// Collapse a full-HTML answer in place into an artifact card. Diffusion keeps the
// raw code visible instead (the trailing MessageHtmlArtifacts appends its card).
function StreamdownBlockContent(props: BlockProps) {
  const blockProps = useAnimationFreeBlockProps(props);

  const activeStreamingBlock = useContext(ActiveStreamingBlockContext);
  const shouldCollapseHtmlArtifacts = useChatRuntimeStore(
    (state) =>
      (state.artifactsEnabled || state.collapseHtmlArtifacts) &&
      !state.loadedIsDiffusion,
  );
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  const hasMermaidFence = props.content.includes("```mermaid");
  const mermaidSource = getMermaidSource(props.content);
  const codeFence = getCodeFence(props.content);

  if (props.isIncomplete && hasMermaidFence) {
    return (
      <div className="my-4 flex h-48 items-center justify-center rounded-xl border border-border bg-muted/30 text-sm text-muted-foreground animate-pulse">
        Loading diagram...
      </div>
    );
  }

  if (props.isIncomplete && codeFence && isSvgFence(codeFence)) {
    return (
      <div className="relative isolate">
        <div className="my-4 rounded-xl border border-border bg-muted/30 p-4">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            svg
          </div>
          <pre className="overflow-x-auto text-xs text-muted-foreground whitespace-pre-wrap break-all">
            <code>{codeFence.source}</code>
          </pre>
        </div>
      </div>
    );
  }

  if (
    shouldCollapseHtmlArtifacts &&
    !messageHasRenderableRenderHtmlTool &&
    props.isIncomplete &&
    codeFence &&
    isHtmlFence(codeFence) &&
    isFullHtmlDocument(codeFence.source)
  ) {
    return (
      <div className="my-4 flex h-48 items-center justify-center rounded-xl border border-border bg-muted/30 text-sm text-muted-foreground animate-pulse">
        Loading canvas preview...
      </div>
    );
  }

  // Re-tokenizing and reconciling a growing Shiki tree is the dominant WebKit
  // cost for long generations. Keep the active fence as one native text node;
  // the completed fence is highlighted (and virtualized when large) once.
  if ((props.isIncomplete || activeStreamingBlock) && codeFence) {
    return (
      <StreamingPlainCodeBlock
        language={codeFence.language}
        source={codeFence.source}
      />
    );
  }

  if (mermaidSource) {
    return (
      <div className="relative isolate">
        <Block {...blockProps} />
        <MermaidCopyButton source={mermaidSource} />
      </div>
    );
  }

  if (codeFence) {
    const svgSource =
      !props.isIncomplete && isSvgFence(codeFence)
        ? sanitizeSvg(codeFence.source)
        : null;
    const htmlSource =
      shouldCollapseHtmlArtifacts &&
      !messageHasRenderableRenderHtmlTool &&
      !props.isIncomplete &&
      isHtmlFence(codeFence) &&
      isFullHtmlDocument(codeFence.source)
        ? codeFence.source
        : null;
    if (htmlSource) {
      return (
        <ArtifactCard code={htmlSource} title="HTML preview" source="fence" />
      );
    }

    return (
      <>
        <div className="relative isolate">
          <Block {...blockProps} />
          <CodeBlockActions
            disabled={props.isIncomplete}
            language={codeFence.language}
            source={codeFence.source}
          />
        </div>
        {svgSource && <SvgPreview source={svgSource} />}
      </>
    );
  }

  return <Block {...blockProps} />;
}
const StreamdownBlock = memo(StreamdownBlockContent);
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

// Limit expensive Markdown work to ten paint-aligned commits per second. The
// transport can deliver hundreds of token events per second; rendering every
// animation frame leaves WebKit no idle time for scrolling or input once the
// active block becomes substantial. Completion and non-prefix replacements
// still bypass the held snapshot in the return path below.
const STREAM_RENDER_INTERVAL_MS = 100;

function useCoalescedStreamingText(
  text: string,
  isStreaming: boolean,
  messageId: string,
): string {
  const [displayed, setDisplayed] = useState({ messageId, text });
  const pendingRef = useRef({ messageId, text });
  const rafRef = useRef<number | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastRenderAtRef = useRef(0);
  const activeMessageIdRef = useRef(messageId);

  const cancelScheduledRender = useCallback(() => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const schedulePaint = useCallback(() => {
    timeoutRef.current = null;
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      lastRenderAtRef.current = performance.now();
      setDisplayed(pendingRef.current);
    });
  }, []);

  useEffect(() => {
    pendingRef.current = { messageId, text };
    if (activeMessageIdRef.current !== messageId) {
      cancelScheduledRender();
      activeMessageIdRef.current = messageId;
      lastRenderAtRef.current = 0;
    }
    if (!isStreaming) {
      cancelScheduledRender();
      return;
    }
    if (timeoutRef.current !== null || rafRef.current !== null) return;

    const elapsed = performance.now() - lastRenderAtRef.current;
    const delay = Math.max(0, STREAM_RENDER_INTERVAL_MS - elapsed);
    if (delay === 0) schedulePaint();
    else timeoutRef.current = setTimeout(schedulePaint, delay);
  }, [cancelScheduledRender, isStreaming, messageId, schedulePaint, text]);

  useEffect(() => cancelScheduledRender, [cancelScheduledRender]);

  // Holding the last painted text is only correct while the reply is being
  // appended to. A running message can also be replaced, as the audio path does
  // when it swaps its placeholder for the player, and that must show at once.
  if (
    isStreaming &&
    displayed.messageId === messageId &&
    text.length >= displayed.text.length &&
    text.startsWith(displayed.text)
  ) {
    return displayed.text;
  }
  return text;
}

export const MARKDOWN_LAYOUT_EVENT = "aui-markdown-layout";

const VIRTUALIZE_AFTER_BLOCKS = 24;
const MARKDOWN_BLOCK_ESTIMATE_PX = 32;
const MARKDOWN_BLOCK_OVERSCAN = 12;

function findVerticalScrollOwner(element: HTMLElement): HTMLElement | null {
  for (let parent = element.parentElement; parent; parent = parent.parentElement) {
    const style = getComputedStyle(parent);
    if (/(auto|scroll)/.test(style.overflowY)) {
      return parent;
    }
  }
  return null;
}

const VIRTUALIZE_CODE_AFTER_LINES = 120;
const VIRTUALIZE_CODE_AFTER_CHARS = 20_000;
const CODE_LINE_ESTIMATE_PX = 20;
const CODE_LINE_OVERSCAN = 24;
const PLAIN_LONG_LINE_AFTER_CHARS = 8_000;
const CODE_LANGUAGE_RE = /language-([^\s]+)/;

function readCodeChildren(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(readCodeChildren).join("");
  if (
    isValidElement<{ children?: ReactNode }>(children) &&
    children.props.children !== undefined
  ) {
    return readCodeChildren(children.props.children);
  }
  return "";
}

function trimTrailingNewlines(source: string): string {
  let end = source.length;
  while (end > 0 && source[end - 1] === "\n") end -= 1;
  return source.slice(0, end);
}

function plainHighlight(source: string): HighlightResult {
  return {
    bg: "transparent",
    fg: "inherit",
    tokens: source.split("\n").map((content) => [
      {
        content,
        offset: 0,
      },
    ]),
  } as HighlightResult;
}

function useHighlightedCode(source: string, language: string): HighlightResult {
  const raw = useMemo(() => plainHighlight(source), [source]);
  const [snapshot, setSnapshot] = useState({ source, result: raw });

  useEffect(() => {
    let active = true;
    const publish = (next: HighlightResult) => {
      if (active) setSnapshot({ source, result: next });
    };
    const immediate = code.highlight(
      {
        code: source,
        language: language as Parameters<typeof code.highlight>[0]["language"],
        themes: STREAMDOWN_SHIKI_THEME,
      },
      publish,
    );
    queueMicrotask(() => publish(immediate ?? raw));
    return () => {
      active = false;
      code.cancelHighlight(publish);
    };
  }, [language, raw, source]);

  return snapshot.source === source ? snapshot.result : raw;
}

function resultRootStyle(result: HighlightResult): CSSProperties {
  const style: Record<string, string> = {};
  if (result.bg) style["--sdm-bg"] = result.bg;
  if (result.fg) style["--sdm-fg"] = result.fg;
  if (result.rootStyle) {
    for (const declaration of result.rootStyle.split(";")) {
      const separator = declaration.indexOf(":");
      if (separator <= 0) continue;
      const property = declaration.slice(0, separator).trim();
      const value = declaration.slice(separator + 1).trim();
      if (property && value) style[property] = value;
    }
  }
  return style as CSSProperties;
}

function tokenStyle(
  token: HighlightResult["tokens"][number][number],
): CSSProperties {
  const style: Record<string, string> = {};
  if (token.color) style["--sdm-c"] = token.color;
  if (token.bgColor) style["--sdm-tbg"] = token.bgColor;
  for (const [property, value] of Object.entries(token.htmlStyle ?? {})) {
    if (property === "color") style["--sdm-c"] = value;
    else if (property === "background-color") style["--sdm-tbg"] = value;
    else style[property] = value;
  }
  return style as CSSProperties;
}

function HighlightedCodeLine({
  line,
  lineNumber,
}: {
  line: HighlightResult["tokens"][number];
  lineNumber: number;
}) {
  const content = line.map((token) => token.content).join("");
  return (
    <span className="block min-w-max" data-code-line={lineNumber}>
      <span
        aria-hidden="true"
        className="mr-4 inline-block w-6 select-none text-right font-mono text-[13px] text-muted-foreground/50"
      >
        {lineNumber}
      </span>
      {content.length > PLAIN_LONG_LINE_AFTER_CHARS ? (
        content
      ) : line.length === 0 || (line.length === 1 && content === "") ? (
        "\n"
      ) : (
        line.map((token, index) => {
          const hasBackground = Boolean(
            token.bgColor ?? token.htmlStyle?.["background-color"],
          );
          return (
            <span
              // A completed Shiki line retains token order and object identity.
              // The line wrapper, not each token, is the virtualized unit.
              key={index}
              className={`text-[var(--sdm-c,inherit)] dark:text-[var(--shiki-dark,var(--sdm-c,inherit))]${
                hasBackground
                  ? " bg-[var(--sdm-tbg)] dark:bg-[var(--shiki-dark-bg,var(--sdm-tbg))]"
                  : ""
              }`}
              style={tokenStyle(token)}
              {...token.htmlAttrs}
            >
              {token.content}
            </span>
          );
        })
      )}
    </span>
  );
}

function VirtualizedCodeLines({ result }: { result: HighlightResult }) {
  const rootRef = useRef<HTMLSpanElement>(null);
  const [scrollElement, setScrollElement] = useState<HTMLElement | null>(null);
  const [scrollMargin, setScrollMargin] = useState(0);

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const owner = findVerticalScrollOwner(root);
    setScrollElement(owner);
    if (!owner) return;
    const measureOffset = () => {
      const rootRect = root.getBoundingClientRect();
      const ownerRect = owner.getBoundingClientRect();
      setScrollMargin(rootRect.top - ownerRect.top + owner.scrollTop);
    };
    measureOffset();
    const resizeObserver = new ResizeObserver(measureOffset);
    resizeObserver.observe(root);
    resizeObserver.observe(owner);
    return () => resizeObserver.disconnect();
  }, []);

  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: result.tokens.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => CODE_LINE_ESTIMATE_PX,
    overscan: CODE_LINE_OVERSCAN,
    scrollMargin,
  });
  const totalSize = virtualizer.getTotalSize();

  useLayoutEffect(() => {
    rootRef.current?.dispatchEvent(
      new CustomEvent(MARKDOWN_LAYOUT_EVENT, { bubbles: true }),
    );
  }, [totalSize]);

  return (
    <span
      ref={rootRef}
      className="block"
      data-virtualized-code="true"
      style={{ height: `${totalSize}px`, position: "relative" }}
    >
      {virtualizer.getVirtualItems().map((virtualLine) => {
        const line = result.tokens[virtualLine.index];
        if (!line) return null;
        return (
          <span
            className="block"
            key={virtualLine.key}
            data-index={virtualLine.index}
            ref={virtualizer.measureElement}
            style={{
              left: 0,
              position: "absolute",
              top: virtualLine.start - scrollMargin,
            }}
          >
            <HighlightedCodeLine
              line={line}
              lineNumber={virtualLine.index + 1}
            />
          </span>
        );
      })}
    </span>
  );
}

function VirtualizedCodeBlock({
  className,
  language,
  source,
}: {
  className?: string;
  language: string;
  source: string;
}) {
  const isIncomplete = useIsCodeFenceIncomplete();
  const result = useHighlightedCode(source, language);

  return (
    <div
      className={`my-4 flex w-full flex-col gap-2 rounded-xl border border-border bg-sidebar p-2 ${className ?? ""}`}
      data-incomplete={isIncomplete || undefined}
      data-language={language}
      data-streamdown="code-block"
    >
      <div
        className="flex h-8 items-center text-xs text-muted-foreground"
        data-language={language}
        data-streamdown="code-block-header"
      >
        <span className="ml-1 font-mono lowercase">{language}</span>
      </div>
      <div
        className="overflow-x-auto rounded-md border border-border bg-background p-4 text-sm"
        data-language={language}
        data-streamdown="code-block-body"
      >
        <pre
          className="bg-[var(--sdm-bg,inherit)] dark:bg-[var(--shiki-dark-bg,var(--sdm-bg,inherit))]"
          style={resultRootStyle(result)}
        >
          <code>
            <VirtualizedCodeLines result={result} />
          </code>
        </pre>
      </div>
    </div>
  );
}

function VirtualizedCode({
  children,
  className,
  node: _node,
  ...props
}: ComponentProps<"code"> & ExtraProps) {

  void _node;
  if (!("data-block" in props)) {
    return (
      <code
        className={`rounded bg-muted px-1.5 py-0.5 font-mono text-sm ${className ?? ""}`}
        data-streamdown="inline-code"
        {...props}
      >
        {children}
      </code>
    );
  }

  return (
    <VirtualizedCodeBlock
      className={className}
      language={className?.match(CODE_LANGUAGE_RE)?.[1] ?? "text"}
      source={trimTrailingNewlines(readCodeChildren(children))}
    />
  );
}

const VIRTUALIZED_CODE_COMPONENTS = {
  ...STREAMDOWN_COMPONENTS,
  code: VirtualizedCode,
} as NonNullable<StreamdownProps["components"]>;

function shouldVirtualizeCode(block: MarkdownBlockSnapshot): boolean {
  const fence = getCodeFence(block.content);
  if (!fence || fence.language?.toLowerCase() === "mermaid") return false;
  return (
    fence.source.length >= VIRTUALIZE_CODE_AFTER_CHARS ||
    fence.source.split("\n").length >= VIRTUALIZE_CODE_AFTER_LINES
  );
}


type VirtualizedMarkdownProps = {
  blocks: readonly MarkdownBlockSnapshot[];
  isStreaming: boolean;
  messageId: string;
};

function VirtualizedMarkdown({
  blocks,
  isStreaming,
  messageId,
}: VirtualizedMarkdownProps) {
  const rootRef = useRef<HTMLDivElement>(null);
  const [scrollElement, setScrollElement] = useState<HTMLElement | null>(null);
  const [scrollMargin, setScrollMargin] = useState(0);

  useLayoutEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    const owner = findVerticalScrollOwner(root);
    setScrollElement(owner);
    if (!owner) return;

    const measureOffset = () => {
      const rootRect = root.getBoundingClientRect();
      const ownerRect = owner.getBoundingClientRect();
      setScrollMargin(rootRect.top - ownerRect.top + owner.scrollTop);
    };
    measureOffset();
    const resizeObserver = new ResizeObserver(measureOffset);
    resizeObserver.observe(root);
    resizeObserver.observe(owner);
    return () => resizeObserver.disconnect();
  }, [messageId]);

  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: blocks.length,
    getScrollElement: () => scrollElement,
    estimateSize: () => MARKDOWN_BLOCK_ESTIMATE_PX,
    getItemKey: (index) => blocks[index]?.id ?? index,
    overscan: MARKDOWN_BLOCK_OVERSCAN,
    scrollMargin,
  });

  const totalSize = virtualizer.getTotalSize();
  useLayoutEffect(() => {
    rootRef.current?.dispatchEvent(
      new CustomEvent(MARKDOWN_LAYOUT_EVENT, { bubbles: true }),
    );
  }, [totalSize]);

  return (
    <div
      ref={rootRef}
      data-virtualized-markdown="true"
      style={{
        height: `${totalSize}px`,
        position: "relative",
        width: "100%",
      }}
    >
      {virtualizer.getVirtualItems().map((virtualBlock) => {
        const block = blocks[virtualBlock.index];
        if (!block) return null;
        const isLast = virtualBlock.index === blocks.length - 1;
        return (
          <div
            key={block.id}
            ref={virtualizer.measureElement}
            data-index={virtualBlock.index}
            data-markdown-block-id={block.id}
            style={{
              left: 0,
              position: "absolute",
              top: virtualBlock.start - scrollMargin,
              paddingBottom: isLast ? 0 : "1rem",
              width: "100%",
            }}
          >
            <ActiveStreamingBlockContext.Provider
              value={isStreaming}
            >

            <Streamdown
              mode="streaming"
              parseIncompleteMarkdown={isStreaming && isLast}
              isAnimating={isStreaming && isLast}
              animated={STREAMDOWN_IMMEDIATE_UPDATES}
              plugins={
                isStreaming
                  ? STREAMDOWN_STREAMING_PLUGINS
                  : STREAMDOWN_PLUGINS
              }
              components={
                !isStreaming && shouldVirtualizeCode(block)
                  ? VIRTUALIZED_CODE_COMPONENTS
                  : STREAMDOWN_COMPONENTS
              }
              urlTransform={safeMarkdownUrl}
              controls={STREAMDOWN_CONTROLS}
              shikiTheme={STREAMDOWN_SHIKI_THEME}
              BlockComponent={StreamdownBlock}
            >
              {block.content}
            </Streamdown>

            </ActiveStreamingBlockContext.Provider>
          </div>
        );
      })}
    </div>
  );
}


const PLAIN_STREAM_CHUNK_SIZE = 4096;
const PlainStreamingChunk = memo(function PlainStreamingChunk({
  text,
}: {
  text: string;
}) {
  return <span>{text}</span>;
});

function PlainStreamingText({ text, status }: { text: string; status: string }) {
  const rootRef = useRef<HTMLDivElement>(null);
  const chunks = useMemo(() => {
    const next: string[] = [];
    for (let offset = 0; offset < text.length; offset += PLAIN_STREAM_CHUNK_SIZE) {
      next.push(text.slice(offset, offset + PLAIN_STREAM_CHUNK_SIZE));
    }
    return next;
  }, [text]);

  useLayoutEffect(() => {
    rootRef.current?.dispatchEvent(
      new CustomEvent(MARKDOWN_LAYOUT_EVENT, { bubbles: true }),
    );
  }, [text]);

  return (
    <div
      ref={rootRef}
      data-status={status}
      data-streaming-plain-text="true"
      className="min-w-0 max-w-full whitespace-pre-wrap break-words"
    >
      {chunks.map((chunk, index) => (
        <PlainStreamingChunk key={index} text={chunk} />
      ))}
    </div>
  );
}


const MarkdownTextImpl = () => {

  const layoutRef = useRef<HTMLDivElement>(null);
  const { text, status } = useMessagePartText();
  // Parts are keyed by index, so switching conversations hands this instance a
  // different message, and Streamdown only extends its parsed blocks: key it per
  // message. The cache generation joins the key for the case the Markdown string
  // cannot express, an edit that drops retained blocks without changing the tail.
  const messageId = useAuiState(({ message }) => message.id);
  const isStreaming = status.type === "running";

  const preferPlainStreaming = useContext(PlainStreamingMarkdownContext);
  const plainStreaming = isStreaming && preferPlainStreaming;
  const displayText = useCoalescedStreamingText(text, isStreaming, messageId);
  const processedText = useMemo(
    () =>
      plainStreaming
        ? ""
        : stabilizeStreamingMarkdown(preprocessLaTeX(displayText), isStreaming),
    [displayText, isStreaming, plainStreaming],
  );
  const incrementalCacheRef = useRef({
    messageId,
    cache: new IncrementalMarkdownCache(),
  });
  if (incrementalCacheRef.current.messageId !== messageId) {
    incrementalCacheRef.current = {
      messageId,
      cache: new IncrementalMarkdownCache(),
    };
  }
  const incrementalCache = incrementalCacheRef.current.cache;
  const incrementalRender = isStreaming && !plainStreaming
    ? incrementalCache.update(processedText)
    : null;
  const blocks = plainStreaming
    ? []
    : incrementalRender?.blocks ??
      parseMarkdownIntoBlocks(processedText).map((content, index) => ({
        id: index + 1,
        content,
      }));
  const shouldVirtualize =
    blocks.length >= VIRTUALIZE_AFTER_BLOCKS || blocks.some(shouldVirtualizeCode);

  useLayoutEffect(() => {
    layoutRef.current?.dispatchEvent(
      new CustomEvent(MARKDOWN_LAYOUT_EVENT, { bubbles: true }),
    );
  }, [processedText, shouldVirtualize]);

  const audioMatch = displayText.match(AUDIO_PLAYER_RE);
  if (audioMatch) {
    return <AudioPlayer src={audioMatch[1]} />;
  }

  if (plainStreaming) {
    return <PlainStreamingText text={displayText} status={status.type} />;
  }

  return (
    <div
      ref={layoutRef}
      data-status={status.type}
      className="min-w-0 max-w-full"
    >
      {shouldVirtualize ? (
        <VirtualizedMarkdown
          blocks={blocks}
          isStreaming={isStreaming}
          messageId={messageId}
        />
      ) : (
        <ActiveStreamingBlockContext.Provider value={isStreaming}>

        <Streamdown
          key={`${messageId}:${incrementalCache.renderGeneration}`}
          mode="streaming"
          parseIncompleteMarkdown={!incrementalRender}
          parseMarkdownIntoBlocksFn={incrementalRender?.parseMarkdownIntoBlocks}
          isAnimating={isStreaming}
          animated={STREAMDOWN_IMMEDIATE_UPDATES}
          plugins={
            isStreaming ? STREAMDOWN_STREAMING_PLUGINS : STREAMDOWN_PLUGINS
          }
          components={STREAMDOWN_COMPONENTS}
          urlTransform={safeMarkdownUrl}
          controls={STREAMDOWN_CONTROLS}
          shikiTheme={STREAMDOWN_SHIKI_THEME}
          BlockComponent={StreamdownBlock}
        >
          {incrementalRender?.markdown ?? processedText}
        </Streamdown>

        </ActiveStreamingBlockContext.Provider>
      )}
    </div>
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
