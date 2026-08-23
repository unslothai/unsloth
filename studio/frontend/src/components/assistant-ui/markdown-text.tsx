// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  ArtifactCard,
  getCodeFence,
  isFullHtmlDocument,
  isHtmlFence,
  isRenderableRenderHtmlToolPart,
  isSvgFence,
  useChatRuntimeStore,
} from "@/features/chat";

import { useT } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { readFencedCodeProvenance } from "@/lib/fenced-code-provenance";
import { preprocessLaTeX } from "@/lib/latex";
import { downloadFile, isDownloadCancelled } from "@/lib/native-files";
import { openLink } from "@/lib/open-link";
import { safeMarkdownUrl } from "@/lib/safe-markdown-url";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import { INTERNAL, useAuiState, useMessagePartText } from "@assistant-ui/react";
import { Copy01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { HighlightResult } from "@streamdown/code";

import { createMathPlugin } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import {
  type ComponentProps,
  type ReactElement,
  type ReactNode,
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Block,
  type BlockProps,
  type CustomRendererProps,
  Streamdown,
  type StreamdownProps,
} from "streamdown";
import {
  DeferredFenceShell,
  fenceMode,
  trimTrailingNewlines,
  useFenceReached,
} from "./code-fence-defer";
import { createCodePlugin } from "./code-plugin";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import { OversizedStreamingCodeBlock } from "./oversized-streaming-code-block";
import {
  createReasoningPageBoundary,
  isReasoningPageBoundaryValid,
  type ReasoningPageBoundary,
  selectReasoningMarkdownPage,
} from "./reasoning-pagination";
import {
  getCodeFenceFilename,
  getStreamingCodeFence,
  isOversizedStreamingCode,
  normalizeCodeFenceLanguage,
} from "./streaming-code-policy";
import { stabilizeStreamingMarkdown } from "./streaming-markdown";
import {
  type IncrementalMarkdownBlock,
  IncrementalMarkdownCache,
  type IncrementalMarkdownChunk,
  type IncrementalMarkdownCodeFence,
  type IncrementalMarkdownRender,
  type IncrementalMarkdownTerminalCodeTail,
  withoutStreamdownAnimationPlugin,
} from "./streaming-render-schedule";
import {
  createStreamingTextPresentationScheduler,
  scheduleAfterPaint,
} from "./streaming-text-presentation";

const math = createMathPlugin({ singleDollarTextMath: true });
const baseCode = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});

type StreamingCodeHighlightObserver = (
  source: string,
  language: string,
) => void;
let streamingCodeHighlightObserver: StreamingCodeHighlightObserver | null =
  null;

// The Chromium harness observes the real plugin boundary; normal bundles install no observer.
export function observeStreamingCodeHighlights(
  observer: StreamingCodeHighlightObserver,
): () => void {
  streamingCodeHighlightObserver = observer;
  return () => {
    if (streamingCodeHighlightObserver === observer) {
      streamingCodeHighlightObserver = null;
    }
  };
}

const code = {
  ...baseCode,
  highlight: (...args: Parameters<typeof baseCode.highlight>) => {
    streamingCodeHighlightObserver?.(args[0].code, String(args[0].language));
    return baseCode.highlight(...args);
  },
} satisfies typeof baseCode;
export type MarkdownCodeHighlighting = "syntax" | "plain";

const PERSISTENT_OVERSIZED_CODE_LANGUAGE = "unsloth-oversized-code";
const STREAMDOWN_SYNTAX_PLUGINS = {
  code,
  math,
  mermaid,
  renderers: [
    {
      component: PersistentOversizedCodeRenderer,
      language: PERSISTENT_OVERSIZED_CODE_LANGUAGE,
    },
  ],
} satisfies NonNullable<StreamdownProps["plugins"]>;
// Without the code plugin, Streamdown still renders code containers and other plugins.
const STREAMDOWN_PLAIN_CODE_PLUGINS = {
  math,
  mermaid,
  renderers: [
    {
      component: PersistentOversizedCodeRenderer,
      language: PERSISTENT_OVERSIZED_CODE_LANGUAGE,
    },
  ],
} satisfies NonNullable<StreamdownProps["plugins"]>;
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

const FINAL_HIGHLIGHT_CHUNK_CHARACTERS = 512;

// Keep open fences plain; highlight completed source in bounded tasks.
const prepareOversizedCodeHighlight = (
  source: string,
  language: string | null,
  onReady: (result: HighlightResult) => void,
): (() => void) => {
  let cancelled = false;
  let cursor = 0;
  let frame: number | null = null;
  let timer: ReturnType<typeof setTimeout> | null = null;

  const schedule = (callback: () => void): void => {
    frame = requestAnimationFrame(() => {
      frame = null;
      timer = setTimeout(() => {
        timer = null;
        callback();
      }, 0);
    });
  };

  const advance = (): void => {
    if (cancelled) return;
    const limit = Math.min(
      source.length,
      cursor + FINAL_HIGHLIGHT_CHUNK_CHARACTERS,
    );
    const newline = source.lastIndexOf("\n", limit);
    cursor = newline >= cursor ? newline + 1 : limit;

    const continueAfterChunk = (result: HighlightResult): void => {
      if (cancelled) return;
      if (cursor >= source.length) {
        onReady(result);
      } else {
        schedule(advance);
      }
    };
    const result = code.highlight(
      {
        code: source.slice(0, cursor),
        // Never the raw info string: ```python startLine=10 matches no grammar
        // and would silently tokenize as plain text.
        language: (normalizeCodeFenceLanguage(language) ??
          "text") as Parameters<typeof code.highlight>[0]["language"],
        themes: STREAMDOWN_SHIKI_THEME,
      },
      continueAfterChunk,
    );
    if (result) continueAfterChunk(result);
  };

  const cancelInitialPaint = scheduleAfterPaint(advance);
  return () => {
    cancelled = true;
    cancelInitialPaint();
    if (frame !== null) cancelAnimationFrame(frame);
    if (timer !== null) clearTimeout(timer);
  };
};
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
            downloadTextFile(getCodeFenceFilename(language), source);
          }}
        >
          <HugeiconsIcon icon={Download01Icon} className="size-icon" />
        </button>
      </div>
    </div>
  );
}

const completedCodeFenceRendererToken = (
  fence: IncrementalMarkdownCodeFence,
): string => `unsloth-fence:${fence.id}`;

const EMPTY_COMPLETED_CODE_FENCES = new Map<
  string,
  IncrementalMarkdownCodeFence
>();
const CompletedCodeFencesContext = createContext<
  ReadonlyMap<string, IncrementalMarkdownCodeFence>
>(EMPTY_COMPLETED_CODE_FENCES);

const EMPTY_CANONICAL_CODE_SOURCES = new Map<string, string>();
const CanonicalCodeSourcesContext = createContext<ReadonlyMap<string, string>>(
  EMPTY_CANONICAL_CODE_SOURCES,
);

const MarkdownCodeHighlightingContext =
  createContext<MarkdownCodeHighlighting>("syntax");

function PersistentOversizedCodeRenderer({
  code: rendererSource,
  isIncomplete,
  meta,
}: CustomRendererProps) {
  const codeFences = useContext(CompletedCodeFencesContext);

  const codeHighlighting = useContext(MarkdownCodeHighlightingContext);
  const withoutParserLineFeed = rendererSource.endsWith("\n")
    ? rendererSource.slice(0, -1)
    : rendererSource;
  // The private metastring contains only this occurrence's deterministic token.
  // Original metadata and source stay in the block-local map and never leak into
  // labels, actions, the DOM, or persisted Markdown.
  const presentation = meta ? codeFences.get(meta) : undefined;
  const source = presentation?.source ?? withoutParserLineFeed;
  const language = presentation?.language ?? null;

  return (
    <div className="relative isolate">
      <OversizedStreamingCodeBlock
        isFenceOpen={false}
        language={language}
        prepareHighlighted={
          codeHighlighting === "syntax"
            ? prepareOversizedCodeHighlight
            : undefined
        }
        source={source}
      />
      <CodeBlockActions
        disabled={isIncomplete}
        language={language}
        source={source}
      />
    </div>
  );
}

const presentCompletedCodeFences = (
  content: string,
  codeFences: readonly IncrementalMarkdownCodeFence[],
): string => {
  let presented = content;
  const oversized = [...codeFences]
    .filter((fence) => isOversizedStreamingCode(fence.source.length))
    .reverse();

  for (const fence of oversized) {
    const lineEnd = content.indexOf("\n", fence.openingOffset);
    const end = lineEnd < 0 ? content.length : lineEnd;
    const openingLine = content.slice(fence.openingOffset, end);
    const marker = openingLine.match(/^( {0,3})(`{3,}|~{3,})/);
    if (!marker) continue;
    const replacement = `${marker[1]}${marker[2]}${PERSISTENT_OVERSIZED_CODE_LANGUAGE} ${completedCodeFenceRendererToken(fence)}`;
    presented =
      presented.slice(0, fence.openingOffset) +
      replacement +
      presented.slice(end);
  }
  return presented;
};

function useAnimationFreeBlockProps(
  props: BlockProps,
  codeFences: readonly IncrementalMarkdownCodeFence[],
): BlockProps {
  // `animated` is needed only to bypass Streamdown's starvable React transition.
  // Its rehype plugin still wraps every word even with duration and stagger set
  // to zero. Remove that one plugin before parsing so long streams do not create
  // thousands of animation spans. Rewrite only completed oversized fence tags
  // to the private renderer language; the context still carries exact source,
  // original language, and metadata for actions and labels.
  const content = useMemo(
    () => presentCompletedCodeFences(props.content, codeFences),
    [codeFences, props.content],
  );
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
    content,
    rehypePlugins,
  } satisfies BlockProps;
}

/**
 * Whether this message carries a renderable render_html tool part, asked once per message part
 * instead of once per markdown block.
 *
 * The value belongs to the MESSAGE, but the block component is mounted per block, so subscribing
 * there minted a subscription per block (800 of 10,193 on the 300K-character heavy thread), each
 * re-scanning `message.parts` on every store update -- and every keystroke is a store update.
 * One subscription in MarkdownTextImpl plus a context read gives the same blocks the same answer.
 *
 * `false` is the right default for a block rendered outside a message part (nothing does today):
 * no render_html part is visible, which is what the artifact collapse below assumes absent
 * evidence.
 */
const RenderHtmlToolPresenceContext = createContext(false);

// Collapse a full-HTML answer in place into an artifact card. Diffusion keeps the
// raw code visible instead (the trailing MessageHtmlArtifacts appends its card).
type StreamdownBlockProps = BlockProps & {
  actionsDisabled?: boolean;
  canonicalCodeSource?: string;
  codeFences?: readonly IncrementalMarkdownCodeFence[];
  isFenceOpen?: boolean;
  sourceContent?: string;
};

function StreamdownBlockContent(props: StreamdownBlockProps): ReactElement;
function StreamdownBlockContent(props: BlockProps) {
  const {
    actionsDisabled,
    canonicalCodeSource,
    codeFences = [],
    isFenceOpen = false,
    sourceContent,
    ...renderProps
  } = props as StreamdownBlockProps;

  const codeHighlighting = useContext(MarkdownCodeHighlightingContext);
  const canonicalCodeSources = useContext(CanonicalCodeSourcesContext);
  const completedCodeFencePresentations = useMemo(
    () =>
      new Map(
        codeFences.map((fence) => [
          completedCodeFenceRendererToken(fence),
          fence,
        ]),
      ),
    [codeFences],
  );
  const blockProps = useAnimationFreeBlockProps(renderProps, codeFences);
  const shouldCollapseHtmlArtifacts = useChatRuntimeStore(
    (state) =>
      (state.artifactsEnabled || state.collapseHtmlArtifacts) &&
      !state.loadedIsDiffusion,
  );
  const messageHasRenderableRenderHtmlTool = useContext(
    RenderHtmlToolPresenceContext,
  );
  const hasMermaidFence = props.content.includes("```mermaid");
  const mermaidSource = getMermaidSource(props.content);
  const codeFence = getCodeFence(props.content);
  const parsedStreamingCodeFence = getStreamingCodeFence(
    sourceContent ?? props.content,
  );
  // A completed code-only provider block takes this direct rendering path rather
  // than the tokenized persistent renderer below. Resolve its canonical record
  // by this block's exact opening offset; semantic mdast source drops the line
  // ending that positions the closer.
  const completedStandaloneFence =
    !props.isIncomplete && parsedStreamingCodeFence
      ? codeFences.find((fence) => fence.openingOffset === 0)
      : undefined;
  const streamingCodeFence =
    completedStandaloneFence ??
    parsedStreamingCodeFence ??
    (codeFence && {
      ...codeFence,
      language: normalizeCodeFenceLanguage(codeFence.language),
    });
  const exactCodeSource =
    canonicalCodeSource ??
    (completedStandaloneFence
      ? canonicalCodeSources.get(completedStandaloneFence.id)
      : undefined) ??
    streamingCodeFence?.source;

  const renderBlock = () => (
    <CompletedCodeFencesContext.Provider
      value={completedCodeFencePresentations}
    >
      <Block {...blockProps} />
    </CompletedCodeFencesContext.Provider>
  );

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

  if (mermaidSource) {
    return (
      <div className="relative isolate">
        {renderBlock()}
        <MermaidCopyButton source={mermaidSource} />
      </div>
    );
  }

  if (streamingCodeFence) {
    const svgSource =
      !isFenceOpen && !props.isIncomplete && isSvgFence(streamingCodeFence)
        ? sanitizeSvg(streamingCodeFence.source)
        : null;
    const htmlSource =
      shouldCollapseHtmlArtifacts &&
      !messageHasRenderableRenderHtmlTool &&
      !isFenceOpen &&
      !props.isIncomplete &&
      isHtmlFence(streamingCodeFence) &&
      isFullHtmlDocument(streamingCodeFence.source)
        ? streamingCodeFence.source
        : null;
    if (htmlSource) {
      return (
        <ArtifactCard code={htmlSource} title="HTML preview" source="fence" />
      );
    }

    return (
      <>
        <FenceBlock
          actionsDisabled={actionsDisabled ?? props.isIncomplete}
          actionsSource={exactCodeSource ?? streamingCodeFence.source}
          isIncomplete={props.isIncomplete}
          language={streamingCodeFence.language}
          renderBlock={renderBlock}
          renderPlainBody={
            codeHighlighting === "plain" ||
            isOversizedStreamingCode(exactCodeSource?.length ?? 0)
              ? () => (
                  <OversizedStreamingCodeBlock
                    isFenceOpen={isFenceOpen}
                    language={streamingCodeFence.language}
                    prepareHighlighted={
                      codeHighlighting === "syntax"
                        ? prepareOversizedCodeHighlight
                        : undefined
                    }
                    source={streamingCodeFence.source}
                  />
                )
              : null
          }
          source={streamingCodeFence.source}
        />
        {svgSource && <SvgPreview source={svgSource} />}
      </>
    );
  }

  return renderBlock();
}

/*
 * The fence branch, extracted so the reach latch can be a hook.
 *
 * With the flag off this renders exactly what the branch rendered before: the
 * same `relative isolate` wrapper, the same `<Block>`, the same action bar. The
 * wrapper is reused as the intersection target rather than a new one being
 * introduced, so the DOM the off arm produces is byte-for-byte what main
 * produces and the on arm differs only in what is INSIDE the wrapper.
 *
 * `renderPlainBody` is the streaming policy's already-plain body (an oversized
 * fence, or the reasoning pane's plain-code policy). It composes rather than
 * competes: that body carries no spans to defer, so the reach gate is switched
 * off for it and every other fence keeps deferral exactly as before.
 */
function FenceBlock({
  actionsDisabled,
  actionsSource,
  isIncomplete,
  language,
  renderBlock,
  renderPlainBody,
  source,
}: {
  actionsDisabled: boolean | undefined;
  actionsSource: string;
  isIncomplete: boolean | undefined;
  language: string | null;
  renderBlock: () => ReactNode;
  renderPlainBody: (() => ReactNode) | null;
  source: string;
}) {
  const host = useRef<HTMLDivElement | null>(null);
  const mode = fenceMode();
  // A streaming fence is the one the reader is watching, so it never defers, and the hook latches
  // it so that finishing the stream cannot hand it back the plain shell.
  const reached = useFenceReached(
    host,
    mode !== "off" && renderPlainBody === null,
    Boolean(isIncomplete),
  );

  /*
   * WHICH FENCES THIS COVERS, and which it does not.
   *
   * `CODE_FENCE_RE` accepts exactly three backticks, unindented. CommonMark also allows tildes,
   * four or more backticks (which is how a model writes a fence whose body contains one), and up
   * to three spaces of indent. Those forms never reach here, so they render exactly as they do
   * today and get no deferral: unrealised benefit, not a wrong result.
   *
   * Left alone deliberately rather than overlooked. `getCodeFence` is also what decides whether a
   * block is an SVG or a full HTML document to be shown as an artifact, and a fence that does not
   * match it renders a bare `<Block>` with no `relative isolate` wrapper and no copy button.
   * Widening the regex would therefore add an artifact path and a copy overlay to blocks that do
   * not have them today, which is a rendering change, and a performance PR is the wrong place to
   * smuggle one in.
   *
   * It also does not move any number here: over the frozen corpus, 2,467,069 characters, all
   * 1,456 fence delimiters are unindented triple backticks. Not one tilde, not one four-backtick
   * fence, not one indented one.
   */
  // `getCodeFence` hands back the WHOLE info string, so a fence opened with metadata such as
  // ```python startLine=10 arrives here as "python startLine=10". Markdown treats everything
  // after the first word as metadata and Streamdown highlights it as `python`, so passing the
  // raw string on would label the shell with the metadata attached and, in the measurement arm,
  // tokenize an unknown language as plain text -- which is exactly the grammar work the arm
  // exists to put back.
  const languageToken = language?.trim().split(/\s+/)[0] || null;

  // MEASUREMENT ARM ONLY. See `FenceMode`: this puts the tokenizer work back while leaving the
  // document at the deferred size, so the two costs can be told apart. `code.highlight` caches
  // on the source string, so the work happens exactly once and the discarded result is the same
  // object the real path would have used.
  const pretokenize = mode === "tokenize" && !reached && renderPlainBody === null;
  useEffect(() => {
    if (!pretokenize) return;
    code.highlight({
      code: trimTrailingNewlines(source),
      language: (languageToken ?? "text") as never,
      themes: STREAMDOWN_SHIKI_THEME,
    }, () => {});
  }, [pretokenize, source, languageToken]);

  return (
    <div className="relative isolate" ref={host}>
      {renderPlainBody !== null ? (
        renderPlainBody()
      ) : reached ? (
        renderBlock()
      ) : (
        <DeferredFenceShell language={languageToken} source={source} />
      )}
      <CodeBlockActions
        disabled={Boolean(actionsDisabled)}
        language={language}
        source={actionsSource}
      />
    </div>
  );
}
const StreamdownBlock = memo(StreamdownBlockContent);
const StreamingMarkdownPlanContext =
  createContext<IncrementalMarkdownRender | null>(null);

const parseProviderShellBlock = (markdown: string): string[] => [markdown];

type StableCommittedChunkProps = {
  chunk: IncrementalMarkdownChunk;
  shellProps: BlockProps;
};

function StableCommittedChunkContent({
  chunk,
  shellProps,
}: StableCommittedChunkProps) {
  return chunk.blocks.map((block, index) => (
    <StreamdownBlock
      {...shellProps}
      codeFences={block.codeFences}
      key={block.id}
      content={block.content}
      index={chunk.startIndex + index}
      isIncomplete={false}
    />
  ));
}

// The provider shell creates a fresh props object when its one live block
// changes. Completed chunks deliberately ignore that wrapper identity: their
// Block configuration is constant for the lifetime of MarkdownText, and
// Streamdown's own contexts still propagate real completion/config changes to
// descendants. Only a changed chunk object should walk its completed blocks.
const StableCommittedChunk = memo(
  StableCommittedChunkContent,
  (previous, next) => previous.chunk === next.chunk,
);

type StableCommittedChunksProps = {
  chunks: readonly IncrementalMarkdownChunk[];
  shellProps: BlockProps;
};

function StableCommittedChunksContent({
  chunks,
  shellProps,
}: StableCommittedChunksProps) {
  return chunks.map((chunk) => (
    <StableCommittedChunk
      key={chunk.id}
      chunk={chunk}
      shellProps={shellProps}
    />
  ));
}

// Ordinary token updates leave the chunks array untouched, so React does not
// even map the growing committed list. A promotion changes the array, while the
// chunk memo above still keeps every closed chunk out of that render.
const StableCommittedChunks = memo(
  StableCommittedChunksContent,
  (previous, next) => previous.chunks === next.chunks,
);

type StableTerminalPrefixProps = {
  blocks: readonly IncrementalMarkdownBlock[];
  index: number;
  shellProps: BlockProps;
};

function StableTerminalPrefixContent({
  blocks,
  index,
  shellProps,
}: StableTerminalPrefixProps) {
  return blocks.map((block, blockIndex) => (
    <StreamdownBlock
      {...shellProps}
      codeFences={block.codeFences}
      key={block.id}
      content={block.content}
      index={index + blockIndex}
      isIncomplete={false}
    />
  ));
}

const StableTerminalPrefix = memo(
  StableTerminalPrefixContent,
  (previous, next) => previous.blocks === next.blocks,
);

type TerminalCodeTailProps = {
  codeTail: IncrementalMarkdownTerminalCodeTail;
  index: number;
  shellProps: BlockProps;
};

function TerminalCodeTail({
  codeTail,
  index,
  shellProps,
}: TerminalCodeTailProps) {
  const isFenceOpen = !codeTail.isClosed;
  const isStreaming = shellProps.isIncomplete;
  const actionsDisabled = isFenceOpen && isStreaming;
  return (
    <>
      <StableTerminalPrefix
        blocks={codeTail.prefixBlocks}
        index={index}
        shellProps={shellProps}
      />
      {/* Syntax keeps the fence plain after stop/cold load; transport state alone
          controls whether exact-source actions are temporarily disabled. */}
      <StreamdownBlock
        {...shellProps}
        key={codeTail.id}
        canonicalCodeSource={codeTail.source}
        content={codeTail.fenceMarkdown}
        sourceContent={codeTail.fenceMarkdown}
        index={index + codeTail.prefixBlocks.length}
        actionsDisabled={actionsDisabled}
        isFenceOpen={isFenceOpen}
        isIncomplete={actionsDisabled}
      />
    </>
  );
}

function PartitionedStreamdownBlock(shellProps: BlockProps) {
  const plan = useContext(StreamingMarkdownPlanContext);
  if (!plan) {
    return null;
  }

  return (
    <>
      <StableCommittedChunks chunks={plan.chunks} shellProps={shellProps} />
      {plan.tail.map((block, index) => {
        const codeTail = plan.terminalCodeTail;
        if (codeTail?.blockId === block.id) {
          return (
            <TerminalCodeTail
              key={codeTail.id}
              codeTail={codeTail}
              index={plan.committedBlockCount + index}
              shellProps={shellProps}
            />
          );
        }
        return (
          <StreamdownBlock
            {...shellProps}
            codeFences={block.codeFences}
            key={block.id}
            content={block.content}
            sourceContent={
              index === plan.tail.length - 1
                ? plan.sourceShellMarkdown
                : undefined
            }
            index={plan.committedBlockCount + index}
            isIncomplete={
              index === plan.tail.length - 1 && shellProps.isIncomplete
            }
          />
        );
      })}
    </>
  );
}

const PartitionedStreamdownShell = memo(PartitionedStreamdownBlock);
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

// Coalesce ordinary token events to paint. Once a cumulative reply reaches
// 24 KiB, publish at 75 ms intervals so reconciliation/mutation work stays in
// the smooth 12-15 Hz band measured by the real trace. Completion, message
// switches, and replacements still return their exact source immediately.
function useCoalescedStreamingText(
  text: string,
  isStreaming: boolean,
  messageId: string,
): string {
  const [displayed, setDisplayed] = useState({ messageId, text });
  const activeMessageIdRef = useRef(messageId);
  const [scheduler] = useState(() =>
    createStreamingTextPresentationScheduler({
      publish: setDisplayed,
      now: () => performance.now(),
      requestFrame: (callback) => requestAnimationFrame(callback),
      cancelFrame: (handle) => cancelAnimationFrame(handle),
      setTimer: (callback, delay) => setTimeout(callback, delay),
      clearTimer: (handle) => clearTimeout(handle),
    }),
  );

  useEffect(() => {
    const pending = { messageId, text };
    if (activeMessageIdRef.current !== messageId) {
      scheduler.cancel();
      activeMessageIdRef.current = messageId;
    }
    if (!isStreaming) {
      scheduler.flush(pending);
      return;
    }
    scheduler.schedule(text.length, pending);
  }, [isStreaming, messageId, scheduler, text]);

  useEffect(() => () => scheduler.cancel(), [scheduler]);

  // Holding the last painted text is only correct while the reply is being
  // appended to. A running message can also be replaced, as the audio path does
  // when it swaps its placeholder for the player, and that must show at once.
  // The length check rejects most of those before the prefix scan runs; the
  // scan itself costs about 59 ms across a 175,000 character stream.
  if (
    isStreaming &&
    displayed.messageId === messageId &&
    text.length >= displayed.text.length &&
    // Not startsWith, which scans a growing reply. See hasPrefix in
    // streaming-render-schedule.ts for the measurement.
    text.slice(0, displayed.text.length) === displayed.text
  ) {
    return displayed.text;
  }
  return text;
}

type PartitionedMarkdownTextProps = {
  codeHighlighting: MarkdownCodeHighlighting;
  isStreaming: boolean;
  markdown: string;
  messageId: string;
  paginateReasoning: boolean;
  persistedTrailingLfOrdinals: readonly number[];
  statusType: string;
};

const EMPTY_FENCE_PROVENANCE: readonly number[] = [];

function PartitionedMarkdownText({
  isStreaming,
  markdown,
  messageId,
  statusType,
  persistedTrailingLfOrdinals,
  codeHighlighting,
  paginateReasoning,
}: PartitionedMarkdownTextProps) {
  const [incrementalCache] = useState(
    () => new IncrementalMarkdownCache(persistedTrailingLfOrdinals),
  );
  const t = useT();
  const [reasoningPageBoundaries, setReasoningPageBoundaries] = useState<
    readonly ReasoningPageBoundary[]
  >([]);
  const pageHistoryInvalid = reasoningPageBoundaries.some(
    (boundary) => !isReasoningPageBoundaryValid(markdown, boundary),
  );
  if (pageHistoryInvalid) setReasoningPageBoundaries([]);
  const reasoningPageEnd = pageHistoryInvalid
    ? null
    : (reasoningPageBoundaries.at(-1)?.end ?? null);
  const reasoningPage = useMemo(
    () =>
      selectReasoningMarkdownPage(markdown, {
        enabled: paginateReasoning,
        end: reasoningPageEnd,

        streaming: isStreaming && reasoningPageEnd === null,
      }),
    [isStreaming, markdown, paginateReasoning, reasoningPageEnd],
  );
  const showEarlierReasoning = useCallback(() => {
    if (!reasoningPage.hasEarlier) return;
    setReasoningPageBoundaries((current) => [
      ...current,
      createReasoningPageBoundary(markdown, reasoningPage.start),
    ]);
  }, [markdown, reasoningPage.hasEarlier, reasoningPage.start]);
  const showNewerReasoning = useCallback(() => {
    setReasoningPageBoundaries((current) => current.slice(0, -1));
  }, []);

  // An older page is immutable even if the live tail keeps growing. Only the
  // latest page participates in streaming repair and presentation cadence.
  const pageIsStreaming = isStreaming && !reasoningPage.hasNewer;
  const pageUsesSourceSlice =
    reasoningPage.start > 0 || reasoningPage.end < markdown.length;
  const incrementalRender = incrementalCache.update(
    reasoningPage.markdown,
    pageIsStreaming,
    pageUsesSourceSlice ? EMPTY_FENCE_PROVENANCE : persistedTrailingLfOrdinals,
  );
  const canonicalCodeSources = useMemo(() => {
    if (!reasoningPage.canonicalCodeSources.some(Boolean)) {
      return EMPTY_CANONICAL_CODE_SOURCES;
    }
    const fences = [
      ...incrementalRender.chunks.flatMap((chunk) =>
        chunk.blocks.flatMap((block) => block.codeFences),
      ),
      ...incrementalRender.tail.flatMap((block) => block.codeFences),
    ];
    const sources = new Map<string, string>();
    reasoningPage.canonicalCodeSources.forEach((source, index) => {
      const fence = fences[index];
      if (source && fence) sources.set(fence.id, source);
    });
    return sources;
  }, [incrementalRender, reasoningPage.canonicalCodeSources]);

  return (
    <div data-status={statusType} className="min-w-0 max-w-full">
      {reasoningPage.hasEarlier && (
        <button
          type="button"
          data-slot="reasoning-show-earlier"
          className="mb-4 w-full cursor-pointer rounded-lg border border-border/60 bg-muted/20 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground"
          onClick={showEarlierReasoning}
        >
          {t("shell.navigation.showMore")}
        </button>
      )}
      <CanonicalCodeSourcesContext.Provider value={canonicalCodeSources}>
        <MarkdownCodeHighlightingContext.Provider value={codeHighlighting}>
          <StreamingMarkdownPlanContext.Provider value={incrementalRender}>
            <Streamdown
              key={messageId}
              mode="streaming"
              parseIncompleteMarkdown={false}
              parseMarkdownIntoBlocksFn={parseProviderShellBlock}
              isAnimating={pageIsStreaming}
              animated={STREAMDOWN_IMMEDIATE_UPDATES}
              plugins={
                codeHighlighting === "syntax"
                  ? STREAMDOWN_SYNTAX_PLUGINS
                  : STREAMDOWN_PLAIN_CODE_PLUGINS
              }
              components={STREAMDOWN_COMPONENTS}
              urlTransform={safeMarkdownUrl}
              controls={STREAMDOWN_CONTROLS}
              shikiTheme={STREAMDOWN_SHIKI_THEME}
              BlockComponent={PartitionedStreamdownShell}
            >
              {incrementalRender.shellMarkdown}
            </Streamdown>
          </StreamingMarkdownPlanContext.Provider>
        </MarkdownCodeHighlightingContext.Provider>
      </CanonicalCodeSourcesContext.Provider>
      {reasoningPage.hasNewer && (
        <button
          type="button"
          data-slot="reasoning-show-newer"
          className="mt-4 w-full cursor-pointer rounded-lg border border-border/60 bg-muted/20 px-3 py-2 text-left text-xs text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground"
          onClick={showNewerReasoning}
        >
          {t("shell.navigation.showLess")}
        </button>
      )}
    </div>
  );
}

type MarkdownTextProps = {
  // Keep this renderer assignable to assistant-ui's text/reasoning component
  // contracts while exposing stable presentation policies to direct callers.
  status?: unknown;
  text?: string;
  codeHighlighting?: MarkdownCodeHighlighting;
  paginateReasoning?: boolean;
};

const MarkdownTextImpl = ({
  codeHighlighting = "syntax",
  paginateReasoning = false,
}: MarkdownTextProps) => {
  const { text, status } = useMessagePartText();
  // Parts are keyed by index, so switching conversations hands this instance a
  // different message. Key the renderer so that message gets its own cache and
  // provider shell, while completion of one message keeps both intact.
  const messageId = useAuiState(({ message }) => message.id);
  // Read once here for every block below: see RenderHtmlToolPresenceContext.
  const persistedFenceProvenance = useAuiState(({ part }) =>
    part.type === "text"
      ? (part as { __unslothFenceProvenance?: unknown })
          .__unslothFenceProvenance
      : undefined,
  );
  const persistedTrailingLfOrdinals = useMemo(
    () =>
      persistedFenceProvenance === undefined
        ? EMPTY_FENCE_PROVENANCE
        : readFencedCodeProvenance({
            __unslothFenceProvenance: persistedFenceProvenance,
          }),
    [persistedFenceProvenance],
  );
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  const isStreaming = status.type === "running";
  const displayText = useCoalescedStreamingText(text, isStreaming, messageId);
  const processedText = useMemo(
    () => stabilizeStreamingMarkdown(preprocessLaTeX(displayText), isStreaming),
    [displayText, isStreaming],
  );

  const audioMatch = displayText.match(AUDIO_PLAYER_RE);
  if (audioMatch) {
    return <AudioPlayer src={audioMatch[1]} />;
  }

  return (
    <RenderHtmlToolPresenceContext.Provider
      value={messageHasRenderableRenderHtmlTool}
    >
      <PartitionedMarkdownText
        key={messageId}
        codeHighlighting={codeHighlighting}
        isStreaming={isStreaming}
        markdown={processedText}
        messageId={messageId}
        statusType={status.type}
        persistedTrailingLfOrdinals={persistedTrailingLfOrdinals}
        paginateReasoning={paginateReasoning}
      />
    </RenderHtmlToolPresenceContext.Provider>
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
