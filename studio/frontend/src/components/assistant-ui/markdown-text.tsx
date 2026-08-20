


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
import { Copy01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { createMathPlugin } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
import {
  type ComponentProps,
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
  Streamdown,
  type StreamdownProps,
} from "streamdown";
import { createCodePlugin } from "./code-plugin";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import { stabilizeStreamingMarkdown } from "./streaming-markdown";
import {
  IncrementalMarkdownCache,
  withoutStreamdownAnimationPlugin,
} from "./streaming-render-schedule";

const math = createMathPlugin({ singleDollarTextMath: true });
const code = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});
const STREAMDOWN_PLUGINS = { code, math, mermaid } satisfies NonNullable<
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
function StreamdownBlockContent(props: BlockProps) {
  const blockProps = useAnimationFreeBlockProps(props);
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

// Coalesce only token events that arrive before the browser's next paint, as
// textgen does. There is no time or length throttle. Incremental block parsing
// bounds the work performed per paint, and completion returns immediately.
function useCoalescedStreamingText(
  text: string,
  isStreaming: boolean,
  messageId: string,
): string {
  const [displayed, setDisplayed] = useState({ messageId, text });
  const pendingRef = useRef({ messageId, text });
  const rafRef = useRef<number | null>(null);
  const activeMessageIdRef = useRef(messageId);

  const cancelScheduledRender = useCallback(() => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  useEffect(() => {
    pendingRef.current = { messageId, text };
    if (activeMessageIdRef.current !== messageId) {
      cancelScheduledRender();
      activeMessageIdRef.current = messageId;
    }
    if (!isStreaming) {
      cancelScheduledRender();
      return;
    }

    if (rafRef.current !== null) {
      return;
    }

    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setDisplayed(pendingRef.current);
    });
  }, [cancelScheduledRender, messageId, text, isStreaming]);

  useEffect(() => {
    return cancelScheduledRender;
  }, [cancelScheduledRender]);

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

const MarkdownTextImpl = () => {
  const { text, status } = useMessagePartText();
  // Parts are keyed by index, so switching conversations hands this instance a
  // different message, and Streamdown only extends its parsed blocks: key it per
  // message. The cache generation joins the key for the case the Markdown string
  // cannot express, an edit that drops retained blocks without changing the tail.
  const messageId = useAuiState(({ message }) => message.id);
  // Read once here for every block below: see RenderHtmlToolPresenceContext.
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  const isStreaming = status.type === "running";
  const displayText = useCoalescedStreamingText(text, isStreaming, messageId);
  const processedText = useMemo(
    () => stabilizeStreamingMarkdown(preprocessLaTeX(displayText), isStreaming),
    [displayText, isStreaming],
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
  const incrementalRender = isStreaming
    ? incrementalCache.update(processedText)
    : null;

  const audioMatch = displayText.match(AUDIO_PLAYER_RE);
  if (audioMatch) {
    return <AudioPlayer src={audioMatch[1]} />;
  }

  return (
    <RenderHtmlToolPresenceContext.Provider
      value={messageHasRenderableRenderHtmlTool}
    >
      <div data-status={status.type} className="min-w-0 max-w-full">
        <Streamdown
          key={`${messageId}:${incrementalCache.renderGeneration}`}
          mode="streaming"
          parseIncompleteMarkdown={!incrementalRender}
          parseMarkdownIntoBlocksFn={incrementalRender?.parseMarkdownIntoBlocks}
          isAnimating={isStreaming}
          animated={STREAMDOWN_IMMEDIATE_UPDATES}
          plugins={STREAMDOWN_PLUGINS}
          components={STREAMDOWN_COMPONENTS}
          urlTransform={safeMarkdownUrl}
          controls={STREAMDOWN_CONTROLS}
          shikiTheme={STREAMDOWN_SHIKI_THEME}
          BlockComponent={StreamdownBlock}
        >
          {incrementalRender?.markdown ?? processedText}
        </Streamdown>
      </div>
    </RenderHtmlToolPresenceContext.Provider>
  );
};

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
