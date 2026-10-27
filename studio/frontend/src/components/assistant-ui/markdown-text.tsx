// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { openImageViewer } from "@/components/image-viewer";
import { filesOpenInBrowser, openFileInBrowser } from "@/features/browser";
import { useT } from "@/i18n";
import {
  ArtifactCard,
  readIncompleteInfo,
  useChatProjectScope,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  getCodeFence,
  isFullHtmlDocument,
  isHtmlFence,
  isSvgFence,
} from "@/features/chat/artifacts/html-fences";
// Leaf module, not the barrel: SEARCH_IMAGE_TAG is read at module scope and the barrel cycles (TDZ).
// eslint-disable-next-line no-restricted-imports
import {
  holdBackPartialSearchImageToken,
  parseSearchImagesSignature,
  placeSubjectImages,
  rewriteSearchImageTokens,
  SEARCH_IMAGE_TAG,
} from "@/features/chat/search-images/search-images";
import {
  partsHaveRenderableRenderHtmlTool,
  partsPrecedingText,
  partsSearchImagesSignature,
  partsTextKey,
} from "@/components/assistant-ui/message-derived";
import { useMessagePartTextSafe } from "@/components/assistant-ui/stale-safe-scopes";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { normalizeEscapedInlineMath } from "@/lib/escaped-inline-math";
import { preprocessLaTeX } from "@/lib/latex";
import { withDataImageSupport } from "@/lib/markdown-data-images";
import { downloadFile, isDownloadCancelled, urlToBlob } from "@/lib/native-files";
import { openLink } from "@/lib/open-link";
import { safeMarkdownUrl } from "@/lib/safe-markdown-url";
import { Tick02Icon } from "@/lib/tick-icon";
import { toast } from "@/lib/toast";
import {
  INTERNAL,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { createMathPlugin } from "@streamdown/math";
import {
  type ComponentProps,
  type ReactNode,
  createContext,
  memo,
  useCallback,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import {
  Block,
  type BlockProps,
  Streamdown,
  type StreamdownProps,
} from "streamdown";
import {
  awaitWorker,
  DeferredFenceShell,
  FenceBody,
  type FenceTokens,
  fenceMode,
  fenceNeedsQuietWork,
  noteStreamingFence,
  PlainFenceShell,
  queueFenceWork,
  trimmedLength,
  trimTrailingNewlines,
  useFenceReached,
} from "./code-fence-defer";
import {
  highlightWorkerState,
  requestFullHighlight,
} from "./use-reasoning-highlight";
import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { markdownBlockFallback } from "./markdown-block-fallback";
import { createCodePlugin } from "./code-plugin";
import { mermaid } from "./mermaid-plugin";
import { CodeHighlightModeContext } from "./code-highlight-context";
import { spellLines } from "./reasoning-code-colour";
import { useThinkingCodeColour } from "./use-thinking-code-colour";
import { withMathBlockMarker } from "./math-block-marker";
import {
  MarkdownBlockBoundary,
  MarkdownBlockFallbackView,
  MarkdownRendererBoundary,
} from "./markdown-block-boundary";
import "katex/dist/katex.min.css";
import { AudioPlayer } from "./audio-player";
import {
  type ContextFile,
  FileContextMenu,
  loadSandboxFile,
  WebLinkContextMenu,
} from "./link-context-menu";
import {
  decodeSegment,
  markdownSandboxImageSrc,
  sandboxFileForHref,
  sandboxSessionIdFor,
  sandboxSessionInSrc,
} from "./sandbox-files";
import { MARKDOWN_TABLE_ICONS, MarkdownTable } from "./markdown-table";
import { SearchImageElement, SearchImagesContext } from "./search-image";
import { useSandboxImage } from "./use-sandbox-image";
import { rehypeSandboxImages } from "./rehype-sandbox-images";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import { stabilizeStreamingMarkdown } from "./streaming-markdown";
import {
  createDiscreteInputTracker,
  type DiscreteInputTracker,
} from "./discrete-input";
import {
  FINISHED_TEXT_REMEND,
  IncrementalMarkdownCache,
  LITERAL_LINK_REMEND,
  hasIncompleteLinkRepair,
  markdownRenderKey,
  parseMarkdownIntoRenderableBlocks,
  withoutStreamdownAnimationPlugin,
} from "./streaming-render-schedule";

const baseMath = createMathPlugin({ singleDollarTextMath: true });
// Composed onto the maths plugin's rehype pass: the only hook after Streamdown's sanitizer, which
// strips unknown classes. See math-block-marker.ts.
const math = {
  ...baseMath,
  rehypePlugin: withMathBlockMarker(baseMath.rehypePlugin),
} satisfies typeof baseMath;
const code = createCodePlugin({
  themes: [unslothLightTheme, unslothDarkTheme],
});
const STREAMDOWN_PLUGINS = { code, math, mermaid } satisfies NonNullable<
  StreamdownProps["plugins"]
>;
const STREAMDOWN_PLAIN_CODE_PLUGINS = { math, mermaid } satisfies NonNullable<
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

const STREAMING_RENDER_DUTY = 3;
const STREAMING_INPUT_YIELDS = 3;

type InputPendingNavigator = Navigator & {
  scheduling?: {
    isInputPending?: (options?: { includeContinuous?: boolean }) => boolean;
  };
};

let discreteInput: DiscreteInputTracker | null = null;

const inputPending = (): boolean => {
  const scheduling = (navigator as InputPendingNavigator).scheduling;
  if (scheduling?.isInputPending) {
    return scheduling.isInputPending();
  }
  discreteInput ??= createDiscreteInputTracker(document);
  return discreteInput.pending();
};

const afterFrame = (callback: () => void): void => {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => {
    channel.port1.close();
    callback();
  };
  channel.port2.postMessage(null);
};

// Streamdown 2.5 schedules ordinary streaming blocks in an interruptible React transition, and a continuous token
// stream can starve that transition for seconds. Its animated path commits every block update directly.
// StreamdownBlock removes the animation transformer while retaining this direct scheduling path.
const STREAMDOWN_IMMEDIATE_UPDATES = {
  duration: 0,
  stagger: 0,
} satisfies NonNullable<StreamdownProps["animated"]>;

/*
 * Registering `img` replaces Streamdown's renderer wholesale, so its behaviour is restated here. A sandbox
 * image src carries no auth header and would 401, so it goes through an authed fetch to an object URL.
 */
const MarkdownImage = memo(function MarkdownImage(props: ComponentProps<"img">) {
  // `node` is Streamdown's ExtraProps; it must not reach the DOM.
  const {
    src,
    alt = "",
    className,
    onLoad,
    onError,
    node: _node,
    ...dom
  } = props as ComponentProps<"img"> & { node?: unknown };
  // Fallback scope for a src that records no session; one that does keeps it.
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const projectId = useChatProjectScope();
  const file = src
    ? markdownSandboxImageSrc(src, {
        threadId: remoteId ?? activeThreadId ?? undefined,
        projectId,
      })
    : null;
  const sandbox = useSandboxImage(file);
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  // A sandbox src stays absent, never raw, until the authed fetch produces a blob.
  const resolved =
    file === null
      ? src
      : sandbox.state.status === "loaded"
        ? sandbox.state.url
        : undefined;
  const failedNow =
    (resolved != null && failedSrc === resolved) ||
    (file !== null && sandbox.state.status === "failed");
  const sized = dom.width != null || dom.height != null;
  // A real extension on the path wins over alt; otherwise infer from blob type. Split before decoding
  // so a raw `#`/`?` keeps its delimiter meaning.
  const downloadName = (blobType: string): string => {
    const tail =
      decodeSegment((file ?? src ?? "").split(/[?#]/)[0].split("/").pop() ?? "") || "";
    const dot = tail.lastIndexOf(".");
    if (dot > -1 && tail.length - dot - 1 <= 4) return tail;
    const ext = /jpe?g/.test(blobType)
      ? "jpg"
      : blobType.includes("svg")
        ? "svg"
        : blobType.includes("gif")
          ? "gif"
          : blobType.includes("webp")
            ? "webp"
            : "png";
    return `${(alt || tail || "image").replace(/\.[^/.]+$/, "")}.${ext}`;
  };
  return (
    <span
      data-streamdown="image-wrapper"
      className="group relative my-4 inline-block"
    >
      <img
        ref={file === null ? undefined : sandbox.ref}
        data-streamdown="image"
        src={resolved}
        alt={alt}
        decoding="async"
        onLoad={(event) => {
          setFailedSrc(null);
          onLoad?.(event);
        }}
        onError={(event) => {
          setFailedSrc(resolved ?? null);
          onError?.(event);
        }}
        className={`max-w-full cursor-zoom-in rounded-lg ${failedNow && !sized ? "hidden" : ""} ${className ?? ""}`}
        {...dom}
        onClick={(event) => {
          // A linked image is the link's: the click opens that, not the viewer.
          if (failedNow || !resolved || event.currentTarget.closest("a")) return;
          const title = alt || "Image";
          openImageViewer([
            {
              key: resolved,
              title,
              fileName: downloadName,
              load: () =>
                file !== null && sandbox.state.status === "loaded"
                  ? Promise.resolve(sandbox.state.blob)
                  : urlToBlob(resolved),
              fallbackUrl: /^https?:/i.test(resolved) ? resolved : undefined,
            },
          ]);
        }}
      />
      {failedNow && (
        <span
          data-streamdown="image-fallback"
          className="text-muted-foreground text-xs italic"
        >
          Image not available
        </span>
      )}
      <span className="pointer-events-none absolute inset-0 hidden rounded-lg bg-black/10 group-hover:block" />
      {!failedNow && resolved ? (
        <button
          type="button"
          title="Download image"
          className="aui-chat-icon-btn absolute right-2 bottom-2 flex h-8 w-8 cursor-pointer items-center justify-center rounded-full bg-background/90 opacity-0 backdrop-blur-sm transition-all duration-200 group-hover:opacity-100"
          onClick={async () => {
            // Reuse fetched bytes under the desktop CSP.
            try {
              const blob =
                file !== null && sandbox.state.status === "loaded"
                  ? sandbox.state.blob
                  : await urlToBlob(resolved);
              await downloadFile(blob, downloadName(blob.type), blob.type);
            } catch (error) {
              if (!isDownloadCancelled(error)) toast.error("Could not save file.");
            }
          }}
        >
          <HugeiconsIcon
            icon={Download01Icon}
            strokeWidth={1.75}
            className="size-icon"
          />
        </button>
      ) : null}
    </span>
  );
});

const LINK_CLASS =
  "text-foreground underline underline-offset-2 decoration-foreground/40 hover:decoration-foreground transition-colors cursor-pointer";

function MarkdownLink({ href, children, ...props }: ComponentProps<"a">) {
  const { node: _node, ...dom } = props as ComponentProps<"a"> & { node?: unknown };
  const file = href ? sandboxFileForHref(href) : null;
  // Only file links subscribe to the chat's scope; links re-render per streamed token.
  if (href && file !== null) {
    return (
      <SandboxFileLink href={href} file={file} dom={dom}>
        {children}
      </SandboxFileLink>
    );
  }
  const link = (
    <a
      href={href}
      rel="noopener noreferrer"
      className={LINK_CLASS}
      onClick={(e) => {
        if (href && openLink(href)) {
          e.preventDefault();
        }
      }}
      {...dom}
    >
      {children}
    </a>
  );
  return href ? <WebLinkContextMenu href={href}>{link}</WebLinkContextMenu> : link;
}

function SandboxFileLink({
  href,
  file,
  dom,
  children,
}: {
  href: string;
  file: string;
  dom: Omit<ComponentProps<"a">, "href" | "children">;
  children: ReactNode;
}) {
  const t = useT();
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const projectId = useChatProjectScope();
  const sessionId =
    sandboxSessionInSrc(href) ??
    sandboxSessionIdFor(remoteId ?? activeThreadId ?? undefined, projectId);
  if (sessionId) {
    const target: ContextFile = {
      name: file.slice(file.lastIndexOf("/") + 1),
      load: () => loadSandboxFile(sessionId, file),
      sandbox: { sessionId, file },
    };
    const openFile = () => {
      if (filesOpenInBrowser()) {
        void target
          .load()
          .then((blob) =>
            openFileInBrowser({ blob, name: target.name, contentType: blob.type, key: `sandbox:${sessionId}:${file}` }),
          )
          .catch(() => toast.error(t("linkMenu.openFailed", { name: target.name })));
        return;
      }
      void target
        .load()
        .then((blob) => downloadFile(blob, target.name, blob.type || undefined))
        .catch((error) => {
          if (!isDownloadCancelled(error)) toast.error(t("linkMenu.saveFailed"));
        });
    };
    return (
      <FileContextMenu file={{ ...target, open: openFile }}>
        <a
          href={href}
          className={LINK_CLASS}
          onClick={(event) => {
            event.preventDefault();
            openFile();
          }}
          {...dom}
        >
          {children}
        </a>
      </FileContextMenu>
    );
  }
  return (
    <a
      href={href}
      rel="noopener noreferrer"
      className={LINK_CLASS}
      onClick={(e) => {
        if (openLink(href)) e.preventDefault();
      }}
      {...dom}
    >
      {children}
    </a>
  );
}

const STREAMDOWN_COMPONENTS = {
  a: MarkdownLink,
  // Module-scoped: Streamdown's memo comparator ignores `components`.
  [SEARCH_IMAGE_TAG]: SearchImageElement,
  img: MarkdownImage,
  table: MarkdownTable,
};
const STREAMDOWN_ALLOWED_TAGS = {
  [SEARCH_IMAGE_TAG]: ["token"],
} satisfies NonNullable<StreamdownProps["allowedTags"]>;

const COPY_RESET_MS = 2000;
const ACTION_PANEL_CLASS =
  "pointer-events-auto flex shrink-0 items-center gap-1";
const ACTION_BUTTON_CLASS =
  "aui-chat-icon-btn flex size-8 cursor-pointer items-center justify-center rounded-full text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover disabled:cursor-not-allowed disabled:opacity-50";

/** Mermaid fence found with fence context, so a ~~~mermaid example inside an outer fence is not a diagram. */
type MermaidFence =
  | { open: true }
  | { open: false; indent: string; body: string };

function findMermaidFence(blockContent: string): MermaidFence {
  const lines = blockContent.split("\n");
  let enclosing: { char: string; run: number } | null = null;
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const match = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
    if (!match) continue;
    const [, marker, rest] = match;
    const isClose =
      rest.replace(/[\t ]*\r?$/, "") === "" && marker.length >= 3;
    if (enclosing === null) {
      if (marker[0] === "`" && rest.includes("`")) continue;
      if (/^[\t ]*mermaid\b/i.test(rest)) {
        const indent = /^( *)/.exec(line)?.[1] ?? "";
        const body = lines.slice(index + 1).join("\n");
        const closeRe = new RegExp(`^ {0,3}${marker[0]}{${marker.length},}[\\t ]*\\r?$`, "m");
        const close = closeRe.exec(body);
        if (close === null) return { open: true };
        const raw = body.slice(0, close.index);
        const stripped = indent
          ? raw
              .split("\n")
              .map((l) => l.slice(Math.min(indent.length, /^ */.exec(l)?.[0].length ?? 0)))
              .join("\n")
          : raw;
        return { open: false, indent, body: stripped.replace(/[\t ]*\r?\n?$/, "") };
      }
      // A bare run at top level is an opener (info may be empty), not a close.
      enclosing = { char: marker[0], run: marker.length };
    } else if (marker[0] === enclosing.char && marker.length >= enclosing.run && isClose) {
      enclosing = null;
    }
  }
  return { open: false, indent: "", body: "" };
}

function mermaidSourceOf(blockContent: string, found: MermaidFence): string | null {
  if (found.open) return null;
  if (found.body.trim().length > 0) return found.body.trim();
  const fence = markdownBlockFallback(blockContent);
  if (fence.fenced && fence.language === "mermaid") {
    const source = fence.text.trim();
    return source.length > 0 ? source : null;
  }
  return null;
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

export function CodeBlockActions({
  disabled,
  language,
  source,
}: {
  disabled: boolean;
  language: string | null;
  source: string | (() => string);
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
            const text = typeof source === "function" ? source() : source;
            if (!(await copyToClipboard(text))) {
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
            downloadTextFile(
              getCodeFilename(language),
              typeof source === "function" ? source() : source,
            );
          }}
        >
          <HugeiconsIcon icon={Download01Icon} className="size-icon" />
        </button>
      </div>
    </div>
  );
}

function useAnimationFreeBlockProps(props: BlockProps): BlockProps {
  // Drop animated's word-wrapping rehype plugin (kept only for direct scheduling); keep the array
  // stable so completed blocks stay memoised.
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

/** Asked once per message, not per block: per-block subscriptions rescanned parts on every keystroke. */
const RenderHtmlToolPresenceContext = createContext(false);

// Diffusion keeps the raw code visible instead (MessageHtmlArtifacts appends its card).
function StreamdownBlockContent(props: BlockProps) {
  const blockProps = useAnimationFreeBlockProps(props);
  const shouldCollapseHtmlArtifacts = useChatRuntimeStore(
    (state) =>
      state.collapseHtmlArtifacts && !state.loadedIsDiffusion,
  );
  const messageHasRenderableRenderHtmlTool = useContext(
    RenderHtmlToolPresenceContext,
  );
  const highlightMode = useContext(CodeHighlightModeContext);
  const highlightCode = highlightMode === "all";
  // ONE walk, both answers. `findMermaidFence` splits the block and scans every line, and asking
  // the two questions separately walked it twice on every render of every block. Measured on a
  // 3,000 line fence: 0.118 ms per walk, so the redundant one cost 0.092 ms per render, about
  // 5.5 ms per second while streaming at 60 fps.
  const mermaidFence = findMermaidFence(props.content);
  const hasMermaidFence = mermaidFence.open;
  const mermaidSource = mermaidSourceOf(props.content, mermaidFence);
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
        Loading HTML preview...
      </div>
    );
  }

  if (mermaidSource) {
    return (
      <div className="relative isolate">
        <MarkdownRendererBoundary
          fallback={
            <DeferredFenceShell language="mermaid" source={mermaidSource} />
          }
        >
          <Block {...blockProps} />
        </MarkdownRendererBoundary>
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
        {highlightCode || highlightMode === "settled" ? (
          <FenceBlock
            // A closed fence in a streaming trace is settled: worker route, not the main thread.
            isIncomplete={highlightCode ? props.isIncomplete : false}
            language={codeFence.language}
            source={codeFence.source}
          />
        ) : (
          <PlainFenceBlock
            actions
            isIncomplete={props.isIncomplete}
            language={codeFence.language}
            source={codeFence.source}
          />
        )}
        {svgSource && <SvgPreview source={svgSource} />}
      </>
    );
  }

  /* Streaming fences reach this bare Block, which first loads the highlighter; guard it so the block can
   * later mount FenceBlock with its controls instead of latching the outer boundary. */
  /* Streaming fence: same per-line FenceBody as a completed one, so the block keeps its shape. */
  /* Fence forms getCodeFence misses (tildes, 4+ backticks, indent) stay on the bounded renderer too. */
  const settledFence = props.isIncomplete ? null : markdownBlockFallback(props.content);
  if (settledFence?.fenced && !(settledFence.language === "mermaid" && mermaidSource)) {
    return highlightCode ? (
      <StreamingFenceBlock
        isIncomplete={false}
        language={settledFence.language}
        source={settledFence.text}
      />
    ) : (
      <LiveThinkingFence
        language={settledFence.language}
        settled
        source={settledFence.text}
      />
    );
  }

  if (props.isIncomplete) {
    const openFence = markdownBlockFallback(props.content);
    if (openFence.fenced) {
      return highlightCode ? (
        <StreamingFenceBlock
          language={openFence.language}
          source={openFence.text}
        />
      ) : (
        <LiveThinkingFence
          language={openFence.language}
          source={openFence.text}
        />
      );
    }
  }

  return (
    <MarkdownRendererBoundary
      fallback={<MarkdownBlockFallbackView content={props.content} />}
    >
      <Block {...blockProps} />
    </MarkdownRendererBoundary>
  );
}

/**
 * This fence's tokens, or null while the grammar loads. Same shape as streamdown's
 * HighlightedCodeBlockBody, which latchNow relies on; `wanted` drops stale streamed callbacks.
 */
const fenceHighlightOptions = (body: string, languageToken: string | null) => ({
  code: body,
  language: (languageToken ?? "text") as never,
  themes: STREAMDOWN_SHIKI_THEME,
});

const tokenizesOffThread = (body: string, streaming: boolean): boolean =>
  !streaming && body.length > MAX_HIGHLIGHT_CHARS;

const slowMainThreadRegex =
  typeof window !== "undefined"
  && typeof window.requestIdleCallback !== "function";
const MAIN_THREAD_TAIL_CHARS = 512;

const settlesOffThread = (
  body: string,
  streaming: boolean,
  options: ReturnType<typeof fenceHighlightOptions>,
): boolean => {
  if (!tokenizesOffThread(body, streaming)) {
    if (streaming || !slowMainThreadRegex) return false;
    const state = highlightWorkerState();
    if (state === "unavailable" || state === "stalled") return false;
  }
  return code.cover(options).uncovered > MAIN_THREAD_TAIL_CHARS;
};

export const highlightFenceSource = (
  body: string,
  language: string | null,
  late: (result: FenceTokens) => void,
): FenceTokens | null =>
  code.highlight(fenceHighlightOptions(body, language), late);

function useFenceTokens(
  source: string,
  languageToken: string | null,
  enabled: boolean,
  streaming: boolean,
): FenceTokens | null {
  const [eager, setEager] = useState(() => !fenceNeedsQuietWork(languageToken, source, streaming));
  if (!eager && !fenceNeedsQuietWork(languageToken, source, streaming)) setEager(true);
  const highlight = useCallback(
    (body: string, late: (result: FenceTokens) => void) =>
      streaming
        ? code.highlight(fenceHighlightOptions(body, languageToken), late)
        : code.highlightExact(fenceHighlightOptions(body, languageToken), late),
    [languageToken, streaming],
  );
  const [seed] = useState<{ body: string; tokens: FenceTokens } | null>(() => {
    if (!enabled || !eager) return null;
    const body = trimTrailingNewlines(source);
    const options = fenceHighlightOptions(body, languageToken);
    const settled = settlesOffThread(body, streaming, options)
      ? code.cached(options)
      : highlight(body, () => {});
    return settled ? { body, tokens: settled } : null;
  });
  const [tokens, setTokens] = useState<FenceTokens | null>(
    seed?.tokens ?? null,
  );
  const wanted = useRef("");
  const seedConsumed = useRef(false);
  const displayed = useRef(seed ? { body: seed.body, language: languageToken } : null);
  useLayoutEffect(() => {
    if (!enabled) return;
    const body = trimTrailingNewlines(source);
    wanted.current = body;
    if (!seedConsumed.current) {
      seedConsumed.current = true;
      if (seed?.body === body) return;
    }
    let active = true;
    let cancel: (() => void) | null = null;
    let release: (() => void) | null = null;
    let cancelStart: (() => void) | null = null;
    let cancelPublish: (() => void) | null = null;
    let cancelFallback: (() => void) | null = null;
    const options = fenceHighlightOptions(body, languageToken);
    const current = (): boolean => active && wanted.current === body;
    const defer = (): boolean => !eager && fenceNeedsQuietWork(languageToken, source, streaming);
    const show = (result: FenceTokens | null): void => {
      if (!current()) return;
      displayed.current = result === null ? null : { body, language: languageToken };
      setTokens(result);
    };
    const publish = (work: () => void): void => {
      if (!current()) return;
      cancelPublish?.();
      cancelPublish = null;
      if (defer()) {
        cancelPublish = queueFenceWork(() => {
          cancelPublish = null;
          if (current()) work();
        });
      } else work();
    };
    const late = (result: FenceTokens) => publish(() => show(result));
    const tokenizeNow = (): boolean => {
      if (!current() || highlight("", () => {}) === null) return false;
      const result = code.highlightExact(options, late);
      if (result === null) return false;
      cancel?.();
      cancelPublish?.();
      cancelPublish = null;
      show(result);
      release?.();
      release = null;
      return true;
    };
    const start = (): void => {
      cancelStart = null;
      if (!current()) return;
      let settled: FenceTokens | null = null;
      if (settlesOffThread(body, streaming, options)) {
        const failover = !tokenizesOffThread(body, streaming);
        settled = code.cached(options);
        if (settled === null) {
          cancel = requestFullHighlight(
            body,
            languageToken,
            (result) => {
              if (!current()) return;
              if (result === null) {
                if (failover) publish(() => show(highlight(body, late)));
                return;
              }
              cancelFallback?.();
              cancelFallback = null;
              release?.();
              release = null;
              publish(() => {
                code.seed(options, result);
                show(result);
              });
            },
            false,
            !failover,
          );
          if (cancel === null && failover) {
            settled = highlight(body, late);
          } else {
            highlight("", () => {});
            settled = code.cover(options).result;
          }
          release = awaitWorker(body.length, () => {
            if (!current()) return false;
            if (!defer()) return tokenizeNow();
            if (cancelFallback === null) {
              cancelFallback = queueFenceWork(() => {
                cancelFallback = null;
                tokenizeNow();
              });
            }
            return false;
          });
        }
      } else {
        settled = highlight(body, late);
      }
      show(settled);
    };
    if (defer()) {
      if (displayed.current !== null
        && (displayed.current.body !== body || displayed.current.language !== languageToken)) {
        show(null);
      }
      cancelStart = queueFenceWork(start);
    } else start();
    return () => {
      active = false;
      cancelStart?.();
      cancelPublish?.();
      cancelFallback?.();
      release?.();
      cancel?.();
    };
  }, [enabled, source, highlight, seed, streaming, languageToken, eager]);
  return tokens;
}

/** A fence still being written: highlighted from the first character, no latch and no action bar. */
function StreamingFenceBlock({
  language,
  source,
  isIncomplete = true,
}: {
  language: string | null;
  source: string;
  isIncomplete?: boolean;
}) {
  const languageToken = language?.trim().split(/\s+/)[0] || null;
  if (isIncomplete) noteStreamingFence(languageToken, source);
  const tokens = useFenceTokens(source, languageToken, true, isIncomplete);
  return (
    <MarkdownRendererBoundary
      fallback={<DeferredFenceShell language={languageToken} source={source} />}
    >
      <FenceBody
        isIncomplete={isIncomplete}
        language={languageToken}
        result={tokens}
        source={source}
        windowing={fenceMode() === "window"}
      />
    </MarkdownRendererBoundary>
  );
}

function PlainFenceBlock({
  actions = false,
  isIncomplete,
  language,
  source,
}: {
  actions?: boolean;
  isIncomplete?: boolean;
  language: string | null;
  source: string;
}) {
  const languageToken = language?.trim().split(/\s+/)[0] || null;
  const shell = <PlainFenceShell language={languageToken} source={source} />;
  if (!actions) return shell;
  return (
    <div className="relative isolate">
      {shell}
      <CodeBlockActions
        disabled={Boolean(isIncomplete)}
        language={language}
        source={source}
      />
    </div>
  );
}

function LiveThinkingFence({
  language,
  settled = false,
  source,
}: {
  language: string | null;
  settled?: boolean;
  source: string;
}) {
  const languageToken = language?.trim().split(/\s+/)[0] || null;
  const host = useRef<HTMLDivElement | null>(null);
  const colour = useThinkingCodeColour(host, source, languageToken, settled);
  const result = useMemo(
    () =>
      colour === null
        ? null
        : spellLines(colour, trimTrailingNewlines(source).split("\n")),
    [colour, source],
  );
  return (
    <div className="relative isolate" ref={host}>
      {result === null ? (
        <PlainFenceShell language={languageToken} source={source} />
      ) : (
        <FenceBody
          isIncomplete={!settled}
          language={languageToken}
          result={result}
          source={source}
          windowing={fenceMode() === "window"}
        />
      )}
    </div>
  );
}

/*
 * The fence branch, extracted so the reach latch can be a hook. With the flag off this renders exactly what the
 * branch rendered before: the same `relative isolate` wrapper, the same action bar, and inside them a body
 * whose elements and classes are streamdown's own. The wrapper is reused as the intersection target rather
 * than a new one being introduced, so the DOM the off arm produces is byte-for-byte what main produces and
 * the on arm differs only in what is INSIDE the wrapper.
 * `<Block>` is gone from this branch. It maps the WHOLE token array on every render and memoizes on the
 * identity of a result object the plugin rebuilds every frame, so the fence re-rendered end to end sixty
 * times a second. See `FenceBody` in `code-fence-defer.tsx` for what replaced it and why.
 */
function FenceBlock({
  isIncomplete,
  language,
  source,
}: {
  isIncomplete: boolean | undefined;
  language: string | null;
  source: string;
}) {
  const host = useRef<HTMLDivElement | null>(null);
  const mode = fenceMode();

  /* CODE_FENCE_RE is deliberately narrow: it also decides the SVG and HTML-artifact paths. */
  // Only the info string's first word is the language (```python startLine=10).
  const languageToken = language?.trim().split(/\s+/)[0] || null;

  /* warm(true) tokenizes now (a sync cache hit for render); warm(false) only loads the grammar. */
  const warm = useCallback(
    (tokens: boolean) => {
      const body = trimTrailingNewlines(source);
      const options = fenceHighlightOptions(body, languageToken);
      const inline =
        tokens
        && (!settlesOffThread(body, Boolean(isIncomplete), options)
          || code.cached(options) !== null);
      if (inline && !isIncomplete) code.highlightExact(options, () => {});
      else code.highlight(inline ? options : fenceHighlightOptions("", languageToken), () => {});
    },
    [source, languageToken, isIncomplete],
  );

  const speculate = useCallback(
    (settle: (seeded: boolean) => void) => {
      const body = trimTrailingNewlines(source);
      const options = fenceHighlightOptions(body, languageToken);
      if (code.cached(options) !== null) {
        settle(true);
        return null;
      }
      const cancel = requestFullHighlight(
        body,
        languageToken,
        (result) => {
          if (result !== null) code.seed(options, result);
          settle(result !== null);
        },
        true,
      );
      if (cancel === null) settle(false);
      return cancel;
    },
    [source, languageToken],
  );

  const reached = useFenceReached(
    host,
    mode !== "off",
    Boolean(isIncomplete),
    languageToken,
    trimmedLength(source),
    source,
    warm,
    speculate,
  );

  // MEASUREMENT ARM ONLY. See `FenceMode`: this puts the tokenizer work back while leaving the document at the
  // deferred size, so the two costs can be told apart. `code.highlight` caches on the source string, so the work
  // happens exactly once and the discarded result is the same object the real path would have used.
  // Asked for only once the fence is reached, so a deferred fence still tokenizes nothing. The
  // latch calls `warm(true)` synchronously on the way in, so this is a cache hit rather than the
  // first tokenization of the body.
  const tokens = useFenceTokens(
    source,
    languageToken,
    reached,
    Boolean(isIncomplete),
  );

  const pretokenize = mode === "tokenize" && !reached;
  useEffect(() => {
    if (!pretokenize) return;
    code.highlight(
      {
        code: trimTrailingNewlines(source),
        language: (languageToken ?? "text") as never,
        themes: STREAMDOWN_SHIKI_THEME,
      },
      () => {},
    );
  }, [pretokenize, source, languageToken]);

  return (
    <div className="relative isolate" ref={host}>
      {/* Only Block loads code at render time; bounding it alone keeps the action bar. */}
      <MarkdownRendererBoundary
        fallback={
          <DeferredFenceShell language={languageToken} source={source} />
        }
      >
        {reached ? (
          <FenceBody
            isIncomplete={isIncomplete}
            language={languageToken}
            result={tokens}
            source={source}
            windowing={mode === "window"}
          />
        ) : (
          <DeferredFenceShell language={languageToken} source={source} />
        )}
      </MarkdownRendererBoundary>
      <CodeBlockActions
        disabled={Boolean(isIncomplete)}
        language={language}
        source={source}
      />
    </div>
  );
}
/** Every block gets a boundary, so a failed lazy chunk costs only that block, not the whole app. */
const StreamdownBlock = memo((props: BlockProps) => (
  <MarkdownBlockBoundary content={props.content}>
    <StreamdownBlockContent {...props} />
  </MarkdownBlockBoundary>
));
StreamdownBlock.displayName = "StreamdownBlock";
// Only the adapter's inline wav: any other src (remote URL, WebKit-followed audio/mpegurl) fetches on render.
const AUDIO_PLAYER_RE = /<audio-player\s+src="(data:audio\/wav;base64,[A-Za-z0-9+/=]+)"\s*\/>/;

// One render per paint, spaced by measured cost; incremental parsing bounds the work per paint.
function useCoalescedStreamingText(
  text: string,
  isStreaming: boolean,
  messageId: string,
): string {
  const [displayed, setDisplayed] = useState({ messageId, text });
  const pendingRef = useRef({ messageId, text });
  const rafRef = useRef<number | null>(null);
  const renderedAtRef = useRef(Number.NEGATIVE_INFINITY);
  const frameAtRef = useRef(Number.NEGATIVE_INFINITY);
  const frameIntervalRef = useRef(Number.POSITIVE_INFINITY);
  const costRef = useRef(0);
  const yieldsRef = useRef(0);
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

    const render = (frameTime: number) => {
      rafRef.current = null;
      if (frameTime > frameAtRef.current) {
        frameIntervalRef.current = Math.min(
          frameIntervalRef.current,
          frameTime - frameAtRef.current,
        );
        frameAtRef.current = frameTime;
      }
      if (
        frameTime - renderedAtRef.current <
        Math.max(
          16 - Math.min(frameIntervalRef.current, 16) / 2,
          STREAMING_RENDER_DUTY * costRef.current,
        )
      ) {
        rafRef.current = requestAnimationFrame(render);
        return;
      }
      if (yieldsRef.current < STREAMING_INPUT_YIELDS && inputPending()) {
        yieldsRef.current += 1;
        rafRef.current = requestAnimationFrame(render);
        return;
      }
      yieldsRef.current = 0;
      renderedAtRef.current = frameTime;
      afterFrame(() => {
        const startedAt = performance.now();
        flushSync(() => setDisplayed(pendingRef.current));
        costRef.current = performance.now() - startedAt;
      });
    };
    rafRef.current = requestAnimationFrame(render);
  }, [cancelScheduledRender, messageId, text, isStreaming]);

  useEffect(() => {
    return cancelScheduledRender;
  }, [cancelScheduledRender]);

  // A running message can also be replaced (audio placeholder to player), which must show at once.
  if (
    isStreaming &&
    displayed.messageId === messageId &&
    text.length >= displayed.text.length &&
    // Not startsWith, which scans a growing reply (see hasPrefix in streaming-render-schedule.ts).
    text.slice(0, displayed.text.length) === displayed.text
  ) {
    return displayed.text;
  }
  return text;
}

/** False inside the reasoning block, which renders through this same component. */
export const SearchImagesEnabledContext = createContext(true);

type MarkdownTextRendererProps = {
  isStreaming: boolean;
  messageHasRenderableRenderHtmlTool: boolean;
  messageId: string;
  messageTextKey: string;
  precedingText: string;
  cut?: boolean;
  searchImagesKey: string;
  statusType: string;
  text: string;
};

function MarkdownTextRenderer({
  isStreaming,
  messageHasRenderableRenderHtmlTool,
  messageId,
  messageTextKey,
  precedingText,
  cut = !isStreaming,
  searchImagesKey,
  statusType,
  text,
}: MarkdownTextRendererProps) {
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const projectId = useChatProjectScope();
  const highlightCode = useContext(CodeHighlightModeContext) === "all";
  const threadId = remoteId ?? activeThreadId ?? undefined;
  // Streamdown's memo comparator ignores rehypePlugins.
  const sandboxScopeKey = JSON.stringify([threadId, projectId]);
  const rehypePlugins = useMemo(
    () =>
      // Streamdown caches processors by plugin name and serialized options.
      withDataImageSupport(STREAMDOWN_ALLOWED_TAGS, [
        [rehypeSandboxImages, { threadId, projectId }],
      ]),
    [threadId, projectId],
  );
  const searchImages = useMemo(
    () => parseSearchImagesSignature(searchImagesKey),
    [searchImagesKey],
  );
  const messageTexts = useMemo(
    () => JSON.parse(messageTextKey) as string[],
    [messageTextKey],
  );
  const displayText = useCoalescedStreamingText(text, isStreaming, messageId);
  const processedText = useMemo(
    () =>
      stabilizeStreamingMarkdown(
        preprocessLaTeX(
          normalizeEscapedInlineMath(
            rewriteSearchImageTokens(
              placeSubjectImages(
                holdBackPartialSearchImageToken(
                  displayText,
                  isStreaming && searchImages.size > 0,
                ),
                searchImages,
                isStreaming,
                precedingText,
                messageTexts,
              ),
              searchImages,
            ),
          ),
          isStreaming,
        ),
        isStreaming,
      ),
    [displayText, isStreaming, messageTexts, precedingText, searchImages],
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
  const renderKey = markdownRenderKey(processedText);

  const audioMatch = displayText.match(AUDIO_PLAYER_RE);
  if (audioMatch) {
    return <AudioPlayer src={audioMatch[1]} />;
  }

  return (
    <RenderHtmlToolPresenceContext.Provider
      value={messageHasRenderableRenderHtmlTool}
    >
      <SearchImagesContext.Provider value={searchImages}>
        <div data-status={statusType} className="min-w-0 max-w-full">
          <Streamdown
            key={`${messageId}:${incrementalCache.renderGeneration}:${renderKey}:${sandboxScopeKey}:${cut}`}
            mode="streaming"
            parseIncompleteMarkdown={!incrementalRender}
            remend={
              !cut
                ? FINISHED_TEXT_REMEND
                : hasIncompleteLinkRepair(processedText)
                  ? LITERAL_LINK_REMEND
                  : undefined
            }
            parseMarkdownIntoBlocksFn={
              incrementalRender?.parseMarkdownIntoBlocks ??
              parseMarkdownIntoRenderableBlocks
            }
            isAnimating={isStreaming}
            animated={STREAMDOWN_IMMEDIATE_UPDATES}
            plugins={
              highlightCode ? STREAMDOWN_PLUGINS : STREAMDOWN_PLAIN_CODE_PLUGINS
            }
            components={STREAMDOWN_COMPONENTS}
            allowedTags={STREAMDOWN_ALLOWED_TAGS}
            rehypePlugins={rehypePlugins}
            urlTransform={safeMarkdownUrl}
            controls={STREAMDOWN_CONTROLS}
            icons={MARKDOWN_TABLE_ICONS}
            shikiTheme={STREAMDOWN_SHIKI_THEME}
            BlockComponent={StreamdownBlock}
          >
            {incrementalRender?.markdown ?? processedText}
          </Streamdown>
        </div>
      </SearchImagesContext.Provider>
    </RenderHtmlToolPresenceContext.Provider>
  );
}

const MarkdownTextImpl = () => {
  const allowSearchImages = useContext(SearchImagesEnabledContext);
  const aui = useAui();
  const { text, status } = useMessagePartTextSafe();
  const partIndex =
    aui.part.source === "message" && aui.part.query.type === "index"
      ? aui.part.query.index
      : 0;
  // Streamdown only extends parsed blocks, so key per message; the cache generation covers edits
  // that drop retained blocks without changing the tail.
  const messageId = useAuiState(({ message }) => message.id);
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    partsHaveRenderableRenderHtmlTool(message.parts),
  );
  // A string, not the Map: selector results are compared by identity.
  const searchImagesKey = useAuiState(({ message }) =>
    allowSearchImages ? partsSearchImagesSignature(message.parts) : "",
  );
  const precedingText = useAuiState(({ message }) =>
    allowSearchImages ? partsPrecedingText(message.parts, partIndex) : "",
  );
  const messageTextKey = useAuiState(({ message }) =>
    allowSearchImages && partsSearchImagesSignature(message.parts) !== ""
      ? partsTextKey(message.parts)
      : "[]",
  );
  // Every streamed yield carries the incomplete stamp and only a normal finish clears it.
  const cut = useAuiState(
    ({ message }) =>
      message.status?.type !== "running" &&
      !!readIncompleteInfo(message.metadata) &&
      !message.parts.slice(partIndex + 1).some((part) => part.type === "text"),
  );

  return (
    <MarkdownTextRenderer
      isStreaming={status.type === "running"}
      messageHasRenderableRenderHtmlTool={messageHasRenderableRenderHtmlTool}
      messageId={messageId}
      messageTextKey={messageTextKey}
      precedingText={precedingText}
      cut={cut}
      searchImagesKey={searchImagesKey}
      statusType={status.type}
      text={text}
    />
  );
};

type MarkdownTextSourceProps = {
  messageHasRenderableRenderHtmlTool: boolean;
  messageId: string;
  sourceText: string;
  streaming: boolean;
};

const MarkdownTextSourceImpl = ({
  messageHasRenderableRenderHtmlTool,
  messageId,
  sourceText,
  streaming,
}: MarkdownTextSourceProps) => (
  <MarkdownTextRenderer
    isStreaming={streaming}
    messageHasRenderableRenderHtmlTool={messageHasRenderableRenderHtmlTool}
    messageId={messageId}
    messageTextKey="[]"
    precedingText=""
    searchImagesKey=""
    statusType={streaming ? "running" : "complete"}
    text={sourceText}
  />
);

export const MarkdownText = withSmoothContextProvider(MarkdownTextImpl);
// Reasoning renders at group scope with no `part`, so this must not use the part adapter or smooth wrapper.
export const MarkdownTextSource = MarkdownTextSourceImpl;
