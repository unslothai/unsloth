// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { openImageViewer } from "@/components/image-viewer";
import { filesOpenInBrowser, openFileInBrowser } from "@/features/browser";
import { useT } from "@/i18n";
import {
  ArtifactCard,
  useChatProjectScope,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  getCodeFence,
  isFullHtmlDocument,
  isHtmlFence,
  isRenderableRenderHtmlToolPart,
  isSvgFence,
} from "@/features/chat/artifacts/html-fences";
// Leaf module, not the barrel: SEARCH_IMAGE_TAG is read at module scope and the barrel cycles (TDZ).
// eslint-disable-next-line no-restricted-imports
import {
  holdBackPartialSearchImageToken,
  parseSearchImagesSignature,
  placeSubjectImages,
  precedingTextForMessagePart,
  rewriteSearchImageTokens,
  SEARCH_IMAGE_TAG,
  searchImagesSignature,
} from "@/features/chat/search-images/search-images";
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
  useMessagePartText,
} from "@assistant-ui/react";
import {
  Copy01Icon,
  Download01Icon,
  ExpandIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { createMathPlugin } from "@streamdown/math";
import { mermaid } from "@streamdown/mermaid";
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
import {
  Block,
  type BlockProps,
  Streamdown,
  type StreamdownProps,
} from "streamdown";
import {
  DeferredFenceShell,
  FenceBody,
  type FenceTokens,
  fenceMode,
  trimmedLength,
  trimTrailingNewlines,
  useFenceReached,
} from "./code-fence-defer";
import { markdownBlockFallback } from "./markdown-block-fallback";
import { createCodePlugin } from "./code-plugin";
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
import { SearchImageElement, SearchImagesContext } from "./search-image";
import { useSandboxImage } from "./use-sandbox-image";
import { rehypeSandboxImages } from "./rehype-sandbox-images";
import { unslothDarkTheme, unslothLightTheme } from "./code-themes";
import { stabilizeStreamingMarkdown } from "./streaming-markdown";
import {
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
// `strokeWidth` is dropped: SVG types it `string | number`, HugeiconsIcon wants a number.
function streamdownIcon(icon: typeof Copy01Icon) {
  return function StreamdownIcon({
    size,
    strokeWidth: _strokeWidth,
    ...props
  }: ComponentProps<"svg"> & { size?: number }) {
    return <HugeiconsIcon icon={icon} size={size} {...props} />;
  };
}
const STREAMDOWN_ICONS = {
  CopyIcon: streamdownIcon(Copy01Icon),
  DownloadIcon: streamdownIcon(Download01Icon),
  Maximize2Icon: streamdownIcon(ExpandIcon),
} satisfies NonNullable<StreamdownProps["icons"]>;
const { withSmoothContextProvider } = INTERNAL;

// Streamdown 2.5 schedules streaming blocks in an interruptible transition a token stream can starve;
// the animated path commits directly, so use it without the animation transformer.
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
          className="absolute right-2 bottom-2 flex h-8 w-8 cursor-pointer items-center justify-center rounded-md border border-border bg-background/90 opacity-0 backdrop-blur-sm transition-all duration-200 group-hover:opacity-100"
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
  "text-primary underline underline-offset-2 decoration-primary/40 hover:decoration-primary transition-colors cursor-pointer";

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
};
const STREAMDOWN_ALLOWED_TAGS = {
  [SEARCH_IMAGE_TAG]: ["token"],
} satisfies NonNullable<StreamdownProps["allowedTags"]>;

const COPY_RESET_MS = 2000;
const ACTION_PANEL_CLASS =
  "pointer-events-auto flex shrink-0 items-center gap-1";
const ACTION_BUTTON_CLASS =
  "flex size-8 cursor-pointer items-center justify-center rounded-[10px] text-chat-icon-fg transition-all hover:bg-chat-icon-bg-hover hover:text-chat-icon-fg-hover disabled:cursor-not-allowed disabled:opacity-50";

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
  // One walk answers both questions; asking separately walked the block twice per render.
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
        <FenceBlock
          isIncomplete={props.isIncomplete}
          language={codeFence.language}
          source={codeFence.source}
        />
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
    return (
      <StreamingFenceBlock
        isIncomplete={false}
        language={settledFence.language}
        source={settledFence.text}
      />
    );
  }

  if (props.isIncomplete) {
    const openFence = markdownBlockFallback(props.content);
    if (openFence.fenced) {
      return (
        <StreamingFenceBlock
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
function useFenceTokens(
  source: string,
  languageToken: string | null,
  enabled: boolean,
): FenceTokens | null {
  const [tokens, setTokens] = useState<FenceTokens | null>(null);
  const wanted = useRef("");
  /* One layout effect: two effects double-rendered the body since approximateResult is a fresh object. */
  useLayoutEffect(() => {
    if (!enabled) return;
    const body = trimTrailingNewlines(source);
    wanted.current = body;
    const settled = code.highlight(
      {
        code: body,
        language: (languageToken ?? "text") as never,
        themes: STREAMDOWN_SHIKI_THEME,
      },
      (late) => {
        if (wanted.current === body) setTokens(late);
      },
    );
    // `settled === null` means a tokenization error; keeping old tokens would show a shorter body.
    setTokens(settled ?? null);
  }, [enabled, source, languageToken]);
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
  const tokens = useFenceTokens(source, languageToken, true);
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

/* Fence branch; the wrapper is the intersection target so the off arm's DOM matches main exactly. */
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
      code.highlight({
        code: tokens ? trimTrailingNewlines(source) : "",
        language: (languageToken ?? "text") as never,
        themes: STREAMDOWN_SHIKI_THEME,
      }, () => {});
    },
    [source, languageToken],
  );

  const reached = useFenceReached(
    host,
    mode !== "off",
    Boolean(isIncomplete),
    languageToken,
    trimmedLength(source),
    warm,
  );

  // Measurement arm only (see FenceMode); a reached fence is already a cache hit from the latch.
  const tokens = useFenceTokens(source, languageToken, reached);

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
const AUDIO_PLAYER_RE = /<audio-player\s+src="([^"]+)"\s*\/>/;

// Coalesce only token events that arrive before the next paint; no time or length throttle.
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
  searchImagesKey,
  statusType,
  text,
}: MarkdownTextRendererProps) {
  const remoteId = useAuiState(({ threadListItem }) => threadListItem.remoteId);
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const projectId = useChatProjectScope();
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
  const pendingLinkRepair = useMemo(
    () => !isStreaming && hasIncompleteLinkRepair(processedText),
    [isStreaming, processedText],
  );
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
            key={`${messageId}:${incrementalCache.renderGeneration}:${renderKey}:${sandboxScopeKey}`}
            mode="streaming"
            parseIncompleteMarkdown={!incrementalRender}
            remend={pendingLinkRepair ? LITERAL_LINK_REMEND : undefined}
            parseMarkdownIntoBlocksFn={
              incrementalRender?.parseMarkdownIntoBlocks ??
              parseMarkdownIntoRenderableBlocks
            }
            isAnimating={isStreaming}
            animated={STREAMDOWN_IMMEDIATE_UPDATES}
            plugins={STREAMDOWN_PLUGINS}
            components={STREAMDOWN_COMPONENTS}
            allowedTags={STREAMDOWN_ALLOWED_TAGS}
            rehypePlugins={rehypePlugins}
            urlTransform={safeMarkdownUrl}
            controls={STREAMDOWN_CONTROLS}
            icons={STREAMDOWN_ICONS}
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
  const { text, status } = useMessagePartText();
  const partIndex =
    aui.part.source === "message" && aui.part.query.type === "index"
      ? aui.part.query.index
      : 0;
  // Streamdown only extends parsed blocks, so key per message; the cache generation covers edits
  // that drop retained blocks without changing the tail.
  const messageId = useAuiState(({ message }) => message.id);
  const messageHasRenderableRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  // A string, not the Map: selector results are compared by identity.
  const searchImagesKey = useAuiState(({ message }) =>
    allowSearchImages ? searchImagesSignature(message.parts) : "",
  );
  const precedingText = useAuiState(({ message }) =>
    allowSearchImages
      ? precedingTextForMessagePart(message.parts, partIndex)
      : "",
  );
  const messageTextKey = useAuiState(({ message }) =>
    allowSearchImages
      ? JSON.stringify(
          message.parts
            .filter((part) => part.type === "text")
            .map((part) => part.text),
        )
      : "[]",
  );

  return (
    <MarkdownTextRenderer
      isStreaming={status.type === "running"}
      messageHasRenderableRenderHtmlTool={messageHasRenderableRenderHtmlTool}
      messageId={messageId}
      messageTextKey={messageTextKey}
      precedingText={precedingText}
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
