// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { LruMap } from "@/features/hub/lib/lru-map";
import { isHuggingFaceOffline } from "@/features/hub/lib/network";
import { fingerprintToken } from "@/features/hub/lib/token-fingerprint";
import { cn } from "@/lib/utils";
import { confirmExternalLink } from "../stores/external-link-confirm";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { code as streamdownCode } from "@streamdown/code";
import { math as streamdownMath } from "@streamdown/math";
import { mermaid as streamdownMermaid } from "@streamdown/mermaid";
import {
  createContext,
  startTransition,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { ComponentProps } from "react";
import { Streamdown, type Components } from "streamdown";
import {
  createReadmeUrlTransform,
  fetchReadme,
  readmeBaseUrl,
  stripChromeHeadings,
  stripFrontmatter,
} from "../lib/hf-readme";

type ReadmePlugins = NonNullable<ComponentProps<typeof Streamdown>["plugins"]>;

interface ReadmeState {
  key: string;
  body: string | null;
  baseUrl: string | null;
  loading: boolean;
  error: string | null;
  plugins: ReadmePlugins | null;
}

type ReadmeCacheEntry = Pick<ReadmeState, "body" | "baseUrl" | "error">;
type ReadmeSubject = "model" | "dataset" | "baseModel";

const README_ALLOWED_TAGS: NonNullable<
  ComponentProps<typeof Streamdown>["allowedTags"]
> = {
  audio: ["src", "controls", "preload", "loop", "muted"],
  video: [
    "src",
    "controls",
    "preload",
    "loop",
    "muted",
    "poster",
    "width",
    "height",
    "playsinline",
  ],
  source: ["src", "type", "media"],
  track: ["src", "kind", "srclang", "label", "default"],
};

const READMEX_NEEDS_MATH = /\$\$|\\\(|\\\[/;
const READMEX_NEEDS_MERMAID = /```mermaid\b/;
const README_RENDER_CHAR_LIMIT = 120_000;
const README_CACHE_TTL_MS = 60_000;
const README_TRUNCATED_NOTICE =
  "\n\n---\n\nCard truncated for performance. Open the repository on Hugging Face to read the full README.";
const ReadmeOnlineContext = createContext(true);

interface ResolvedReadmeCacheEntry {
  entry: ReadmeCacheEntry;
  expiresAt: number;
}

type ReadmeCacheValue =
  | ResolvedReadmeCacheEntry
  | Promise<ReadmeCacheEntry>;

const readmeCache = new LruMap<string, ReadmeCacheValue>(64);

function isReadmeCachePromise(
  entry: ReadmeCacheValue,
): entry is Promise<ReadmeCacheEntry> {
  return typeof (entry as Promise<ReadmeCacheEntry>).then === "function";
}

function readResolvedReadmeCache(cacheKey: string): ReadmeCacheEntry | null {
  const cached = readmeCache.get(cacheKey);
  if (!cached || isReadmeCachePromise(cached)) {
    return null;
  }
  if (cached.expiresAt <= Date.now()) {
    readmeCache.delete(cacheKey);
    return null;
  }
  return cached.entry;
}

function readmeStateFromEntry(
  key: string,
  entry: ReadmeCacheEntry,
): ReadmeState {
  return {
    key,
    body: entry.body,
    baseUrl: entry.baseUrl,
    loading: false,
    error: entry.error,
    plugins: null,
  };
}

function hasReadmeContent(state: Pick<ReadmeState, "body" | "error">): boolean {
  return state.error == null && state.body != null;
}

function prepareReadmeBody(markdown: string): string {
  const { body } = stripFrontmatter(markdown);
  const cleaned = stripChromeHeadings(body).trim();
  if (cleaned.length <= README_RENDER_CHAR_LIMIT) return cleaned;
  return `${cleaned.slice(0, README_RENDER_CHAR_LIMIT).trimEnd()}${README_TRUNCATED_NOTICE}`;
}

const PROSE = cn(
  "max-w-none text-ui-13p5 leading-[1.7] text-foreground/85",
  "[&_h1]:text-ui-18 [&_h1]:font-semibold [&_h1]:tracking-tight [&_h1]:mt-2 [&_h1]:mb-3",
  "[&_h2]:text-ui-15p5 [&_h2]:font-semibold [&_h2]:tracking-tight [&_h2]:mt-5 [&_h2]:mb-2",
  "[&_h3]:text-ui-14 [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-1.5",
  "[&_p]:my-2.5 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0.5",
  "[&_a]:text-primary [&_a:hover]:underline",
  "[&_code]:rounded-md [&_code]:bg-muted/60 [&_code]:px-1 [&_code]:py-0.5 [&_code]:text-ui-12 [&_code]:font-mono",
  "[&_pre]:rounded-[12px] [&_pre]:border [&_pre]:border-border/60 [&_pre]:bg-muted/40 [&_pre]:p-3 [&_pre]:text-ui-12 [&_pre]:overflow-x-auto",
  "[&_pre_code]:bg-transparent [&_pre_code]:p-0",
  "[&_blockquote]:border-l-2 [&_blockquote]:border-border/60 [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground",
  "[&_table]:my-3 [&_table]:text-ui-12p5",
  "[&_th]:px-2 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold [&_th]:border-b [&_th]:border-border/60",
  "[&_td]:px-2 [&_td]:py-1.5 [&_td]:border-b [&_td]:border-border/40",
  "[&_img]:rounded-[10px] [&_img]:my-2 [&_img]:max-w-full",
  "[&_audio]:my-1 [&_audio]:max-w-full [&_audio]:h-9 [&_audio]:align-middle",
  "[&_video]:my-2 [&_video]:max-w-full [&_video]:rounded-[10px]",
  "[&_hr]:my-4 [&_hr]:border-border/40",
);

function ReadmeLink({
  node,
  href,
  target,
  rel,
  className,
  title,
  children,
  ...props
}: ComponentProps<"a"> & { node?: unknown }) {
  void node;
  const online = useContext(ReadmeOnlineContext);
  const rawHref = typeof href === "string" ? href.trim() : "";
  const disabledExternal =
    !online &&
    rawHref.length > 0 &&
    !rawHref.startsWith("#") &&
    (/^(https?:)?\/\//i.test(rawHref) || rawHref.startsWith("/"));
  if (disabledExternal) {
    return (
      <span
        aria-disabled="true"
        title={title ?? "Unavailable offline"}
        className={cn(className, "cursor-not-allowed opacity-60")}
      >
        {children}
      </span>
    );
  }
  return (
    <a
      {...props}
      href={href}
      rel={rel ?? "noreferrer noopener"}
      target={target ?? "_blank"}
      className={className}
      title={title}
      onClick={(event) => {
        if (confirmExternalLink(rawHref)) {
          event.preventDefault();
        }
      }}
    >
      {children}
    </a>
  );
}

function ReadmeImage({
  node,
  alt,
  loading,
  decoding,
  ...props
}: ComponentProps<"img"> & { node?: unknown }) {
  void node;
  return (
    <img
      {...props}
      alt={alt ?? ""}
      decoding={decoding ?? "async"}
      loading={loading ?? "lazy"}
    />
  );
}

const README_COMPONENTS: Components = {
  a: ReadmeLink,
  img: ReadmeImage,
};

function readmeLoadingMessage(subject: ReadmeSubject): string {
  if (subject === "dataset") return "Loading dataset card...";
  if (subject === "baseModel") return "Loading base model card...";
  return "Loading model card...";
}

function readmePreparingMessage(subject: ReadmeSubject): string {
  if (subject === "dataset") return "Preparing dataset card...";
  if (subject === "baseModel") return "Preparing base model card...";
  return "Preparing model card...";
}

function readmeOfflineMessage(subject: ReadmeSubject): string {
  if (subject === "dataset") return "Dataset card unavailable offline.";
  if (subject === "baseModel") return "Base model card unavailable offline.";
  return "Model card unavailable offline.";
}

function readmeMissingMessage(subject: ReadmeSubject): string {
  if (subject === "dataset") return "This dataset has no README.";
  if (subject === "baseModel") return "This base model has no README.";
  return "This repository has no README.";
}

function readmeUnavailableMessage(subject: ReadmeSubject): string {
  if (subject === "dataset") {
    return "Dataset card unavailable. It may be private, gated, or temporarily unreachable.";
  }
  if (subject === "baseModel") {
    return "Base model card unavailable. It may be private, gated, or temporarily unreachable.";
  }
  return "Model card unavailable. It may be private, gated, or temporarily unreachable.";
}

function isMissingReadmeError(error: string): boolean {
  return /^No (dataset|model) card available\.$/.test(error);
}

function scheduleIdleTask(callback: () => void, timeout = 250): () => void {
  let canceled = false;
  const run = () => {
    if (!canceled) callback();
  };

  if (typeof window === "undefined") {
    run();
    return () => {
      canceled = true;
    };
  }

  const idleWindow = window as Window & {
    requestIdleCallback?: Window["requestIdleCallback"];
    cancelIdleCallback?: Window["cancelIdleCallback"];
  };

  if (idleWindow.requestIdleCallback && idleWindow.cancelIdleCallback) {
    const handle = idleWindow.requestIdleCallback(run, { timeout });
    return () => {
      canceled = true;
      idleWindow.cancelIdleCallback?.(handle);
    };
  }

  const handle = globalThis.setTimeout(run, Math.min(timeout, 120));
  return () => {
    canceled = true;
    globalThis.clearTimeout(handle);
  };
}

function ReadmePlaceholder({
  kind,
  message,
}: {
  kind: "model" | "dataset";
  message?: string;
}) {
  return (
    <div
      className="min-h-[108px] space-y-3 py-0.5"
      aria-busy="true"
      aria-live="polite"
    >
      <div className="flex items-center gap-2 text-ui-12p5 text-muted-foreground">
        <Spinner className="size-3.5" />
        {message ?? `Loading ${kind === "dataset" ? "dataset" : "model"} card…`}
      </div>
      <div className="space-y-2" aria-hidden="true">
        <div className="h-2.5 w-11/12 rounded-full bg-muted/60" />
        <div className="h-2.5 w-4/5 rounded-full bg-muted/50" />
        <div className="h-2.5 w-2/3 rounded-full bg-muted/40" />
      </div>
    </div>
  );
}

async function loadReadmeFromCache({
  cacheKey,
  repoId,
  kind,
  hfToken,
}: {
  cacheKey: string;
  repoId: string;
  kind: "model" | "dataset";
  hfToken: string | undefined;
}): Promise<ReadmeCacheEntry> {
  const cached = readmeCache.get(cacheKey);
  if (cached) {
    if (isReadmeCachePromise(cached)) return cached;
    if (cached.expiresAt > Date.now()) return cached.entry;
    readmeCache.delete(cacheKey);
  }

  const promise = fetchReadme(repoId, kind, hfToken || null).then((fetched) => {
    if (fetched == null) {
      return {
        body: null,
        baseUrl: null,
        error: `No ${kind === "dataset" ? "dataset" : "model"} card available.`,
      };
    }
    return {
      body: prepareReadmeBody(fetched.markdown),
      baseUrl: readmeBaseUrl(repoId, kind, fetched.branch),
      error: null,
    };
  });

  readmeCache.set(cacheKey, promise);
  try {
    const entry = await promise;
    if (entry.error == null) {
      readmeCache.set(cacheKey, {
        entry,
        expiresAt: Date.now() + README_CACHE_TTL_MS,
      });
    } else if (readmeCache.get(cacheKey) === promise) {
      readmeCache.delete(cacheKey);
    }
    return entry;
  } catch (err) {
    if (readmeCache.get(cacheKey) === promise) {
      readmeCache.delete(cacheKey);
    }
    throw err;
  }
}

function loadPlugins(needs: {
  math: boolean;
  mermaid: boolean;
}): ReadmePlugins {
  const plugins: ReadmePlugins = { code: streamdownCode };
  if (needs.math) plugins.math = streamdownMath;
  if (needs.mermaid) plugins.mermaid = streamdownMermaid;
  return plugins;
}

export function ModelReadme({
  repoId,
  kind = "model",
  subject = kind,
}: {
  repoId: string;
  kind?: "model" | "dataset";
  subject?: ReadmeSubject;
}) {
  const hfToken = useHfTokenStore((s) => s.token);
  const online = useOnlineStatus();
  const tokenFingerprint = useMemo(() => fingerprintToken(hfToken), [hfToken]);
  const stateKey = useMemo(
    () => `${kind}::${repoId}::${tokenFingerprint}`,
    [kind, repoId, tokenFingerprint],
  );
  const [state, setState] = useState<ReadmeState>(() => {
    const cached = readResolvedReadmeCache(stateKey);
    if (cached) return readmeStateFromEntry(stateKey, cached);
    return {
      key: stateKey,
      body: null,
      baseUrl: null,
      loading: !isHuggingFaceOffline(),
      error: null,
      plugins: null,
    };
  });
  const matchingState = state.key === stateKey ? state : null;
  const cachedEntry = readResolvedReadmeCache(stateKey);
  const cachedState = cachedEntry
    ? readmeStateFromEntry(stateKey, cachedEntry)
    : null;
  const current = matchingState && hasReadmeContent(matchingState)
    ? matchingState
    : cachedState && hasReadmeContent(cachedState)
      ? cachedState
      : !online
        ? {
            key: stateKey,
            body: null,
            baseUrl: null,
            loading: false,
            error: readmeOfflineMessage(subject),
            plugins: null,
          }
        : matchingState ?? {
            key: stateKey,
            body: null,
            baseUrl: null,
            loading: true,
            error: null,
            plugins: null,
          };

  const urlTransform = useMemo(
    () =>
      current.baseUrl ? createReadmeUrlTransform(current.baseUrl) : undefined,
    [current.baseUrl],
  );

  useEffect(() => {
    let canceled = false;
    if (!online) return;
    void loadReadmeFromCache({
      cacheKey: stateKey,
      repoId,
      kind,
      hfToken,
    })
      .then((entry) => {
        if (canceled) return;
        startTransition(() => {
          setState({
            key: stateKey,
            body: entry.body,
            baseUrl: entry.baseUrl,
            loading: false,
            error: entry.error,
            plugins: null,
          });
        });
      })
      .catch((err) => {
        if (canceled) return;
        startTransition(() => {
          setState({
            key: stateKey,
            body: null,
            baseUrl: null,
            loading: false,
            error:
              err instanceof Error ? err.message : "Failed to load model card",
            plugins: null,
          });
        });
      });
    return () => {
      canceled = true;
    };
  }, [repoId, kind, hfToken, stateKey, online]);

  useEffect(() => {
    // Plugin assembly is synchronous, so it runs regardless of connectivity: a
    // body served from the README cache while offline still clears the preparing gate.
    if (!current.body || current.loading || current.error || current.plugins) {
      return;
    }
    let canceled = false;
    const key = current.key;
    const body = current.body;
    const baseUrl = current.baseUrl;
    const applyPlugins = (next: ReadmePlugins) => {
      if (canceled) return;
      startTransition(() => {
        setState((prev) => {
          if (prev.key === key) {
            return prev.plugins ? prev : { ...prev, plugins: next };
          }
          return {
            key,
            body,
            baseUrl,
            loading: false,
            error: null,
            plugins: next,
          };
        });
      });
    };
    const cancelIdle = scheduleIdleTask(() => {
      applyPlugins(
        loadPlugins({
          math: READMEX_NEEDS_MATH.test(body),
          mermaid: READMEX_NEEDS_MERMAID.test(body),
        }),
      );
    }, 400);
    return () => {
      canceled = true;
      cancelIdle();
    };
  }, [
    current.body,
    current.loading,
    current.error,
    current.plugins,
    current.key,
    current.baseUrl,
  ]);

  if (current.loading) {
    return (
      <ReadmePlaceholder kind={kind} message={readmeLoadingMessage(subject)} />
    );
  }

  if (current.error) {
    const errorMessage = isMissingReadmeError(current.error)
      ? readmeMissingMessage(subject)
      : current.error.toLowerCase().includes("offline")
        ? current.error
        : readmeUnavailableMessage(subject);
    return (
      <p className="min-h-[44px] text-ui-12p5 text-muted-foreground">
        {errorMessage}
      </p>
    );
  }

  if (!current.body) {
    return (
      <p className="min-h-[44px] text-ui-12p5 text-muted-foreground">
        {readmeMissingMessage(subject)}
      </p>
    );
  }

  if (!current.plugins) {
    return (
      <ReadmePlaceholder
        kind={kind}
        message={readmePreparingMessage(subject)}
      />
    );
  }

  return (
    <div className={cn("hub-readme-prose", PROSE)}>
      <ReadmeOnlineContext.Provider value={online}>
        <Streamdown
          mode="static"
          plugins={current.plugins}
          controls={false}
          components={README_COMPONENTS}
          allowedTags={README_ALLOWED_TAGS}
          urlTransform={urlTransform}
        >
          {current.body}
        </Streamdown>
      </ReadmeOnlineContext.Provider>
    </div>
  );
}
