// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  defaultRangeExtractor,
  observeElementOffset,
  useVirtualizer,
  type VirtualItem,
} from "@tanstack/react-virtual";
import {
  Fragment as InlineFragment,
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  CodeBlockActions,
  MarkdownTextSource,
  SearchImagesEnabledContext,
} from "./markdown-text";
import { FenceLine } from "./code-fence-defer";
import {
  useReasoningHighlight,
  type ReasoningLineTokens,
} from "./use-reasoning-highlight";
import {
  ReasoningTranscriptIndex,
  resolveReasoningAnchor,
  type ReasoningReadingAnchor,
  type ReasoningFragment,
} from "./reasoning-transcript-index";
import {
  captureReasoningAnchor,
  reasoningTextRange,
} from "./reasoning-reading-anchor";
import {
  useAdjustForContentInsertedAbove,
  useDetachThreadFromBottom,
} from "./use-intent-aware-autoscroll";
import { cn } from "@/lib/utils";

type Props = {
  initialAnchor?: ReasoningReadingAnchor;
  documents: readonly string[];
  messageId: string;
  messageHasRenderableRenderHtmlTool: boolean;
  streaming: boolean;
};

const CodeFragment = memo(
  function CodeFragment({
    fragment,
    result,
  }: { fragment: ReasoningFragment; result: ReasoningLineTokens }) {
    const code = fragment.code!;
    return (
      <>
        {code.lines.map(({ line, column, text }) => {
          const tokens = result.get(line);
          // A delayed grammar must never display the previous, shorter source.
          if (
            !tokens ||
            tokens
              .map((token) => token.content)
              .join("")
              .slice(column, column + text.length) !== text
          ) {
            return (
              <InlineFragment key={`${line}:${column}`}>
                {column === 0 && line > 0 ? "\n" : ""}
                {text}
              </InlineFragment>
            );
          }
          let offset = 0;
          const clipped = tokens.flatMap((token) => {
            const start = offset;
            offset += token.content.length;
            const content = token.content.slice(
              Math.max(0, column - start),
              Math.max(
                0,
                Math.min(token.content.length, column + text.length - start),
              ),
            );
            return content ? [{ ...token, content }] : [];
          });
          return (
            <InlineFragment key={`${line}:${column}`}>
              {column === 0 && line > 0 ? "\n" : ""}
              <FenceLine line={clipped} windowed inline />
            </InlineFragment>
          );
        })}
      </>
    );
  },
  (previous, next) => {
    // Appending later lines cannot change this fragment's grammar or text. Keep its
    // highlighted subtree and selection alive instead of repainting it for every token.
    const a = previous.fragment;
    const b = next.fragment;
    return (
      a.key === b.key &&
      a.text === b.text &&
      a.first === b.first &&
      a.last === b.last &&
      a.code?.language === b.code?.language &&
      previous.result.get(a.code!.lines[0].line) ===
        next.result.get(b.code!.lines[0].line)
    );
  },
);

type RowProps = {
  item: VirtualItem;
  fragment: ReasoningFragment;
  top: number;
  measure: (element: HTMLElement | null) => void;
  children: React.ReactNode;
};

function Row({ item, fragment, top, measure, children }: RowProps) {
  return (
    <div
      ref={measure}
      data-index={item.index}
      data-reasoning-fragment={fragment.key}
      className="absolute top-0 left-0 w-full min-w-0"
      style={{
        transform: `translateY(${item.start - top}px)`,
        contain: "layout style",
      }}
    >
      {children}
    </div>
  );
}

function CodeGroup({
  items,
  fragments,
  top,
  measure,
  streaming,
}: {
  items: VirtualItem[];
  fragments: ReasoningFragment[];
  top: number;
  measure: RowProps["measure"];
  streaming: boolean;
}) {
  const surface = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const element = surface.current;
    if (!element) return;
    // ResizeObserver does not report inline span geometry. Observe their shared
    // formatting surface and feed each bounded row's measured contribution back.
    const observer = new ResizeObserver(() => {
      for (const row of element.querySelectorAll<HTMLElement>(
        "[data-reasoning-code-row]",
      ))
        measure(row);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [measure]);
  const code = fragments[items[0].index].code!;
  // One highlighter subscription per visible fence, with its complete grammar context.
  const lines = [
    ...new Set(
      items.flatMap((item) =>
        fragments[item.index].code!.lines.map((line) => line.line),
      ),
    ),
  ];
  const result = useReasoningHighlight(code.source, code.language, lines);
  const first = fragments[items[0].index];
  const last = fragments[items.at(-1)!.index];
  return (
    <div
      ref={surface}
      data-slot="reasoning-code-fragment"
      data-language={code.language ?? undefined}
      className={cn(
        "aui-reasoning-code-fragment absolute top-0 left-0 w-full min-w-0",
        first.first && "aui-reasoning-code-first",
        last.last && "aui-reasoning-code-last",
      )}
      style={{ transform: `translateY(${items[0].start - top}px)` }}
    >
      {first.first && (
        <CodeBlockActions
          disabled={streaming}
          language={code.language}
          source={code.source}
        />
      )}
      {first.first && (
        <div className="mb-2 min-h-4 pr-20 text-xs text-muted-foreground">
          {code.language}
        </div>
      )}
      <pre className="!m-0 min-h-[1lh] whitespace-pre-wrap [overflow-wrap:anywhere] font-mono">
        <code>
          {items.map((item, i) => (
            <InlineFragment key={item.key}>
              {i > 0 && items[i - 1].index + 1 !== item.index && (
                <span
                  aria-hidden="true"
                  data-reasoning-code-gap=""
                  className="block"
                  style={{ height: Math.max(0, item.start - items[i - 1].end) }}
                />
              )}
              <span
                ref={measure}
                data-index={item.index}
                data-reasoning-fragment={fragments[item.index].key}
                data-reasoning-code-row=""
              >
                <CodeFragment
                  fragment={fragments[item.index]}
                  result={result}
                />
              </span>
            </InlineFragment>
          ))}
        </code>
      </pre>
    </div>
  );
}

const Fragment = memo(function Fragment({
  fragment,
  messageId,
  messageHasRenderableRenderHtmlTool,
  streaming,
}: Omit<Props, "documents"> & { fragment: ReasoningFragment }) {
  const root = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    // Continuation containers preserve indentation/numbering but must not paint
    // another bullet for the same item. Later, real siblings retain their markers.
    const items: HTMLElement[] = [];
    let container: Element | null | undefined = root.current?.firstElementChild;
    for (
      let depth = 0;
      depth < (fragment.listContinuationDepth ?? 0);
      depth += 1
    ) {
      while (
        container?.firstElementChild &&
        ["DIV", "BLOCKQUOTE"].includes(container.firstElementChild.tagName)
      )
        container = container.firstElementChild;
      const item: HTMLElement | null | undefined = container?.querySelector(
        ":scope > :is(ol, ul):first-child > li:first-child",
      );
      if (!item) break;
      item.dataset.reasoningListContinuation = "";
      items.push(item);
      container = item;
    }
    return () => {
      for (const item of items) delete item.dataset.reasoningListContinuation;
    };
  }, [fragment]);
  if (fragment.hidden) return null;
  return (
    <div
      ref={root}
      data-table-continuation={fragment.tableContinuation || undefined}
      className={cn("aui-reasoning-prose-fragment", fragment.first && "pt-4")}
    >
      <MarkdownTextSource
        messageId={messageId}
        messageHasRenderableRenderHtmlTool={messageHasRenderableRenderHtmlTool}
        sourceText={fragment.renderText ?? fragment.text}
        streaming={streaming}
      />
    </div>
  );
});

export function ReasoningTranscript({
  initialAnchor,
  documents,
  messageId,
  messageHasRenderableRenderHtmlTool,
  streaming,
}: Props) {
  const root = useRef<HTMLDivElement>(null);
  const [index] = useState(() => new ReasoningTranscriptIndex());
  const fragments = useMemo(() => index.update(documents), [documents, index]);
  const [anchor, setAnchor] = useState(
    () => initialAnchor && resolveReasoningAnchor(fragments, initialAnchor),
  );
  const [anchoring, setAnchoring] = useState(
    Boolean(anchor && anchor.index >= 0),
  );
  const [geometry, setGeometry] = useState({
    top: 0,
    width: 640,
    lineHeight: 24,
    fontPixels: 15,
  });
  const [protectedKeys, setProtectedKeys] = useState<Set<string>>(
    () => new Set(),
  );
  const adjustAbove = useAdjustForContentInsertedAbove();
  const detach = useDetachThreadFromBottom();
  const viewport = useCallback(
    () => root.current?.closest<HTMLDivElement>(".aui-thread-viewport") ?? null,
    [],
  );

  const virtualizer = useVirtualizer<HTMLDivElement, HTMLElement>({
    count: fragments.length,
    getScrollElement: viewport,
    getItemKey: (i) => fragments[i].key,
    scrollMargin: geometry.top,
    initialOffset: () => viewport()?.scrollTop ?? 0,
    observeElementOffset: (instance, callback) => {
      // The viewport predates this transcript. Subscribe with its current offset;
      // waiting for the next scroll event leaves TanStack's initial zero cached.
      callback(instance.scrollElement?.scrollTop ?? 0, false);
      return observeElementOffset(instance, callback);
    },
    overscan: 2,
    measureElement: (element) => {
      if (!element.hasAttribute("data-reasoning-code-row"))
        return element.getBoundingClientRect().height;
      const surface = element.closest<HTMLElement>(
        '[data-slot="reasoning-code-fragment"]',
      )!;
      const previous = element.previousElementSibling;
      const bottom = element.nextElementSibling
        ? element.getBoundingClientRect().bottom
        : surface.getBoundingClientRect().bottom;
      const top = previous
        ? previous.getBoundingClientRect().bottom
        : surface.getBoundingClientRect().top;
      const margin = previous
        ? 0
        : Number.parseFloat(getComputedStyle(surface).marginTop) || 0;
      return Math.max(0, bottom - top + margin);
    },
    estimateSize: (i) => {
      const fragment = fragments[i];
      if (fragment.hidden) return 0;
      const code = fragment.code;
      const columns = Math.max(
        12,
        Math.floor(
          (geometry.width - (code ? 32 : 0)) /
            (geometry.fontPixels * (code ? 0.58 : 0.48)),
        ),
      );
      const lines = fragment.text
        .split("\n")
        .reduce(
          (sum, line) => sum + Math.max(1, Math.ceil(line.length / columns)),
          0,
        );
      return code
        ? lines * geometry.lineHeight +
            (fragment.first ? 40 : 0) +
            (fragment.last ? 16 : 0)
        : Math.max(1, lines - 2) * geometry.lineHeight +
            (fragment.first ? 16 : 0);
    },
    rangeExtractor: (range) => {
      const mounted = new Set(defaultRangeExtractor(range));
      if (anchoring && anchor) mounted.add(anchor.index);
      if (protectedKeys.size)
        fragments.forEach((fragment, i) => {
          if (protectedKeys.has(fragment.key)) mounted.add(i);
        });
      return [...mounted].sort((a, b) => a - b);
    },
    // TanStack reports measurement corrections; the thread controller owns all scroll writes.
    // Its own initial-position writes are intentionally ignored.
    scrollToFn: (offset, { adjustments }) => {
      const element = viewport();
      if (element && adjustments !== undefined)
        adjustAbove(offset + adjustments - element.scrollTop);
    },
  });
  virtualizer.shouldAdjustScrollPositionOnItemSizeChange = (item) =>
    // Ref measurements run before the virtualizer observes the shared viewport.
    // Its offset is still zero then: applying that correction would send a reader
    // back to the beginning when a live trace first crosses the threshold.
    virtualizer.scrollElement !== null &&
    item.end < (viewport()?.scrollTop ?? 0);

  useLayoutEffect(() => {
    const element = root.current;
    const scroll = viewport();
    if (!element || !scroll) return;
    let frame = 0;
    let width = 0;
    let readingAnchor: (ReasoningReadingAnchor & { index: number }) | undefined;
    const measure = () => {
      frame = 0;
      const rect = element.getBoundingClientRect();
      if (!rect.width) return;
      const style = getComputedStyle(element);
      const next = {
        top: rect.top - scroll.getBoundingClientRect().top + scroll.scrollTop,
        width: rect.width,
        lineHeight: Number.parseFloat(style.lineHeight) || 24,
        fontPixels: Number.parseFloat(style.fontSize) || 15,
      };
      setGeometry((old) =>
        Object.keys(next).every(
          (key) =>
            next[key as keyof typeof next] === old[key as keyof typeof old],
        )
          ? old
          : next,
      );
      if (width && width !== rect.width) {
        // Capture before reflow: a long user message above us may push the whole
        // transcript offscreen at the new width before ResizeObserver runs.
        if (readingAnchor) {
          setAnchor(readingAnchor);
          setAnchoring(true);
        }
        virtualizer.measure();
      } else {
        readingAnchor = undefined;
        for (const row of element.querySelectorAll<HTMLElement>(
          "[data-index]",
        )) {
          const captured = captureReasoningAnchor(row, scroll);
          if (captured) {
            readingAnchor = { ...captured, index: Number(row.dataset.index) };
            break;
          }
        }
      }
      width = rect.width;
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    const observer = new ResizeObserver(schedule);
    observer.observe(element);
    observer.observe(scroll);
    if (scroll.firstElementChild) observer.observe(scroll.firstElementChild);
    scroll.addEventListener("scroll", schedule, { passive: true });
    measure();
    return () => {
      observer.disconnect();
      scroll.removeEventListener("scroll", schedule);
      cancelAnimationFrame(frame);
    };
  }, [viewport, virtualizer]);

  useLayoutEffect(() => {
    if (!anchoring || !anchor) return;
    // Keep this passage mounted through the initial estimates, then reconcile its
    // measured position through the thread's single scroll owner before releasing it.
    const frame = requestAnimationFrame(() => {
      const row = root.current?.querySelector(`[data-index="${anchor.index}"]`);
      const passage =
        row && reasoningTextRange(row, anchor.text, anchor.occurrence);
      if (passage)
        adjustAbove(passage.getBoundingClientRect().top - anchor.top);
      setAnchoring(false);
    });
    return () => cancelAnimationFrame(frame);
  }, [anchor, anchoring, adjustAbove]);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const protect = () => {
      const selection = window.getSelection();
      const keys = new Set<string>();
      for (const row of element.querySelectorAll<HTMLElement>(
        "[data-reasoning-fragment]",
      )) {
        const focused =
          document.activeElement && row.contains(document.activeElement);
        let selected = false;
        if (selection && !selection.isCollapsed) {
          for (let i = 0; i < selection.rangeCount; i += 1)
            selected ||= selection.getRangeAt(i).intersectsNode(row);
        }
        if (selected || focused) keys.add(row.dataset.reasoningFragment!);
      }
      if (keys.size) detach();
      setProtectedKeys((old) =>
        old.size === keys.size && [...old].every((key) => keys.has(key))
          ? old
          : keys,
      );
    };
    document.addEventListener("selectionchange", protect);
    element.addEventListener("focusin", protect);
    element.addEventListener("focusout", protect);
    return () => {
      document.removeEventListener("selectionchange", protect);
      element.removeEventListener("focusin", protect);
      element.removeEventListener("focusout", protect);
    };
  }, [detach]);

  const groups: {
    key: string;
    code: boolean;
    items: VirtualItem[];
  }[] = [];
  for (const item of virtualizer.getVirtualItems()) {
    const fragment = fragments[item.index];
    const key = fragment.code
      ? `${fragment.document}:${fragment.start}`
      : fragment.key;
    const previous = groups.at(-1);
    if (previous?.key === key) previous.items.push(item);
    else
      groups.push({
        key,
        code: Boolean(fragment.code),
        items: [item],
      });
  }

  return (
    <SearchImagesEnabledContext.Provider value={false}>
      <div
        ref={root}
        data-slot="reasoning-transcript"
        className="relative min-w-0"
        style={{ height: virtualizer.getTotalSize(), overflowAnchor: "none" }}
      >
        {groups.map((group) => {
          if (group.code)
            return (
              <CodeGroup
                key={group.key}
                items={group.items}
                fragments={fragments}
                top={geometry.top}
                measure={virtualizer.measureElement}
                streaming={streaming}
              />
            );
          const item = group.items[0];
          const fragment = fragments[item.index];
          return (
            <Row
              key={item.key}
              item={item}
              fragment={fragment}
              top={geometry.top}
              measure={virtualizer.measureElement}
            >
              <Fragment
                fragment={fragment}
                messageId={messageId}
                messageHasRenderableRenderHtmlTool={
                  messageHasRenderableRenderHtmlTool
                }
                streaming={streaming && item.index === fragments.length - 1}
              />
            </Row>
          );
        })}
      </div>
    </SearchImagesEnabledContext.Provider>
  );
}
