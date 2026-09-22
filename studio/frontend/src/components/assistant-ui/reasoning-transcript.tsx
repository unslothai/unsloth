// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  defaultRangeExtractor,
  observeElementOffset,
  useVirtualizer,
  type VirtualItem,
} from "@tanstack/react-virtual";
import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
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
  findReasoningAnchor,
  type ReasoningReadingAnchor,
  type ReasoningFragment,
} from "./reasoning-transcript-index";
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
      <div
        data-slot="reasoning-code-fragment"
        data-language={code.language ?? undefined}
        className={cn(
          "aui-reasoning-code-fragment",
          fragment.first && "aui-reasoning-code-first",
          fragment.last && "aui-reasoning-code-last",
        )}
      >
        {fragment.first && code.language && (
          <div className="mb-2 text-xs text-muted-foreground">
            {code.language}
          </div>
        )}
        <pre className="!m-0 whitespace-pre-wrap [overflow-wrap:anywhere] font-mono">
          <code>
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
                  <span key={`${line}:${column}`} className="block min-h-[1lh]">
                    {text || "\n"}
                  </span>
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
                    Math.min(
                      token.content.length,
                      column + text.length - start,
                    ),
                  ),
                );
                return content ? [{ ...token, content }] : [];
              });
              return (
                <FenceLine key={`${line}:${column}`} line={clipped} windowed />
              );
            })}
          </code>
        </pre>
      </div>
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
  measure: (element: HTMLDivElement | null) => void;
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
}: {
  items: VirtualItem[];
  fragments: ReasoningFragment[];
  top: number;
  measure: RowProps["measure"];
}) {
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
  return items.map((item) => (
    <Row
      key={item.key}
      item={item}
      fragment={fragments[item.index]}
      top={top}
      measure={measure}
    >
      <CodeFragment fragment={fragments[item.index]} result={result} />
    </Row>
  ));
}

const Fragment = memo(function Fragment({
  fragment,
  messageId,
  messageHasRenderableRenderHtmlTool,
  streaming,
}: Omit<Props, "documents"> & { fragment: ReasoningFragment }) {
  return (
    <div
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
    () =>
      initialAnchor && {
        ...initialAnchor,
        index: findReasoningAnchor(fragments, initialAnchor.text),
      },
  );
  const [anchoring, setAnchoring] = useState(
    Boolean(anchor && anchor.index >= 0),
  );
  const [geometry, setGeometry] = useState({
    top: 0,
    width: 640,
    lineHeight: 24,
    fontSize: 15,
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

  const virtualizer = useVirtualizer<HTMLDivElement, HTMLDivElement>({
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
    measureElement: (element) => element.getBoundingClientRect().height,
    estimateSize: (i) => {
      const fragment = fragments[i];
      const code = fragment.code;
      const columns = Math.max(
        12,
        Math.floor(
          (geometry.width - (code ? 32 : 0)) /
            (geometry.fontSize * (code ? 0.58 : 0.48)),
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
        fontSize: Number.parseFloat(style.fontSize) || 15,
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
        const viewportTop = scroll.getBoundingClientRect().top;
        const passage = [
          ...element.querySelectorAll<HTMLElement>(
            "p, pre, li, h1, h2, h3, h4, h5, h6",
          ),
        ].find(
          (node) =>
            node.textContent &&
            node.getBoundingClientRect().top >= viewportTop &&
            node.getBoundingClientRect().top <
              viewportTop + scroll.clientHeight,
        );
        const row = passage?.closest<HTMLElement>("[data-index]");
        readingAnchor =
          passage && row
            ? {
                text: passage.textContent!.slice(0, 200),
                top: passage.getBoundingClientRect().top,
                index: Number(row.dataset.index),
              }
            : undefined;
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
        row &&
        [
          ...row.querySelectorAll<HTMLElement>(
            "p, pre, li, h1, h2, h3, h4, h5, h6",
          ),
        ].find((node) => node.textContent?.includes(anchor.text));
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

  const groups: { key: string; code: boolean; items: VirtualItem[] }[] = [];
  for (const item of virtualizer.getVirtualItems()) {
    const fragment = fragments[item.index];
    const key = fragment.code
      ? `${fragment.document}:${fragment.start}`
      : fragment.key;
    const previous = groups.at(-1);
    if (previous?.key === key) previous.items.push(item);
    else groups.push({ key, code: Boolean(fragment.code), items: [item] });
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
