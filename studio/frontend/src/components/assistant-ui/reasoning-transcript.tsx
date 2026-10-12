// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Component,
  type CSSProperties,
  memo,
  startTransition,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { flushSync } from "react-dom";
import {
  CodeBlockActions,
  MarkdownTextSource,
  SearchImagesEnabledContext,
} from "./markdown-text";
import { CodeHighlightModeContext } from "./code-highlight-context";
import { FenceLine } from "./code-fence-defer";
import { clipLineTokens, type TokenLine } from "./reasoning-code-colour";
import { useThinkingCodeColour } from "./use-thinking-code-colour";
import {
  cachedReasoningTranscriptIndex,
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
import {
  addIsland,
  anchorAtSamePlace,
  buildChunkGrid,
  chunkEndAt,
  chunkPieces,
  chunkStartAt,
  type ChunkPiece,
  type ChunkSlice,
  closedChunkUnchanged,
  estimateFragmentHeights,
  type FragmentGeometry,
  frameBudget,
  INITIAL_VIEWPORTS,
  initialRows,
  isCovered,
  type Island,
  mountPlan,
  NO_ISLANDS,
  WIDEN_CHARACTERS_PER_FRAME,
  WIDEN_FRAGMENTS_PER_FRAME,
  widenBudget,
} from "./reasoning-mount-controller";
import {
  CHUNK_ESTIMATE_PROPERTY,
  ROW_SETTLED_ATTRIBUTE,
} from "./reasoning-row-containment-mode";
import { createRemeasureQueue, createSettleQueue } from "./reasoning-settle-queue";
import { hasPendingProgressiveMounts } from "./progressive-messages";
import { panelDragInProgress } from "@/components/ui/panel-drag-overlay";
import { cn } from "@/lib/utils";

type Reserve = { element: HTMLDivElement; from: number; to: number };
type Reading = ReasoningReadingAnchor & {
  key: string;
  row: Element;
  offset: number;
};

type Props = {
  initialAnchor?: ReasoningReadingAnchor;
  documents: readonly string[];
  indexKey: string;
  messageId: string;
  messageHasRenderableRenderHtmlTool: boolean;
  streaming: boolean;
};

const hiddenTranscripts = new WeakSet<Element>();
const blindTranscripts = new WeakSet<Element>();
const rowRemeasurers = new WeakMap<Element, (row: HTMLElement) => void>();
const withoutIdleCallback =
  typeof window !== "undefined" &&
  typeof window.requestIdleCallback !== "function";

const settleQueue = createSettleQueue<HTMLElement>(
  (callback) => requestAnimationFrame(callback),
  (row) => {
    if (!row.isConnected) return;
    row.setAttribute(ROW_SETTLED_ATTRIBUTE, "");
    const transcript = row.parentElement;
    if (transcript && hiddenTranscripts.has(transcript))
      blindTranscripts.add(transcript);
  },
);

const remeasureQueue = createRemeasureQueue<HTMLElement>(
  (callback) => requestAnimationFrame(callback),
  (row) => {
    if (!row.isConnected) return;
    if (row.parentElement) rowRemeasurers.get(row.parentElement)?.(row);
  },
  panelDragInProgress,
);

const resettleRows = (root: HTMLElement): void => {
  for (const row of root.querySelectorAll<HTMLElement>(
    `[${ROW_SETTLED_ATTRIBUTE}]`,
  )) {
    remeasureQueue.add(row);
  }
};

type Spent = { characters: number; rows: number };
type Widener = (characters: number, rows: number) => Spent | null;

const NOTHING_SPENT: Spent = { characters: 0, rows: 0 };
const wideners = new Set<Widener>();
let widenFrame = 0;
const WIDEN_COMMIT_TARGET_MS = 16;
const MIN_WIDEN_SCALE = 1 / 8;
let widenScale = 1;
let commitStartedAt = 0;
let widenCommitMs = 0;

const widenAll = (): void => {
  widenFrame = 0;
  if (widenCommitMs > 0) {
    widenScale = Math.min(
      1,
      Math.max(
        MIN_WIDEN_SCALE,
        widenScale * Math.min(2, WIDEN_COMMIT_TARGET_MS / widenCommitMs),
      ),
    );
    widenCommitMs = 0;
  }
  let characters = Math.ceil(WIDEN_CHARACTERS_PER_FRAME * widenScale);
  let rows = Math.ceil(WIDEN_FRAGMENTS_PER_FRAME * widenScale);
  const pending = [...wideners];
  for (let at = 0; at < pending.length; at += 1) {
    if (characters <= 0 || rows <= 0) break;
    const remaining = pending.length - at;
    const widen = pending[at];
    const spent = widen(
      Math.ceil(characters / remaining),
      Math.ceil(rows / remaining),
    );
    if (spent === null) wideners.delete(widen);
    else {
      characters -= spent.characters;
      rows -= spent.rows;
    }
  }
  if (wideners.size > 0) widenFrame = requestAnimationFrame(widenAll);
};

const requestWiden = (widen: Widener): void => {
  wideners.add(widen);
  if (widenFrame === 0) widenFrame = requestAnimationFrame(widenAll);
};

const cancelWiden = (widen: Widener): void => {
  wideners.delete(widen);
};

const RESETTLE_DELAY_MS = 150;
const STARVED_MS = 200;
const SHIFT_EPSILON_PX = 0.5;

let passageRange: Range | null = null;

const printers = new Set<() => void>();

const printTranscripts = (): void => {
  flushSync(() => {
    for (const print of printers) print();
  });
};

if (typeof window !== "undefined" && typeof window.addEventListener === "function") {
  window.addEventListener("beforeprint", printTranscripts, { capture: true });
  window.matchMedia?.("print")?.addEventListener?.(
    "change",
    (event) => {
      if (event.matches) printTranscripts();
    },
    { capture: true },
  );
}

class CommitBounds extends Component<{
  before: () => void;
  after: () => void;
  children: React.ReactNode;
}> {
  getSnapshotBeforeUpdate() {
    this.props.before();
    return null;
  }
  componentDidUpdate() {
    this.props.after();
  }
  render() {
    return this.props.children;
  }
}

function useSettledRow(
  row: React.RefObject<HTMLElement | null>,
  active: boolean,
): void {
  useLayoutEffect(() => {
    const element = row.current;
    if (!element || !active) return;
    settleQueue.add(element);
    const changed = new MutationObserver(() => {
      if (!element.hasAttribute(ROW_SETTLED_ATTRIBUTE)) settleQueue.add(element);
    });
    changed.observe(element, {
      subtree: true,
      childList: true,
      characterData: true,
    });
    return () => {
      changed.disconnect();
      remeasureQueue.forget(element);
      settleQueue.forget(element);
    };
  }, [row, active]);
}

const codeRowText = (
  fragment: ReasoningFragment,
  leading: boolean,
): string[] => {
  const nodes: string[] = [];
  for (const [at, { line, column, text }] of fragment.code!.lines.entries()) {
    if (column === 0 && line > 0 && !(leading && at === 0)) nodes.push("\n");
    if (text !== "") nodes.push(text);
  }
  return nodes;
};

const colouredRow = (
  fragment: ReasoningFragment,
  leading: boolean,
  tokens: readonly TokenLine[],
): React.ReactNode[] => {
  const nodes: React.ReactNode[] = [];
  for (const [at, { line, column, text }] of fragment.code!.lines.entries()) {
    if (column === 0 && line > 0 && !(leading && at === 0)) nodes.push("\n");
    if (text === "") continue;
    const clipped = clipLineTokens(tokens[line], column, text);
    nodes.push(
      clipped === null ? (
        text
      ) : (
        <FenceLine key={`${line}:${column}`} line={clipped} windowed inline />
      ),
    );
  }
  return nodes;
};

const CodeRow = memo(function CodeRow({
  index,
  fragment,
  leading,
  tokens,
}: {
  index: number;
  fragment: ReasoningFragment;
  leading: boolean;
  tokens: readonly TokenLine[] | null;
}) {
  return (
    <span
      data-index={index}
      data-reasoning-fragment={fragment.key}
      data-reasoning-code-row=""
    >
      {tokens === null
        ? codeRowText(fragment, leading)
        : colouredRow(fragment, leading, tokens)}
    </span>
  );
});

function Row({
  index,
  fragment,
  children,
}: {
  index: number;
  fragment: ReasoningFragment;
  children: React.ReactNode;
}) {
  if (fragment.hidden) return null;
  return (
    <div
      data-index={index}
      data-reasoning-fragment={fragment.key}
      className="min-w-0"
    >
      {children}
    </div>
  );
}

function CodeGroup({
  indices,
  fragments,
  streaming,
}: {
  indices: number[];
  fragments: readonly ReasoningFragment[];
  streaming: boolean;
}) {
  const code = fragments[indices[0]].code!;
  const first = fragments[indices[0]];
  const last = fragments[indices[indices.length - 1]];
  const surface = useRef<HTMLDivElement>(null);
  const colour = useThinkingCodeColour(
    surface,
    code.source,
    code.language,
    // A trace that stopped inside a fence never closes it; it is still finished.
    !code.incomplete || !streaming,
  );
  const breaksAfter =
    !last.last &&
    fragments[indices[indices.length - 1] + 1]?.code?.lines[0]?.column === 0;
  return (
    <div
      ref={surface}
      data-slot="reasoning-code-fragment"
      data-language={code.language ?? undefined}
      className={cn(
        "aui-reasoning-code-fragment relative isolate min-w-0",
        first.first && "aui-reasoning-code-first",
        last.last && "aui-reasoning-code-last",
      )}
    >
      {first.first && (
        <CodeBlockActions
          disabled={streaming}
          language={code.language}
          source={() => code.source}
        />
      )}
      {first.first && (
        <div className="mb-2 min-h-4 pr-20 text-xs text-muted-foreground">
          {code.language}
        </div>
      )}
      <div className="aui-reasoning-code-lines !m-0 min-h-[1lh] whitespace-pre-wrap [overflow-wrap:anywhere] font-mono">
        <code>
          {indices.map((index, at) => (
            <CodeRow
              key={fragments[index].key}
              index={index}
              fragment={fragments[index]}
              leading={at === 0}
              tokens={colour?.tokens ?? null}
            />
          ))}
          {breaksAfter && "\n"}
        </code>
      </div>
    </div>
  );
}

const Fragment = memo(function Fragment({
  fragment,
  messageId,
  messageHasRenderableRenderHtmlTool,
  streaming,
}: Omit<Props, "documents" | "indexKey"> & { fragment: ReasoningFragment }) {
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

type ChunkProps = {
  slices: readonly ChunkSlice[];
  fragments: readonly ReasoningFragment[];
  closed: boolean;
  estimate: number;
  messageId: string;
  messageHasRenderableRenderHtmlTool: boolean;
};

const Chunk = memo(function Chunk({
  slices,
  fragments,
  closed,
  estimate,
  messageId,
  messageHasRenderableRenderHtmlTool,
}: ChunkProps) {
  const box = useRef<HTMLDivElement>(null);
  useSettledRow(box, closed);
  return (
    <div
      ref={box}
      data-reasoning-chunk=""
      className="min-w-0"
      style={{ [CHUNK_ESTIMATE_PROPERTY]: `${estimate}px` } as CSSProperties}
    >
      {slices.map(({ key, code, first, end, streaming }) => {
        if (code) {
          const indices: number[] = [];
          for (let i = first; i < end; i += 1) indices.push(i);
          return (
            <CodeGroup
              key={key}
              indices={indices}
              fragments={fragments}
              streaming={streaming}
            />
          );
        }
        const fragment = fragments[first];
        return (
          <Row key={key} index={first} fragment={fragment}>
            <Fragment
              fragment={fragment}
              messageId={messageId}
              messageHasRenderableRenderHtmlTool={
                messageHasRenderableRenderHtmlTool
              }
              streaming={streaming}
            />
          </Row>
        );
      })}
    </div>
  );
}, closedChunkUnchanged);

const DEFAULT_GEOMETRY: FragmentGeometry = {
  width: 640,
  lineHeight: 24,
  fontPixels: 15,
  codeLineHeight: 19,
  codeFontPixels: 12,
  codeFooter: 12,
};

const measureGeometry = (
  element: HTMLElement,
  codeProbe: HTMLElement | null,
): FragmentGeometry | null => {
  const width = element.getBoundingClientRect().width;
  if (!width) return null;
  const style = getComputedStyle(element);
  const lines = codeProbe?.firstElementChild;
  const code = lines ? getComputedStyle(lines) : null;
  const footer = codeProbe ? getComputedStyle(codeProbe) : null;
  return {
    width,
    lineHeight: Number.parseFloat(style.lineHeight) || DEFAULT_GEOMETRY.lineHeight,
    fontPixels: Number.parseFloat(style.fontSize) || DEFAULT_GEOMETRY.fontPixels,
    codeLineHeight:
      Number.parseFloat(code?.lineHeight ?? "") || DEFAULT_GEOMETRY.codeLineHeight,
    codeFontPixels:
      Number.parseFloat(code?.fontSize ?? "") || DEFAULT_GEOMETRY.codeFontPixels,
    codeFooter: footer
      ? (Number.parseFloat(footer.paddingBottom) || 0) +
        (Number.parseFloat(footer.borderBottomWidth) || 0)
      : DEFAULT_GEOMETRY.codeFooter,
  };
};

const keepGeometry =
  (next: FragmentGeometry) =>
  (old: FragmentGeometry): FragmentGeometry =>
    (Object.keys(next) as (keyof FragmentGeometry)[]).every(
      (key) => old[key] === next[key],
    )
      ? old
      : next;

export function ReasoningTranscript({
  initialAnchor,
  documents,
  indexKey,
  messageId,
  messageHasRenderableRenderHtmlTool,
  streaming,
}: Props) {
  const root = useRef<HTMLDivElement>(null);
  const codeProbe = useRef<HTMLDivElement>(null);
  const [index] = useState(() => cachedReasoningTranscriptIndex(indexKey));
  const fragments = useMemo(() => index.update(documents), [documents, index]);
  const fragmentsRef = useRef(fragments);
  const [geometry, setGeometry] = useState(DEFAULT_GEOMETRY);
  const estimates = useMemo(
    () => estimateFragmentHeights(fragments, geometry),
    [fragments, geometry],
  );
  const grid = useMemo(
    () => buildChunkGrid(fragments, estimates),
    [fragments, estimates],
  );
  const [restore, setRestore] = useState(() => {
    const anchor =
      initialAnchor && resolveReasoningAnchor(fragments, initialAnchor);
    return anchor && anchor.index >= 0 ? anchor : null;
  });
  const origin = `${index.generation}:${fragments[0]?.key ?? ""}`;
  const originRef = useRef(origin);
  const [mount, setMount] = useState(() => {
    const rows = chunkEndAt(
      grid,
      initialRows(
        estimates,
        INITIAL_VIEWPORTS * (typeof window === "undefined" ? 900 : window.innerHeight),
        restore?.index ?? -1,
        fragments.map((fragment) => fragment.text.length),
      ),
    );
    return {
      origin,
      limit: isCovered(rows, fragments.length) ? Number.POSITIVE_INFINITY : rows,
    };
  });
  if (mount.origin !== origin) {
    const value = chunkEndAt(
      grid,
      initialRows(
        estimates,
        INITIAL_VIEWPORTS * window.innerHeight,
        -1,
        fragments.map((fragment) => fragment.text.length),
      ),
    );
    setMount({
      origin,
      limit: isCovered(value, fragments.length) ? Number.POSITIVE_INFINITY : value,
    });
  }
  const limit = mount.limit;
  const setLimit = useCallback((value: number) => {
    setMount({ origin: originRef.current, limit: value });
  }, []);
  const limitRef = useRef(limit);
  const committedRef = useRef(limit);
  const streamingRef = useRef(streaming);
  const estimatesRef = useRef(estimates);
  const gridRef = useRef(grid);
  const [islandState, setIslandState] = useState<{
    origin: string;
    list: readonly Island[];
  }>(() => ({ origin, list: NO_ISLANDS }));
  const islands = islandState.origin === origin ? islandState.list : NO_ISLANDS;
  const setIslands = useCallback((list: readonly Island[]) => {
    setIslandState({ origin: originRef.current, list });
  }, []);
  const islandsRef = useRef<readonly Island[]>(NO_ISLANDS);
  const mountedIslands = useRef<readonly Island[]>(NO_ISLANDS);
  const recapture = useRef<(fresh?: boolean) => void>(() => {});
  const carry = useRef<(previous: readonly ReasoningFragment[]) => boolean>(
    () => false,
  );
  const demandNow = useRef<() => void>(() => {});
  const requestedAt = useRef(0);
  const reserves = useRef(new Map<string, Reserve>());
  const commitEdge = useRef({ before: () => {}, after: () => {} });
  const beforeCommit = useCallback(() => {
    commitStartedAt = performance.now();
    commitEdge.current.before();
  }, []);
  const afterCommit = useCallback(() => commitEdge.current.after(), []);
  const mountedAt = useRef(0);
  useLayoutEffect(() => {
    mountedAt.current = performance.now();
  }, []);
  useLayoutEffect(() => {
    const previous = fragmentsRef.current;
    fragmentsRef.current = fragments;
    streamingRef.current = streaming;
    estimatesRef.current = estimates;
    gridRef.current = grid;
    const restarted = originRef.current !== origin;
    if (restarted) {
      originRef.current = origin;
      mountedAt.current = performance.now();
      islandsRef.current = NO_ISLANDS;
      limitRef.current = limit;
    }
    const grew = committedRef.current !== limit;
    committedRef.current = limit;
    if (limit > limitRef.current) limitRef.current = limit;
    const carried = restarted && carry.current(previous);
    if (grew || mountedIslands.current !== islands) {
      mountedIslands.current = islands;
      if (!carried) recapture.current();
    }
    if (grew && withoutIdleCallback)
      widenCommitMs = Math.max(widenCommitMs, performance.now() - commitStartedAt);
    if (restarted && !carried) demandNow.current();
  });
  useEffect(() => {
    const print = () => {
      const value = Number.POSITIVE_INFINITY;
      if (committedRef.current === value) return;
      limitRef.current = value;
      setLimit(value);
    };
    printers.add(print);
    return () => {
      printers.delete(print);
    };
  }, [setLimit]);
  const adjustAbove = useAdjustForContentInsertedAbove();
  const detach = useDetachThreadFromBottom();
  const covered = isCovered(limit, fragments.length);

  useEffect(() => {
    if (covered) return;
    const widen: Widener = (characters, share) => {
      const all = fragmentsRef.current;
      const current = limitRef.current;
      if (committedRef.current !== current) {
        if (performance.now() - requestedAt.current < STARVED_MS)
          return NOTHING_SPENT;
        requestedAt.current = performance.now();
        setLimit(current);
        return NOTHING_SPENT;
      }
      if (isCovered(current, all.length)) return null;
      const allowed = frameBudget(performance.now() - mountedAt.current);
      const budgeted = widenBudget(
        all.map((fragment) => fragment.text.length),
        current,
        Math.min(characters, allowed.characters),
        Math.min(share, allowed.fragments),
      );
      const spent = { characters: budgeted.spent, rows: budgeted.rows - current };
      let rows = chunkEndAt(gridRef.current, budgeted.rows);
      for (const skip of islandsRef.current)
        if (rows >= skip.start && rows < skip.end) rows = skip.end;
      requestedAt.current = performance.now();
      limitRef.current = isCovered(rows, all.length)
        ? Number.POSITIVE_INFINITY
        : rows;
      const value = limitRef.current;
      if (streamingRef.current) setLimit(value);
      else startTransition(() => setLimit(value));
      return spent;
    };
    requestWiden(widen);
    return () => cancelWiden(widen);
  }, [covered, setLimit]);

  useLayoutEffect(() => {
    const element = root.current;
    const next = element && measureGeometry(element, codeProbe.current);
    if (next) setGeometry(keepGeometry(next));
  }, []);

  useLayoutEffect(() => {
    const pending = restore;
    const element = root.current;
    if (!pending || !element) return;
    const row = element.querySelector(`[data-index="${pending.index}"]`);
    if (!row) return;
    setRestore(null);
    requestAnimationFrame(() => {
      if (!row.isConnected) return;
      const passage = reasoningTextRange(row, pending.text, pending.occurrence);
      if (passage)
        adjustAbove(passage.getBoundingClientRect().top - pending.top);
      recapture.current(true);
    });
  }, [restore, limit, adjustAbove]);

  useLayoutEffect(() => {
    const element = root.current;
    const scroll = element?.closest<HTMLElement>(".aui-thread-viewport");
    if (!element || !scroll) return;
    let width = element.getBoundingClientRect().width;
    let frame = 0;
    let reading: Reading | undefined;
    const firstEndingBelow = (items: ArrayLike<Element>, top: number) => {
      let low = 0;
      let high = items.length - 1;
      let first = items.length;
      while (low <= high) {
        const middle = (low + high) >> 1;
        if (items[middle].getBoundingClientRect().bottom > top) {
          first = middle;
          high = middle - 1;
        } else {
          low = middle + 1;
        }
      }
      return first;
    };
    const visibleAnchor = (): Reading | undefined => {
      const chunks = element.querySelectorAll<HTMLElement>(
        ":scope > [data-reasoning-chunk]",
      );
      const bounds = scroll.getBoundingClientRect();
      const rootTop = element.getBoundingClientRect().top;
      for (
        let at = firstEndingBelow(chunks, bounds.top);
        at < chunks.length;
        at += 1
      ) {
        const chunk = chunks[at];
        if (chunk.getBoundingClientRect().top >= bounds.bottom) break;
        const rows = chunk.querySelectorAll<HTMLElement>(
          "[data-reasoning-fragment]",
        );
        for (let i = firstEndingBelow(rows, bounds.top); i < rows.length; i += 1) {
          const row = rows[i];
          if (row.getBoundingClientRect().top >= bounds.bottom) return undefined;
          const key = row.dataset.reasoningFragment;
          const captured = captureReasoningAnchor(row, scroll);
          if (captured && key !== undefined)
            return { ...captured, key, row, offset: captured.top - rootTop };
        }
      }
      return undefined;
    };
    const passageOf = (anchor: Reading) => {
      const row =
        anchor.row.isConnected
        && element.contains(anchor.row)
        && anchor.row.getAttribute("data-reasoning-fragment") === anchor.key
          ? anchor.row
          : element.querySelector(
              `[data-reasoning-fragment="${CSS.escape(anchor.key)}"]`,
            );
      if (!row) return null;
      anchor.row = row;
      passageRange ??= document.createRange();
      return reasoningTextRange(row, anchor.text, anchor.occurrence, passageRange);
    };
    let residue = 0;
    const hold = (anchor: Reading): boolean => {
      residue = 0;
      const rootTop = element.getBoundingClientRect().top;
      const was = rootTop + anchor.offset;
      const bounds = scroll.getBoundingClientRect();
      if (was < bounds.top - bounds.height || was > bounds.bottom + bounds.height)
        return false;
      const passage = passageOf(anchor);
      if (!passage) return false;
      const top = passage.getBoundingClientRect().top;
      const shift = top - was;
      if (Math.abs(shift) < SHIFT_EPSILON_PX) {
        residue = shift;
        return true;
      }
      anchor.offset = top - rootTop;
      anchor.top = was;
      const before = scroll.scrollTop;
      const rounded = Math.round(shift);
      const acted = adjustAbove(rounded);
      const applied = scroll.scrollTop - before;
      if (applied !== 0 || (acted && rounded !== 0)) {
        anchor.offset -= shift - applied;
        residue = shift - applied;
      }
      return true;
    };
    const rowAt = (from: number, to: number, offset: number): number => {
      const sizes = estimatesRef.current;
      let at = from;
      for (let covered = 0; at < to && covered + sizes[at] <= offset; at += 1) {
        covered += sizes[at];
      }
      return at;
    };
    const demand = () => {
      const all = fragmentsRef.current.length;
      const mounted = committedRef.current;
      const prefix = isCovered(mounted, all) ? all : mounted;
      if (prefix >= all) return;
      const bounds = scroll.getBoundingClientRect();
      const above = bounds.top - bounds.height;
      const below = bounds.bottom + bounds.height;
      let next = islandsRef.current;
      for (const { element: reserve, from, to } of reserves.current.values()) {
        if (from >= to || to > all || !reserve.isConnected) continue;
        const rect = reserve.getBoundingClientRect();
        if (!rect.height || rect.bottom <= above || rect.top >= below) continue;
        const start = chunkStartAt(
          gridRef.current,
          rowAt(from, to, Math.max(0, above - rect.top)),
        );
        const end = chunkEndAt(
          gridRef.current,
          Math.min(to, rowAt(from, to, Math.max(0, below - rect.top)) + 1),
        );
        if (end > start) next = addIsland(next, { start, end }, prefix);
      }
      if (next === islandsRef.current) return;
      islandsRef.current = next;
      setIslands(next);
    };
    demandNow.current = () => {
      if (element.getBoundingClientRect().width) demand();
    };
    const laidOut = () => {
      const now = element.getBoundingClientRect().width;
      return now !== 0 && now === width;
    };
    const capture = () => {
      frame = 0;
      if (!laidOut()) return;
      const held = reading !== undefined && hold(reading);
      demand();
      reading = visibleAnchor();
      if (held && reading) reading.offset -= residue;
    };
    recapture.current = (fresh = false) => {
      if (!laidOut()) return;
      if (!fresh && reading && hold(reading)) return;
      reading = visibleAnchor();
    };
    let rootBefore: number | null = null;
    let carrying = false;
    const keepOnScreen = () => {
      if (reading && rootBefore !== null)
        reading.offset += rootBefore - element.getBoundingClientRect().top;
      rootBefore = null;
    };
    carry.current = (previous) => {
      keepOnScreen();
      const anchor = reading;
      const from = anchor && previous.find((fragment) => fragment.key === anchor.key);
      if (!from || !laidOut()) return false;
      const all = fragmentsRef.current;
      const place = anchorAtSamePlace(
        previous,
        all,
        from,
        anchor.text,
        anchor.occurrence ?? 0,
      );
      if (!place) return false;
      const at = place.index;
      anchor.key = all[at].key;
      anchor.occurrence = place.occurrence;
      const mounted = committedRef.current;
      if (isCovered(mounted, all.length) || at < mounted) return false;
      const below = 2 * scroll.getBoundingClientRect().height;
      const next = addIsland(
        islandsRef.current,
        {
          start: chunkStartAt(gridRef.current, at),
          end: chunkEndAt(
            gridRef.current,
            Math.min(all.length, rowAt(at, all.length, below) + 1),
          ),
        },
        mounted,
      );
      islandsRef.current = next;
      setIslands(next);
      carrying = true;
      return true;
    };
    const schedule = () => {
      if (frame === 0) frame = requestAnimationFrame(capture);
    };
    let edge: number | null = null;
    commitEdge.current = {
      before: () => {
        edge = null;
        rootBefore = null;
        if (!laidOut()) return;
        reading ??= visibleAnchor();
        if (reading) {
          rootBefore = element.getBoundingClientRect().top;
          return;
        }
        if (hasPendingProgressiveMounts()) return;
        const bounds = scroll.getBoundingClientRect();
        const box = element.getBoundingClientRect();
        if (box.top < bounds.top && box.bottom <= bounds.bottom)
          edge = box.bottom - bounds.top;
      },
      after: () => {
        if (carrying) {
          carrying = false;
          keepOnScreen();
        }
        const was = edge;
        edge = null;
        if (was === null || !laidOut()) return;
        const shift =
          element.getBoundingClientRect().bottom -
          scroll.getBoundingClientRect().top -
          was;
        if (Math.abs(shift) >= SHIFT_EPSILON_PX) adjustAbove(Math.round(shift));
      },
    };
    let height = element.getBoundingClientRect().height;
    let settleTimer = 0;
    const resettleLater = () => {
      settleTimer = 0;
      if (panelDragInProgress()) {
        settleTimer = window.setTimeout(resettleLater, RESETTLE_DELAY_MS);
        return;
      }
      resettleRows(element);
      const next = measureGeometry(element, codeProbe.current);
      if (next) setGeometry(keepGeometry(next));
      if (reading) hold(reading);
    };
    const atBottom = () =>
      scroll.scrollTop + scroll.clientHeight >= scroll.scrollHeight - 1;
    let remeasureResidue = 0;
    rowRemeasurers.set(element, (row) => {
      const bounds = scroll.getBoundingClientRect();
      const box = element.getBoundingClientRect();
      row.removeAttribute(ROW_SETTLED_ATTRIBUTE);
      settleQueue.add(row);
      if (!box.width || box.bottom > bounds.top) {
        remeasureResidue = 0;
        return;
      }
      const shift =
        element.getBoundingClientRect().bottom -
        scroll.getBoundingClientRect().top -
        (box.bottom - bounds.top) +
        remeasureResidue;
      const before = scroll.scrollTop;
      const acted = adjustAbove(Math.round(shift));
      remeasureResidue = acted ? shift - (scroll.scrollTop - before) : 0;
    });
    const observer = new ResizeObserver(() => {
      const box = element.getBoundingClientRect();
      if (!box.width) {
        hiddenTranscripts.add(element);
        return;
      }
      hiddenTranscripts.delete(element);
      const widened = box.width !== width;
      if (blindTranscripts.delete(element) && !widened) {
        if (settleTimer !== 0) clearTimeout(settleTimer);
        settleTimer = window.setTimeout(resettleLater, RESETTLE_DELAY_MS);
      }
      const grown = box.height !== height;
      width = box.width;
      height = box.height;
      if (!widened && !grown) return;
      const anchor = reading;
      if (!widened) {
        if (anchor && !atBottom()) hold(anchor);
        return;
      }
      if (settleTimer !== 0) clearTimeout(settleTimer);
      settleTimer = window.setTimeout(resettleLater, RESETTLE_DELAY_MS);
      if (!anchor) return;
      const passage = passageOf(anchor);
      if (!passage) return;
      const top = passage.getBoundingClientRect().top;
      anchor.offset = top - box.top;
      adjustAbove(top - anchor.top);
    });
    observer.observe(element);
    scroll.addEventListener("scroll", schedule, { passive: true });
    return () => {
      commitEdge.current = { before: () => {}, after: () => {} };
      observer.disconnect();
      scroll.removeEventListener("scroll", schedule);
      if (frame !== 0) cancelAnimationFrame(frame);
      if (settleTimer !== 0) clearTimeout(settleTimer);
      hiddenTranscripts.delete(element);
      blindTranscripts.delete(element);
      rowRemeasurers.delete(element);
    };
  }, [adjustAbove, setIslands]);

  useEffect(() => {
    const element = root.current;
    if (!element) return;
    const protect = () => {
      const active = document.activeElement;
      let touches = Boolean(active && element.contains(active));
      const selection = window.getSelection();
      if (!touches && selection && !selection.isCollapsed) {
        for (let i = 0; i < selection.rangeCount && !touches; i += 1)
          touches = selection.getRangeAt(i).intersectsNode(element);
      }
      if (touches) detach();
    };
    document.addEventListener("selectionchange", protect);
    element.addEventListener("focusin", protect);
    return () => {
      document.removeEventListener("selectionchange", protect);
      element.removeEventListener("focusin", protect);
    };
  }, [detach]);

  const shown = covered ? fragments.length : limit;
  const sealed = covered && !streaming;
  const fragmentKey = (at: number) => fragments[at].key;
  const sliceOf = ({
    index,
    first,
    end,
  }: ChunkPiece["groups"][number]): ChunkSlice => {
    const group = grid.groups[index];
    return {
      key: group.key,
      code: group.code,
      first,
      end,
      streaming:
        streaming &&
        (group.code
          ? fragments[first].code!.incomplete
          : first === fragments.length - 1),
    };
  };
  const rendered: React.ReactNode[] = [];
  for (const { from, to, reserve } of mountPlan(
    shown,
    islands,
    fragments.length,
  )) {
    if (reserve === undefined) {
      for (const piece of chunkPieces(grid, fragmentKey, from, to)) {
        rendered.push(
          <Chunk
            key={piece.key}
            slices={piece.groups.map(sliceOf)}
            fragments={fragments}
            closed={piece.whole && (!piece.last || sealed)}
            estimate={grid.heights[piece.end] - grid.heights[piece.first]}
            messageId={messageId}
            messageHasRenderableRenderHtmlTool={
              messageHasRenderableRenderHtmlTool
            }
          />,
        );
      }
      continue;
    }
    const key = `reserve:${reserve}`;
    rendered.push(
      <div
        key={key}
        ref={(element) => {
          if (!element) return;
          const registry = reserves.current;
          registry.set(key, { element, from, to });
          return () => {
            if (registry.get(key)?.element === element) registry.delete(key);
          };
        }}
        aria-hidden="true"
        data-reasoning-reserve=""
        style={{ height: grid.heights[to] - grid.heights[from] }}
      />,
    );
  }

  return (
    <SearchImagesEnabledContext.Provider value={false}>
      <CodeHighlightModeContext.Provider value="settled">
        <div
          ref={root}
          data-slot="reasoning-transcript"
          className="relative min-w-0"
          style={{ overflowAnchor: "none" }}
        >
          <div
            ref={codeProbe}
            aria-hidden
            className="aui-reasoning-code-fragment aui-reasoning-code-last pointer-events-none invisible absolute"
          >
            <div className="aui-reasoning-code-lines" />
          </div>
          <CommitBounds before={beforeCommit} after={afterCommit}>
            {rendered}
          </CommitBounds>
        </div>
      </CodeHighlightModeContext.Provider>
    </SearchImagesEnabledContext.Provider>
  );
}
