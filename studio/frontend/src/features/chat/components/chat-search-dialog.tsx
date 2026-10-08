// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Command,
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandList,
} from "@/components/ui/command";
import {
  type ShortcutId,
  isImeComposing,
  triggerShortcut,
  useShortcut,
  useShortcutAvailable,
  useShortcutLabel,
} from "@/features/settings";
import { type TranslationKey, useT } from "@/i18n";
import { MessageCircleIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  DashboardCircleIcon,
  FlimSlateIcon,
  Folder01Icon,
  Image03Icon,
  PencilEdit02Icon,
  Search01Icon,
  BubbleChatTemporaryIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { useNavigate } from "@tanstack/react-router";
import { Command as CommandPrimitive } from "cmdk";
import { Columns2Icon, GitBranchIcon, type LucideIcon } from "lucide-react";
import {
  type KeyboardEvent,
  type ReactNode,
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  type ModelInventoryFormat,
  isHiddenModelId,
  useHubInventory,
} from "@/features/hub";
import { modelLabelKey } from "@/features/library";
import { useChatProjects } from "../hooks/use-chat-projects";
import { useChatSidebarItems } from "../hooks/use-chat-sidebar-items";
import {
  type ChatSearchItem,
  chatSearchIndexHasRows,
  useChatSearchIndex,
} from "../hooks/use-chat-search-index";
import {
  type LibrarySearchEntry,
  libraryTime,
  useChatSearchSources,
} from "../hooks/use-chat-search-sources";
import { useChatSearchStore } from "../stores/chat-search-store";
import { isCompactChatSearchList } from "../utils/chat-search-list-height";
import {
  ALL_TAB_GROUP_LIMIT,
  CHAT_SEARCH_TABS,
  type ChatSearchKind,
  type ChatSearchRow,
  type ChatSearchTab,
  filterRows,
  haystackMatches,
  queryTokens,
  recentRows,
  stepTab,
  tabStepForKey,
} from "../utils/chat-search-tabs";

// Rows mounted while the dialog animates open; the rest follow once it settles, so a long
// history never lays out hundreds of rows mid-transition.
const INITIAL_ROW_COUNT = 24;
const FULL_ROW_REVEAL_MS = 220;

const KINDS = CHAT_SEARCH_TABS.filter(
  (tab): tab is ChatSearchKind => tab !== "all",
);

// We filter rows here (cmdk runs with shouldFilter=false) to control the two-tier behavior and
// avoid cmdk's fuzzy scorer keeping non-matches visible (#5572): every whitespace token must
// be a substring. User messages are searched first; expand to the full conversation only when
// user text alone matches nothing anywhere.
export function selectVisibleChats<
  T extends { userSearchText: string; searchText: string },
>(items: T[], search: string): T[] {
  const tokens = queryTokens(search);
  if (tokens.length === 0) return items;
  const userHits = items.filter((it) =>
    haystackMatches(it.userSearchText, tokens),
  );
  if (userHits.length > 0) return userHits;
  return items.filter((it) => haystackMatches(it.searchText, tokens));
}

type WhenKey = "today" | "pastWeek" | "pastMonth" | "older";

function whenKey(time: number): WhenKey {
  const diff = Date.now() - time;
  const day = 86_400_000;
  if (diff < day) return "today";
  if (diff < 7 * day) return "pastWeek";
  if (diff < 30 * day) return "pastMonth";
  return "older";
}

/** Hugeicons path data, or a lucide icon (compare and fork, as in the Library). */
type RowIcon = IconSvgElement | LucideIcon;

interface Row extends ChatSearchRow {
  icon: RowIcon;
  /** Replaces the age at the row's end. */
  meta?: string;
  /** Not a runnable model (embedder, STT, probe); listed only when queried. */
  hidden?: boolean;
  open: () => void;
}

type ActionId = Extract<
  ShortcutId,
  "newChat" | "newTemporaryChat" | "switchToTrain" | "switchToImages" | "switchToVideo"
>;

interface Action {
  id: ActionId;
  icon: IconSvgElement;
  labelKey: TranslationKey;
}

// Run through the shortcut registry, like the command palette.
const ACTIONS: Action[] = [
  { id: "newChat", icon: PencilEdit02Icon, labelKey: "shell.search.newChat" },
  {
    id: "newTemporaryChat",
    icon: BubbleChatTemporaryIcon,
    labelKey: "shell.search.newTemporaryChat",
  },
  {
    id: "switchToTrain",
    icon: TestTubeOutlineIcon,
    labelKey: "shell.search.fineTune",
  },
  {
    id: "switchToImages",
    icon: Image03Icon,
    labelKey: "shell.search.generateImage",
  },
  {
    id: "switchToVideo",
    icon: FlimSlateIcon,
    labelKey: "shell.search.generateVideo",
  },
];

const NAVIGATION_KEYS = new Set(["ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

const ROW_CLASS =
  "relative flex cursor-pointer select-none items-center gap-3 rounded-full px-3 py-2.5 text-sm outline-hidden data-selected:bg-muted data-selected:text-foreground";

export function ChatSearchDialog() {
  const t = useT();
  const isOpen = useChatSearchStore((s) => s.isOpen);
  const setOpen = useChatSearchStore((s) => s.setOpen);
  const close = useChatSearchStore((s) => s.close);
  const opener = useChatSearchStore((s) => s.opener);
  const navigate = useNavigate();
  const { items, loading } = useChatSearchIndex(isOpen);
  // Every thread, empty ones too: the index skips those, the sidebar's project ages do not.
  const { items: threads } = useChatSidebarItems({
    enabled: isOpen,
    requireMessages: false,
  });
  const { projects, hasLoaded: projectsLoaded } = useChatProjects();
  const sources = useChatSearchSources(isOpen);
  const { cachedRows, localRows, downloadedReady } = useHubInventory({
    kind: "models",
    enabled: isOpen,
  });
  const [query, setQuery] = useState("");
  const [tab, setTab] = useState<ChatSearchTab>("all");
  // Pin the highlight to the first row until the user moves it: later-loading kinds can sort above it.
  const [selected, setSelected] = useState("");
  const [moved, setMoved] = useState(false);
  // Filtering scans every conversation's text, so keep it off the keystroke path.
  const deferredQuery = useDeferredValue(query);
  // An empty query needs no scan, and the deferred value must not hold a previous filter over a reopened dialog.
  const activeQuery = query === "" ? "" : deferredQuery;
  const [rowLimit, setRowLimit] = useState(INITIAL_ROW_COUNT);
  // Centered, so the height is decided per open and only ever relaxed compact -> fixed.
  const [compactList, setCompactList] = useState(() =>
    isCompactChatSearchList(true, chatSearchIndexHasRows()),
  );

  // One call per action (hooks cannot run in a loop). Availability can change while open, so it
  // gates the selection keys too, not just the rendered rows.
  const available: Record<ActionId, boolean> = {
    newChat: useShortcutAvailable("newChat", false),
    newTemporaryChat: useShortcutAvailable("newTemporaryChat", false),
    switchToTrain: useShortcutAvailable("switchToTrain", false),
    switchToImages: useShortcutAvailable("switchToImages", false),
    switchToVideo: useShortcutAvailable("switchToVideo", false),
  };
  // Compact only while no kind has rows: projects, Library, downloads and actions fill the list too.
  const otherRows =
    projects.length > 0 ||
    sources.files.length > 0 ||
    sources.fineTunes.length > 0 ||
    cachedRows.length > 0 ||
    localRows.length > 0 ||
    Object.values(available).some(Boolean);

  // Reset in the opening render, not an effect: Radix mounts the portal as this render commits, so
  // an effect would trim rows only after the previous set was in the DOM. Resetting on close
  // instead would tear rows down inside the exit animation.
  const [wasOpen, setWasOpen] = useState(isOpen);
  if (isOpen !== wasOpen) {
    setWasOpen(isOpen);
    if (isOpen) {
      setQuery("");
      setTab("all");
      setMoved(false);
      setSelected("");
      setRowLimit(INITIAL_ROW_COUNT);
      setCompactList(
        isCompactChatSearchList(true, otherRows || chatSearchIndexHasRows()),
      );
    }
  } else if (
    compactList !==
    isCompactChatSearchList(compactList, otherRows || items.length > 0)
  ) {
    // Backstop for an open with no hint at all: the fixed height is taken when the first build lands.
    setCompactList(false);
  }

  useEffect(() => {
    if (!isOpen) return;
    const timer = setTimeout(
      () => setRowLimit(Number.POSITIVE_INFINITY),
      FULL_ROW_REVEAL_MS,
    );
    return () => clearTimeout(timer);
  }, [isOpen]);

  // skipInTextFields keeps the composer's own Cmd-K and any browser find intact while the user is
  // typing, as the hand-rolled handler did.
  useShortcut("searchChats", () => useChatSearchStore.getState().open(), {
    skipInTextFields: true,
  });

  const go = (to: () => void) => () => {
    to();
    close();
  };

  // The labels a row shows but its text lacks, so typing them finds it.
  const chats = useMemo(() => {
    const untitled = t("shell.search.untitledChat").toLowerCase();
    const compare = t("shell.search.compare").toLowerCase();
    return items.map((item) => {
      const extra = [
        item.title ? "" : untitled,
        item.type === "compare" ? compare : "",
      ].join(" ").trim();
      return extra
        ? {
            ...item,
            userSearchText: `${item.userSearchText} ${extra}`,
            searchText: `${item.searchText} ${extra}`,
          }
        : item;
    });
  }, [items, t]);

  // Unfiltered rows per kind. Chats are matched separately (two tiers).
  const rowsByKind = useMemo<Record<Exclude<ChatSearchKind, "chats">, Row[]>>(() => {
    // As the sidebar: a project's own edits or its newest chat, whichever is later.
    const newestChat = new Map<string, number>();
    for (const thread of threads) {
      if (!thread.projectId) continue;
      newestChat.set(
        thread.projectId,
        Math.max(newestChat.get(thread.projectId) ?? 0, thread.updatedAt),
      );
    }
    const activityAt = (project: (typeof projects)[number]) =>
      Math.max(project.updatedAt ?? project.createdAt, newestChat.get(project.id) ?? 0);
    return {
      projects: projects
        .filter((project) => !project.archived)
        .map((project) => ({
          key: `project:${project.id}`,
          kind: "projects" as const,
          title: project.name,
          time: activityAt(project),
          haystack: project.name.toLowerCase(),
          icon: Folder01Icon,
          open: () =>
            navigate({ to: "/chat", search: { project: project.id } }),
        }))
        .sort((a, b) => b.time - a.time),
      files: sources.files.map((entry) => libraryRow(entry, "files", navigate)),
      // Hub downloads plus Library fine-tunes.
      models: [
        ...downloadedModelRows(cachedRows, localRows, navigate),
        ...sources.fineTunes.map((entry) => {
          const row = libraryRow(entry, "models", navigate);
          // How it was made, as the Library labels it (LoRA, GGUF export, ...).
          const label = t(modelLabelKey(entry.item) ?? "library.modelKind.model");
          return { ...row, meta: label, haystack: `${row.haystack} ${label.toLowerCase()}` };
        }),
      ].sort((a, b) => b.time - a.time),
    };
  }, [projects, threads, sources, cachedRows, localRows, navigate, t]);

  const matchAll = useCallback(
    (search: string): Record<ChatSearchKind, Row[]> => ({
      chats: selectVisibleChats(chats, search)
        .map((item) => chatRow(item, t, navigate))
        .sort((a, b) => b.time - a.time),
      projects: filterRows(rowsByKind.projects, search),
      files: filterRows(rowsByKind.files, search),
      // Infrastructure models stay out until a query names them, as in the Hub.
      models: filterRows(
        queryTokens(search).length > 0
          ? rowsByKind.models
          : rowsByKind.models.filter((row) => !row.hidden),
        search,
      ),
    }),
    [chats, rowsByKind, t, navigate],
  );
  const matched = useMemo(() => matchAll(activeQuery), [matchAll, activeQuery]);

  const hasQuery = queryTokens(activeQuery).length > 0;
  // Live query, not the deferred one: Enter must never run an action the input no longer matches.
  const visibleActions = ACTIONS.filter(
    (action) =>
      available[action.id] &&
      haystackMatches(t(action.labelKey).toLowerCase(), queryTokens(query)),
  );

  // All: one headed group per kind. A kind's tab: one list.
  const groupsFor = (
    rows: Record<ChatSearchKind, Row[]>,
    queried: boolean,
  ): { heading?: string; rows: Row[] }[] =>
    tab !== "all"
      ? [{ rows: rows[tab].slice(0, rowLimit) }]
      : queried
        ? KINDS.map((kind) => ({
            heading: t(`shell.search.tabs.${kind}`),
            rows: rows[kind].slice(0, ALL_TAB_GROUP_LIMIT),
          }))
        : [{ heading: t("shell.search.recents"), rows: recentRows(rows) }];
  const groups = groupsFor(matched, hasQuery);
  const showActions = tab === "all" && visibleActions.length > 0;
  const rowCount = groups.reduce((sum, group) => sum + group.rows.length, 0);
  const firstKey =
    groups.find((group) => group.rows.length > 0)?.rows[0].key ??
    (showActions ? `action:${visibleActions[0].id}` : "");
  const shownKeys = new Set([
    ...groups.flatMap((group) => group.rows.map((row) => row.key)),
    ...(showActions ? visibleActions.map((action) => `action:${action.id}`) : []),
  ]);

  const activate = (row: Row) => {
    // The list can trail the input by a render; only open rows the live query still matches.
    if (query === activeQuery) return go(row.open)();
    const live = matchAll(query);
    if (live[row.kind].some((match) => match.key === row.key)) return go(row.open)();
    // A stale auto-highlight: run what the caught-up list will show first.
    if (moved) return;
    const first = groupsFor(live, queryTokens(query).length > 0).find(
      (group) => group.rows.length > 0,
    )?.rows[0];
    if (first) go(first.open)();
    else if (showActions) go(() => void triggerShortcut(visibleActions[0].id))();
  };

  const switchTab = (next: ChatSearchTab) => {
    setTab(next);
    setMoved(false);
    setSelected("");
  };

  const onInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    // An arrow mid-composition moves the IME caret, not the tab.
    if (isImeComposing(event.nativeEvent)) return;
    if (event.altKey || event.metaKey || event.ctrlKey || event.shiftKey) return;
    const step = tabStepForKey(event.key, event.currentTarget);
    if (step === null) return;
    const next = stepTab(tab, step);
    if (next === tab) return;
    event.preventDefault();
    switchTab(next);
  };

  const loadingByTab: Record<ChatSearchTab, boolean> = {
    chats: loading,
    projects: !projectsLoaded,
    files: !sources.ready,
    models: !downloadedReady || !sources.ready,
    all: loading || !projectsLoaded || !sources.ready || !downloadedReady,
  };
  const sourceLoading = loadingByTab[tab];
  const emptyText = sourceLoading
    ? t("shell.search.loading")
    : hasQuery
      ? t("shell.search.noMatches")
      : t(`shell.search.empty.${tab}`);

  return (
    <CommandDialog
      open={isOpen}
      onOpenChange={setOpen}
      onCloseAutoFocus={(event) => {
        if (opener?.isConnected) {
          event.preventDefault();
          opener.focus({ preventScroll: true });
        }
      }}
      className="chat-search-surface rounded-3xl! max-sm:rounded-none! top-[calc(50%+var(--studio-window-chrome-top,0px)/2)] -translate-y-1/2 w-[calc(635px*var(--ui-space-scale,1))] max-w-[calc(100%-2rem)] gap-0 p-0 ring-0 duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] sm:max-w-[calc(635px*var(--ui-space-scale,1))]"
      overlayClassName="bg-transparent supports-backdrop-filter:backdrop-blur-none"
    >
      <Command
        className="rounded-3xl p-0"
        shouldFilter={false}
        value={moved && shownKeys.has(selected) ? selected : firstKey}
        onValueChange={setSelected}
        onKeyDown={(e) => {
          if (NAVIGATION_KEYS.has(e.key)) setMoved(true);
        }}
      >
        <div className="flex items-center gap-3 border-b border-border/40 px-4 py-3">
          <HugeiconsIcon
            icon={Search01Icon}
            strokeWidth={2}
            className="size-4 shrink-0 text-muted-foreground"
          />
          <CommandPrimitive.Input
            placeholder={t("shell.search.placeholder")}
            // Controlled: reopening inside the exit animation reuses the mounted tree, so cmdk would keep
            // the previous text while the filter state is clear.
            value={query}
            onValueChange={(value) => {
              setQuery(value);
              setMoved(false);
              setSelected("");
            }}
            onKeyDown={onInputKeyDown}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          <button
            type="button"
            onClick={close}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.stopPropagation();
            }}
            className="flex size-6 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-label={t("common.close")}
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={2}
              className="size-4"
            />
          </button>
        </div>
        <div
          role="tablist"
          aria-label={t("shell.search.placeholder")}
          className="no-scrollbar flex items-center gap-1 overflow-x-auto border-b border-border/40 px-3 py-2"
        >
          {CHAT_SEARCH_TABS.map((entry) => (
            <button
              key={entry}
              type="button"
              role="tab"
              aria-selected={entry === tab}
              // Keep focus in the input.
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => switchTab(entry)}
              // cmdk's root runs the highlighted row on Enter; a focused tab must only switch.
              onKeyDown={(e) => {
                if (e.key === "Enter") e.stopPropagation();
              }}
              className={cn(
                "shrink-0 rounded-full px-3 py-1 text-ui-13 font-medium transition-colors",
                entry === tab
                  ? "bg-muted text-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {t(`shell.search.tabs.${entry}`)}
            </button>
          ))}
        </div>
        <CommandList
          onPointerMove={() => setMoved(true)}
          className={cn(
            "cmd-native-scrollbar hover-scrollbar p-1",
            compactList ? "max-h-[calc(420px*var(--ui-space-scale,1))]" : "h-[calc(420px*var(--ui-space-scale,1))] max-h-[60dvh]",
          )}
        >
          <CommandEmpty className="py-6 text-center text-xs text-muted-foreground">
            {emptyText}
          </CommandEmpty>
          {groups.map(
            (group) =>
              group.rows.length > 0 && (
                <CommandGroup
                  key={group.heading ?? tab}
                  heading={group.heading}
                  className="p-0"
                >
                  {group.rows.map((row) => (
                    <CommandPrimitive.Item
                      key={row.key}
                      value={row.key}
                      onSelect={() => activate(row)}
                      className={ROW_CLASS}
                    >
                      <RowGlyph icon={row.icon} />
                      <span className="min-w-0 flex-1 truncate text-ui-13 font-medium">
                        {row.title}
                      </span>
                      <span className="shrink-0 text-ui-11 text-muted-foreground">
                        {row.meta ?? t(`shell.search.when.${whenKey(row.time)}`)}
                      </span>
                    </CommandPrimitive.Item>
                  ))}
                </CommandGroup>
              ),
          )}
          {showActions && (
            <CommandGroup
              heading={t("shell.search.actions")}
              className={cn("p-0", rowCount > 0 && "mt-2")}
            >
              {visibleActions.map((action) => (
                <ActionItem
                  key={action.id}
                  action={action}
                  onSelect={go(() => void triggerShortcut(action.id))}
                />
              ))}
            </CommandGroup>
          )}
        </CommandList>
        <div className="flex items-center gap-5 border-t border-border/40 px-5 py-2.5 text-ui-11 text-muted-foreground max-sm:hidden">
          <FooterHint label={t("shell.search.footer.close")}>
            <Key>Esc</Key>
          </FooterHint>
          <FooterHint label={t("shell.search.footer.changeType")}>
            <Key>←</Key>
            <Key>→</Key>
          </FooterHint>
          <FooterHint label={t("shell.search.footer.open")}>
            <Key>↵</Key>
          </FooterHint>
        </div>
      </Command>
    </CommandDialog>
  );
}

const FORMAT_LABELS: Partial<Record<ModelInventoryFormat, string>> = {
  gguf: "GGUF",
  safetensors: "Safetensors",
  checkpoint: "Safetensors",
  adapter: "Adapter",
};

/** Complete downloads only. */
function downloadedModelRows(
  cachedRows: readonly CachedInventoryRow[],
  localRows: readonly LocalInventoryRow[],
  navigate: ReturnType<typeof useNavigate>,
): Row[] {
  const open = (id: string) => () =>
    navigate({ to: "/hub", search: { tab: "downloaded", model: id } });
  // The server already hides infrastructure cache rows, but not optimistic ones.
  const cached = cachedRows
    .filter(
      (row) =>
        !row.partial &&
        !(row.optimistic && isHiddenModelId(row.id, row.repoId, row.cachePath)),
    )
    .map(
      (row): Row => ({
        key: `hub-cache:${row.id}`,
        kind: "models",
        title: row.repoId,
        time: row.lastModified ?? 0,
        haystack: [
          row.repoId,
          row.formatVariant ?? "",
          row.modelFormat,
          FORMAT_LABELS[row.modelFormat] ?? "",
        ]
          .join(" ")
          .toLowerCase(),
        icon: DashboardCircleIcon,
        meta: FORMAT_LABELS[row.modelFormat],
        open: open(row.id),
      }),
    );
  const local = localRows
    .filter((row) => !row.partial)
    .map((row): Row => {
      const title = row.displayName || row.title;
      return {
        key: `hub-local:${row.id}`,
        kind: "models",
        title,
        time: row.updatedAt ?? 0,
        haystack: [
          title,
          row.repoId ?? "",
          row.sourceLabel,
          row.path,
          row.modelFormat,
          FORMAT_LABELS[row.modelFormat] ?? "",
        ]
          .join(" ")
          .toLowerCase(),
        icon: DashboardCircleIcon,
        meta: row.sourceLabel,
        hidden: isHiddenModelId(row.id, row.repoId, row.path, row.title),
        open: open(row.id),
      };
    });
  return [...cached, ...local];
}

function chatRow(
  item: ChatSearchItem,
  t: ReturnType<typeof useT>,
  navigate: ReturnType<typeof useNavigate>,
): Row {
  return {
    key: `chat:${item.id}`,
    kind: "chats",
    title: item.title || t("shell.search.untitledChat"),
    time: item.updatedAt ?? item.createdAt,
    haystack: item.searchText,
    icon:
      item.type === "compare"
        ? Columns2Icon
        : item.isFork
          ? GitBranchIcon
          : MessageCircleIcon,
    meta: item.type === "compare" ? t("shell.search.compare") : undefined,
    open: () =>
      navigate({
        to: "/chat",
        search:
          item.type === "single"
            ? {
                thread: item.id,
                ...(item.projectId ? { project: item.projectId } : {}),
              }
            : {
                compare: item.id,
                ...(item.projectId ? { project: item.projectId } : {}),
              },
      }),
  };
}

function libraryRow(
  { item, icon }: LibrarySearchEntry,
  kind: "files" | "models",
  navigate: ReturnType<typeof useNavigate>,
): Row {
  return {
    key: `library:${item.id}`,
    kind,
    title: item.name,
    time: libraryTime(item),
    haystack: [
      item.name,
      item.fileName ?? "",
      item.threadTitle ?? "",
      item.model?.baseModel ?? "",
    ]
      .join(" ")
      .toLowerCase(),
    icon,
    open: () =>
      navigate({
        to: "/library",
        search: { show: kind === "models" ? "models" : "all", item: item.id },
      }),
  };
}

function ActionItem({
  action,
  onSelect,
}: {
  action: Action;
  onSelect: () => void;
}) {
  const t = useT();
  const label = useShortcutLabel(action.id);
  return (
    <CommandPrimitive.Item
      value={`action:${action.id}`}
      onSelect={onSelect}
      className={ROW_CLASS}
    >
      <RowGlyph icon={action.icon} />
      <span className="min-w-0 flex-1 truncate text-ui-13 font-medium">
        {t(action.labelKey)}
      </span>
      {label && (
        <span className="shrink-0 text-ui-11 text-muted-foreground">
          {label}
        </span>
      )}
    </CommandPrimitive.Item>
  );
}

const GLYPH_CLASS = "size-4 shrink-0 text-muted-foreground";

// Same stroke as the sidebar and Library.
function RowGlyph({ icon }: { icon: RowIcon }) {
  if (isHugeicon(icon)) {
    return <HugeiconsIcon icon={icon} strokeWidth={1.75} className={GLYPH_CLASS} />;
  }
  const Icon = icon;
  return <Icon strokeWidth={1.75} className={GLYPH_CLASS} />;
}

// Hugeicons glyphs are arrays; lucide glyphs are components.
function isHugeicon(icon: RowIcon): icon is IconSvgElement {
  return Array.isArray(icon);
}

function FooterHint({ label, children }: { label: string; children: ReactNode }) {
  return (
    <span className="flex items-center gap-1.5">
      {label}
      <span className="flex items-center gap-1">{children}</span>
    </span>
  );
}

function Key({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-5 min-w-5 items-center justify-center rounded-md border border-border/60 px-1 font-sans text-ui-11 text-muted-foreground">
      {children}
    </kbd>
  );
}
