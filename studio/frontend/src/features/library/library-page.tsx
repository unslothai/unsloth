// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import {
  type NativeIntent,
  consumeNativePathToken,
  useNativeFileDrop,
} from "@/features/native-intents";
import { useSettingsDialogStore } from "@/features/settings";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";
import {
  AiBrain01Icon,
  ArrowRight01Icon,
  AudioWave01Icon,
  Cancel01Icon,
  FlimSlateIcon,
  Delete02Icon,
  Folder01Icon,
  FolderExportIcon,
  Image02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { StarPointedIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  chatAboutItems,
  chatWithModel,
  downloadLibraryItem,
} from "./actions";
import type { LibraryFolder, LibraryItem, LibraryUploadBatch } from "./api";
import { fileKind, hasImagePreview, isDeletable, isFileItem, isModelItem } from "./file-kind";
import {
  type LibraryActions,
  LibraryActionsProvider,
  type LibraryTarget,
} from "./actions-context";
import { FolderGrid, ItemCard, Masonry } from "./components/library-cards";
import { ConfirmDeleteDialog, NameDialog } from "./components/library-dialogs";
import { LibraryList } from "./components/library-list";
import { LibraryPreview } from "./components/library-preview";
import { LibraryToolbar, type NewAction } from "./components/library-toolbar";
import { EMPTY_FILTERS, type LibraryFilters, filtersActive, matchesFilters } from "./filters";
import { LIBRARY_TABS, type LibrarySearch, type LibraryTab } from "./search";
import { useLibraryStore } from "./store";
import {
  compareBySort,
  nextSort,
  sortState,
  useLibraryViewStore,
  includedBySettings,
  lastActivity,
  type LibrarySortState,
  useLibrarySettingsStore,
} from "./settings-store";

const TAB_LABELS: Record<LibraryTab, string> = {
  suggested: "Suggested",
  favorites: "Favorites",
  folders: "Folders",
  images: "Images",
  videos: "Videos",
  audio: "Audio",
  models: "Fine-tunes",
  all: "All",
};

const EMPTY_ICONS: Record<LibraryTab, typeof Folder01Icon> = {
  suggested: Upload01Icon,
  favorites: StarPointedIcon,
  folders: Folder01Icon,
  images: Image02Icon,
  videos: FlimSlateIcon,
  audio: AudioWave01Icon,
  models: AiBrain01Icon,
  all: Upload01Icon,
};

const EMPTY_COPY: Record<LibraryTab, [title: string, description: string]> = {
  suggested: ["Your library is empty", "Files you upload or create in chats show up here."],
  favorites: ["No favorites yet", "Add files to your favorites to find them here quickly."],
  folders: ["Create your first folder", "Create folders to organize items in your library."],
  images: ["No images yet", "Images you upload or generate show up here."],
  videos: ["No videos yet", "Videos you upload or generate show up here."],
  audio: ["No audio yet", "Speech you generate on the Audio page shows up here."],
  models: ["No fine-tuned models yet", "Models you train or export in Studio show up here."],
  all: ["Your library is empty", "Files you upload or create in chats show up here."],
};

/** What each single-kind tab holds. Only the Source filter means anything on these. */
const KIND_TABS: Partial<Record<LibraryTab, (item: LibraryItem) => boolean>> = {
  images: hasImagePreview,
  videos: (item) => fileKind(item) === "video",
  audio: (item) => fileKind(item) === "audio",
  models: isModelItem,
};


const DELETE_NOTES: Record<string, string> = {
  upload: "This permanently deletes the file.",
  attachment: "It is also removed from the chat it was attached to.",
  image: "It is also removed from your Images gallery.",
  video: "It is also removed from your Video gallery.",
  audio: "It is also removed from your Audio gallery.",
  sandbox: "It is also removed from the chat that created it.",
  model: "This permanently deletes the model from disk.",
};

type NameDialogState =
  | { mode: "create"; parentId: string | null; thenMove?: LibraryTarget }
  | { mode: "rename"; target: LibraryTarget };

function nameMatches(name: string, needle: string): boolean {
  return !needle || name.toLowerCase().includes(needle);
}

function targetName(target: LibraryTarget): string {
  return target.kind === "item" ? target.item.name : target.folder.name;
}


function Tabs({
  tabs,
  active,
  onChange,
}: {
  tabs: readonly LibraryTab[];
  active: LibraryTab;
  onChange: (tab: LibraryTab) => void;
}) {
  return (
    <nav className="mt-6 flex flex-wrap gap-1" aria-label="Library sections">
      {tabs.map((tab) => (
        <button
          key={tab}
          type="button"
          aria-current={tab === active ? "page" : undefined}
          onClick={() => onChange(tab)}
          className={cn(
            "rounded-full px-3.5 py-1.5 text-[15px] text-foreground/80 transition-colors hover:text-foreground",
            tab === active && "bg-muted font-medium text-foreground",
          )}
        >
          {TAB_LABELS[tab]}
        </button>
      ))}
    </nav>
  );
}

function EmptyState({
  icon,
  title,
  description,
  action,
}: {
  icon: typeof Folder01Icon;
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="mx-auto mt-16 flex max-w-md flex-col items-center gap-2 text-center">
      <HugeiconsIcon icon={icon} strokeWidth={1.5} className="mb-2 size-7" />
      <h2 className="font-medium text-[20px] text-foreground">{title}</h2>
      <p className="text-[15px] text-muted-foreground">{description}</p>
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

function SectionHeading({ children }: { children: ReactNode }) {
  return <h2 className="mb-4 mt-8 font-medium text-[18px] text-foreground">{children}</h2>;
}

function LoadingGrid() {
  return (
    <div className="mt-6 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
      {Array.from({ length: 10 }, (_, index) => (
        <Skeleton key={index} className="aspect-square rounded-xl" />
      ))}
    </div>
  );
}

export function LibraryPage() {
  const search = useSearch({ from: "/library" });
  const refresh = useLibraryStore((s) => s.refresh);

  useEffect(() => {
    void refresh();
    // Files land in the Library from chats and the Images page, so look again on return.
    const onFocus = () => void refresh();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refresh]);

  // Search, filters and selection belong to the view they were made in, so each tab and folder
  // gets a fresh one.
  return <LibraryView key={`${search.show ?? ""}:${search.folder ?? ""}`} search={search} />;
}

function LibraryView({ search }: { search: LibrarySearch }) {
  const navigate = useNavigate();
  const { items: allItems, folders, status, error, refresh, patchItem, removeItem, upload, addFolder, patchFolder, removeFolder, markOpened } =
    useLibraryStore();
  const settings = useLibrarySettingsStore();
  // Sources switched off in settings are left out everywhere, folder counts included.
  const items = useMemo(
    () => allItems.filter((item) => includedBySettings(item.id, settings)),
    [allItems, settings],
  );
  const view = useLibraryViewStore((s) => s.view);
  const setView = useLibraryViewStore((s) => s.setView);
  const openSettings = useSettingsDialogStore((s) => s.openDialog);

  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<LibraryFilters>(EMPTY_FILTERS);
  const [selection, setSelection] = useState<Set<string>>(new Set());
  const [nameDialog, setNameDialog] = useState<NameDialogState | null>(null);
  const [pendingDelete, setPendingDelete] = useState<LibraryTarget[] | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const loaded = status === "ready";
  // "auto" tabs count as shown until the listing lands, so a start tab is not swapped mid-load.
  const tabVisible = (entry: LibraryTab) => {
    const visibility = settings.tabs[entry] ?? "always";
    if (visibility === "hidden" || (entry === "models" && !settings.showFineTunes)) return false;
    return visibility === "always" || !loaded || items.some(KIND_TABS[entry] ?? (() => true));
  };
  const preferred = settings.startTab === "last" ? settings.lastTab : settings.startTab;
  const tab: LibraryTab =
    search.show ??
    (tabVisible(preferred) ? preferred : (LIBRARY_TABS.find(tabVisible) ?? "all"));
  // A column click (or a ?sort link) wins over the Sort setting until the view changes.
  const [sortOverride, setSortOverride] = useState<LibrarySortState | null>(
    search.sort ? sortState(search.sort) : null,
  );
  const sort = sortOverride ?? sortState(settings.sort);
  const folderId = search.folder ?? null;
  const folderById = useMemo(() => new Map(folders.map((f) => [f.id, f])), [folders]);
  const currentFolder = folderId ? (folderById.get(folderId) ?? null) : null;

  const go = useCallback(
    (next: LibrarySearch, replace = false) =>
      void navigate({ to: "/library", search: next, replace }),
    [navigate],
  );

  // ── Derived views ──────────────────────────────────────────────

  const counts = useMemo(() => {
    const map = new Map<string, number>();
    const bump = (id: string | null) => id && map.set(id, (map.get(id) ?? 0) + 1);
    for (const item of items) bump(item.folderId);
    for (const folder of folders) bump(folder.parentId);
    return map;
  }, [items, folders]);

  const needle = query.trim().toLowerCase();

  const imagesTab = tab === "images" && !folderId;
  const kindFilter = folderId ? undefined : KIND_TABS[tab];

  // The open tab always shows, so a link straight to a hidden or empty one still lands somewhere.
  const shownTabs = LIBRARY_TABS.filter((entry) => entry === tab || tabVisible(entry));

  const setLibrarySettings = settings.set;
  useEffect(() => {
    if (!folderId) setLibrarySettings({ lastTab: tab });
  }, [tab, folderId, setLibrarySettings]);
  const visibleItems = useMemo(() => {
    let pool = items;
    if (folderId) pool = pool.filter((item) => item.folderId === folderId);
    else if (tab === "favorites") pool = pool.filter((item) => item.favorite);
    else if (kindFilter) pool = pool.filter(kindFilter);
    else if (tab === "folders") pool = [];
    pool = pool.filter(
      (item) => nameMatches(item.name, needle) && matchesFilters(item, filters, !kindFilter),
    );
    // Suggested is the most recently active slice, where opening a file counts as activity.
    if (tab === "suggested" && !folderId) {
      const byActivity = [...pool].sort((a, b) => lastActivity(b) - lastActivity(a));
      pool = needle || filtersActive(filters) ? byActivity : byActivity.slice(0, settings.suggestedLimit);
      if (sort.key === "modified") return sort.desc ? pool : pool.reverse();
    }
    return [...pool].sort(compareBySort(sort));
  }, [items, folderId, tab, needle, filters, kindFilter, settings.suggestedLimit, sort]);

  const visibleFolders = useMemo(() => {
    const showsFolders = folderId || tab === "folders" || tab === "all";
    if (!showsFolders || (filtersActive(filters) && tab !== "folders")) return [];
    return folders
      .filter((folder) => folder.parentId === folderId)
      .filter((folder) => nameMatches(folder.name, needle))
      // Folders have no size of their own.
      .sort(compareBySort(sort.key === "size" ? { key: "name", desc: false } : sort));
  }, [folders, folderId, tab, needle, filters, sort]);

  const previewItem = search.item ? (items.find((item) => item.id === search.item) ?? null) : null;

  // ── Actions ────────────────────────────────────────────────────

  const fail = (message: string) => (err: unknown) =>
    toast.error(message, { description: err instanceof Error ? err.message : String(err) });

  // Only files can be attached; a model in the folder stays behind.
  const filesInFolder = (id: string) =>
    items.filter((item) => item.folderId === id && isFileItem(item));

  const chatAbout = (item: LibraryItem) =>
    void (item.model ? chatWithModel(navigate, item) : chatAboutItems(navigate, [item]));

  const moveTo = async (target: LibraryTarget, destination: string | null) => {
    const label = destination ? (folderById.get(destination)?.name ?? "folder") : "Library";
    try {
      if (target.kind === "item") await patchItem(target.item.id, { folderId: destination });
      else await patchFolder(target.folder.id, { parentId: destination });
      toast.success(`Moved to ${label}`);
    } catch (err) {
      fail("Could not move")(err);
    }
  };

  const actions: LibraryActions = {
    folders,
    openItem: (item) => {
      markOpened(item.id);
      go({ ...search, item: item.id });
    },
    openFolder: (id) => go({ folder: id }),
    chatAbout: (target) =>
      target.kind === "item"
        ? chatAbout(target.item)
        : void chatAboutItems(navigate, filesInFolder(target.folder.id)),
    toggleFavorite: (item) =>
      void patchItem(item.id, { favorite: !item.favorite }).catch(fail("Could not update favorites")),
    download: (item) => void downloadLibraryItem(item),
    rename: (target) => setNameDialog({ mode: "rename", target }),
    moveTo: (target, destination) => void moveTo(target, destination),
    moveToNewFolder: (target) =>
      setNameDialog({ mode: "create", parentId: folderId, thenMove: target }),
    remove: (target) => requestDelete([target]),
  };

  async function uploadBatch(batch: LibraryUploadBatch, count: number, label: string) {
    if (count === 0) return;
    const id = toast.loading(`Uploading ${count === 1 ? label : `${count} files`}…`);
    try {
      await upload(batch, folderId);
      toast.success(count === 1 ? "File uploaded" : `${count} files uploaded`, { id });
    } catch (err) {
      toast.error("Upload failed", {
        id,
        description: err instanceof Error ? err.message : String(err),
      });
    }
  }

  const uploadFiles = (files: File[]) =>
    uploadBatch({ files }, files.length, files[0]?.name ?? "");

  // Desktop drops arrive as paths. The webview can only read media back, so each path is traded for
  // a signed grant and the backend reads the file itself.
  async function uploadNativeDrops(intents: NativeIntent[]) {
    try {
      const leases = await Promise.all(
        intents.map((intent) => consumeNativePathToken(intent.path.token, "attach")),
      );
      await uploadBatch(
        { nativePathLeases: leases.map((lease) => lease.nativePathLease) },
        leases.length,
        leases[0]?.displayLabel ?? "",
      );
    } catch (err) {
      fail("Could not read the dropped files")(err);
    }
  }

  // Anywhere on the page takes a file drop, the desktop app's native drops included.
  const { ref: dropRef, dragging, dragHandlers } = useNativeFileDrop({
    onFiles: uploadFiles,
    onNativeIntents: uploadNativeDrops,
  });

  async function createNote() {
    try {
      const note = new File([""], "Untitled note.md", { type: "text/markdown" });
      const [id] = await upload({ files: [note] }, folderId);
      if (id) go({ ...search, item: id });
    } catch (err) {
      fail("Could not create the note")(err);
    }
  }

  function handleNew(action: NewAction) {
    switch (action) {
      case "note":
        void createNote();
        break;
      case "image":
        void navigate({ to: "/images" });
        break;
      case "folder":
        setNameDialog({ mode: "create", parentId: folderId });
        break;
      case "upload":
        fileInput.current?.click();
        break;
    }
  }

  async function submitName(name: string) {
    if (!nameDialog) return;
    try {
      if (nameDialog.mode === "rename") {
        const { target } = nameDialog;
        if (target.kind === "item") await patchItem(target.item.id, { name });
        else await patchFolder(target.folder.id, { name });
        return;
      }
      const folder = await addFolder(name, nameDialog.parentId);
      if (nameDialog.thenMove) await moveTo(nameDialog.thenMove, folder.id);
    } catch (err) {
      fail(nameDialog.mode === "rename" ? "Could not rename" : "Could not create the folder")(err);
      throw err;
    }
  }

  const requestDelete = (targets: LibraryTarget[]) =>
    settings.confirmDelete ? setPendingDelete(targets) : void confirmDelete(targets);

  async function confirmDelete(targets: LibraryTarget[]) {
    setPendingDelete(null);
    setSelection(new Set());
    const results = await Promise.allSettled(
      targets.map((target) =>
        target.kind === "item" ? removeItem(target.item.id) : removeFolder(target.folder.id),
      ),
    );
    const failed = results.find((result) => result.status === "rejected");
    if (failed) fail("Some items could not be deleted")(failed.reason);
    // A folder being viewed that was just deleted leaves nothing to show.
    if (folderId && targets.some((t) => t.kind === "folder" && t.folder.id === folderId)) {
      go({ folder: currentFolder?.parentId ?? undefined, show: "folders" }, true);
    }
  }

  const deleteCopy = (targets: LibraryTarget[]) => {
    if (targets.length > 1) {
      return {
        title: `Delete ${targets.length} items?`,
        description:
          "Files are deleted from where they live. Anything inside a deleted folder moves up a level.",
      };
    }
    const target = targets[0]!;
    if (target.kind === "folder") {
      const parent = target.folder.parentId ? folderById.get(target.folder.parentId) : null;
      return {
        title: `Delete "${target.folder.name}"?`,
        description: `Everything inside moves to ${parent ? `"${parent.name}"` : "your Library"}. No files are deleted.`,
      };
    }
    const source = target.item.id.split(":", 1)[0]!;
    return {
      title: `Delete "${target.item.name}"?`,
      description: DELETE_NOTES[source] ?? "This permanently deletes the file.",
    };
  };

  const selectedTargets = (): LibraryTarget[] => {
    const out: LibraryTarget[] = [];
    for (const folder of visibleFolders)
      if (selection.has(`folder:${folder.id}`)) out.push({ kind: "folder", folder });
    for (const item of visibleItems)
      if (selection.has(`item:${item.id}`)) out.push({ kind: "item", item });
    return out;
  };

  // ── Header ─────────────────────────────────────────────────────

  const breadcrumb: LibraryFolder[] = [];
  for (let at = currentFolder; at; at = at.parentId ? (folderById.get(at.parentId) ?? null) : null) {
    breadcrumb.unshift(at);
    if (breadcrumb.length > 32) break;
  }

  const title = folderId ? (
    <nav className="flex min-w-0 items-center gap-2 text-ui-25 font-semibold tracking-[-0.028em]" aria-label="Breadcrumb">
      <button
        type="button"
        onClick={() => go({ show: "folders" })}
        className="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
      >
        Library
      </button>
      {breadcrumb.map((folder, index) => (
        <span key={folder.id} className="flex min-w-0 items-center gap-2">
          <HugeiconsIcon icon={ArrowRight01Icon} strokeWidth={2} className="size-5 shrink-0 text-muted-foreground" />
          {index === breadcrumb.length - 1 ? (
            <span className="truncate text-foreground">{folder.name}</span>
          ) : (
            <button
              type="button"
              onClick={() => go({ folder: folder.id })}
              className="truncate text-muted-foreground transition-colors hover:text-foreground"
            >
              {folder.name}
            </button>
          )}
        </span>
      ))}
    </nav>
  ) : (
    <h1 className="text-ui-25 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground">
      Library
    </h1>
  );

  // ── Body ───────────────────────────────────────────────────────

  const uploadButton = (
    <Button variant="muted" className="rounded-full px-5" onClick={() => fileInput.current?.click()}>
      Upload files
    </Button>
  );

  function renderGrid() {
    const itemsGrid = visibleItems.length > 0 && (
      <Masonry
        items={visibleItems}
        getKey={(item) => item.id}
        render={(item) => <ItemCard item={item} />}
      />
    );
    if (folderId || tab === "all") {
      return (
        <>
          {visibleFolders.length > 0 && (
            <>
              {tab === "all" && !folderId && <SectionHeading>Folders</SectionHeading>}
              <div className={cn(!(tab === "all" && !folderId) && "mt-6")}>
                <FolderGrid folders={visibleFolders} counts={counts} />
              </div>
            </>
          )}
          {visibleItems.length > 0 && (
            <>
              {tab === "all" && !folderId ? (
                <SectionHeading>Items</SectionHeading>
              ) : (
                <div className="mt-6" />
              )}
              {itemsGrid}
            </>
          )}
        </>
      );
    }
    if (tab === "folders") {
      return (
        <div className="mt-6">
          <FolderGrid folders={visibleFolders} counts={counts} />
        </div>
      );
    }
    return <div className="mt-6">{itemsGrid}</div>;
  }

  function emptyAction(): ReactNode {
    switch (tab) {
      case "favorites":
        return undefined;
      case "folders":
        return (
          <Button
            variant="dark"
            className="rounded-full px-5"
            onClick={() => setNameDialog({ mode: "create", parentId: null })}
          >
            Create folder
          </Button>
        );
      case "videos":
        return (
          <Button variant="muted" className="rounded-full px-5" onClick={() => void navigate({ to: "/video" })}>
            Generate a video
          </Button>
        );
      case "audio":
        return (
          <Button variant="muted" className="rounded-full px-5" onClick={() => void navigate({ to: "/audio" })}>
            Generate audio
          </Button>
        );
      case "models":
        return (
          <Button variant="muted" className="rounded-full px-5" onClick={() => void navigate({ to: "/studio" })}>
            Train a model
          </Button>
        );
      default:
        return uploadButton;
    }
  }

  function renderBody() {
    if (status === "loading" || status === "idle") return <LoadingGrid />;
    if (status === "error") {
      return (
        <EmptyState
          icon={Folder01Icon}
          title="Could not load your Library"
          description={error ?? "Something went wrong."}
          action={<Button variant="muted" className="rounded-full" onClick={() => void refresh()}>Try again</Button>}
        />
      );
    }
    const empty = visibleItems.length === 0 && visibleFolders.length === 0;
    const narrowed = Boolean(needle) || filtersActive(filters);
    if (empty && narrowed) {
      return (
        <EmptyState
          icon={Folder01Icon}
          title="No matches"
          description="Try a different search or clear the filters."
        />
      );
    }
    if (empty && folderId) {
      return (
        <div className="mt-6 flex min-h-[calc(420px*var(--ui-space-scale,1))] flex-col items-center justify-center gap-5 rounded-xl border border-dashed border-border bg-muted/50">
          <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.5} className="size-8" />
          {uploadButton}
          <p className="text-sm text-muted-foreground">or drop files here</p>
        </div>
      );
    }
    if (empty) {
      const [emptyTitle, emptyDescription] = EMPTY_COPY[tab];
      return (
        <EmptyState
          icon={EMPTY_ICONS[tab]}
          title={emptyTitle}
          description={emptyDescription}
          action={emptyAction()}
        />
      );
    }
    if (view === "list" && !imagesTab) {
      return (
        <div className="mt-6">
          <LibraryList
            folders={visibleFolders}
            items={visibleItems}
            counts={counts}
            selection={selection}
            onSelectionChange={setSelection}
            sort={sort}
            onSortChange={(key) => setSortOverride(nextSort(sort, key))}
            activity={tab === "suggested" && !folderId}
          />
        </div>
      );
    }
    return renderGrid();
  }

  const nameDialogProps =
    nameDialog?.mode === "rename"
      ? {
          title: nameDialog.target.kind === "folder" ? "Rename folder" : "Rename file",
          submitLabel: "Rename",
          initialValue: targetName(nameDialog.target),
        }
      : { title: "New folder", submitLabel: "Create", initialValue: "" };

  const selectedCount = selection.size;
  const deletableSelection = () =>
    selectedTargets().filter((t) => t.kind === "folder" || isDeletable(t.item));
  const deleteText = pendingDelete ? deleteCopy(pendingDelete) : null;

  return (
    <LibraryActionsProvider value={actions}>
      <main
        ref={dropRef}
        {...dragHandlers}
        className="relative mx-auto w-full max-w-[calc(1560px*var(--ui-space-scale,1))] px-6 pb-24 pt-8 font-heading sm:px-10"
      >
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(event) => {
            void uploadFiles([...(event.target.files ?? [])]);
            event.target.value = "";
          }}
        />
        <LibraryToolbar
          title={title}
          filters={filters}
          onFiltersChange={setFilters}
          filterMode={!folderId && tab === "folders" ? "none" : kindFilter ? "source" : "all"}
          view={imagesTab ? null : view}
          onViewChange={setView}
          search={query}
          onSearchChange={setQuery}
          searchPlaceholder={folderId ? "Search folder" : "Search library"}
          onNew={handleNew}
          onSettings={() => openSettings("library")}
        />
        {!folderId && <Tabs tabs={shownTabs} active={tab} onChange={(next) => go({ show: next })} />}
        {renderBody()}

        {dragging && (
          <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-background/70 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3 rounded-xl border-2 border-dashed border-muted-foreground px-16 py-12">
              <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.5} className="size-8" />
              <p className="font-medium text-lg">
                Drop to upload{currentFolder ? ` to "${currentFolder.name}"` : ""}
              </p>
            </div>
          </div>
        )}

        {selectedCount > 0 && (
          <div className="fixed bottom-8 left-1/2 z-30 flex -translate-x-1/2 items-center gap-1 rounded-full border border-border bg-popover p-1.5 pl-5 shadow-lg">
            <span className="mr-2 text-sm">{selectedCount} selected</span>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="rounded-full">
                  <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className="size-4" />
                  Add to folder
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="center" side="top" className="max-h-[min(--spacing(80),var(--radix-dropdown-menu-content-available-height))] w-56">
                <DropdownMenuItem
                  onSelect={() => {
                    void Promise.all(selectedTargets().map((t) => moveTo(t, null)));
                    setSelection(new Set());
                  }}
                >
                  <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                  Library (no folder)
                </DropdownMenuItem>
                {folders
                  .filter((folder) => !selection.has(`folder:${folder.id}`))
                  .map((folder) => (
                    <DropdownMenuItem
                      key={folder.id}
                      onSelect={() => {
                        void Promise.all(selectedTargets().map((t) => moveTo(t, folder.id)));
                        setSelection(new Set());
                      }}
                    >
                      <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                      <span className="truncate">{folder.name}</span>
                    </DropdownMenuItem>
                  ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              variant="ghost"
              size="sm"
              className="rounded-full text-destructive hover:text-destructive"
              disabled={deletableSelection().length === 0}
              onClick={() => requestDelete(deletableSelection())}
            >
              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-4" />
              Delete
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              className="rounded-full"
              aria-label="Clear selection"
              onClick={() => setSelection(new Set())}
            >
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-4" />
            </Button>
          </div>
        )}
      </main>

      <NameDialog
        open={nameDialog !== null}
        {...nameDialogProps}
        onSubmit={submitName}
        onOpenChange={(open) => !open && setNameDialog(null)}
      />
      <ConfirmDeleteDialog
        open={pendingDelete !== null}
        title={deleteText?.title ?? ""}
        description={deleteText?.description ?? ""}
        confirmLabel="Delete"
        onConfirm={() => pendingDelete && void confirmDelete(pendingDelete)}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      />
      <LibraryPreview
        item={previewItem}
        onOpenChange={(open) => !open && go({ ...search, item: undefined }, true)}
        onChat={chatAbout}
        onDownload={(item) => void downloadLibraryItem(item)}
        onOpenThread={(threadId) => void navigate({ to: "/chat", search: { thread: threadId } })}
        onSaved={() => void refresh()}
      />
    </LibraryActionsProvider>
  );
}
