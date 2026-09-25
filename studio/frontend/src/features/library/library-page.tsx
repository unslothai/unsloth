// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Skeleton } from "@/components/ui/skeleton";
import {
  type NativeIntent,
  consumeNativePathToken,
  useNativeFileDrop,
} from "@/features/native-intents";
import { useSettingsDialogStore } from "@/features/settings";
import { type TranslationKey, useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { toast } from "@/lib/toast";
import {
  AudioWave01Icon,
  Cancel01Icon,
  FlimSlateIcon,
  Delete02Icon,
  Download01Icon,
  Folder01Icon,
  FolderExportIcon,
  Image02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { ChevronRightStandardIcon } from "@/lib/chevron-icons";
import { StarPointedIcon, TestTubeOutlineIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouter, useSearch } from "@tanstack/react-router";
import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  chatAboutItems,
  chatWithModel,
  downloadLibraryItem,
  downloadLibraryItems,
} from "./actions";
import { type LibraryFolder, type LibraryItem, type LibraryUploadBatch, errorMessage } from "./api";
import {
  type LibraryTypeFilter,
  fileKind,
  isDeletable,
  isFileItem,
  isModelItem,
} from "./file-kind";
import {
  type LibraryActions,
  LibraryActionsProvider,
  type LibraryTarget,
} from "./actions-context";
import { CardSelectionContext } from "./components/card-selection";
import { FolderGrid, ItemCard, Masonry } from "./components/library-cards";
import { ConfirmDeleteDialog, NameDialog } from "./components/library-dialogs";
import { LibraryList } from "./components/library-list";
import { LibraryPreview } from "./components/library-preview";
import { LibraryHeader } from "./components/library-header";
import { LibraryToolbar, type NewAction } from "./components/library-toolbar";
import { EMPTY_FILTERS, type LibraryFilters, filtersActive, matchesFilters } from "./filters";
import { LIBRARY_TABS, type LibrarySearch, type LibraryTab } from "./search";
import { useLibraryStore } from "./store";
import {
  SORT_STATES,
  compareBySort,
  includedBySettings,
  lastActivity,
  nextSort,
  sortParam,
  useLibrarySettingsStore,
  useLibraryViewStore,
  useLibraryVisitStore,
} from "./settings-store";

const TAB_LABELS: Record<LibraryTab, TranslationKey> = {
  suggested: "library.tabs.suggested",
  favorites: "library.tabs.favorites",
  folders: "library.tabs.folders",
  images: "library.tabs.images",
  videos: "library.tabs.videos",
  audio: "library.tabs.audio",
  models: "library.tabs.models",
  all: "library.tabs.all",
};

type EmptyCopy = [icon: typeof Folder01Icon, title: TranslationKey, description: TranslationKey];
const EMPTY_COPY: Record<LibraryTab, EmptyCopy> = {
  suggested: [Upload01Icon, "library.empty.suggestedTitle", "library.empty.suggestedDescription"],
  favorites: [StarPointedIcon, "library.empty.favoritesTitle", "library.empty.favoritesDescription"],
  folders: [Folder01Icon, "library.empty.foldersTitle", "library.empty.foldersDescription"],
  images: [Image02Icon, "library.empty.imagesTitle", "library.empty.imagesDescription"],
  videos: [FlimSlateIcon, "library.empty.videosTitle", "library.empty.videosDescription"],
  audio: [AudioWave01Icon, "library.empty.audioTitle", "library.empty.audioDescription"],
  models: [TestTubeOutlineIcon, "library.empty.modelsTitle", "library.empty.modelsDescription"],
  all: [Upload01Icon, "library.empty.suggestedTitle", "library.empty.suggestedDescription"],
};

// Where an empty media tab sends you to make something for it.
const EMPTY_LINKS = {
  videos: ["/video", "library.empty.generateVideo"],
  audio: ["/audio", "library.empty.generateAudio"],
  models: ["/studio", "library.empty.trainModel"],
} as const;

/** What each single-kind tab holds. Only the Source filter means anything on these. */
const KIND_TABS: Partial<Record<LibraryTab, (item: LibraryItem) => boolean>> = {
  // Every image, SVG included; hasImagePreview only picks how its card draws.
  images: (item) => fileKind(item) === "image",
  videos: (item) => fileKind(item) === "video",
  audio: (item) => fileKind(item) === "audio",
  models: isModelItem,
};

const DELETE_NOTES: Record<string, TranslationKey> = {
  upload: "library.dialog.deleteUpload",
  attachment: "library.dialog.deleteAttachment",
  image: "library.dialog.deleteImage",
  video: "library.dialog.deleteVideo",
  audio: "library.dialog.deleteAudio",
  sandbox: "library.dialog.deleteSandbox",
  model: "library.dialog.deleteModel",
};

type NameDialogState =
  | { mode: "create"; parentId: string | null; thenMove?: LibraryTarget }
  | { mode: "rename"; target: LibraryTarget };

function nameMatches(name: string, needle: string): boolean {
  return !needle || name.toLowerCase().includes(needle);
}

// The selection bar's labelled buttons, and its round icon ones.
const BAR_PILL =
  "flex h-9 items-center gap-2 rounded-full px-4 text-sm font-medium outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50";
const BAR_ROUND =
  "flex size-9 items-center justify-center rounded-full outline-none transition-colors hover:bg-sidebar-accent focus-visible:ring-2 focus-visible:ring-ring";

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
    <div className="mt-6 grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-5">
      {Array.from({ length: 10 }, (_, index) => (
        <Skeleton key={index} className="aspect-square rounded-xl" />
      ))}
    </div>
  );
}

// Everything but media and models: the Storage Files row.
const FILE_TYPES: LibraryTypeFilter[] = ["documents", "spreadsheets", "presentations", "pdfs"];

export function LibraryPage() {
  const search = useSearch({ from: "/library" });
  const refresh = useLibraryStore((s) => s.refresh);
  const visit = useLibraryVisitStore((s) => s.visit);

  useEffect(() => {
    void refresh();
    // Files land in the Library from chats and the Images page, so look again on return.
    const onFocus = () => void refresh();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [refresh]);

  // Search, filters and selection belong to the view they were made in, so each tab and folder
  // gets a fresh one, as does a link from Settings.
  return (
    <LibraryView
      key={`${visit}:${search.show ?? ""}:${search.folder ?? ""}:${search.filter ?? ""}`}
      search={search}
    />
  );
}

function LibraryView({ search }: { search: LibrarySearch }) {
  const t = useT();
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
  const [filters, setFilters] = useState<LibraryFilters>(() =>
    search.filter === "files" ? { sources: new Set(), types: new Set(FILE_TYPES) } : EMPTY_FILTERS,
  );
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
  // Read once: changing Open on in settings must not move a Library that is already open.
  const [preferred] = useState(() =>
    settings.startTab === "last" ? settings.lastTab : settings.startTab,
  );
  const tab: LibraryTab =
    search.show ??
    (tabVisible(preferred) ? preferred : (LIBRARY_TABS.find(tabVisible) ?? "all"));
  // A column click or ?sort link (in the URL, so a Storage link always lands by size) beats the setting.
  const sort = SORT_STATES[search.sort ?? settings.sort];
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

  // A search, filter or Content setting that hides a selected entry deselects it, so bulk actions
  // only ever act on what is on screen and clearing the filter never brings a selection back.
  const visibleKeys = useMemo(
    () =>
      new Set([
        ...visibleFolders.map((folder) => `folder:${folder.id}`),
        ...visibleItems.map((item) => `item:${item.id}`),
      ]),
    [visibleFolders, visibleItems],
  );
  if ([...selection].some((key) => !visibleKeys.has(key))) {
    setSelection(new Set([...selection].filter((key) => visibleKeys.has(key))));
  }

  const previewItem = search.item ? (items.find((item) => item.id === search.item) ?? null) : null;
  // Recorded whenever the preview opens, so a shared link or history entry counts, not only a click.
  const previewId = previewItem?.id ?? null;
  useEffect(() => {
    if (previewId) markOpened(previewId);
  }, [previewId, markOpened]);

  // Opening a file pushes a history entry; closing one this view opened goes back off it, so Back
  // afterwards leaves the Library instead of landing on the same page again.
  const router = useRouter();
  const pushedPreview = useRef<string | null>(null);
  // Closed here but still in the URL until Back lands: not a missing file.
  const [closingPreview, setClosingPreview] = useState<string | null>(null);
  if (!search.item && closingPreview !== null) setClosingPreview(null);
  useEffect(() => {
    if (!search.item) pushedPreview.current = null;
  }, [search.item]);
  const openPreview = (id: string) => {
    pushedPreview.current = id;
    go({ ...search, item: id });
  };
  const closePreview = () => {
    if (!search.item) return;
    setClosingPreview(search.item);
    if (pushedPreview.current === search.item) {
      pushedPreview.current = null;
      router.history.back();
    } else {
      go({ ...search, item: undefined }, true);
    }
  };

  // A link to a file that has since gone (deleted, or another account's) says so. The snapshot
  // may just predate it (a note that was just made), so look once more first.
  const missingItem =
    search.item && status === "ready" && !previewItem && closingPreview !== search.item
      ? search.item
      : null;
  useEffect(() => {
    if (!missingItem) return;
    let cancelled = false;
    void refresh().then(() => {
      if (cancelled) return;
      if (useLibraryStore.getState().items.some((item) => item.id === missingItem)) return;
      toast(t("library.toast.missingItem"));
      void navigate({
        to: "/library",
        search: (prev) => ({ ...prev, item: undefined }),
        replace: true,
      });
    });
    return () => {
      cancelled = true;
    };
  }, [missingItem, refresh, navigate, t]);

  // ── Actions ────────────────────────────────────────────────────

  const fail = (message: string) => (err: unknown) =>
    toast.error(message, { description: errorMessage(err) });

  // Every file under the folder, subfolders included. Only files can be attached; a model in the
  // folder stays behind.
  const filesInFolder = (id: string) => {
    const inside = new Set([id]);
    for (let grew = true; grew; ) {
      grew = false;
      for (const folder of folders) {
        if (folder.parentId && inside.has(folder.parentId) && !inside.has(folder.id)) {
          inside.add(folder.id);
          grew = true;
        }
      }
    }
    return items.filter(
      (item) => item.folderId !== null && inside.has(item.folderId) && isFileItem(item),
    );
  };

  const chatAbout = (item: LibraryItem) =>
    void (item.model ? chatWithModel(navigate, item) : chatAboutItems(navigate, [item]));

  // One toast for the batch, however many moved.
  const moveAll = async (targets: LibraryTarget[], destination: string | null) => {
    const folder = destination
      ? (folderById.get(destination)?.name ?? t("library.toast.folderFallback"))
      : null;
    const results = await Promise.allSettled(
      targets.map((target) =>
        target.kind === "item"
          ? patchItem(target.item.id, { folderId: destination })
          : patchFolder(target.folder.id, { parentId: destination }),
      ),
    );
    const failed = results.find((result) => result.status === "rejected");
    if (failed) fail(t("library.toast.moveFailed"))(failed.reason);
    else if (folder === null) toast.success(t("library.toast.movedToLibrary"));
    else toast.success(t("library.toast.movedToFolder", { folder }));
  };

  const actions: LibraryActions = {
    folders,
    openItem: (item) => openPreview(item.id),
    openFolder: (id) => go({ folder: id }),
    chatAbout: (target) =>
      target.kind === "item"
        ? chatAbout(target.item)
        : void chatAboutItems(navigate, filesInFolder(target.folder.id)),
    toggleFavorite: (item) =>
      void patchItem(item.id, { favorite: !item.favorite }).catch(fail(t("library.toast.favoritesFailed"))),
    download: (item) => void downloadLibraryItem(item),
    rename: (target) => setNameDialog({ mode: "rename", target }),
    moveTo: (target, destination) => void moveAll([target], destination),
    moveToNewFolder: (target) =>
      setNameDialog({ mode: "create", parentId: folderId, thenMove: target }),
    remove: (target) => requestDelete([target]),
  };

  async function uploadBatch(batch: LibraryUploadBatch, count: number, label: string) {
    if (count === 0) return;
    const id = toast.loading(
      count === 1
        ? t("library.toast.uploadingOne", { name: label })
        : t("library.toast.uploadingMany", { count }),
    );
    try {
      await upload(batch, folderId);
      toast.success(
        count === 1 ? t("library.toast.uploadedOne") : t("library.toast.uploadedMany", { count }),
        { id },
      );
    } catch (err) {
      toast.error(t("library.toast.uploadFailed"), { id, description: errorMessage(err) });
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
      fail(t("library.toast.readDropsFailed"))(err);
    }
  }

  // Anywhere on the page takes a file drop, the desktop app's native drops included.
  const { ref: dropRef, dragging, dragHandlers } = useNativeFileDrop({
    onFiles: uploadFiles,
    onNativeIntents: uploadNativeDrops,
  });

  async function createNote() {
    try {
      const note = new File([""], `${t("library.create.untitledNote")}.md`, {
        type: "text/markdown",
      });
      const [id] = await upload({ files: [note] }, folderId);
      if (id) openPreview(id);
    } catch (err) {
      fail(t("library.toast.createNoteFailed"))(err);
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
      case "video":
        void navigate({ to: "/video" });
        break;
      case "audio":
        // Speak mode, where generated clips are made and listed.
        void navigate({ to: "/audio", search: { task: "text-to-speech" } });
        break;
      case "model":
        // A model is made by training one.
        void navigate({ to: "/studio" });
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
      if (nameDialog.thenMove) await moveAll([nameDialog.thenMove], folder.id);
    } catch (err) {
      const renaming = nameDialog.mode === "rename";
      fail(t(renaming ? "library.toast.renameFailed" : "library.toast.createFolderFailed"))(err);
      throw err;
    }
  }

  const requestDelete = (targets: LibraryTarget[]) =>
    settings.confirmDelete ? setPendingDelete(targets) : void confirmDelete(targets);

  async function confirmDelete(targets: LibraryTarget[]) {
    setPendingDelete(null);
    setSelection(new Set());
    // The open file is going: close it first, so it is not reported as missing.
    if (targets.some((t) => t.kind === "item" && t.item.id === search.item)) closePreview();
    const results = await Promise.allSettled(
      targets.map((target) =>
        target.kind === "item" ? removeItem(target.item.id) : removeFolder(target.folder.id),
      ),
    );
    const failed = results.find((result) => result.status === "rejected");
    if (failed) fail(t("library.toast.deleteFailed"))(failed.reason);
    // A folder being viewed that was just deleted leaves nothing to show.
    if (folderId && targets.some((t) => t.kind === "folder" && t.folder.id === folderId)) {
      go({ folder: currentFolder?.parentId ?? undefined, show: "folders" }, true);
    }
  }

  const deleteCopy = (targets: LibraryTarget[]) => {
    if (targets.length > 1) {
      return {
        title: t("library.dialog.deleteManyTitle", { count: targets.length }),
        description: t("library.dialog.deleteManyDescription"),
      };
    }
    const target = targets[0]!;
    if (target.kind === "folder") {
      const parent = target.folder.parentId ? folderById.get(target.folder.parentId) : null;
      return {
        title: t("library.dialog.deleteTitle", { name: target.folder.name }),
        description: parent
          ? t("library.dialog.deleteFolderIntoParent", { folder: parent.name })
          : t("library.dialog.deleteFolderIntoLibrary"),
      };
    }
    const source = target.item.id.split(":", 1)[0]!;
    return {
      title: t("library.dialog.deleteTitle", { name: target.item.name }),
      description: t(DELETE_NOTES[source] ?? "library.dialog.deleteUpload"),
    };
  };

  const cardSelection = {
    selection,
    toggle: (key: string) =>
      setSelection((current) => {
        const next = new Set(current);
        if (!next.delete(key)) next.add(key);
        return next;
      }),
  };

  const selectedTargets = (): LibraryTarget[] => {
    const out: LibraryTarget[] = [];
    for (const folder of visibleFolders)
      if (selection.has(`folder:${folder.id}`)) out.push({ kind: "folder", folder });
    for (const item of visibleItems)
      if (selection.has(`item:${item.id}`)) out.push({ kind: "item", item });
    return out;
  };

  // A selected folder can't move into itself or anything under it.
  const bulkDestinations = () =>
    folders.filter((folder) => {
      for (let at: LibraryFolder | undefined = folder; at; at = at.parentId ? folderById.get(at.parentId) : undefined)
        if (selection.has(`folder:${at.id}`)) return false;
      return true;
    });

  // ── Header ─────────────────────────────────────────────────────

  const breadcrumb: LibraryFolder[] = [];
  for (let at = currentFolder; at; at = at.parentId ? (folderById.get(at.parentId) ?? null) : null) {
    breadcrumb.unshift(at);
    if (breadcrumb.length > 32) break;
  }

  const title = folderId ? (
    <nav className="flex min-w-0 items-center gap-2 text-[calc(1.6875rem*var(--ui-font-scale,1))] font-semibold tracking-[-0.028em]" aria-label={t("library.breadcrumb")}>
      <button
        type="button"
        onClick={() => go({ show: "folders" })}
        className="shrink-0 text-muted-foreground transition-colors hover:text-foreground"
      >
        {t("shell.navigation.library")}
      </button>
      {breadcrumb.map((folder, index) => (
        <span key={folder.id} className="flex min-w-0 items-center gap-2">
          <HugeiconsIcon icon={ChevronRightStandardIcon} strokeWidth={2} className="size-5 shrink-0 text-muted-foreground" />
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
    <h1 className="text-[calc(1.6875rem*var(--ui-font-scale,1))] font-semibold leading-[1.04] tracking-[-0.028em] text-foreground">
      {t("shell.navigation.library")}
    </h1>
  );

  // ── Body ───────────────────────────────────────────────────────

  const uploadButton = (
    <Button variant="muted" className="rounded-full px-5" onClick={() => fileInput.current?.click()}>
      {t("library.empty.uploadFiles")}
    </Button>
  );

  function renderCards() {
    const itemsGrid = visibleItems.length > 0 && (
      <Masonry
        items={visibleItems}
        getKey={(item) => item.id}
        render={(item) => <ItemCard item={item} />}
      />
    );
    if (folderId || tab === "all") {
      // All heads each section; a folder just lists its subfolders above its files.
      const sectioned = !folderId;
      return (
        <>
          {visibleFolders.length > 0 && (
            <>
              {sectioned && <SectionHeading>{t("library.tabs.folders")}</SectionHeading>}
              <div className={cn(!sectioned && "mt-6")}>
                <FolderGrid folders={visibleFolders} counts={counts} />
              </div>
            </>
          )}
          {visibleItems.length > 0 && (
            <>
              {sectioned ? (
                <SectionHeading>{t("library.sections.items")}</SectionHeading>
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
    if (tab === "favorites") return undefined;
    if (tab === "folders") {
      return (
        <Button
          variant="dark"
          className="rounded-full px-5"
          onClick={() => setNameDialog({ mode: "create", parentId: null })}
        >
          {t("library.empty.createFolder")}
        </Button>
      );
    }
    if (!(tab in EMPTY_LINKS)) return uploadButton;
    const [to, label] = EMPTY_LINKS[tab as keyof typeof EMPTY_LINKS];
    return (
      <Button variant="muted" className="rounded-full px-5" onClick={() => void navigate({ to })}>
        {t(label)}
      </Button>
    );
  }

  function renderBody() {
    if (status === "loading" || status === "idle") return <LoadingGrid />;
    if (status === "error") {
      return (
        <EmptyState
          icon={Folder01Icon}
          title={t("library.empty.loadErrorTitle")}
          description={error ?? t("library.empty.loadErrorFallback")}
          action={
            <Button variant="muted" className="rounded-full" onClick={() => void refresh()}>
              {t("library.empty.tryAgain")}
            </Button>
          }
        />
      );
    }
    const empty = visibleItems.length === 0 && visibleFolders.length === 0;
    const narrowed = Boolean(needle) || filtersActive(filters);
    if (empty && narrowed) {
      return (
        <EmptyState
          icon={Folder01Icon}
          title={t("library.empty.noMatchesTitle")}
          description={t("library.empty.noMatchesDescription")}
        />
      );
    }
    if (empty && folderId) {
      return (
        <div className="mt-6 flex min-h-[calc(420px*var(--ui-space-scale,1))] flex-col items-center justify-center gap-5 rounded-xl border border-dashed border-border bg-muted/50">
          <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.5} className="size-8" />
          {uploadButton}
          <p className="text-sm text-muted-foreground">{t("library.empty.dropHere")}</p>
        </div>
      );
    }
    if (empty) {
      const [emptyIcon, emptyTitle, emptyDescription] = EMPTY_COPY[tab];
      return (
        <EmptyState
          icon={emptyIcon}
          title={t(emptyTitle)}
          description={t(emptyDescription)}
          action={emptyAction()}
        />
      );
    }
    if (view === "list") {
      return (
        <div className="mt-6">
          <LibraryList
            folders={visibleFolders}
            items={visibleItems}
            counts={counts}
            selection={selection}
            onSelectionChange={setSelection}
            sort={sort}
            onSortChange={(key) => go({ ...search, sort: sortParam(nextSort(sort, key)) }, true)}
            activity={tab === "suggested" && !folderId}
          />
        </div>
      );
    }
    return (
      <CardSelectionContext.Provider value={cardSelection}>{renderCards()}</CardSelectionContext.Provider>
    );
  }

  const nameDialogProps =
    nameDialog?.mode === "rename"
      ? {
          title: t(
            nameDialog.target.kind === "folder"
              ? "library.dialog.renameFolder"
              : "library.dialog.renameFile",
          ),
          submitLabel: t("common.rename"),
          initialValue:
            nameDialog.target.kind === "item" ? nameDialog.target.item.name : nameDialog.target.folder.name,
        }
      : {
          title: t("library.menu.newFolder"),
          submitLabel: t("library.dialog.create"),
          initialValue: "",
        };

  const selectedCount = selection.size;
  // Selected files and the files in selected folders, each once.
  const selectedFiles = () => {
    const out = new Map<string, LibraryItem>();
    for (const target of selectedTargets()) {
      const files =
        target.kind === "item"
          ? [target.item].filter(isFileItem)
          : filesInFolder(target.folder.id);
      for (const file of files) out.set(file.id, file);
    }
    return [...out.values()];
  };
  // A lone fine-tune opens with its run settings, as its own menu does.
  const selectedModel = () => {
    const targets = selectedTargets();
    const only = targets.length === 1 ? targets[0] : undefined;
    return only?.kind === "item" && isModelItem(only.item) ? only.item : null;
  };
  const bulkChat = () => {
    const model = selectedModel();
    if (model) chatAbout(model);
    else void chatAboutItems(navigate, selectedFiles());
    setSelection(new Set());
  };
  const bulkDownload = () => {
    const files = selectedFiles();
    setSelection(new Set());
    void downloadLibraryItems(files);
  };
  const bulkMove = (destination: string | null) => {
    void moveAll(selectedTargets(), destination);
    setSelection(new Set());
  };
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
        <LibraryHeader
          title={title}
          controls={
            <LibraryToolbar
              filters={filters}
              onFiltersChange={setFilters}
              filterMode={!folderId && tab === "folders" ? "none" : kindFilter ? "source" : "all"}
              view={view}
              onViewChange={setView}
              search={query}
              onSearchChange={setQuery}
              searchPlaceholder={t(folderId ? "library.searchFolder" : "library.searchLibrary")}
              onNew={handleNew}
              onSettings={() => openSettings("library")}
            />
          }
          tabs={
            folderId
              ? null
              : {
                  items: shownTabs.map((key) => ({ key, label: t(TAB_LABELS[key]) })),
                  active: tab,
                  onChange: (next) => go({ show: next as LibraryTab }),
                }
          }
        />
        <div className="pl-3">{renderBody()}</div>

        {dragging && (
          <div className="pointer-events-none fixed inset-0 z-40 flex items-center justify-center bg-background/70 backdrop-blur-sm">
            <div className="flex flex-col items-center gap-3 rounded-xl border-2 border-dashed border-muted-foreground px-16 py-12">
              <HugeiconsIcon icon={Upload01Icon} strokeWidth={1.5} className="size-8" />
              <p className="font-medium text-lg">
                {currentFolder
                  ? t("library.dropToUploadInto", { folder: currentFolder.name })
                  : t("library.dropToUpload")}
              </p>
            </div>
          </div>
        )}

        {selectedCount > 0 && (
          // The side menu's color with the dropdowns' shadow: the composer's in light mode, the page
          // color in dark.
          <div className="fixed bottom-8 left-1/2 z-30 flex -translate-x-1/2 items-center gap-2 rounded-full bg-sidebar py-2 pl-6 pr-2 text-sidebar-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:shadow-[0_8px_28px_-6px_var(--background)]">
            <span className="mr-4 whitespace-nowrap text-sm font-medium">
              {t("shell.selection.countSelected", { count: selectedCount })}
            </span>
            <button
              type="button"
              disabled={selectedModel() === null && selectedFiles().length === 0}
              onClick={bulkChat}
              className={cn(BAR_PILL, "bg-foreground text-background transition-opacity hover:opacity-85")}
            >
              <HugeiconsIcon icon={PencilEdit02Icon} strokeWidth={1.75} className="size-4" />
              {t("library.selection.startChat")}
            </button>
            <button
              type="button"
              disabled={selectedFiles().length === 0}
              onClick={bulkDownload}
              // Dark mode: borderless, filled like the model picker's search field.
              className={cn(
                BAR_PILL,
                "border border-border transition-colors hover:bg-sidebar-accent dark:border-transparent dark:bg-accent/60 dark:hover:bg-accent",
              )}
            >
              <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4" />
              {t("library.menu.download")}
            </button>
            <button
              type="button"
              disabled={deletableSelection().length === 0}
              onClick={() => requestDelete(deletableSelection())}
              className={cn(
                BAR_PILL,
                "border border-red-500/70 text-red-600 transition-colors hover:bg-red-500/15 dark:text-red-400",
              )}
            >
              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-4" />
              {t("common.delete")}
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  type="button"
                  aria-label={t("library.menu.moreActions")}
                  className={cn(BAR_ROUND, "data-[state=open]:bg-sidebar-accent")}
                >
                  <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-5" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="center" side="top" className="w-48">
                <DropdownMenuSub>
                  <DropdownMenuSubTrigger className="gap-2.5">
                    <HugeiconsIcon icon={FolderExportIcon} strokeWidth={1.75} className="size-icon" />
                    {t("library.selection.move")}
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent className="max-h-[min(--spacing(80),var(--radix-dropdown-menu-content-available-height))] w-56">
                    <DropdownMenuItem onSelect={() => bulkMove(null)}>
                      <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                      {t("library.menu.noFolder")}
                    </DropdownMenuItem>
                    {bulkDestinations().map((folder) => (
                      <DropdownMenuItem key={folder.id} onSelect={() => bulkMove(folder.id)}>
                        <HugeiconsIcon icon={Folder01Icon} strokeWidth={1.75} className="size-icon" />
                        <span className="truncate">{folder.name}</span>
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              </DropdownMenuContent>
            </DropdownMenu>
            <button
              type="button"
              aria-label={t("library.selection.clear")}
              onClick={() => setSelection(new Set())}
              className={BAR_ROUND}
            >
              <HugeiconsIcon icon={Cancel01Icon} strokeWidth={1.75} className="size-5" />
            </button>
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
        confirmLabel={t("common.delete")}
        onConfirm={() => pendingDelete && void confirmDelete(pendingDelete)}
        onOpenChange={(open) => !open && setPendingDelete(null)}
      />
      <LibraryPreview
        item={previewItem}
        onOpenChange={(open) => !open && closePreview()}
        onChat={chatAbout}
        onDownload={(item) => void downloadLibraryItem(item)}
        onOpenThread={(threadId) => void navigate({ to: "/chat", search: { thread: threadId } })}
        onToggleFavorite={actions.toggleFavorite}
        onDelete={(item) => actions.remove({ kind: "item", item })}
        onSaved={() => void refresh()}
      />
    </LibraryActionsProvider>
  );
}
