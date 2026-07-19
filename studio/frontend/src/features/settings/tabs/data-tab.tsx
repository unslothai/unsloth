// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Spinner } from "@/components/ui/spinner";
import { Switch } from "@/components/ui/switch";
import { usePlatformStore } from "@/config/env";
import {
  EXPORT_FORMATS_LIST,
  type FineTuneFormat,
  archiveAllChatItems,
  bulkExportConversationsByScope,
  clearAllChats,
  countAllChats,
  downloadArchivedChatExport,
  downloadChatExport,
  exportFineTuneJsonl,
  importConversationsFromFile,
  useChatPreferencesStore,
  useChatRuntimeStore,
  useChatSidebarItems,
} from "@/features/chat";
import { useT } from "@/i18n";
import {
  ChevronDownStandardIcon,
  ChevronRightStandardIcon,
} from "@/lib/chevron-icons";
import { toast } from "@/lib/toast";
import {
  Archive02Icon,
  ArrowLeft01Icon,
  Delete02Icon,
  Download01Icon,
  Tick02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { ArchivedChatsView } from "../components/archived-chats-dialog";
import {
  createFineTuneRecipeFromChats,
  loadFineTuneDatasetInTrainTab,
} from "../components/finetune-recipe";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import { UploadedFilesView } from "../components/uploaded-files-dialog";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

export function DataTab() {
  const t = useT();
  const navigate = useNavigate();
  const archivedChatsRequested = useSettingsDialogStore(
    (s) => s.archivedChatsRequested,
  );
  const consumeArchivedChatsRequest = useSettingsDialogStore(
    (s) => s.consumeArchivedChatsRequest,
  );
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [archiveConfirmOpen, setArchiveConfirmOpen] = useState(false);
  // Subpages swap the Data tab body instead of opening nested dialogs.
  const [subpage, setSubpage] = useState<"main" | "archived" | "files">(
    archivedChatsRequested ? "archived" : "main",
  );
  const [count, setCount] = useState<number | null>(null);
  const [exporting, setExporting] = useState(false);
  const [archivedExporting, setArchivedExporting] = useState(false);
  // Gates the archived subpage Export button.
  const { archivedItems } = useChatSidebarItems({ requireMessages: false });
  const [clearing, setClearing] = useState(false);
  const [archiving, setArchiving] = useState(false);
  const [fineTuneExporting, setFineTuneExporting] = useState(false);
  const [openingRecipe, setOpeningRecipe] = useState(false);
  const [loadingTraining, setLoadingTraining] = useState(false);
  // Chat-only hosts redirect /studio back to /chat, so loading a dataset in
  // the Train tab would upload it and then strand the user; gate the action
  // the same way the sidebar gates Train.
  const chatOnly = usePlatformStore((s) => s.isChatOnly());
  const [fineTuneAction, setFineTuneAction] = useState<
    "train" | "recipes" | "export"
  >(chatOnly ? "export" : "train");
  // Chat Completions (OpenAI messages) is the only export format we ship.
  const fineTuneFormat: FineTuneFormat = "openai";

  // The MLX self-heal can flip chat-only while the dialog is open.
  useEffect(() => {
    if (chatOnly) {
      setFineTuneAction((a) => (a === "train" ? "export" : a));
    }
  }, [chatOnly]);
  // Requests can arrive after Data is already mounted (for example from the
  // archive-all toast), so always switch before consuming the flag.
  useEffect(() => {
    if (!archivedChatsRequested) return;
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      setSubpage("archived");
      consumeArchivedChatsRequest();
    });
    return () => {
      cancelled = true;
    };
  }, [archivedChatsRequested, consumeArchivedChatsRequest]);

  const confirmDeleteChats = useChatPreferencesStore(
    (state) => state.confirmDeleteChats,
  );
  const setConfirmDeleteChats = useChatPreferencesStore(
    (state) => state.setConfirmDeleteChats,
  );

  const storeThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  // Open chat id from the route (single thread or compare pair), mirroring
  // ArchivedChatsView: compare panes only live in the search params.
  const openChatId = useRouterState({
    select: (s) => {
      if (!s.location.pathname.startsWith("/chat")) return undefined;
      const search = s.location.search as Record<string, string | undefined>;
      return search.thread ?? search.compare ?? storeThreadId ?? undefined;
    },
  });

  useEffect(() => {
    void countAllChats().then(setCount);
  }, []);

  const handleExport = async () => {
    setExporting(true);
    try {
      await downloadChatExport();
    } finally {
      setExporting(false);
    }
  };

  const handleExportArchived = async () => {
    setArchivedExporting(true);
    try {
      const exported = await downloadArchivedChatExport();
      toast.success(
        exported === 0
          ? t("settings.data.noArchivedChatsToExport")
          : exported === 1
            ? t("settings.data.exportedOneArchivedChat")
            : t("settings.data.exportedArchivedChatCount", { count: exported }),
      );
    } catch (error) {
      toast.error(t("settings.data.failedToExportArchivedChats"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setArchivedExporting(false);
    }
  };

  const importInputRef = useRef<HTMLInputElement>(null);
  const handleImport = async (file: File) => {
    try {
      const imported = await importConversationsFromFile(file, null);
      if (imported === 0) {
        toast.info(t("settings.chat.importNoConversations"));
      } else {
        toast.success(
          imported === 1
            ? t("settings.chat.importedOneChat")
            : t("settings.chat.importedChatCount", { count: imported }),
        );
        setCount(await countAllChats().catch(() => count));
      }
    } catch {
      toast.error(t("settings.chat.importFailed"));
    }
  };

  const handleArchiveAll = async () => {
    setArchiving(true);
    try {
      const archived = await archiveAllChatItems(openChatId, (view) => {
        navigate({ to: "/chat", search: { new: view.newThreadNonce } });
      });
      setArchiveConfirmOpen(false);
      toast.success(
        archived === 0
          ? t("settings.data.noChatsToArchive")
          : archived === 1
            ? t("settings.data.archivedOneChat")
            : t("settings.data.archivedChatCount", { count: archived }),
      );
    } catch (error) {
      toast.error(t("settings.data.failedToArchiveChats"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setArchiving(false);
    }
  };

  const handleFineTuneExport = async () => {
    setFineTuneExporting(true);
    try {
      await exportFineTuneJsonl(fineTuneFormat);
    } catch (error) {
      toast.error(t("settings.data.fineTuneExportFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setFineTuneExporting(false);
    }
  };

  const handleOpenInRecipes = async () => {
    setOpeningRecipe(true);
    try {
      const recipeId = await createFineTuneRecipeFromChats(fineTuneFormat);
      if (!recipeId) return;
      useSettingsDialogStore.getState().closeDialog();
      void navigate({ to: "/data-recipes/$recipeId", params: { recipeId } });
    } catch (error) {
      toast.error(t("settings.data.fineTuneRecipeFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setOpeningRecipe(false);
    }
  };

  const handleUseInTraining = async () => {
    setLoadingTraining(true);
    try {
      const loaded = await loadFineTuneDatasetInTrainTab(fineTuneFormat);
      if (!loaded) return;
      useSettingsDialogStore.getState().closeDialog();
      void navigate({ to: "/studio" });
    } catch (error) {
      toast.error(t("settings.data.fineTuneTrainFailed"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setLoadingTraining(false);
    }
  };

  const fineTuneActionLabels = {
    train: t("settings.data.fineTuneTrainAction"),
    recipes: t("settings.data.fineTuneOpenRecipesAction"),
    export: t("settings.data.fineTuneExportAction"),
  } as const;
  const fineTuneBusy = loadingTraining || openingRecipe || fineTuneExporting;
  const runFineTuneAction = () => {
    if (fineTuneAction === "train") {
      if (chatOnly) return;
      void handleUseInTraining();
    } else if (fineTuneAction === "recipes") void handleOpenInRecipes();
    else void handleFineTuneExport();
  };

  const handleClear = async () => {
    setClearing(true);
    try {
      const result = await clearAllChats();
      const clearedCount = result.deletedThreadIds.length;
      const hasFailedStore =
        result.backend === "failed" || result.legacy === "failed";
      if (!hasFailedStore && result.failedThreadIds.length === 0) {
        setCount(0);
        setConfirmOpen(false);
        toast.success(
          clearedCount === 0
            ? t("settings.chat.clearedAllChats")
            : clearedCount === 1
              ? t("settings.chat.clearedOneChat")
              : t("settings.chat.clearedChatCount", { count: clearedCount }),
        );
        return;
      }

      const fallbackRemaining =
        result.failedThreadIds.length > 0
          ? result.failedThreadIds.length
          : (count ?? 0);
      const remaining = await countAllChats().catch(() => fallbackRemaining);
      setCount(remaining);
      setConfirmOpen(false);
      toast.warning(t("settings.chat.someChatsCouldNotBeCleared"), {
        description:
          result.failedThreadIds.length > 0
            ? clearedCount === 1 && result.failedThreadIds.length === 1
              ? t("settings.chat.oneChatClearedRemainOne")
              : clearedCount === 1
                ? t("settings.chat.oneChatClearedRemain", {
                    remainingCount: result.failedThreadIds.length,
                  })
                : result.failedThreadIds.length === 1
                  ? t("settings.chat.chatsClearedRemainOne", { clearedCount })
                  : t("settings.chat.chatsClearedRemain", {
                      clearedCount,
                      remainingCount: result.failedThreadIds.length,
                    })
            : remaining === 1
              ? t("settings.chat.storageClearFailedOne")
              : t("settings.chat.storageClearFailed", { count: remaining }),
      });
    } catch (error) {
      const remaining = await countAllChats().catch(() => count);
      setCount(remaining);
      toast.error(t("settings.chat.failedToClearChats"), {
        description: error instanceof Error ? error.message : undefined,
      });
    } finally {
      setClearing(false);
    }
  };

  if (subpage === "archived") {
    return (
      <div className="flex flex-col gap-6">
        <header className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSubpage("main")}
            aria-label={`Back to ${t("settings.data.title")}`}
            className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
          </button>
          <h1 className="text-xl font-semibold font-heading">
            {t("settings.data.title")}
          </h1>
        </header>
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1">
            <h2 className="text-sm font-semibold">
              {t("settings.data.archivedChats")}
            </h2>
            <p className="text-xs text-muted-foreground">
              {t("settings.data.archivedChatsDescription")}
            </p>
          </div>
          {archivedItems.length > 0 && (
            <Button
              variant="outline"
              size="sm"
              className="shrink-0"
              onClick={handleExportArchived}
              disabled={archivedExporting}
            >
              {archivedExporting ? (
                <Spinner className="size-4" />
              ) : (
                <HugeiconsIcon
                  icon={Download01Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              )}
              {archivedExporting
                ? t("settings.data.exportingArchivedChats")
                : t("settings.data.exportArchivedChats")}
            </Button>
          )}
        </div>
        <ArchivedChatsView />
      </div>
    );
  }

  if (subpage === "files") {
    return (
      <div className="flex flex-col gap-6">
        <header className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSubpage("main")}
            aria-label={`Back to ${t("settings.data.title")}`}
            className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
          </button>
          <h1 className="text-xl font-semibold font-heading">
            {t("settings.data.title")}
          </h1>
        </header>
        <div className="flex flex-col gap-1">
          <h2 className="text-sm font-semibold">
            {t("settings.data.uploadedFiles")}
          </h2>
          <p className="text-xs text-muted-foreground">
            {t("settings.data.uploadedFilesDescription")}
          </p>
        </div>
        <UploadedFilesView />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.data.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.data.description")}
        </p>
      </header>

      <div className="flex flex-col divide-y divide-border/60">
        <SettingsRow
          alignTop={true}
          label={t("settings.data.fineTuneExport")}
          description={t("settings.data.fineTuneExportDescription")}
        >
          <div className="flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                {/* Fixed width so switching actions never resizes the row. */}
                <Button
                  variant="outline"
                  size="sm"
                  disabled={count === 0}
                  className="w-44 justify-between"
                >
                  <span className="truncate">
                    {fineTuneActionLabels[fineTuneAction]}
                  </span>
                  <HugeiconsIcon
                    icon={ChevronDownStandardIcon}
                    className="size-3.5 shrink-0"
                  />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                {(["export", "train", "recipes"] as const).map((action) => (
                  <DropdownMenuItem
                    key={action}
                    disabled={action === "train" && chatOnly}
                    onSelect={() => setFineTuneAction(action)}
                  >
                    <span className="flex-1">
                      {fineTuneActionLabels[action]}
                    </span>
                    {fineTuneAction === action ? (
                      <HugeiconsIcon icon={Tick02Icon} className="size-4" />
                    ) : null}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <Button
              size="icon-sm"
              onClick={runFineTuneAction}
              disabled={fineTuneBusy || count === 0}
              aria-label={t("settings.data.fineTuneRunAction")}
              title={`${t("settings.data.fineTuneRunAction")}: ${fineTuneActionLabels[fineTuneAction]}`}
              className="shrink-0 rounded-full"
            >
              {fineTuneBusy ? (
                <Spinner className="size-4" />
              ) : (
                <HugeiconsIcon
                  icon={ChevronRightStandardIcon}
                  strokeWidth={2.5}
                  className="size-4"
                />
              )}
            </Button>
          </div>
        </SettingsRow>

        <SettingsRow
          label={t("settings.data.archivedChats")}
          description={t("settings.data.archivedChatsDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSubpage("archived")}
          >
            {t("settings.data.manageAction")}
          </Button>
        </SettingsRow>

        <SettingsRow
          label={t("settings.data.archiveAllChats")}
          description={t("settings.data.archiveAllChatsDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setArchiveConfirmOpen(true)}
          >
            <HugeiconsIcon icon={Archive02Icon} className="size-3.5 mr-1.5" />
            {t("settings.data.archiveAllAction")}
          </Button>
        </SettingsRow>

        <SettingsRow
          label={t("settings.data.confirmBeforeDeleting")}
          description={t("settings.data.confirmBeforeDeletingDescription")}
        >
          <Switch
            checked={confirmDeleteChats}
            onCheckedChange={setConfirmDeleteChats}
          />
        </SettingsRow>

        <SettingsRow
          label={t("settings.chat.exportHistory")}
          description={t("settings.chat.exportHistoryDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            disabled={exporting || count === 0}
          >
            <HugeiconsIcon icon={Download01Icon} className="size-3.5 mr-1.5" />
            {exporting
              ? t("settings.chat.exportingAction")
              : t("settings.chat.exportAction")}
          </Button>
        </SettingsRow>

        <SettingsRow
          label={t("settings.chat.exportConversations")}
          description={t("settings.chat.exportConversationsDescription")}
        >
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <Button variant="outline" size="sm" disabled={count === 0}>
                <HugeiconsIcon
                  icon={Download01Icon}
                  className="size-3.5 mr-1.5"
                />
                {t("settings.chat.exportConversationsAction")}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              {(
                [
                  { scope: "recents", label: "exportScopeRecents" },
                  { scope: "all", label: "exportScopeAll" },
                ] as const
              ).map(({ scope, label }) => (
                <DropdownMenuSub key={scope}>
                  <DropdownMenuSubTrigger>
                    <HugeiconsIcon
                      icon={Download01Icon}
                      className="size-3.5 mr-1"
                    />
                    {t(`settings.chat.${label}`)}
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent className="w-56">
                    {EXPORT_FORMATS_LIST.map(({ fmt, label: fmtLabel }) => (
                      <DropdownMenuItem
                        key={`${scope}-m-${fmt}`}
                        onSelect={() =>
                          void bulkExportConversationsByScope(scope, fmt, true)
                        }
                      >
                        {fmtLabel} {t("settings.chat.exportCombinedSuffix")}
                      </DropdownMenuItem>
                    ))}
                    <DropdownMenuSeparator />
                    {EXPORT_FORMATS_LIST.map(({ fmt, label: fmtLabel }) => (
                      <DropdownMenuItem
                        key={`${scope}-s-${fmt}`}
                        onSelect={() =>
                          void bulkExportConversationsByScope(scope, fmt, false)
                        }
                      >
                        {fmtLabel} {t("settings.chat.exportPerChatSuffix")}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        </SettingsRow>

        <SettingsRow
          destructive={true}
          // divide-y already draws the row separator; drop the extra border.
          className="border-t-0 mt-0 pt-3"
          label={t("settings.chat.clearAllChats")}
          description={
            count === null
              ? t("settings.chat.clearAllChatsDescription")
              : count === 0
                ? t("settings.chat.noChatsToClear")
                : count === 1
                  ? t("settings.chat.clearOneChatDescription")
                  : t("settings.chat.clearChatCountDescription", { count })
          }
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setConfirmOpen(true)}
            disabled={count === 0}
            className="text-destructive hover:text-destructive hover:border-destructive/60"
          >
            <HugeiconsIcon icon={Delete02Icon} className="size-3.5 mr-1.5" />
            {t("settings.chat.clearChatsAction")}
          </Button>
        </SettingsRow>

        <SettingsRow
          label={t("settings.chat.importChats")}
          description={t("settings.chat.importChatsDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => importInputRef.current?.click()}
          >
            <HugeiconsIcon icon={Upload01Icon} className="size-3.5 mr-1.5" />
            {t("settings.chat.importChatsAction")}
          </Button>
          <input
            ref={importInputRef}
            type="file"
            accept=".jsonl,.ndjson,.csv"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void handleImport(file);
            }}
          />
        </SettingsRow>
      </div>

      <SettingsSection title={t("settings.data.filesSection")}>
        <SettingsRow
          label={t("settings.data.uploadedFiles")}
          description={t("settings.data.uploadedFilesDescription")}
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSubpage("files")}
          >
            {t("settings.data.manageAction")}
          </Button>
        </SettingsRow>
      </SettingsSection>

      <Dialog open={archiveConfirmOpen} onOpenChange={setArchiveConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{t("settings.data.archiveAllChatsTitle")}</DialogTitle>
            <DialogDescription>
              {t("settings.data.archiveAllChatsConfirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setArchiveConfirmOpen(false)}
            >
              {t("common.cancel")}
            </Button>
            <Button onClick={handleArchiveAll} disabled={archiving}>
              {archiving
                ? t("settings.data.archivingAction")
                : t("settings.data.archiveAllAction")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {count === 1
                ? t("settings.chat.clearOneChatTitle")
                : t("settings.chat.clearChatsTitle", { count: count ?? 0 })}
            </DialogTitle>
            <DialogDescription>
              {t("settings.chat.clearChatsConfirmDescription")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirmOpen(false)}>
              {t("common.cancel")}
            </Button>
            <Button
              onClick={handleClear}
              disabled={clearing}
              className="bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              {clearing
                ? t("settings.chat.clearingAction")
                : count === 1
                  ? t("settings.chat.clearOneChatAction")
                  : t("settings.chat.clearChatCountAction", {
                      count: count ?? 0,
                    })}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
