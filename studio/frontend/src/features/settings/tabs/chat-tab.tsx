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
import { Switch } from "@/components/ui/switch";
import {
  EXPORT_FORMATS_LIST,
  type PlusMenuItemId,
  bulkExportConversationsByScope,
  clearAllChats,
  countAllChats,
  downloadChatExport,
  importConversationsFromFile,
  useChatPreferencesStore,
  useChatRuntimeStore,
  usePlusMenuPrefsStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  Bookmark02Icon,
  Delete02Icon,
  Download01Icon,
  FileDatabaseIcon,
  Folder01Icon,
  McpServerIcon,
  PencilRulerIcon,
  Settings02Icon,
  ShieldBanIcon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Columns2Icon, PlusIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { ArchivedChatsDialog } from "../components/archived-chats-dialog";
import { SettingsRow } from "../components/settings-row";
import {
  SettingsGroupDivider,
  SettingsSection,
} from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

// Adjustable "+" menu items shown in settings, in display order. Icons mirror
// the ones used in the composer + menu itself.
const PLUS_MENU_ICON_CLASS = "size-[18px]";
const PLUS_MENU_SETTINGS: {
  id: PlusMenuItemId;
  label: string;
  icon: ReactNode;
}[] = [
  {
    id: "chatWithFiles",
    label: "Chat with Files (RAG)",
    icon: (
      <HugeiconsIcon
        icon={FileDatabaseIcon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
  {
    id: "mcp",
    label: "MCP",
    icon: (
      <HugeiconsIcon
        icon={McpServerIcon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
  {
    id: "savedPrompts",
    label: "Saved prompts",
    icon: (
      <HugeiconsIcon
        icon={Bookmark02Icon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
  {
    id: "compareChat",
    label: "Compare chat",
    icon: <Columns2Icon className={PLUS_MENU_ICON_CLASS} />,
  },
  {
    id: "exportChat",
    label: "Export chat",
    icon: (
      <HugeiconsIcon
        icon={Download01Icon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
  {
    id: "canvas",
    label: "Canvas",
    icon: (
      <HugeiconsIcon
        icon={PencilRulerIcon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
  {
    id: "projects",
    label: "Projects",
    icon: (
      <HugeiconsIcon
        icon={Folder01Icon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
  {
    id: "bypassPermissions",
    label: "Bypass permissions",
    icon: (
      <HugeiconsIcon
        icon={ShieldBanIcon}
        strokeWidth={2}
        className={PLUS_MENU_ICON_CLASS}
      />
    ),
  },
];

export function ChatTab() {
  const t = useT();
  const plusPins = usePlusMenuPrefsStore((state) => state.pins);
  const togglePlusPin = usePlusMenuPrefsStore((state) => state.togglePin);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [archivedOpen, setArchivedOpen] = useState(false);
  const [count, setCount] = useState<number | null>(null);
  const archivedChatsRequested = useSettingsDialogStore(
    (s) => s.archivedChatsRequested,
  );
  const consumeArchivedChatsRequest = useSettingsDialogStore(
    (s) => s.consumeArchivedChatsRequest,
  );

  // Open the archived list when the archive toast asked to jump here.
  useEffect(() => {
    if (!archivedChatsRequested) return;
    setArchivedOpen(true);
    consumeArchivedChatsRequest();
  }, [archivedChatsRequested, consumeArchivedChatsRequest]);
  const [exporting, setExporting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const autoTitle = useChatRuntimeStore((state) => state.autoTitle);
  const setAutoTitle = useChatRuntimeStore((state) => state.setAutoTitle);
  const showCanvasMenuItem = useChatRuntimeStore(
    (state) => state.showCanvasMenuItem,
  );
  const setShowCanvasMenuItem = useChatRuntimeStore(
    (state) => state.setShowCanvasMenuItem,
  );
  const collapseHtmlArtifacts = useChatRuntimeStore(
    (state) => state.collapseHtmlArtifacts,
  );
  const setCollapseHtmlArtifacts = useChatRuntimeStore(
    (state) => state.setCollapseHtmlArtifacts,
  );
  const allowArtifactNetworkAccess = useChatRuntimeStore(
    (state) => state.allowArtifactNetworkAccess,
  );
  const setAllowArtifactNetworkAccess = useChatRuntimeStore(
    (state) => state.setAllowArtifactNetworkAccess,
  );
  const hydratePersistedSettings = useChatRuntimeStore(
    (state) => state.hydratePersistedSettings,
  );
  const loadOnSelection = useChatRuntimeStore((state) => state.loadOnSelection);
  const setLoadOnSelection = useChatRuntimeStore(
    (state) => state.setLoadOnSelection,
  );
  const expandQuantizations = useChatRuntimeStore(
    (state) => state.expandQuantizations,
  );
  const setExpandQuantizations = useChatRuntimeStore(
    (state) => state.setExpandQuantizations,
  );
  const showAllQuantizations = useChatRuntimeStore(
    (state) => state.showAllQuantizations,
  );
  const setShowAllQuantizations = useChatRuntimeStore(
    (state) => state.setShowAllQuantizations,
  );
  const confirmDeleteChats = useChatPreferencesStore(
    (state) => state.confirmDeleteChats,
  );
  const setConfirmDeleteChats = useChatPreferencesStore(
    (state) => state.setConfirmDeleteChats,
  );
  const showModelDisclaimer = useChatPreferencesStore(
    (state) => state.showModelDisclaimer,
  );
  const setShowModelDisclaimer = useChatPreferencesStore(
    (state) => state.setShowModelDisclaimer,
  );
  const showResponseModel = useChatPreferencesStore(
    (state) => state.showResponseModel,
  );
  const setShowResponseModel = useChatPreferencesStore(
    (state) => state.setShowResponseModel,
  );

  useEffect(() => {
    void countAllChats().then(setCount);
    void hydratePersistedSettings();
  }, [hydratePersistedSettings]);

  const handleExport = async () => {
    setExporting(true);
    try {
      await downloadChatExport();
    } finally {
      setExporting(false);
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

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.chat.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.chat.description")}
        </p>
      </header>

      <SettingsSection title="Select model settings">
        <SettingsRow
          label="Load on selection"
          alignTop={true}
          description={
            <span>
              On: Unsloth auto-picks the best settings and loads it.
              <br />
              Off: opens Run settings to customize, then load.
              <br />
              The gear always opens Run settings:{" "}
              <span className="ml-2 inline-flex items-center gap-3 align-middle">
                <span className="font-mono text-xs text-foreground">
                  Q4_K_M
                </span>
                <span className="text-[9px] font-medium text-green-400">
                  downloaded
                </span>
                <span className="text-[10px] text-muted-foreground">16 GB</span>
                <span className="inline-flex size-4 items-center justify-center rounded bg-black/[0.06] dark:bg-white/[0.08]">
                  <HugeiconsIcon
                    icon={Settings02Icon}
                    strokeWidth={1.75}
                    className="size-2.5 text-muted-foreground/80"
                  />
                </span>
              </span>
            </span>
          }
        >
          <Switch
            checked={loadOnSelection}
            onCheckedChange={setLoadOnSelection}
          />
        </SettingsRow>
        <SettingsRow
          label="Expand quantizations"
          description={
            <span>
              On: On Device GGUF models show their quantizations right away.
              <br />
              Off: click a model to view its quantizations.
            </span>
          }
        >
          <Switch
            checked={expandQuantizations}
            onCheckedChange={setExpandQuantizations}
          />
        </SettingsRow>
        <SettingsRow
          label="Show all quantizations"
          description={
            <span>
              On: list every On Device quantization, including not downloaded.
              <br />
              Off: show only downloaded quantizations.
            </span>
          }
        >
          <Switch
            checked={showAllQuantizations}
            onCheckedChange={setShowAllQuantizations}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title="Chat menu"
        description={
          <>
            Pin items to chat's{" "}
            <PlusIcon
              aria-label="+"
              className="inline size-3.5 align-[-2px] stroke-[2px]"
            />{" "}
            side menu. Others move into “More”.
          </>
        }
      >
        {PLUS_MENU_SETTINGS.map((item) => (
          <SettingsRow key={item.id} label={item.label} icon={item.icon}>
            {/* Canvas toggles menu visibility; the rest toggle pin placement. */}
            <Switch
              checked={
                item.id === "canvas" ? showCanvasMenuItem : plusPins[item.id]
              }
              onCheckedChange={
                item.id === "canvas"
                  ? setShowCanvasMenuItem
                  : () => togglePlusPin(item.id)
              }
            />
          </SettingsRow>
        ))}
        <SettingsGroupDivider />
        <SettingsRow
          label={t("settings.chat.modelDisclaimer")}
          description={t("settings.chat.modelDisclaimerDescription")}
        >
          <Switch
            checked={showModelDisclaimer}
            onCheckedChange={setShowModelDisclaimer}
          />
        </SettingsRow>
        <SettingsRow
          label="Show response model"
          description="Show model metadata in assistant responses."
        >
          <Switch
            checked={showResponseModel}
            onCheckedChange={setShowResponseModel}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.general.chatDefaults")}>
        <SettingsRow
          label={t("settings.general.autoTitleNewChats")}
          description={t("settings.general.autoTitleNewChatsDescription")}
        >
          <Switch checked={autoTitle} onCheckedChange={setAutoTitle} />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.chat.artifacts.title")}>
        <SettingsRow
          label={t("settings.chat.artifacts.collapseHtmlBlocks")}
          description={t(
            "settings.chat.artifacts.collapseHtmlBlocksDescription",
          )}
        >
          <Switch
            checked={collapseHtmlArtifacts}
            onCheckedChange={setCollapseHtmlArtifacts}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.artifacts.allowNetworkAccess")}
          description={t(
            "settings.chat.artifacts.allowNetworkAccessDescription",
          )}
        >
          <Switch
            checked={allowArtifactNetworkAccess}
            onCheckedChange={setAllowArtifactNetworkAccess}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.chat.data")}>
        <SettingsRow
          label="Archived chats"
          description="View and manage chats you have archived."
        >
          <Button
            variant="outline"
            size="sm"
            onClick={() => setArchivedOpen(true)}
          >
            Manage
          </Button>
        </SettingsRow>

        <SettingsRow
          label="Confirm before deleting"
          description="Ask for confirmation before a chat is deleted. Turn off to delete instantly."
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

        <SettingsRow
          destructive={true}
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
      </SettingsSection>

      <ArchivedChatsDialog open={archivedOpen} onOpenChange={setArchivedOpen} />

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
