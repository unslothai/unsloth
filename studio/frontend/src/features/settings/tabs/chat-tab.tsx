// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { toast } from "@/lib/toast";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  clearAllChats,
  countAllChats,
  downloadChatExport,
  useChatRuntimeStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import { Delete02Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

export function ChatTab() {
  const t = useT();
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [count, setCount] = useState<number | null>(null);
  const [exporting, setExporting] = useState(false);
  const [clearing, setClearing] = useState(false);
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
        <h1 className="text-lg font-semibold font-heading">
          {t("settings.chat.title")}
        </h1>
        <p className="text-xs text-muted-foreground">
          {t("settings.chat.description")}
        </p>
      </header>

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
          destructive
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
