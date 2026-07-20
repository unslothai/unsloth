// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Switch } from "@/components/ui/switch";
import {
  type PlusMenuItemId,
  useChatPreferencesStore,
  useChatRuntimeStore,
  usePlusMenuPrefsStore,
} from "@/features/chat";
import { useT } from "@/i18n";
import {
  Bookmark02Icon,
  Download01Icon,
  FileDatabaseIcon,
  Folder01Icon,
  McpServerIcon,
  PencilRulerIcon,
  Settings02Icon,
  ShieldBanIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Columns2Icon, PlusIcon } from "lucide-react";
import { useEffect } from "react";
import type { ReactNode } from "react";
import { SettingsRow } from "../components/settings-row";
import {
  SettingsGroupDivider,
  SettingsSection,
} from "../components/settings-section";

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
    void hydratePersistedSettings();
  }, [hydratePersistedSettings]);

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
                <span className="text-[9px] font-medium text-green-600/90 dark:text-green-400/80">
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
    </div>
  );
}
