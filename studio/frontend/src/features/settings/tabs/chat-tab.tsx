// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  type PlusMenuItemId,
  refreshModelDisclaimerPreference,
  saveModelDisclaimerPreference,
  useChatPreferencesStore,
  useChatRuntimeStore,
  usePlusMenuPrefsStore,
  useSidebarOrganizationStore,
} from "@/features/chat";
import {
  compactionStyleValue,
  parseCompactionStyle,
} from "@/features/chat/utils/auto-compaction";
import { PASTED_TEXT_THRESHOLD_CHOICES } from "@/features/chat/utils/pasted-text";
import { formatBindingLabel, isMacPlatform } from "../lib/keyboard-shortcuts";
import { useUserProfileStore } from "@/features/profile";
import { type TranslationKey, useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  Bookmark02Icon,
  Download01Icon,
  FileDatabaseIcon,
  Folder01Icon,
  McpServerIcon,
  PencilRulerIcon,
  ShieldBanIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Columns2Icon } from "lucide-react";
import { useEffect } from "react";
import type { ReactNode } from "react";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";

// Adjustable "+" menu items shown in settings, in display order. Icons mirror
// the ones used in the composer + menu itself.
const PLUS_MENU_ICON_CLASS = "size-[18px]";
const PLUS_MENU_SETTINGS: {
  id: PlusMenuItemId;
  labelKey: TranslationKey;
  icon: ReactNode;
}[] = [
  {
    id: "chatWithFiles",
    labelKey: "settings.chat.menu.chatWithFiles",
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
    labelKey: "settings.chat.menu.mcp",
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
    labelKey: "settings.chat.menu.savedPrompts",
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
    labelKey: "settings.chat.menu.compareChat",
    icon: <Columns2Icon className={PLUS_MENU_ICON_CLASS} />,
  },
  {
    id: "exportChat",
    labelKey: "settings.chat.menu.exportChat",
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
    labelKey: "settings.chat.artifacts.title",
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
    labelKey: "shell.navigation.projects",
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
    labelKey: "settings.general.permissions.bypassLabel",
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
  const projectAttachmentTarget = useChatRuntimeStore(
    (state) => state.projectAttachmentTarget,
  );
  const setProjectAttachmentTarget = useChatRuntimeStore(
    (state) => state.setProjectAttachmentTarget,
  );
  const rememberParamsPerModel = useChatRuntimeStore(
    (state) => state.rememberParamsPerModel,
  );
  const setRememberParamsPerModel = useChatRuntimeStore(
    (state) => state.setRememberParamsPerModel,
  );
  const autoCompactEnabled = useChatRuntimeStore(
    (state) => state.autoCompactEnabled,
  );
  const setAutoCompactEnabled = useChatRuntimeStore(
    (state) => state.setAutoCompactEnabled,
  );
  const contextPolicy = useChatRuntimeStore((state) => state.contextPolicy);
  const compactionHeadroomRatio = useChatRuntimeStore(
    (state) => state.compactionHeadroomRatio,
  );
  const setContextPolicy = useChatRuntimeStore(
    (state) => state.setContextPolicy,
  );
  const setCompactionHeadroomRatio = useChatRuntimeStore(
    (state) => state.setCompactionHeadroomRatio,
  );
  const showGreetingSloth = useUserProfileStore((s) => s.showGreetingSloth);
  const setShowGreetingSloth = useUserProfileStore(
    (s) => s.setShowGreetingSloth,
  );
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
  const searchImages = useChatRuntimeStore((state) => state.searchImages);
  const setSearchImages = useChatRuntimeStore(
    (state) => state.setSearchImages,
  );
  const hydratePersistedSettings = useChatRuntimeStore(
    (state) => state.hydratePersistedSettings,
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
  const showMemoryBar = useChatRuntimeStore((state) => state.showMemoryBar);
  const setShowMemoryBar = useChatRuntimeStore(
    (state) => state.setShowMemoryBar,
  );
  const organizeBy = useSidebarOrganizationStore((s) => s.organizeBy);
  const setOrganizeBy = useSidebarOrganizationStore((s) => s.setOrganizeBy);
  const showModelDisclaimer = useChatPreferencesStore(
    (state) => state.showModelDisclaimer,
  );
  const showResponseModel = useChatPreferencesStore(
    (state) => state.showResponseModel,
  );
  const setShowResponseModel = useChatPreferencesStore(
    (state) => state.setShowResponseModel,
  );
  const collapseThinkingByDefault = useChatPreferencesStore(
    (state) => state.collapseThinkingByDefault,
  );
  const setCollapseThinkingByDefault = useChatPreferencesStore(
    (state) => state.setCollapseThinkingByDefault,
  );
  const collapseToolActivityByDefault = useChatPreferencesStore(
    (state) => state.collapseToolActivityByDefault,
  );
  const setCollapseToolActivityByDefault = useChatPreferencesStore(
    (state) => state.setCollapseToolActivityByDefault,
  );
  const pastedTextMinChars = useChatPreferencesStore(
    (state) => state.pastedTextMinChars,
  );
  const setPastedTextMinChars = useChatPreferencesStore(
    (state) => state.setPastedTextMinChars,
  );
  // The platform's own paste-without-formatting chord, which the composer reads
  // as "put it in the box" whatever this threshold says. macOS carries it on
  // Option, that being the chord its Edit menu binds.
  const macPlatform = isMacPlatform();
  const plainPasteLabel = formatBindingLabel(
    { code: "KeyV", mod: true, ctrl: false, shift: true, alt: macPlatform },
    macPlatform,
  );

  useEffect(() => {
    void hydratePersistedSettings();
    refreshModelDisclaimerPreference().catch(() => undefined);
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

      <SettingsSection title={t("settings.chat.modelSelection.title")}>
        <SettingsRow
          label={t("settings.chat.modelSelection.expandQuantizations")}
          description={t(
            "settings.chat.modelSelection.expandQuantizationsDescription",
          )}
        >
          <Switch
            checked={expandQuantizations}
            onCheckedChange={setExpandQuantizations}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.modelSelection.showAllQuantizations")}
          description={t(
            "settings.chat.modelSelection.showAllQuantizationsDescription",
          )}
        >
          <Switch
            checked={showAllQuantizations}
            onCheckedChange={setShowAllQuantizations}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.modelSelection.showMemoryBar")}
          description={t(
            "settings.chat.modelSelection.showMemoryBarDescription",
          )}
        >
          <Switch checked={showMemoryBar} onCheckedChange={setShowMemoryBar} />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection
        title={t("settings.chat.menu.title")}
        description={t("settings.chat.menu.description")}
      >
        {PLUS_MENU_SETTINGS.map((item) => (
          <SettingsRow key={item.id} label={t(item.labelKey)} icon={item.icon}>
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
      </SettingsSection>

      <SettingsSection title={t("settings.general.chatDefaults")}>
        <SettingsRow
          label={t("settings.chat.projectsSection")}
          description={t("settings.chat.projectsSectionDescription")}
        >
          <Switch
            checked={organizeBy === "project"}
            onCheckedChange={(checked) =>
              setOrganizeBy(checked ? "project" : "list")
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.thinking.collapseByDefault")}
          description={t("settings.chat.thinking.collapseByDefaultDescription")}
        >
          <Switch
            checked={collapseThinkingByDefault}
            onCheckedChange={setCollapseThinkingByDefault}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.tools.collapseByDefault")}
          description={t(
            "settings.chat.tools.collapseByDefaultDescription",
          )}
        >
          <Switch
            checked={collapseToolActivityByDefault}
            onCheckedChange={setCollapseToolActivityByDefault}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.modelDisclaimer")}
          description={t("settings.chat.modelDisclaimerDescription")}
        >
          <Switch
            checked={showModelDisclaimer}
            onCheckedChange={(checked) => {
              return saveModelDisclaimerPreference(checked).catch(() => {
                toast.error("Could not save the model disclaimer setting.");
              });
            }}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.showResponseModel")}
          description={t("settings.chat.showResponseModelDescription")}
        >
          <Switch
            checked={showResponseModel}
            onCheckedChange={setShowResponseModel}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.general.autoTitleNewChats")}
          description={t("settings.general.autoTitleNewChatsDescription")}
        >
          <Switch checked={autoTitle} onCheckedChange={setAutoTitle} />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.projectAttachments")}
          description={t("settings.chat.projectAttachmentsDescription")}
        >
          <Switch
            checked={projectAttachmentTarget === "project"}
            onCheckedChange={(checked) =>
              setProjectAttachmentTarget(checked ? "project" : "thread")
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.rememberParamsPerModel")}
          description={t("settings.chat.rememberParamsPerModelDescription")}
        >
          <Switch
            checked={rememberParamsPerModel}
            onCheckedChange={setRememberParamsPerModel}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.autoCompact")}
          description={t("settings.chat.autoCompactDescription")}
        >
          <Switch
            checked={autoCompactEnabled}
            onCheckedChange={setAutoCompactEnabled}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.compactionStyle")}
          description={t("settings.chat.compactionStyleDescription")}
        >
          <Select
            value={compactionStyleValue(contextPolicy, compactionHeadroomRatio)}
            onValueChange={(value) => {
              const next = parseCompactionStyle(value);
              setContextPolicy(next.contextPolicy);
              setCompactionHeadroomRatio(next.compactionHeadroomRatio);
            }}
            disabled={!autoCompactEnabled}
          >
            <SelectTrigger
              className="w-64"
              aria-label={t("settings.chat.compactionStyle")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="inherit">
                {t("settings.chat.compactionStyleInherit")}
              </SelectItem>
              <SelectItem value="checkpoint">
                {t("settings.chat.compactionStyleCheckpoint")}
              </SelectItem>
              <SelectItem value="rolling:0.25">
                {t("settings.chat.compactionStyleRollingDefault")}
              </SelectItem>
              <SelectItem value="rolling:0.1">
                {t("settings.chat.compactionStyleRolling10")}
              </SelectItem>
              <SelectItem value="rolling:0.05">
                {t("settings.chat.compactionStyleRolling5")}
              </SelectItem>
              <SelectItem value="rolling:0">
                {t("settings.chat.compactionStyleRollingNone")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.pastedTextThreshold")}
          description={t("settings.chat.pastedTextThresholdDescription", {
            shortcut: plainPasteLabel,
          })}
        >
          <Select
            value={String(pastedTextMinChars)}
            onValueChange={(value) => setPastedTextMinChars(Number(value))}
          >
            <SelectTrigger
              className="w-36"
              aria-label={t("settings.chat.pastedTextThreshold")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {PASTED_TEXT_THRESHOLD_CHOICES.map((choice) => (
                <SelectItem key={choice} value={String(choice)}>
                  {choice === 0
                    ? t("settings.chat.pastedTextThresholdOff")
                    : choice.toLocaleString()}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("settings.profile.greetingSloth")}
          description={t("settings.profile.greetingSlothDescription")}
        >
          <Switch
            id="profile-greeting-sloth"
            checked={showGreetingSloth}
            onCheckedChange={setShowGreetingSloth}
          />
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

      <SettingsSection title={t("settings.chat.webSearch.title")}>
        <SettingsRow
          label={t("settings.chat.webSearch.images")}
          description={t("settings.chat.webSearch.imagesDescription")}
        >
          <Switch checked={searchImages} onCheckedChange={setSearchImages} />
        </SettingsRow>
      </SettingsSection>
    </div>
  );
}
