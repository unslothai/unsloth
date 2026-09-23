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
  DEFAULT_THINKING_VISIBILITY,
  DEFAULT_TOOL_VISIBILITY,
  type PlusMenuItemId,
  normaliseDisplayVisibility,
  refreshModelDisclaimerPreference,
  saveModelDisclaimerPreference,
  useChatPreferencesStore,
  useChatRuntimeStore,
  usePlusMenuPrefsStore,
  useSidebarOrganizationStore,
} from "@/features/chat";
import { PASTED_TEXT_THRESHOLD_CHOICES } from "@/features/chat/utils/pasted-text";
import { refreshContextUsage } from "@/features/chat/utils/refresh-context-usage";
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
  Scroll01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Columns2Icon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  type CurrentDatePromptSettings,
  loadCurrentDatePrompt,
  updateCurrentDatePrompt,
} from "../api/current-date-prompt";
import { SettingsRow } from "../components/settings-row";
import { ComposerSettings } from "../components/composer-settings";
import { SettingsSection } from "../components/settings-section";
import { useSettingsDialogStore } from "../stores/settings-dialog-store";

// Adjustable "+" menu items shown in settings, in display order. Icons mirror
// the ones used in the composer + menu itself.
const PLUS_MENU_ICON_CLASS = "size-[calc(18px*var(--ui-space-scale,1))]";
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
    id: "skills",
    labelKey: "settings.chat.menu.skills",
    icon: (
      <HugeiconsIcon
        icon={Scroll01Icon}
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
  const setSearchImages = useChatRuntimeStore((state) => state.setSearchImages);
  const networkAccessRowRef = useRef<HTMLDivElement | null>(null);
  const scrollTarget = useSettingsDialogStore((s) => s.scrollTarget);
  const consumeScrollTarget = useSettingsDialogStore(
    (s) => s.consumeScrollTarget,
  );
  useEffect(() => {
    if (scrollTarget !== "chat-canvas-network") return;
    const frame = window.requestAnimationFrame(() => {
      networkAccessRowRef.current?.scrollIntoView({
        block: "center",
        behavior: "smooth",
      });
      consumeScrollTarget("chat-canvas-network");
    });
    return () => window.cancelAnimationFrame(frame);
  }, [consumeScrollTarget, scrollTarget]);
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
  const thinkingVisibility = useChatPreferencesStore(
    (state) => state.thinkingVisibility,
  );
  const setThinkingVisibility = useChatPreferencesStore(
    (state) => state.setThinkingVisibility,
  );
  const [currentDatePrompt, setCurrentDatePrompt] =
    useState<CurrentDatePromptSettings | null>(null);
  const [currentDatePromptError, setCurrentDatePromptError] = useState<
    string | null
  >(null);
  const [isSavingCurrentDatePrompt, setIsSavingCurrentDatePrompt] =
    useState(false);
  const toolVisibility = useChatPreferencesStore(
    (state) => state.toolVisibility,
  );
  const setToolVisibility = useChatPreferencesStore(
    (state) => state.setToolVisibility,
  );
  const foldToolActivityIntoThinking = useChatPreferencesStore(
    (state) => state.foldToolActivityIntoThinking,
  );
  const setFoldToolActivityIntoThinking = useChatPreferencesStore(
    (state) => state.setFoldToolActivityIntoThinking,
  );
  // Cannot coexist with always expanded. The stored preference is left alone so it comes back.
  const foldBlockedByAlwaysExpanded = toolVisibility === "expanded";
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

  useEffect(() => {
    let cancelled = false;
    void loadCurrentDatePrompt(t("settings.chat.currentDate.loadError"))
      .then((settings) => {
        if (cancelled) return;
        setCurrentDatePrompt(settings);
        setCurrentDatePromptError(null);
      })
      .catch((error) => {
        if (cancelled) return;
        setCurrentDatePromptError(
          error instanceof Error
            ? error.message
            : t("settings.chat.currentDate.loadError"),
        );
      });
    return () => {
      cancelled = true;
    };
  }, [t]);

  const saveCurrentDatePrompt = async (enabled: boolean) => {
    setIsSavingCurrentDatePrompt(true);
    setCurrentDatePromptError(null);
    try {
      const settings = await updateCurrentDatePrompt(
        enabled,
        t("settings.chat.currentDate.saveError"),
      );
      setCurrentDatePrompt(settings);
      void refreshContextUsage({ invalidate: true });
    } catch (error) {
      setCurrentDatePromptError(
        error instanceof Error
          ? error.message
          : t("settings.chat.currentDate.saveError"),
      );
    } finally {
      setIsSavingCurrentDatePrompt(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold font-heading">
          {t("settings.chat.title")}
        </h1>
      </header>

      <SettingsSection title={t("settings.general.chatDefaults")}>
        <ComposerSettings embedded={true} />
        <SettingsRow
          label={t("settings.chat.pastedTextThreshold")}
          description={
            pastedTextMinChars > 0
              ? t("settings.chat.pastedTextShortDescription", {
                  count: pastedTextMinChars.toLocaleString(),
                })
              : t("settings.chat.pastedTextOffDescription")
          }
          hint={t("settings.chat.pastedTextThresholdDescription", {
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
          label={t("settings.chat.autoCompact")}
          description={t("settings.chat.autoCompactDescription")}
          hint={t("settings.chat.autoCompactHint")}
        >
          <Switch
            aria-label={t("settings.chat.autoCompact")}
            checked={autoCompactEnabled}
            onCheckedChange={setAutoCompactEnabled}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.chat.groups.conversations.title")}>
        <SettingsRow
          label={t("settings.chat.rememberParamsPerModel")}
          description={t("settings.chat.rememberParamsPerModelDescription")}
          hint={t("settings.chat.rememberParamsPerModelHint")}
        >
          <Switch
            aria-label={t("settings.chat.rememberParamsPerModel")}
            checked={rememberParamsPerModel}
            onCheckedChange={setRememberParamsPerModel}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.currentDate.label")}
          description={t("settings.chat.currentDate.description")}
        >
          <div className="flex flex-col items-end gap-1">
            <Switch
              aria-label={t("settings.chat.currentDate.label")}
              checked={currentDatePrompt?.enabled ?? false}
              disabled={!currentDatePrompt || isSavingCurrentDatePrompt}
              onCheckedChange={(enabled) => void saveCurrentDatePrompt(enabled)}
            />
            {currentDatePromptError ? (
              <span
                role="alert"
                className="max-w-[calc(260px*var(--ui-space-scale,1))] text-right text-xs text-destructive"
              >
                {currentDatePromptError}
              </span>
            ) : null}
          </div>
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.projectAttachments")}
          description={t("settings.chat.projectAttachmentsDescription")}
          hint={t("settings.chat.projectAttachmentsHint")}
        >
          <Switch
            aria-label={t("settings.chat.projectAttachments")}
            checked={projectAttachmentTarget === "project"}
            onCheckedChange={(checked) =>
              setProjectAttachmentTarget(checked ? "project" : "thread")
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.general.autoTitleNewChats")}
          description={t("settings.general.autoTitleNewChatsDescription")}
        >
          <Switch
            aria-label={t("settings.general.autoTitleNewChats")}
            checked={autoTitle}
            onCheckedChange={setAutoTitle}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.chat.groups.display.title")}>
        <SettingsRow
          label={t("settings.chat.webSearch.images")}
          description={t("settings.chat.webSearch.imagesDescription")}
        >
          <Switch checked={searchImages} onCheckedChange={setSearchImages} />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.thinking.visibility")}
          description={t("settings.chat.thinking.visibilityDescription")}
        >
          <Select
            value={thinkingVisibility}
            onValueChange={(value) =>
              setThinkingVisibility(
                normaliseDisplayVisibility(value, DEFAULT_THINKING_VISIBILITY),
              )
            }
          >
            <SelectTrigger
              className="w-64"
              aria-label={t("settings.chat.thinking.visibility")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="collapsed">
                {t("settings.chat.visibility.collapsed")}
              </SelectItem>
              <SelectItem value="auto">
                {t("settings.chat.visibility.auto")}
              </SelectItem>
              <SelectItem value="expanded">
                {t("settings.chat.visibility.expanded")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.tools.visibility")}
          description={t("settings.chat.tools.visibilityDescription")}
        >
          <Select
            value={toolVisibility}
            onValueChange={(value) =>
              setToolVisibility(
                normaliseDisplayVisibility(value, DEFAULT_TOOL_VISIBILITY),
              )
            }
          >
            <SelectTrigger
              className="w-64"
              aria-label={t("settings.chat.tools.visibility")}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="collapsed">
                {t("settings.chat.visibility.collapsed")}
              </SelectItem>
              <SelectItem value="auto">
                {t("settings.chat.visibility.auto")}
              </SelectItem>
              <SelectItem value="expanded">
                {t("settings.chat.visibility.expanded")}
              </SelectItem>
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.tools.foldIntoThinking")}
          // Says why the row is off rather than letting a checked switch do nothing.
          description={t(
            foldBlockedByAlwaysExpanded
              ? "settings.chat.tools.foldIntoThinkingBlocked"
              : "settings.chat.tools.foldIntoThinkingDescription",
          )}
        >
          <Switch
            aria-label={t("settings.chat.tools.foldIntoThinking")}
            checked={foldToolActivityIntoThinking && !foldBlockedByAlwaysExpanded}
            disabled={foldBlockedByAlwaysExpanded}
            onCheckedChange={setFoldToolActivityIntoThinking}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.showResponseModel")}
          description={t("settings.chat.showResponseModelDescription")}
        >
          <Switch
            aria-label={t("settings.chat.showResponseModel")}
            checked={showResponseModel}
            onCheckedChange={setShowResponseModel}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.projectsSection")}
          description={t("settings.chat.projectsSectionDescription")}
        >
          <Switch
            aria-label={t("settings.chat.projectsSection")}
            checked={organizeBy === "project"}
            onCheckedChange={(checked) =>
              setOrganizeBy(checked ? "project" : "list")
            }
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.chat.modelDisclaimer")}
          description={t("settings.chat.modelDisclaimerDescription")}
        >
          <Switch
            aria-label={t("settings.chat.modelDisclaimer")}
            checked={showModelDisclaimer}
            onCheckedChange={(checked) => {
              return saveModelDisclaimerPreference(checked).catch(() => {
                toast.error("Could not save the model disclaimer setting.");
              });
            }}
          />
        </SettingsRow>
        <SettingsRow
          label={t("settings.profile.greetingSloth")}
          description={t("settings.profile.greetingSlothDescription")}
        >
          <Switch
            aria-label={t("settings.profile.greetingSloth")}
            id="profile-greeting-sloth"
            checked={showGreetingSloth}
            onCheckedChange={setShowGreetingSloth}
          />
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.chat.modelSelection.title")}>
        <SettingsRow
          label={t("settings.chat.modelSelection.showMemoryBar")}
          description={t(
            "settings.chat.modelSelection.showMemoryBarDescription",
          )}
        >
          <Switch checked={showMemoryBar} onCheckedChange={setShowMemoryBar} />
        </SettingsRow>
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
      </SettingsSection>

      <SettingsSection title={t("settings.chat.artifacts.title")}>
        <div ref={networkAccessRowRef}>
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
        </div>
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
      </SettingsSection>

      <SettingsSection title={t("settings.chat.groups.menu.title")}>
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
    </div>
  );
}
