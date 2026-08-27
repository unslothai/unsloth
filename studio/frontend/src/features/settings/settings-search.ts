// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TranslationKey } from "@/i18n";
import type { SettingsTab } from "./stores/settings-dialog-store";

/**
 * Searchable entries per tab: the label/title keys rendered by each tab.
 * Tab names themselves always match, so tabs without translatable rows
 * (profile, connections) are still reachable from search.
 */
export const SETTINGS_SEARCH_INDEX: Record<SettingsTab, TranslationKey[]> = {
  general: [
    "settings.general.account",
    "settings.general.password",
    "settings.general.huggingFaceToken",
    "settings.appearance.language.title",
    "settings.appearance.language.label",
    "settings.general.notifications.sectionTitle",
    "settings.general.notifications.showLlamaUpdates",
    "settings.general.previewSharing.sectionTitle",
    "settings.general.previewSharing.enableLabel",
    "settings.general.previewSharing.revokeLabel",
    "settings.general.rag.sectionTitle",
    "settings.general.rag.embeddingModel",
    "settings.general.helperLlm.sectionTitle",
    "settings.general.helperLlm.preloadOnStartup",
    "settings.general.downloads.sectionTitle",
    "settings.general.downloads.transport",
    "settings.general.downloads.https",
    "settings.general.downloads.xet",
    "settings.general.downloads.auto",
    "settings.general.uploads.sectionTitle",
    "settings.general.uploads.maxUploadSize",
    "settings.general.resetPreferences.sectionTitle",
    "settings.general.resetPreferences.label",
  ],
  profile: [
    "settings.profile.title",
    "settings.profile.description",
    "settings.profile.displayName",
    "settings.profile.nickname",
    // avatarShape lives inside the avatar edit popover, so it has no
    // always-rendered label for search to scroll to.
    // The stats heading and highlight tiles render for every profile; the
    // insight and training cards are conditional, so they stay out.
    "settings.profile.stats.title",
    "settings.profile.stats.lifetimeTokens",
    "settings.profile.stats.peakTokens",
    "settings.profile.stats.longestChat",
    "settings.profile.stats.currentStreak",
    "settings.profile.stats.longestStreak",
  ],
  appearance: [
    "settings.appearance.theme.label",
    "settings.appearance.palette.label",
    "settings.appearance.custom.colors.accent",
    "settings.appearance.custom.colors.background",
    "settings.appearance.custom.colors.foreground",
    "settings.appearance.custom.uiFont.label",
    "settings.appearance.custom.headingFont.label",
    "settings.appearance.custom.chatFont.label",
    "settings.appearance.custom.codeFont.label",
    "settings.appearance.custom.contrast.label",
    "settings.appearance.custom.pointerCursors.label",
    "settings.appearance.custom.reduceMotion.label",
    "settings.appearance.custom.uiFontSize.label",
    "settings.appearance.custom.codeFontSize.label",
    "settings.appearance.custom.fontSmoothing.label",
    "settings.appearance.layout.compactSidebar",
    "settings.appearance.sidebarNav.title",
    "settings.appearance.sidebarMenu.title",
    "settings.appearance.sidebarMenu.darkModeToggle",
  ],
  resources: [
    "settings.resources.liveMonitor.title",
    "settings.resources.liveMonitor.cpu",
    "settings.resources.liveMonitor.ram",
    "settings.resources.liveMonitor.vram",
    "settings.resources.liveMonitor.disk",
    "settings.resources.gpu.title",
    "settings.resources.modelMemory.title",
    "settings.resources.modelMemory.keepResident",
    "settings.resources.modelMemory.noRamReserve",
    "settings.resources.storage.title",
    "settings.resources.storage.modelsFolder",
    "settings.resources.storage.futureDownloads",
    "settings.resources.storage.systemDisk",
    "settings.resources.environment.title",
    "settings.resources.environment.backend",
    "settings.resources.environment.python",
    "settings.resources.environment.torch",
    "settings.resources.environment.transformers",
    "settings.resources.liveUpdates",
  ],
  chat: [
    "settings.general.chatDefaults",
    "settings.general.autoTitleNewChats",
    "settings.chat.projectAttachments",
    "settings.chat.rememberParamsPerModel",
    "settings.chat.autoCompact",
    "settings.chat.compactionStyle",
    "settings.profile.greetingSloth",
    "settings.chat.thinking.collapseByDefault",
    "settings.chat.tools.collapseByDefault",
    "settings.chat.artifacts.title",
    "settings.chat.artifacts.collapseHtmlBlocks",
    "settings.chat.artifacts.allowNetworkAccess",
    "settings.chat.webSearch.title",
    "settings.chat.webSearch.images",
    "settings.chat.modelDisclaimer",
    "settings.chat.projectsSection",
  ],
  // Chat data management moved to the Data tab; keep these rows findable there.
  data: [
    "settings.data.fineTuneExport",
    "settings.data.archivedChats",
    "settings.data.archiveAllChats",
    "settings.data.confirmBeforeDeleting",
    "settings.data.alwaysDeleteFiles",
    "settings.data.uploadedFiles",
    "settings.chat.exportHistory",
    "settings.chat.exportConversations",
    "settings.chat.importChats",
    "settings.chat.clearAllChats",
  ],
  "api-keys": [
    "settings.apiKeys.title",
    "settings.apiKeys.description",
    "settings.apiKeys.accessTokens",
  ],
  // The two cards label themselves in English in every locale, so keys naming them
  // would never match their own anchor. The header carries both entries instead.
  "remote-lan": ["settings.remoteLan.title", "settings.remoteLan.description"],
  agents: [
    // Every key needs a rendered data-settings-label, or a hit has nothing to scroll to.
    "settings.agents.title",
    "settings.agents.description",
    "settings.agents.intro",
    "settings.agents.agent",
    "settings.agents.model",
    "settings.agents.quantization",
    // subagent.title is deliberately absent: its label only mounts for the agents
    // that support subagents, so a hit would have nothing to scroll to otherwise.
    "settings.agents.options.title",
    "settings.agents.remote.title",
    "settings.agents.passthrough.title",
    "settings.agents.dryRun.title",
  ],
  connections: [],
  voice: [
    "settings.voice.dictation.sectionTitle",
    "settings.voice.dictation.microphoneLabel",
    "settings.voice.dictation.languageLabel",
    "settings.voice.dictionary.sectionTitle",
    "settings.voice.recents.sectionTitle",
    "settings.voice.readAloud.sectionTitle",
    "settings.voice.readAloud.buttonLabel",
    "settings.voice.readAloud.engineLabel",
    "settings.voice.readAloud.voiceLabel",
    "settings.voice.readAloud.speedLabel",
    "settings.voice.readAloud.pitchLabel",
    "settings.voice.readAloud.volumeLabel",
    "settings.voice.readAloud.previewLabel",
  ],
  "keyboard-shortcuts": [
    // The tab title and every action label, so searching "sidebar" or
    // "new chat" from the settings search lands on the row itself.
    "settings.keyboardShortcuts.title",
    "settings.keyboardShortcuts.actions.newChat.label",
    "settings.keyboardShortcuts.actions.newTemporaryChat.label",
    "settings.keyboardShortcuts.actions.archiveChat.label",
    "settings.keyboardShortcuts.actions.newStandaloneChat.label",
    "settings.keyboardShortcuts.actions.markChatUnread.label",
    "settings.keyboardShortcuts.actions.togglePinChat.label",
    "settings.keyboardShortcuts.actions.selectAllChats.label",
    "settings.keyboardShortcuts.actions.clearChatSelection.label",
    "settings.keyboardShortcuts.actions.deleteSelectedChats.label",
    "settings.keyboardShortcuts.actions.nextRecentlyViewedChat.label",
    "settings.keyboardShortcuts.actions.nextChat.label",
    "settings.keyboardShortcuts.actions.nextChatNeedingAttention.label",
    "settings.keyboardShortcuts.actions.previousRecentlyViewedChat.label",
    "settings.keyboardShortcuts.actions.previousChat.label",
    "settings.keyboardShortcuts.actions.goToRecentChat1.label",
    "settings.keyboardShortcuts.actions.goToRecentChat2.label",
    "settings.keyboardShortcuts.actions.goToRecentChat3.label",
    "settings.keyboardShortcuts.actions.goToRecentChat4.label",
    "settings.keyboardShortcuts.actions.goToRecentChat5.label",
    "settings.keyboardShortcuts.actions.goToRecentChat6.label",
    "settings.keyboardShortcuts.actions.switchToChat.label",
    "settings.keyboardShortcuts.actions.switchToProjects.label",
    "settings.keyboardShortcuts.actions.switchToHub.label",
    "settings.keyboardShortcuts.actions.switchToTrain.label",
    "settings.keyboardShortcuts.actions.switchToRecipes.label",
    "settings.keyboardShortcuts.actions.switchToImages.label",
    "settings.keyboardShortcuts.actions.switchToVideo.label",
    "settings.keyboardShortcuts.actions.switchToAudio.label",
    "settings.keyboardShortcuts.actions.switchToExport.label",
    "settings.keyboardShortcuts.actions.toggleApiMonitor.label",
    "settings.keyboardShortcuts.actions.toggleSidebar.label",
    "settings.keyboardShortcuts.actions.openMcpServers.label",
    "settings.keyboardShortcuts.actions.clearAllUnreads.label",
    "settings.keyboardShortcuts.actions.logOut.label",
    "settings.keyboardShortcuts.actions.openSettings.label",
    "settings.keyboardShortcuts.actions.approveToolRequest.label",
    "settings.keyboardShortcuts.actions.declineToolRequest.label",
    "settings.keyboardShortcuts.actions.attachFiles.label",
    "settings.keyboardShortcuts.actions.cycleReasoningEffort.label",
    "settings.keyboardShortcuts.actions.decreaseReasoningEffort.label",
    "settings.keyboardShortcuts.actions.increaseReasoningEffort.label",
    "settings.keyboardShortcuts.actions.openModelPicker.label",
    "settings.keyboardShortcuts.actions.openProjectPicker.label",
    "settings.keyboardShortcuts.actions.startDictation.label",
    "settings.keyboardShortcuts.actions.sendMessage.label",
    "settings.keyboardShortcuts.actions.toggleFastMode.label",
    "settings.keyboardShortcuts.actions.copyChatAsMarkdown.label",
    "settings.keyboardShortcuts.actions.copySessionId.label",
    "settings.keyboardShortcuts.actions.forkChat.label",
    "settings.keyboardShortcuts.actions.searchChats.label",
    "settings.keyboardShortcuts.actions.renameChat.label",
    "settings.keyboardShortcuts.actions.openKeyboardShortcuts.label",
  ],
  debugging: [
    "settings.debugging.logSection",
    "settings.debugging.source",
    "settings.debugging.path",
    "settings.debugging.refreshSection",
    "settings.debugging.mode",
    "settings.debugging.keywords",
  ],
  about: [
    "settings.about.updates",
    "settings.about.releaseNotes",
    "settings.about.documentation",
    "settings.about.help",
    "settings.about.feedback",
    "settings.about.hardware",
    "settings.about.license.sectionTitle",
    "settings.about.dangerZone",
    "settings.about.shutDownStudio",
  ],
};

export function createSettingsSearchIndex({
  desktop,
  closeToTray,
}: {
  desktop: boolean;
  closeToTray: boolean;
}): Record<SettingsTab, TranslationKey[]> {
  if (!desktop) {
    return SETTINGS_SEARCH_INDEX;
  }
  return {
    ...SETTINGS_SEARCH_INDEX,
    general: [
      ...SETTINGS_SEARCH_INDEX.general,
      "settings.about.updates",
      "settings.general.startup.sectionTitle",
      "settings.general.startup.launchAtLogin",
      ...(closeToTray ? (["settings.general.startup.closeToTray"] as const) : []),
    ],
    about: SETTINGS_SEARCH_INDEX.about.filter(
      (key) => key !== "settings.about.updates",
    ),
  };
}

/**
 * Extra terms a row matches on, beyond its own label. The value is a
 * translation key holding space-separated synonyms; it is never rendered.
 * Search matched labels only, so "models folder" or "directory" found nothing.
 */
export const SETTINGS_SEARCH_KEYWORDS: Partial<
  Record<TranslationKey, TranslationKey>
> = {
  "settings.resources.storage.modelsFolder":
    "settings.resources.storage.modelsFolderKeywords",
  // mlock, vram, ulimit and pin are in none of these labels, so search
  // missed the rows the feature is named after.
  "settings.resources.modelMemory.title":
    "settings.resources.modelMemory.modelMemoryKeywords",
  "settings.resources.modelMemory.keepResident":
    "settings.resources.modelMemory.modelMemoryKeywords",
  "settings.resources.modelMemory.noRamReserve":
    "settings.resources.modelMemory.modelMemoryKeywords",
  "settings.chat.autoCompact": "settings.chat.autoCompactKeywords",
  "settings.chat.compactionStyle": "settings.chat.autoCompactKeywords",
};
