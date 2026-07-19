// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { ChatPage, validateChatSearch, type ChatSearch } from "./chat-page";
export {
  addScanFolder,
  browseFolders,
  deleteCachedModel,
  deleteFineTunedModel,
  fetchGgufStagedMetadata,
  getInferenceStatus,
  listGgufVariants,
  listLocalModels,
  listRecommendedFolders,
  listScanFolders,
  loadModel,
  removeScanFolder,
  type BrowseFoldersResponse,
  type CachedGgufRepo,
  type CachedModelRepo,
  type LocalModelInfo,
  type ScanFolderInfo,
} from "./api/chat-api";
export type { GgufVariantDetail } from "./types/api";
export {
  ChatSettingsPanel,
  defaultInferenceParams,
  type InferenceParams,
  type Preset,
} from "./chat-settings-sheet";
export { useChatRuntimeStore } from "./stores/chat-runtime-store";
export {
  normalizeSpeculativeType,
  readPersistedSpeculativeType,
  readPersistedGpuMemoryMode,
  reconcilePersistedGpuIds,
  GPU_LAYERS_AUTO,
} from "./stores/chat-runtime-store";
export {
  preferFullToolOutput,
  toolOutputKey,
  useToolPaneScope,
} from "./tool-output-scope";
export { PermissionModeDropdown } from "./permission-mode-select";
export { useChatSearchStore } from "./stores/chat-search-store";
export { usePinnedChatsStore } from "./stores/pinned-chats-store";
export { useChatPreferencesStore } from "./stores/chat-preferences-store";
export {
  PLUS_MENU_ORDER,
  usePlusMenuPrefsStore,
  type PlusMenuItemId,
} from "./stores/plus-menu-prefs-store";
export {
  useChatModelRuntime,
  resyncInferenceStatusAfterServerModelChange,
} from "./hooks/use-chat-model-runtime";
export {
  customProviderDisplayName,
  isCustomProviderType,
  isExternalModelId,
  parseExternalModelId,
} from "./external-providers";
export { ApiProviderLogo } from "./api-provider-logo";
export { useExternalProvidersStore } from "./stores/external-providers-store";
export { ChatSearchDialog } from "./components/chat-search-dialog";
export { setTrainingCompareHandoff } from "./lib/training-compare-handoff";
export type { ProjectRecord } from "./types";
export { clearAllChats, countAllChats } from "./utils/clear-all-chats";
export { listStoredChatThreads } from "./utils/chat-history-storage";
export { ArtifactCard } from "./artifacts/artifact-card";
export {
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "./artifacts/store";
export { downloadChatExport } from "./utils/export-chat-history";
export {
  clearNewChatDraft,
  composerDraftKey,
  readComposerDraft,
  writeComposerDraft,
} from "./utils/composer-draft";
export {
  EXPORT_FORMATS_LIST,
  bulkExportConversationsByScope,
  importConversationsFromFile,
} from "./prompt-storage/prompt-storage-dialog";
export {
  archiveChatItem,
  deleteChatItem,
  renameChatItem,
  unarchiveChatItem,
  useChatSidebarItems,
  type SidebarItem,
} from "./hooks/use-chat-sidebar-items";
export {
  createChatProject,
  deleteChatProject,
  moveChatItemToProject,
  renameChatProject,
  updateChatProjectInstructions,
  useChatProjects,
} from "./hooks/use-chat-projects";
