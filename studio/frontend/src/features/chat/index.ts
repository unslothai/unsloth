// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { ChatPage, validateChatSearch, type ChatSearch } from "./chat-page";
export {
  addScanFolder,
  browseFolders,
  deleteChatAttachment,
  deleteFineTunedModel,
  fetchChatAttachmentBlob,
  fetchGgufStagedMetadata,
  getCachedModelPath,
  getInferenceStatus,
  listChatAttachments,
  listGgufVariants,
  listLocalModels,
  listRecommendedFolders,
  listScanFolders,
  loadModel,
  notifyChatHistoryUpdated,
  removeScanFolder,
  revealCachedModel,
  type BrowseFoldersResponse,
  type CachedGgufRepo,
  type CachedModelRepo,
  type ChatAttachmentPage,
  type ChatAttachmentRecord,
  type LocalModelInfo,
  type ScanFolderInfo,
} from "./api/chat-api";
export type { GgufVariantDetail } from "./types/api";
export {
  ChatSettingsPanel,
  ParamSlider,
  defaultInferenceParams,
  type InferenceParams,
  type Preset,
} from "./chat-settings-sheet";
export { useChatRuntimeStore } from "./stores/chat-runtime-store";
export {
  CHAT_RAG_CAPTION_KEY,
  CHAT_RAG_OCR_KEY,
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
export { usePinnedProjectsStore } from "./stores/pinned-projects-store";
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
export { emitChatAttachmentDeleted } from "./utils/chat-attachment-events";
export { ArtifactCard } from "./artifacts/artifact-card";
export {
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "./artifacts/store";
export {
  downloadArchivedChatExport,
  downloadChatExport,
} from "./utils/export-chat-history";
export {
  clearNewChatDraft,
  composerDraftKey,
  readComposerDraft,
  writeComposerDraft,
} from "./utils/composer-draft";
export {
  EXPORT_FORMATS_LIST,
  buildFineTuneJsonl,
  bulkExportConversationsByScope,
  exportFineTuneJsonl,
  importConversationsFromFile,
  type FineTuneFormat,
} from "./prompt-storage/prompt-storage-dialog";
export {
  archiveAllChatItems,
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
export { subscribeDictationLevel } from "./adapters/dictation-level";
export {
  StudioDictationAdapter,
  cancelActiveStudioDictation,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "./adapters/studio-dictation-adapter";
export {
  StudioModelDictationAdapter,
  fetchSttStatus,
  loadSttModel,
  startSttDownload,
  unloadSttModel,
  validateSttModel,
  type SttDownloadStatus,
} from "./adapters/studio-model-dictation-adapter";
export {
  StudioSpeechSynthesisAdapter,
  createConfiguredUtterance,
  curateSystemVoices,
  generateStudioTtsAudio,
} from "./adapters/studio-speech-synthesis-adapter";
