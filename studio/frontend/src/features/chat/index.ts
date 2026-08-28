// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { ChatPage, validateChatSearch, type ChatSearch } from "./chat-page";
export { NewProjectDialog } from "./components/new-project-dialog";
export {
  addScanFolder,
  browseFolders,
  ChatThreadDeletedError,
  deleteChatAttachment,
  deleteFineTunedModel,
  fetchChatAttachmentBlob,
  fetchGgufStagedMetadata,
  getCachedModelPath,
  getInferenceStatus,
  listCachedGguf,
  listChatAttachments,
  listGgufVariants,
  listLocalModels,
  estimateKvCache,
  listLoras,
  listModels,
  listRecommendedFolders,
  listScanFolders,
  loadModel,
  unloadModel,
  notifyChatHistoryUpdated,
  removeScanFolder,
  revealCachedModel,
  type BrowseFoldersResponse,
  type CachedGgufRepo,
  type CachedModelRepo,
  type ChatAttachmentPage,
  type ChatAttachmentRecord,
  type KvCacheEstimate,
  type LocalModelInfo,
  type ScanFolderInfo,
} from "./api/chat-api";
export type {
  ApiMonitorEntry,
  BackendModelDetails,
  GgufVariantDetail,
  InferenceStatusResponse,
} from "./types/api";
export {
  applyActiveModelStatusToStore,
  resolveInferenceCheckpointId,
} from "./lib/apply-inference-status-to-store";
export { isSpeechOnlyStatus } from "./lib/speech-only-status";
export {
  ChatSettingsPanel,
  ParamSlider,
  defaultInferenceParams,
  type InferenceParams,
  type Preset,
} from "./chat-settings-sheet";
export { useChatRuntimeStore } from "./stores/chat-runtime-store";
export {
  hydrateModelDisclaimerPreference,
  refreshModelDisclaimerPreference,
  saveModelDisclaimerPreference,
} from "./sync-model-disclaimer-preference";
export { useChatActive, useInComparePane } from "./runtime-provider";
export {
  CHAT_RAG_CAPTION_KEY,
  CHAT_RAG_OCR_KEY,
  normalizeSpeculativeType,
  readPersistedSpeculativeType,
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
  readPersistedGpuMemoryMode,
  reconcilePersistedGpuIds,
  reconcilePersistedGpuSelection,
  GPU_LAYERS_AUTO,
} from "./stores/chat-runtime-store";
export { resolveStagedDiffusionClassification } from "./lib/gpu-placement";
export {
  preferFullToolOutput,
  preferSanitizedFullToolOutput,
  toolOutputKey,
  toolThreadScope,
  useToolOutputFor,
  useUnresolvedToolPaneScope,
  useToolPaneScope,
} from "./tool-output-scope";
export { useToolAwaitingApproval } from "./tool-approval";
export { PermissionModeDropdown } from "./permission-mode-select";
export { useChatSearchStore } from "./stores/chat-search-store";
export type { ChatNavigationState } from "./stores/chat-navigation-store";
export {
  adjacentChatItem,
  countUnreadRows,
  nextAttentionChatItem,
  openChatItemById,
  recentChatItemAtSlot,
  useChatNavigationStore,
  visibleChatItems,
} from "./stores/chat-navigation-store";
export { usePinnedChatsStore } from "./stores/pinned-chats-store";
export { usePinnedProjectsStore } from "./stores/pinned-projects-store";
export {
  applyManualOrder,
  dropEdgeFor,
  moveIdBy,
  showsInRecents,
  PINNED_ORDER_SCOPE,
  PROJECT_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  reorderIds,
  SIDEBAR_ORGANIZATION_STORAGE_KEY,
  useSidebarOrganizationStore,
} from "./stores/sidebar-organization-store";
export type {
  SidebarChatSort,
  SidebarOrganizeBy,
} from "./stores/sidebar-organization-store";
export { useChatPreferencesStore } from "./stores/chat-preferences-store";
export {
  usePromptQueueUI,
  type PromptQueueUIEntry,
  type PromptQueueUIItem,
  type PromptQueueUIItemStatus,
  type PromptQueueUIState,
} from "./stores/prompt-queue-ui-store";
export {
  notifyPromptQueueRunFailed,
  PROMPT_QUEUE_RUN_FAILED_EVENT,
  PROMPT_QUEUE_STOP_EVENT,
  requestLocalPromptQueueStop,
  type PromptQueueRunFailedEventDetail,
  type PromptQueueStopEventDetail,
} from "./utils/prompt-queue-boundary";
export {
  adoptPreStreamRunReservation,
  cancelPreStreamRunReservations,
  findPreStreamRunReservation,
  hasPreStreamRunReservation,
  preStreamRunThreadIdsForAdapter,
  preStreamRunThreadIdsForRuntime,
  releasePreStreamRunForThreadIds,
  releasePreStreamRunReservation,
  reservePreStreamRun,
} from "./utils/pre-stream-run-reservation";
export { claimThreadCreation } from "./utils/chat-thread-creation-claim";
export { useChatProjectScope } from "./chat-project-scope";
// Audio swaps the same llama-server Chat decodes on, so it needs the same confirmation.
export {
  confirmStopRunningChatsIfNeeded,
  type StopRunningChatsDecision,
} from "./utils/confirm-stop-running-chats";
export {
  promptQueueActiveItemChanged,
  reorderPromptQueueItems,
} from "./utils/prompt-queue-reorder";
export {
  PROMPT_QUEUE_DRAG_TYPE,
  hasPendingPromptQueueStart,
  isPromptQueueChord,
  isPromptQueueDragTypes,
  pastedTextQueueKey,
} from "./utils/prompt-queue-input";
export {
  localPromptQueueModelBoundary,
  planLocalPromptQueueStop,
  shouldAbortPendingQueueForModelBoundary,
  shouldAbortPendingQueueForSettingsChange,
} from "./utils/prompt-queue-model-boundary";
export { chatHistoryClearBoundary } from "./utils/chat-history-clear-boundary";
export { rangeBetween, toggleSelected } from "./utils/row-selection";
export {
  addQueuedChatRunSettingsThreadIds,
  consumeQueuedChatRunSettings,
  discardQueuedChatRunSettings,
  discardQueuedChatRunSettingsForThread,
  registerQueuedChatRunSettings,
  snapshotQueuedChatRunSettings,
  type QueuedChatRunSettings,
} from "./utils/queued-chat-run-settings";
export {
  PLUS_MENU_ORDER,
  usePlusMenuPrefsStore,
  type PlusMenuItemId,
} from "./stores/plus-menu-prefs-store";
export {
  useChatModelRuntime,
  resyncInferenceStatusAfterServerModelChange,
} from "./hooks/use-chat-model-runtime";
export { compareModelDisplayName } from "./lib/external-model-label";
export { chatModelLoaded } from "./lib/chat-model-loaded";
export type { ChatModelLoadedInput } from "./lib/chat-model-loaded";
export {
  customProviderDisplayName,
  isCustomProviderType,
  isExternalModelId,
  parseExternalModelId,
} from "./external-providers";
export {
  type AttachmentText,
  assertDocumentAttachmentSize,
  attachmentAudioSrc,
  attachmentTextLanguage,
  countAttachmentTextLines,
  isAudioAttachment,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} from "./attachment-content";
export { ApiProviderLogo } from "./api-provider-logo";
export { useExternalProvidersStore } from "./stores/external-providers-store";
export { DeleteChatFilesSwitch } from "./components/delete-chat-files-switch";
export { ChatSearchDialog } from "./components/chat-search-dialog";
export { StopRunningChatsDialog } from "./components/stop-running-chats-dialog";
export { setTrainingCompareHandoff } from "./lib/training-compare-handoff";
export type { ProjectRecord } from "./types";
export { clearAllChats, countAllChats } from "./utils/clear-all-chats";
export { offerToDeleteKeptSandboxes } from "./utils/offer-kept-sandbox-files";
export { pasteClipboardFiles } from "./utils/clipboard-files";
export {
  extractYoutubeVideoId,
  extractYoutubeVideoUrlFromClipboard,
} from "./utils/youtube-url";
export {
  isSearchImagesToolResult,
  searchImagePath,
  stripSearchImageTokens,
  type SearchImageEntry,
} from "./search-images/search-images";
export { YoutubeTranscriptPrompt } from "./components/youtube-transcript-prompt";
export {
  formatMcpToolName,
  mcpServerFromProvenance,
} from "./utils/mcp-tool-name";
export {
  PASTED_TEXT_PREVIEW_MAX_CHARS,
  attachmentContentText,
  attachmentsPastedText,
  createPastedTextFile,
  isPastedTextContent,
  isPastedTextFile,
  isPlainPasteChord,
  pasteLongTextAsFile,
  plainPasteStillCounts,
  pastedTextContentBytes,
  pastedTextContentPreview,
  pastedTextOf,
  pastedTextPreview,
  shouldAttachPastedText,
} from "./utils/pasted-text";
export {
  deleteStoredChatThreads,
  ensureStoredChatThread,
  getStoredChatProject,
  getStoredChatThread,
  isThreadIncognito,
  listStoredChatMessages,
  listStoredChatThreads,
  markThreadIncognito,
} from "./utils/chat-history-storage";
export { allRecordedSandboxSessionIds } from "./utils/recorded-sandbox-session";
export {
  markChatThreadDeleted,
  removeChatThreadTombstones,
} from "./utils/chat-thread-tombstones";
export { emitChatAttachmentDeleted } from "./utils/chat-attachment-events";
export {
  forkCountFor,
  subscribeForkCounts,
} from "./utils/fork-count-store";
export { resolveReasoningGroupDuration } from "./utils/reasoning-duration";
export {
  reasoningAutoOpensWhileStreaming,
  resolveReasoningOpen,
  resolveReasoningToggle,
  startsNewReasoningRound,
} from "./utils/reasoning-visibility";
export { ArtifactCard } from "./artifacts/artifact-card";
export { ResearchMessage } from "./components/research-message";
export {
  ResearchActivityPanel,
  ResearchActivitySheet,
} from "./components/research-activity-panel";
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
  composerPasteDraftKey,
  readComposerDraft,
  readPasteDraft,
  writeComposerDraft,
  writePasteDraft,
} from "./utils/composer-draft";
export {
  CONVERSATION_MARKDOWN_FORMAT,
  CONVERSATION_MARKDOWN_LABEL,
} from "./utils/conversation-markdown";
export {
  COMBINED_EXPORT_FORMATS_LIST,
  EXPORT_FORMATS_LIST,
  buildFineTuneJsonl,
  bulkExportConversationsByScope,
  exportBulkConversationsMerged,
  exportBulkConversationsSeparate,
  exportFineTuneJsonl,
  type ConvExportFormat,
  type FineTuneFormat,
} from "./prompt-storage/prompt-storage-dialog";
export {
  fileImportSource,
  importConversationsFromFile,
  importConversationsFromSource,
  nativeImportSource,
  type ImportProgress,
  type ImportResult,
  type ImportSource,
} from "./utils/chat-import";
export {
  archiveAllChatItems,
  archiveChatItem,
  archiveChatItems,
  deleteChatItem,
  deleteChatItems,
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
  setChatProjectWorkspace,
  updateChatProjectInstructions,
  useChatProjects,
} from "./hooks/use-chat-projects";
export { subscribeDictationLevel } from "./adapters/dictation-level";
export {
  dictationFailed,
  dictationProducedTranscript,
} from "./adapters/dictation-outcome";
export {
  StudioDictationAdapter,
  cancelActiveStudioDictation,
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
} from "./adapters/studio-dictation-adapter";
export {
  StudioModelDictationAdapter,
  SttModelNotDownloadedError,
  cancelSttDownload,
  fetchSttStatus,
  loadSttModel,
  startSttDownload,
  sttEngineFor,
  sttEngineStatusFor,
  unloadSttModel,
  validateSttModel,
  type SttDownloadStatus,
  type SttEngine,
} from "./adapters/studio-model-dictation-adapter";
export {
  StudioSpeechSynthesisAdapter,
  createConfiguredUtterance,
  curateSystemVoices,
  generateCustomTtsAudio,
  generateStudioTtsAudio,
  releaseTtsAudioUrl,
} from "./adapters/studio-speech-synthesis-adapter";
