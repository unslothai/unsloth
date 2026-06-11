// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { ChatPage } from "./chat-page";
export {
  getInferenceStatus,
  listGgufVariants,
  listLocalModels,
  loadModel,
  type LocalModelInfo,
} from "./api/chat-api";
export type { GgufVariantDetail } from "./types/api";
export {
  ChatSettingsPanel,
  defaultInferenceParams,
  type InferenceParams,
  type Preset,
} from "./chat-settings-sheet";
export { useChatRuntimeStore } from "./stores/chat-runtime-store";
export { useChatSearchStore } from "./stores/chat-search-store";
export { useChatModelRuntime } from "./hooks/use-chat-model-runtime";
export { isExternalModelId } from "./external-providers";
export { ChatSearchDialog } from "./components/chat-search-dialog";
export { setTrainingCompareHandoff } from "./lib/training-compare-handoff";
export type { ProjectRecord } from "./types";
export { clearAllChats, countAllChats } from "./utils/clear-all-chats";
export { ArtifactCard } from "./artifacts/artifact-card";
export {
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "./artifacts/store";
export { downloadChatExport } from "./utils/export-chat-history";
export {
  EXPORT_FORMATS_LIST,
  bulkExportConversationsByScope,
  importConversationsFromFile,
} from "./prompt-storage/prompt-storage-dialog";
export {
  deleteChatItem,
  renameChatItem,
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
export {
  AttachmentChipBody,
  AttachmentChipButton,
  AttachmentChipProgress,
  AttachmentChipRemoveButton,
  AttachmentChipTitle,
  attachmentChipTokens,
} from "./components/attachment-chip-primitives";
export { DocumentStack } from "./components/document-stack";
export { DocumentPreviewSheet } from "./components/document-preview-panel";
export {
  isDocumentAttachment,
  type DocumentPendingAttachment,
  type ExtractedDocument,
  type PendingDocumentAttachment,
} from "./types";
export {
  documentFigureImageDataUrl,
  formatDocumentTokens,
} from "./utils/document-extraction";
