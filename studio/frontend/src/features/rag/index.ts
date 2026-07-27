// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export { KnowledgeBaseComposerButton } from "./components/knowledge-base-composer-button";
export { KnowledgeBaseDialog } from "./components/knowledge-base-dialog";
export { RetrievalSettingsSection } from "./components/retrieval-settings-section";
export { ThreadDocumentsBar } from "./components/thread-documents-bar";
export {
  deleteDocument,
  getDocumentFileUrl,
  listAllDocuments,
} from "./api/rag-api";
export type { KnowledgeBase, RagDocument, UploadedDocument } from "./types/rag";
