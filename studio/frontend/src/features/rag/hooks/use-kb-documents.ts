// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import type { RagDocument } from "../api/rag-api";
import { kbScopeKey, threadScopeKey, useRagStore } from "../stores/rag-store";

// Module-scope sentinel so the selector returns a stable reference
// when the scope key isn't populated yet. A `[]` literal in the
// selector returns a new array on every call → Zustand's Object.is
// snapshot check flags it as changed → re-render → selector reruns
// → new `[]` → infinite loop → React error #185.
const EMPTY_DOCS: RagDocument[] = [];

export function useKBDocuments(kbId: string | null) {
  const scopeKey = kbId ? kbScopeKey(kbId) : "";
  const documents = useRagStore((s) =>
    scopeKey ? (s.documentsByScope[scopeKey] ?? EMPTY_DOCS) : EMPTY_DOCS,
  );
  const loading = useRagStore((s) => (scopeKey ? !!s.docsLoading[scopeKey] : false));
  const error = useRagStore((s) =>
    scopeKey ? (s.docsError[scopeKey] ?? null) : null,
  );
  const loadKBDocuments = useRagStore((s) => s.loadKBDocuments);
  const uploadDocument = useRagStore((s) => s.uploadDocument);
  const deleteDocument = useRagStore((s) => s.deleteDocument);

  useEffect(() => {
    if (kbId) void loadKBDocuments(kbId);
  }, [kbId, loadKBDocuments]);

  return {
    documents,
    loading,
    error,
    refresh: () => (kbId ? loadKBDocuments(kbId) : Promise.resolve()),
    upload: (file: File) =>
      kbId
        ? uploadDocument({ kind: "kb", kbId }, file)
        : Promise.reject(new Error("no KB selected")),
    remove: (documentId: string) =>
      scopeKey ? deleteDocument(documentId, scopeKey) : Promise.resolve(),
  };
}

export function useThreadDocuments(threadId: string | null) {
  const scopeKey = threadId ? threadScopeKey(threadId) : "";
  const documents = useRagStore((s) =>
    scopeKey ? (s.documentsByScope[scopeKey] ?? EMPTY_DOCS) : EMPTY_DOCS,
  );
  const loading = useRagStore((s) => (scopeKey ? !!s.docsLoading[scopeKey] : false));
  const error = useRagStore((s) =>
    scopeKey ? (s.docsError[scopeKey] ?? null) : null,
  );
  const loadThreadDocuments = useRagStore((s) => s.loadThreadDocuments);
  const uploadDocument = useRagStore((s) => s.uploadDocument);
  const deleteDocument = useRagStore((s) => s.deleteDocument);

  useEffect(() => {
    if (threadId) void loadThreadDocuments(threadId);
  }, [threadId, loadThreadDocuments]);

  return {
    documents,
    loading,
    error,
    refresh: () =>
      threadId ? loadThreadDocuments(threadId) : Promise.resolve(),
    upload: (file: File) =>
      threadId
        ? uploadDocument({ kind: "thread", threadId }, file)
        : Promise.reject(new Error("no thread selected")),
    remove: (documentId: string) =>
      scopeKey ? deleteDocument(documentId, scopeKey) : Promise.resolve(),
  };
}
