// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  clearThreadDocuments as apiClearThreadDocuments,
  createKnowledgeBase,
  type CreateKnowledgeBaseRequest,
  deleteDocument as apiDeleteDocument,
  deleteKnowledgeBase as apiDeleteKB,
  getRagDefaults as apiGetRagDefaults,
  getThreadRagSettings as apiGetThreadSettings,
  type JobEvent,
  type KnowledgeBase,
  listKBDocuments,
  listKnowledgeBases,
  listThreadDocuments,
  listThreadIndexes,
  type RagDefaults,
  type RagDocument,
  type ReingestKBOptions,
  reingestKnowledgeBase as apiReingestKB,
  reingestThreadDocuments as apiReingestThread,
  setRagDefaults as apiSetRagDefaults,
  setThreadRagSettings as apiSetThreadSettings,
  subscribeToJobEvents,
  type ThreadIndexSummary,
  type ThreadRagSettings,
  type UpdateRagDefaultsRequest,
  type UpdateThreadRagSettingsRequest,
  uploadKBDocument,
  uploadThreadDocument,
} from "../api/rag-api";

interface RagStoreState {
  knowledgeBases: KnowledgeBase[];
  kbsLoading: boolean;
  kbsError: string | null;

  documentsByScope: Record<string, RagDocument[]>;
  docsLoading: Record<string, boolean>;
  docsError: Record<string, string | null>;

  jobs: Record<string, JobEvent>;
  jobUnsubscribers: Record<string, () => void>;

  threadIndexes: ThreadIndexSummary[];
  threadIndexesLoading: boolean;

  loadKnowledgeBases: () => Promise<void>;
  createKB: (req: CreateKnowledgeBaseRequest) => Promise<KnowledgeBase>;
  deleteKB: (kbId: string) => Promise<void>;

  loadKBDocuments: (kbId: string) => Promise<void>;
  loadThreadDocuments: (threadId: string) => Promise<void>;
  uploadDocument: (
    scope: { kind: "kb"; kbId: string } | { kind: "thread"; threadId: string },
    file: File,
  ) => Promise<{ documentId: string; jobId: string }>;
  deleteDocument: (documentId: string, scopeKey: string) => Promise<void>;

  loadThreadIndexes: () => Promise<void>;
  clearThreadIndex: (threadId: string) => Promise<void>;

  reingestKB: (kbId: string, opts?: ReingestKBOptions) => Promise<string[]>;
  reingestThread: (
    threadId: string,
    opts?: UpdateThreadRagSettingsRequest,
  ) => Promise<string[]>;

  defaults: RagDefaults | null;
  loadDefaults: () => Promise<void>;
  updateDefaults: (patch: UpdateRagDefaultsRequest) => Promise<void>;

  threadSettings: Record<string, ThreadRagSettings>;
  loadThreadSettings: (threadId: string) => Promise<void>;
  updateThreadSettings: (
    threadId: string,
    patch: UpdateThreadRagSettingsRequest,
  ) => Promise<ThreadRagSettings>;

  subscribeJob: (jobId: string, onComplete?: () => void) => void;
}

function kbScopeKey(kbId: string): string {
  return `kb:${kbId}`;
}
function threadScopeKey(threadId: string): string {
  return `thread:${threadId}`;
}

export const useRagStore = create<RagStoreState>((set, get) => ({
  knowledgeBases: [],
  kbsLoading: false,
  kbsError: null,

  documentsByScope: {},
  docsLoading: {},
  docsError: {},

  jobs: {},
  jobUnsubscribers: {},

  threadIndexes: [],
  threadIndexesLoading: false,

  defaults: null,

  threadSettings: {},

  async loadKnowledgeBases() {
    set({ kbsLoading: true, kbsError: null });
    try {
      const kbs = await listKnowledgeBases();
      set({ knowledgeBases: kbs, kbsLoading: false });
    } catch (err) {
      set({
        kbsLoading: false,
        kbsError: err instanceof Error ? err.message : String(err),
      });
    }
  },

  async createKB(req) {
    const kb = await createKnowledgeBase(req);
    set((state) => ({ knowledgeBases: [kb, ...state.knowledgeBases] }));
    return kb;
  },

  async deleteKB(kbId) {
    await apiDeleteKB(kbId);
    set((state) => {
      const scopeKey = kbScopeKey(kbId);
      const { [scopeKey]: _docs, ...restDocs } = state.documentsByScope;
      return {
        knowledgeBases: state.knowledgeBases.filter((k) => k.id !== kbId),
        documentsByScope: restDocs,
      };
    });
  },

  async loadKBDocuments(kbId) {
    const key = kbScopeKey(kbId);
    set((state) => ({
      docsLoading: { ...state.docsLoading, [key]: true },
      docsError: { ...state.docsError, [key]: null },
    }));
    try {
      const docs = await listKBDocuments(kbId);
      set((state) => ({
        documentsByScope: { ...state.documentsByScope, [key]: docs },
        docsLoading: { ...state.docsLoading, [key]: false },
      }));
    } catch (err) {
      set((state) => ({
        docsLoading: { ...state.docsLoading, [key]: false },
        docsError: {
          ...state.docsError,
          [key]: err instanceof Error ? err.message : String(err),
        },
      }));
    }
  },

  async loadThreadDocuments(threadId) {
    const key = threadScopeKey(threadId);
    set((state) => ({
      docsLoading: { ...state.docsLoading, [key]: true },
      docsError: { ...state.docsError, [key]: null },
    }));
    try {
      const docs = await listThreadDocuments(threadId);
      set((state) => ({
        documentsByScope: { ...state.documentsByScope, [key]: docs },
        docsLoading: { ...state.docsLoading, [key]: false },
      }));
    } catch (err) {
      set((state) => ({
        docsLoading: { ...state.docsLoading, [key]: false },
        docsError: {
          ...state.docsError,
          [key]: err instanceof Error ? err.message : String(err),
        },
      }));
    }
  },

  async uploadDocument(scope, file) {
    const result =
      scope.kind === "kb"
        ? await uploadKBDocument(scope.kbId, file)
        : await uploadThreadDocument(scope.threadId, file);
    const scopeKey =
      scope.kind === "kb"
        ? kbScopeKey(scope.kbId)
        : threadScopeKey(scope.threadId);
    // Refresh the doc list so the new pending row appears.
    if (scope.kind === "kb") {
      void get().loadKBDocuments(scope.kbId);
    } else {
      void get().loadThreadDocuments(scope.threadId);
    }
    // Subscribe to the job; refresh docs on completion to pick up
    // the final num_chunks / status.
    get().subscribeJob(result.job_id, () => {
      if (scope.kind === "kb") {
        void get().loadKBDocuments(scope.kbId);
      } else {
        void get().loadThreadDocuments(scope.threadId);
      }
    });
    return { documentId: result.document_id, jobId: result.job_id, scopeKey } as {
      documentId: string;
      jobId: string;
    };
  },

  async deleteDocument(documentId, scopeKey) {
    await apiDeleteDocument(documentId);
    set((state) => {
      const current = state.documentsByScope[scopeKey] ?? [];
      return {
        documentsByScope: {
          ...state.documentsByScope,
          [scopeKey]: current.filter((d) => d.id !== documentId),
        },
      };
    });
  },

  async loadThreadIndexes() {
    set({ threadIndexesLoading: true });
    try {
      const threads = await listThreadIndexes();
      set({ threadIndexes: threads, threadIndexesLoading: false });
    } catch {
      set({ threadIndexesLoading: false });
    }
  },

  async clearThreadIndex(threadId) {
    await apiClearThreadDocuments(threadId);
    const scopeKey = threadScopeKey(threadId);
    set((state) => {
      const { [scopeKey]: _docs, ...restDocs } = state.documentsByScope;
      return {
        documentsByScope: restDocs,
        threadIndexes: state.threadIndexes.filter(
          (t) => t.thread_id !== threadId,
        ),
      };
    });
  },

  async reingestKB(kbId, opts) {
    const response = await apiReingestKB(kbId, opts ?? {});
    // Refresh the KB list so updated chunking_strategy / mode / embedder
    // values flow back into the UI, and the doc list so old chunk
    // counts reset to 0 until each job completes.
    void get().loadKnowledgeBases();
    void get().loadKBDocuments(kbId);
    // Subscribe to every new job so progress chips render and the doc
    // list refreshes on completion (mirrors uploadDocument's pattern).
    for (const jobId of response.job_ids) {
      get().subscribeJob(jobId, () => {
        void get().loadKBDocuments(kbId);
      });
    }
    return response.job_ids;
  },

  async loadDefaults() {
    try {
      const defaults = await apiGetRagDefaults();
      set({ defaults });
    } catch {
      // Defaults endpoint is best-effort; a 401 / network blip just
      // leaves defaults null and dialogs fall back to hard-coded
      // ('standard', 'text', no embedder override).
    }
  },

  async updateDefaults(patch) {
    const defaults = await apiSetRagDefaults(patch);
    set({ defaults });
  },

  async reingestThread(threadId, opts) {
    const response = await apiReingestThread(threadId, opts ?? {});
    void get().loadThreadDocuments(threadId);
    void get().loadThreadIndexes();
    if (opts) {
      // The reingest endpoint persists the new settings as a side
      // effect; refresh the local copy so the UI reflects them.
      void get().loadThreadSettings(threadId);
    }
    for (const jobId of response.job_ids) {
      get().subscribeJob(jobId, () => {
        void get().loadThreadDocuments(threadId);
      });
    }
    return response.job_ids;
  },

  async loadThreadSettings(threadId) {
    try {
      const settings = await apiGetThreadSettings(threadId);
      set((state) => ({
        threadSettings: { ...state.threadSettings, [threadId]: settings },
      }));
    } catch {
      // Best-effort — falls back to defaults UI-side when missing.
    }
  },

  async updateThreadSettings(threadId, patch) {
    const settings = await apiSetThreadSettings(threadId, patch);
    set((state) => ({
      threadSettings: { ...state.threadSettings, [threadId]: settings },
    }));
    return settings;
  },

  subscribeJob(jobId, onComplete) {
    // Object indexing in TS returns V (not V|undefined) without
    // noUncheckedIndexedAccess, so we test membership explicitly to
    // avoid the "always-truthy function reference" lint.
    if (jobId in get().jobUnsubscribers) return;
    const unsubscribe = subscribeToJobEvents(jobId, {
      onEvent: (event) => {
        set((state) => ({ jobs: { ...state.jobs, [jobId]: event } }));
        if (event.type === "complete" || event.type === "error") {
          onComplete?.();
        }
      },
      onClose: () => {
        set((state) => {
          const { [jobId]: _gone, ...rest } = state.jobUnsubscribers;
          return { jobUnsubscribers: rest };
        });
      },
    });
    set((state) => ({
      jobUnsubscribers: { ...state.jobUnsubscribers, [jobId]: unsubscribe },
    }));
  },
}));

export { kbScopeKey, threadScopeKey };
