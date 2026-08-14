import {
  deleteDatasetDocuments,
  isPlatformApiError,
  listDatasetDocuments,
  parseDatasetDocuments,
  stopDatasetDocuments,
  updateDatasetDocument,
  uploadDatasetDocuments,
  validatePlatformDocumentFile,
  type PlatformDocument,
  type PlatformFileValidationFailure,
} from "@/integrations/platform-backend";
import { listAllKnowledgeBases } from "@/features/rag/api/platform-dataset-adapter";
import type { KnowledgeBase } from "@/features/rag/types/rag";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const POLL_START_MS = 1_500;
const POLL_MAX_MS = 15_000;

export interface DocumentLibraryError {
  kind: "permission" | "timeout" | "request";
  message: string;
}

export function nextDocumentPollDelay(delayMs: number): number {
  return Math.min(POLL_MAX_MS, Math.max(POLL_START_MS, delayMs * 2));
}

function libraryError(error: unknown): DocumentLibraryError {
  if (isPlatformApiError(error)) {
    return {
      kind: error.isPermissionError
        ? "permission"
        : error.isTimeout
          ? "timeout"
          : "request",
      message: error.message,
    };
  }
  return {
    kind: "request",
    message: error instanceof Error ? error.message : String(error),
  };
}

export interface UploadSummary {
  accepted: File[];
  rejected: PlatformFileValidationFailure[];
  partialFailure: string | null;
}

export function useDocumentLibrary(initialDatasetId?: string) {
  const [datasets, setDatasets] = useState<KnowledgeBase[]>([]);
  const [datasetId, setDatasetIdState] = useState(initialDatasetId ?? "");
  const [documents, setDocuments] = useState<PlatformDocument[]>([]);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const [page, setPageState] = useState(1);
  const [pageSize, setPageSizeState] = useState(10);
  const [keywords, setKeywordsState] = useState("");
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [mutating, setMutating] = useState(false);
  const [error, setError] = useState<DocumentLibraryError | null>(null);
  const mountedRef = useRef(true);
  const requestRef = useRef<AbortController | null>(null);
  const mutationRef = useRef<AbortController | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pollDelayRef = useRef(POLL_START_MS);

  const clearPoll = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const refresh = useCallback(
    async (quiet = false) => {
      if (!datasetId) {
        setDocuments([]);
        setTotalDocuments(0);
        return [] as PlatformDocument[];
      }
      requestRef.current?.abort();
      const controller = new AbortController();
      requestRef.current = controller;
      if (!quiet) setLoadingDocuments(true);
      try {
        const documentResult = await listDatasetDocuments(
          datasetId,
          { page, pageSize, keywords },
          controller.signal,
        );
        if (!mountedRef.current || controller.signal.aborted) return [];
        setDocuments(documentResult.items);
        setTotalDocuments(documentResult.total);
        setError(null);
        return documentResult.items;
      } catch (refreshError) {
        if (
          !mountedRef.current ||
          (isPlatformApiError(refreshError) && refreshError.isAbort)
        ) {
          return [];
        }
        setError(libraryError(refreshError));
        return [];
      } finally {
        if (mountedRef.current && !quiet) setLoadingDocuments(false);
        if (requestRef.current === controller) requestRef.current = null;
      }
    },
    [datasetId, keywords, page, pageSize],
  );

  const setDatasetId = useCallback((value: string) => {
    setDatasetIdState(value);
    setPageState(1);
    setKeywordsState("");
  }, []);

  const setPage = useCallback((value: number) => {
    setPageState(Math.max(1, Math.trunc(value)));
  }, []);

  const setPageSize = useCallback((value: number) => {
    setPageSizeState(Math.min(100, Math.max(1, Math.trunc(value))));
    setPageState(1);
  }, []);

  const setKeywords = useCallback((value: string) => {
    setKeywordsState(value.trim());
    setPageState(1);
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    const controller = new AbortController();
    setLoadingDatasets(true);
    listAllKnowledgeBases(controller.signal)
      .then((rows) => {
        if (controller.signal.aborted) return;
        setDatasets(rows);
        setDatasetIdState((current) => {
          if (current && rows.some((row) => row.id === current)) return current;
          return rows[0]?.id ?? "";
        });
        setError(null);
      })
      .catch((loadError) => {
        if (!isPlatformApiError(loadError) || !loadError.isAbort) {
          setError(libraryError(loadError));
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingDatasets(false);
      });
    return () => controller.abort();
  }, []);

  const totalPages = Math.max(1, Math.ceil(totalDocuments / pageSize));

  useEffect(() => {
    if (page > totalPages) setPageState(totalPages);
  }, [page, totalPages]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const hasActiveDocuments = documents.some(
    (document) =>
      document.status === "pending" || document.status === "running",
  );

  useEffect(() => {
    clearPoll();
    if (!datasetId || !hasActiveDocuments) {
      pollDelayRef.current = POLL_START_MS;
      return;
    }

    let stopped = false;
    const schedule = () => {
      if (stopped || document.visibilityState === "hidden") return;
      pollTimerRef.current = setTimeout(async () => {
        const rows = await refresh(true);
        if (stopped) return;
        const stillActive = rows.some(
          (row) => row.status === "pending" || row.status === "running",
        );
        if (!stillActive) {
          pollDelayRef.current = POLL_START_MS;
          return;
        }
        pollDelayRef.current = nextDocumentPollDelay(pollDelayRef.current);
        schedule();
      }, pollDelayRef.current);
    };
    const onVisibility = () => {
      clearPoll();
      if (document.visibilityState === "visible") {
        pollDelayRef.current = POLL_START_MS;
        void refresh(true).then(schedule);
      }
    };
    document.addEventListener("visibilitychange", onVisibility);
    schedule();
    return () => {
      stopped = true;
      clearPoll();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [clearPoll, datasetId, hasActiveDocuments, refresh]);

  useEffect(
    () => () => {
      mountedRef.current = false;
      requestRef.current?.abort();
      mutationRef.current?.abort();
      clearPoll();
    },
    [clearPoll],
  );

  const runMutation = useCallback(
    async (operation: (signal: AbortSignal) => Promise<unknown>) => {
      mutationRef.current?.abort();
      const controller = new AbortController();
      mutationRef.current = controller;
      setMutating(true);
      setError(null);
      try {
        await operation(controller.signal);
        await refresh(true);
      } catch (mutationError) {
        if (!isPlatformApiError(mutationError) || !mutationError.isAbort) {
          const mapped = libraryError(mutationError);
          setError(mapped);
          throw mutationError;
        }
      } finally {
        if (mountedRef.current) setMutating(false);
        if (mutationRef.current === controller) mutationRef.current = null;
      }
    },
    [refresh],
  );

  const upload = useCallback(
    async (files: File[], parseAfterUpload: boolean): Promise<UploadSummary> => {
      const rejected = files
        .map(validatePlatformDocumentFile)
        .filter(
          (failure): failure is PlatformFileValidationFailure => failure !== null,
        );
      const rejectedFiles = new Set(rejected.map((failure) => failure.file));
      const accepted = files.filter((file) => !rejectedFiles.has(file));
      if (!datasetId || accepted.length === 0) {
        return { accepted, rejected, partialFailure: null };
      }
      let partialFailure: string | null = null;
      await runMutation(async (signal) => {
        const result = await uploadDatasetDocuments(datasetId, accepted, signal);
        partialFailure = result.partialFailure;
        if (parseAfterUpload && result.documents.length > 0) {
          await parseDatasetDocuments(
            datasetId,
            result.documents.map((document) => document.id),
            signal,
          );
        }
      });
      return { accepted, rejected, partialFailure };
    },
    [datasetId, runMutation],
  );

  const parse = useCallback(
    (documentIds: string[]) =>
      runMutation((signal) =>
        parseDatasetDocuments(datasetId, documentIds, signal),
      ),
    [datasetId, runMutation],
  );

  const stop = useCallback(
    (documentIds: string[]) =>
      runMutation((signal) =>
        stopDatasetDocuments(datasetId, documentIds, signal),
      ),
    [datasetId, runMutation],
  );

  const remove = useCallback(
    (documentIds: string[]) =>
      runMutation((signal) =>
        deleteDatasetDocuments(datasetId, documentIds, signal),
      ),
    [datasetId, runMutation],
  );

  const rename = useCallback(
    (documentId: string, name: string) =>
      runMutation((signal) =>
        updateDatasetDocument(datasetId, documentId, { name }, signal),
      ),
    [datasetId, runMutation],
  );

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === datasetId) ?? null,
    [datasetId, datasets],
  );

  return {
    datasets,
    datasetId,
    setDatasetId,
    selectedDataset,
    documents,
    totalDocuments,
    page,
    pageSize,
    totalPages,
    keywords,
    setPage,
    setPageSize,
    setKeywords,
    loadingDatasets,
    loadingDocuments,
    mutating,
    error,
    refresh,
    upload,
    parse,
    stop,
    remove,
    rename,
  };
}
