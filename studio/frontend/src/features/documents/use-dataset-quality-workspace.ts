import {
  createDocumentChunk,
  deleteDocumentChunks,
  deleteDocumentStructureGraph,
  getDocumentChunk,
  getDocumentStructureGraph,
  isPlatformApiError,
  listDocumentChunks,
  retrievePlatformChunks,
  setDocumentChunksEnabled,
  updateDocumentChunk,
  type PlatformChunk,
  type PlatformChunkDraft,
  type PlatformRetrievalRequest,
  type PlatformRetrievalResult,
  type PlatformStructureGraph,
} from "@/integrations/platform-backend";
import { useCallback, useEffect, useRef, useState } from "react";

export interface DatasetQualityError {
  kind: "permission" | "timeout" | "request";
  message: string;
}

function qualityError(error: unknown): DatasetQualityError {
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

export function useDatasetQualityWorkspace(
  datasetId: string,
  documentId: string,
) {
  const [chunks, setChunks] = useState<PlatformChunk[]>([]);
  const [chunkTotal, setChunkTotal] = useState(0);
  const [chunkPage, setChunkPageState] = useState(1);
  const [chunkPageSize, setChunkPageSizeState] = useState(50);
  const [chunkKeywords, setChunkKeywordsState] = useState("");
  const [chunkAvailability, setChunkAvailabilityState] = useState<
    "all" | "enabled" | "disabled"
  >("all");
  const [loadingChunks, setLoadingChunks] = useState(false);
  const [chunkError, setChunkError] = useState<DatasetQualityError | null>(
    null,
  );
  const [mutating, setMutating] = useState(false);

  const [structureGraph, setStructureGraph] = useState<PlatformStructureGraph>({
    templates: [],
  });
  const [structureKeywords, setStructureKeywords] = useState("");
  const [loadingStructure, setLoadingStructure] = useState(false);
  const [structureError, setStructureError] =
    useState<DatasetQualityError | null>(null);

  const [retrieval, setRetrieval] = useState<PlatformRetrievalResult | null>(
    null,
  );
  const [retrieving, setRetrieving] = useState(false);
  const [retrievalError, setRetrievalError] =
    useState<DatasetQualityError | null>(null);

  const mountedRef = useRef(true);
  const chunkRequestRef = useRef<AbortController | null>(null);
  const structureRequestRef = useRef<AbortController | null>(null);
  const retrievalRequestRef = useRef<AbortController | null>(null);
  const mutationRequestRef = useRef<AbortController | null>(null);

  const abortAll = useCallback(() => {
    chunkRequestRef.current?.abort();
    structureRequestRef.current?.abort();
    retrievalRequestRef.current?.abort();
    mutationRequestRef.current?.abort();
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortAll();
    };
  }, [abortAll]);

  useEffect(() => {
    abortAll();
    setChunks([]);
    setChunkTotal(0);
    setChunkPageState(1);
    setChunkError(null);
    setLoadingChunks(false);
    setMutating(false);
    setStructureGraph({ templates: [] });
    setStructureError(null);
    setLoadingStructure(false);
    setRetrieval(null);
    setRetrievalError(null);
    setRetrieving(false);
  }, [abortAll, datasetId, documentId]);

  const refreshChunks = useCallback(async () => {
    if (!datasetId || !documentId) {
      setChunks([]);
      setChunkTotal(0);
      return [] as PlatformChunk[];
    }
    chunkRequestRef.current?.abort();
    const controller = new AbortController();
    chunkRequestRef.current = controller;
    setLoadingChunks(true);
    try {
      const result = await listDocumentChunks(
        datasetId,
        documentId,
        {
          page: chunkPage,
          pageSize: chunkPageSize,
          keywords: chunkKeywords,
          available:
            chunkAvailability === "all"
              ? undefined
              : chunkAvailability === "enabled",
        },
        controller.signal,
      );
      if (!mountedRef.current || controller.signal.aborted) return [];
      setChunks(result.items);
      setChunkTotal(result.total);
      setChunkError(null);
      return result.items;
    } catch (error) {
      if (!isPlatformApiError(error) || !error.isAbort) {
        if (mountedRef.current) setChunkError(qualityError(error));
      }
      return [];
    } finally {
      if (mountedRef.current && !controller.signal.aborted)
        setLoadingChunks(false);
      if (chunkRequestRef.current === controller)
        chunkRequestRef.current = null;
    }
  }, [
    chunkAvailability,
    chunkKeywords,
    chunkPage,
    chunkPageSize,
    datasetId,
    documentId,
  ]);

  useEffect(() => {
    void refreshChunks();
  }, [refreshChunks]);

  const loadStructure = useCallback(
    async (keywords: string) => {
      if (!datasetId || !documentId) {
        setStructureGraph({ templates: [] });
        return;
      }
      structureRequestRef.current?.abort();
      const controller = new AbortController();
      structureRequestRef.current = controller;
      setLoadingStructure(true);
      try {
        const graph = await getDocumentStructureGraph(
          datasetId,
          documentId,
          keywords,
          controller.signal,
        );
        if (mountedRef.current && !controller.signal.aborted) {
          setStructureGraph(graph);
          setStructureError(null);
        }
      } catch (error) {
        if (
          mountedRef.current &&
          (!isPlatformApiError(error) || !error.isAbort)
        ) {
          setStructureError(qualityError(error));
        }
      } finally {
        if (mountedRef.current && !controller.signal.aborted)
          setLoadingStructure(false);
        if (structureRequestRef.current === controller)
          structureRequestRef.current = null;
      }
    },
    [datasetId, documentId],
  );

  const refreshStructure = useCallback(
    () => loadStructure(structureKeywords),
    [loadStructure, structureKeywords],
  );

  useEffect(() => {
    void loadStructure("");
  }, [loadStructure]);

  const runMutation = useCallback(
    async (operation: (signal: AbortSignal) => Promise<unknown>) => {
      mutationRequestRef.current?.abort();
      const controller = new AbortController();
      mutationRequestRef.current = controller;
      setMutating(true);
      try {
        const result = await operation(controller.signal);
        if (!controller.signal.aborted) await refreshChunks();
        return result;
      } catch (error) {
        if (!isPlatformApiError(error) || !error.isAbort) {
          if (mountedRef.current) setChunkError(qualityError(error));
        }
        throw error;
      } finally {
        if (mountedRef.current && !controller.signal.aborted)
          setMutating(false);
        if (mutationRequestRef.current === controller)
          mutationRequestRef.current = null;
      }
    },
    [refreshChunks],
  );

  const createChunk = useCallback(
    (draft: PlatformChunkDraft) =>
      runMutation((signal) =>
        createDocumentChunk(datasetId, documentId, draft, signal),
      ),
    [datasetId, documentId, runMutation],
  );

  const loadChunk = useCallback(
    async (chunkId: string): Promise<PlatformChunk> => {
      mutationRequestRef.current?.abort();
      const controller = new AbortController();
      mutationRequestRef.current = controller;
      setMutating(true);
      try {
        return await getDocumentChunk(
          datasetId,
          documentId,
          chunkId,
          controller.signal,
        );
      } finally {
        if (mountedRef.current && !controller.signal.aborted)
          setMutating(false);
        if (mutationRequestRef.current === controller)
          mutationRequestRef.current = null;
      }
    },
    [datasetId, documentId],
  );

  const updateChunk = useCallback(
    (chunkId: string, draft: Partial<PlatformChunkDraft>) =>
      runMutation((signal) =>
        updateDocumentChunk(datasetId, documentId, chunkId, draft, signal),
      ),
    [datasetId, documentId, runMutation],
  );

  const setChunksEnabled = useCallback(
    (chunkIds: string[], enabled: boolean) =>
      runMutation((signal) =>
        setDocumentChunksEnabled(
          datasetId,
          documentId,
          chunkIds,
          enabled,
          signal,
        ),
      ),
    [datasetId, documentId, runMutation],
  );

  const removeChunks = useCallback(
    (chunkIds: string[]) =>
      runMutation((signal) =>
        deleteDocumentChunks(datasetId, documentId, chunkIds, signal),
      ),
    [datasetId, documentId, runMutation],
  );

  const removeStructureTemplate = useCallback(
    async (templateId: string) => {
      mutationRequestRef.current?.abort();
      const controller = new AbortController();
      mutationRequestRef.current = controller;
      setMutating(true);
      try {
        const deleted = await deleteDocumentStructureGraph(
          datasetId,
          documentId,
          templateId,
          controller.signal,
        );
        if (!controller.signal.aborted) await refreshStructure();
        return deleted;
      } catch (error) {
        if (!isPlatformApiError(error) || !error.isAbort) {
          if (mountedRef.current) setStructureError(qualityError(error));
        }
        throw error;
      } finally {
        if (mountedRef.current && !controller.signal.aborted)
          setMutating(false);
        if (mutationRequestRef.current === controller)
          mutationRequestRef.current = null;
      }
    },
    [datasetId, documentId, refreshStructure],
  );

  const runRetrieval = useCallback(
    async (request: Omit<PlatformRetrievalRequest, "datasetIds">) => {
      if (!datasetId) return null;
      retrievalRequestRef.current?.abort();
      const controller = new AbortController();
      retrievalRequestRef.current = controller;
      setRetrieving(true);
      setRetrievalError(null);
      try {
        const result = await retrievePlatformChunks(
          { ...request, datasetIds: [datasetId] },
          controller.signal,
        );
        if (!mountedRef.current || controller.signal.aborted) return null;
        setRetrieval(result);
        return result;
      } catch (error) {
        if (
          mountedRef.current &&
          (!isPlatformApiError(error) || !error.isAbort)
        ) {
          setRetrievalError(qualityError(error));
        }
        return null;
      } finally {
        if (mountedRef.current && !controller.signal.aborted)
          setRetrieving(false);
        if (retrievalRequestRef.current === controller)
          retrievalRequestRef.current = null;
      }
    },
    [datasetId],
  );

  const setChunkPage = useCallback((value: number) => {
    setChunkPageState(Math.max(1, Math.trunc(value)));
  }, []);
  const setChunkPageSize = useCallback((value: number) => {
    setChunkPageSizeState(Math.min(200, Math.max(1, Math.trunc(value))));
    setChunkPageState(1);
  }, []);
  const setChunkKeywords = useCallback((value: string) => {
    setChunkKeywordsState(value.trim());
    setChunkPageState(1);
  }, []);
  const setChunkAvailability = useCallback(
    (value: "all" | "enabled" | "disabled") => {
      setChunkAvailabilityState(value);
      setChunkPageState(1);
    },
    [],
  );

  return {
    chunks,
    chunkTotal,
    chunkPage,
    chunkPageSize,
    chunkKeywords,
    chunkAvailability,
    loadingChunks,
    chunkError,
    mutating,
    structureGraph,
    structureKeywords,
    loadingStructure,
    structureError,
    retrieval,
    retrieving,
    retrievalError,
    setChunkPage,
    setChunkPageSize,
    setChunkKeywords,
    setChunkAvailability,
    setStructureKeywords,
    refreshChunks,
    refreshStructure,
    createChunk,
    loadChunk,
    updateChunk,
    setChunksEnabled,
    removeChunks,
    removeStructureTemplate,
    runRetrieval,
  };
}
