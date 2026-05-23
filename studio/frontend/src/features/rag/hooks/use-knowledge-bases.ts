// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { useRagStore } from "../stores/rag-store";

export function useKnowledgeBases() {
  const knowledgeBases = useRagStore((s) => s.knowledgeBases);
  const loading = useRagStore((s) => s.kbsLoading);
  const error = useRagStore((s) => s.kbsError);
  const load = useRagStore((s) => s.loadKnowledgeBases);
  const createKB = useRagStore((s) => s.createKB);
  const deleteKB = useRagStore((s) => s.deleteKB);

  useEffect(() => {
    if (knowledgeBases.length === 0 && !loading) {
      void load();
    }
    // Only run on mount — store-level cache prevents refetch loops.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { knowledgeBases, loading, error, refresh: load, createKB, deleteKB };
}
