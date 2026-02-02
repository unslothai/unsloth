import { listDatasets } from "@huggingface/hub";
import { useEffect, useState } from "react";

export interface HfDatasetResult {
  id: string;
  downloads: number;
  likes: number;
}

interface HfSearchState {
  results: HfDatasetResult[];
  isLoading: boolean;
  error: string | null;
}

export function useHfDatasetSearch(
  query: string,
  options?: { limit?: number; accessToken?: string },
): HfSearchState {
  const { limit = 20, accessToken } = options ?? {};
  const [state, setState] = useState<HfSearchState>({
    results: [],
    isLoading: false,
    error: null,
  });

  useEffect(() => {
    if (!query.trim()) {
      setState({ results: [], isLoading: false, error: null });
      return;
    }

    let cancelled = false;
    setState((prev) => ({ ...prev, isLoading: true, error: null }));

    (async () => {
      try {
        const results: HfDatasetResult[] = [];
        const iter = listDatasets({
          search: { query },
          limit,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        });
        for await (const ds of iter) {
          if (cancelled) return;
          results.push({
            id: ds.id,
            downloads: ds.downloads,
            likes: ds.likes,
          });
        }
        if (!cancelled) {
          setState({ results, isLoading: false, error: null });
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            results: [],
            isLoading: false,
            error: err instanceof Error ? err.message : "Search failed",
          });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [query, limit, accessToken]);

  return state;
}
