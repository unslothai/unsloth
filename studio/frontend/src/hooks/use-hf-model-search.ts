import { listModels } from "@huggingface/hub";
import { useEffect, useState } from "react";

export interface HfModelResult {
  id: string;
  downloads: number;
  likes: number;
  task?: string;
}

interface HfSearchState {
  results: HfModelResult[];
  isLoading: boolean;
  error: string | null;
}

export function useHfModelSearch(
  query: string,
  options?: { task?: string; limit?: number; accessToken?: string },
): HfSearchState {
  const { task, limit = 20, accessToken } = options ?? {};
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
        const results: HfModelResult[] = [];
        const iter = listModels({
          search: { query, ...(task ? { task } : {}) },
          limit,
          ...(accessToken ? { credentials: { accessToken } } : {}),
        });
        for await (const model of iter) {
          if (cancelled) return;
          results.push({
            id: model.id,
            downloads: model.downloads,
            likes: model.likes,
            task: model.task,
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
  }, [query, task, limit, accessToken]);

  return state;
}
