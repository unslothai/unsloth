// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useTrainingConfigStore } from "@/features/training";
import { useT } from "@/i18n";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import { useCallback } from "react";

export interface NotebookCatalogEntry {
  id: string;
  title: string;
  notebook_file: string;
  category: string;
  featured: boolean;
  studio_model: string | null;
  colab_url: string;
  github_url: string;
}

export interface NotebookCatalogResponse {
  notebooks: NotebookCatalogEntry[];
  categories: string[];
}

export async function fetchNotebookCatalog(
  query?: string,
): Promise<NotebookCatalogResponse> {
  const params = new URLSearchParams();
  const normalizedQuery = query?.trim();
  if (normalizedQuery) {
    params.set("q", normalizedQuery);
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await authFetch(`/api/notebooks${suffix}`);
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as NotebookCatalogResponse;
}

function normalizeNotebookSearch(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

export function notebookMatchesQuery(
  notebook: NotebookCatalogEntry,
  query: string,
): boolean {
  const normalizedQuery = normalizeNotebookSearch(query);
  if (!normalizedQuery) return true;
  const haystack = [
    notebook.title,
    notebook.notebook_file,
    notebook.studio_model ?? "",
    notebook.category,
  ]
    .map(normalizeNotebookSearch)
    .join("");
  return haystack.includes(normalizedQuery);
}

export function useOpenNotebookInStudio(): {
  openInStudio: (notebook: NotebookCatalogEntry) => void;
} {
  const t = useT();
  const navigate = useNavigate();
  const setSelectedModel = useTrainingConfigStore((s) => s.setSelectedModel);

  const openInStudio = useCallback(
    (notebook: NotebookCatalogEntry) => {
      if (notebook.studio_model) {
        setSelectedModel(notebook.studio_model);
      } else {
        toast.message(t("notebooks.studioOpenedWithoutModel"));
      }
      void navigate({ to: "/studio" });
    },
    [navigate, setSelectedModel, t],
  );

  return { openInStudio };
}
