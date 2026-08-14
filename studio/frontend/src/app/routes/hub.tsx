


import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const DocumentLibraryPage = lazyRouteComponent(
  () => import("@/features/documents/document-library-page"),
  "DocumentLibraryPage",
);

export interface DocumentLibrarySearch {
  // Legacy model-hub keys remain accepted so existing deep links and internal
  // model-picker navigation degrade to the document landing page instead of
  // becoming type-invalid while the old page is no longer mounted here.
  tab?: "discover" | "downloaded";
  dataset?: string;
  model?: string;
  file?: string;
  intent?: number;
  section?: "trending" | "latest" | "finetune";
  kind?: "models" | "datasets";
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/hub",
  beforeLoad: () => requireAuth(),
  component: DocumentLibraryPage,
  validateSearch: (search: Record<string, unknown>): DocumentLibrarySearch => {
    const next: DocumentLibrarySearch = {};
    if (
      search.tab === "discover" ||
      search.tab === "downloaded"
    ) next.tab = search.tab;
    if (typeof search.dataset === "string" && search.dataset.trim()) next.dataset = search.dataset.trim();
    if (typeof search.model === "string" && search.model.trim()) next.model = search.model.trim();
    if (next.model && typeof search.file === "string" && search.file.trim()) next.file = search.file.trim();
    if (next.file && typeof search.intent === "number" && Number.isSafeInteger(search.intent)) next.intent = search.intent;
    if (search.section === "trending" || search.section === "latest" || search.section === "finetune") next.section = search.section;
    if (search.kind === "models" || search.kind === "datasets") next.kind = search.kind;
    return next;
  },
});
