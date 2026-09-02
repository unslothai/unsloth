// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Browser harness for tests/studio/playwright_mtp_download_visibility.py. It runs
// the same staging predicate as the chat model picker and renders the real
// bottom-right download manager without needing a GPU or a multi-gigabyte pull.
/* eslint-disable react-refresh/only-export-components -- standalone Playwright harness */

import { TooltipProvider } from "@/components/ui/tooltip";
/* eslint-disable no-restricted-imports -- the harness exercises these internal boundaries directly */
import { AUTH_TOKEN_KEY } from "@/features/auth/session";
import { wantsDownloadManagerStaging } from "@/features/chat/utils/model-download-staging";
import {
  DownloadManagerPanel,
  __resetDownloadManagerForTests,
  downloadManager,
  useDownloadManagerStore,
} from "@/features/hub/download-manager";
/* eslint-enable no-restricted-imports */
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

const REPO_ID = "unsloth/Qwen3.8-Flash-Next-GGUF";

type Variant = {
  quant: string;
  downloaded: boolean;
} & Record<"download_size_bytes", number>;

declare global {
  interface Window {
    __mtpDownloadSmoke?: {
      reproduce: () => Promise<{
        backendDownloaded: boolean;
        staged: boolean;
        outcome: string | null;
      }>;
      jobs: () => ReturnType<typeof useDownloadManagerStore.getState>;
    };
  }
}

localStorage.setItem(AUTH_TOKEN_KEY, "mtp-download-smoke");
__resetDownloadManagerForTests();

window.__mtpDownloadSmoke = {
  reproduce: async () => {
    const response = await fetch(
      `/api/models/gguf-variants?repo_id=${encodeURIComponent(REPO_ID)}`,
    );
    if (!response.ok) {
      throw new Error(`Variant request failed: ${response.status}`);
    }
    const payload = (await response.json()) as { variants: Variant[] };
    const variant = payload.variants[0];
    if (!variant) {
      throw new Error("The deterministic backend returned no variant");
    }

    const staged = wantsDownloadManagerStaging({
      id: REPO_ID,
      source: "hub",
      isGguf: true,
      ggufVariant: variant.quant,
      isDownloaded: variant.downloaded,
    });
    let outcome: string | null = null;
    if (staged) {
      outcome = await downloadManager.requestStart({
        kind: "model",
        repoId: REPO_ID,
        variant: variant.quant,
        expectedBytes: variant.download_size_bytes,
      });
    }
    return {
      backendDownloaded: variant.downloaded,
      staged,
      outcome,
    };
  },
  jobs: () => useDownloadManagerStore.getState(),
};

function Harness() {
  const side = new URLSearchParams(window.location.search).get("side") ?? "A/B";
  return (
    <TooltipProvider>
      <main className="min-h-dvh bg-background p-10 text-foreground">
        <div className="max-w-2xl rounded-2xl border border-border bg-card p-7 shadow-sm">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            {side} · cached quant, missing MTP companion
          </p>
          <h1 className="mt-3 text-2xl font-semibold">Qwen3.8 Flash Next</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Selecting the already-cached model should stage its required 2.6 GiB
            MTP sidecar in Downloads before loading.
          </p>
        </div>
        <DownloadManagerPanel />
      </main>
    </TooltipProvider>
  );
}

const rootRoute = createRootRoute({ component: Harness });
const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: () => null,
});
const router = createRouter({
  routeTree: rootRoute.addChildren([indexRoute]),
  history: createMemoryHistory({ initialEntries: ["/"] }),
});

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}
createRoot(rootElement).render(
  <StrictMode>
    <RouterProvider router={router as unknown as never} />
  </StrictMode>,
);
