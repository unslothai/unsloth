// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";
import {
  type StubElement,
  loadWithStubs,
  stubJsxRuntime,
} from "./helpers/module-stubs.ts";

registerBundlerResolver();

const [ggufFit, variantSort, modelIdentity, ggufFilename, format] =
  await Promise.all([
    import("../src/lib/gguf-fit.ts"),
    import("../src/features/hub/lib/gguf-variant-sort.ts"),
    import("../src/features/hub/lib/model-identity.ts"),
    import("../src/features/hub/lib/gguf-filename.ts"),
    import("../src/features/hub/lib/format.ts"),
  ]);

const GiB = 1024 ** 3;
const VARIANTS = [
  { quant: "Q4_K_M", filename: "Qwen3-8B-Q4_K_M.gguf", size_bytes: 4.68 * GiB },
];

type Card = {
  GgufDownloadCard: (props: Record<string, unknown>) => StubElement;
};

type Cleanup = (() => void) | undefined;
type Effect = () => Cleanup;

/** Enough of React's effect pass for one component: in order, on commit, when deps change. */
function effectRuntime() {
  const slots: { deps?: unknown[]; cleanup?: Cleanup }[] = [];
  let cursor = 0;
  let pending: (() => void)[] = [];
  return {
    begin() {
      cursor = 0;
    },
    useEffect(effect: Effect, deps?: unknown[]) {
      let slot = slots[cursor];
      if (!slot) {
        slot = {};
        slots[cursor] = slot;
      }
      cursor += 1;
      const changed =
        !slot.deps ||
        !deps ||
        deps.length !== slot.deps.length ||
        deps.some((dep, i) => dep !== slot.deps?.[i]);
      if (!changed) return;
      pending.push(() => {
        slot.cleanup?.();
        slot.deps = deps;
        slot.cleanup = effect();
      });
    },
    commit() {
      const run = pending;
      pending = [];
      for (const fn of run) fn();
    },
    unmount() {
      for (const slot of slots) slot.cleanup?.();
    },
  };
}

function loadCard(
  effects: ReturnType<typeof effectRuntime>,
  calls: unknown[],
): Card {
  const noop = () => undefined;
  const anyExport = new Proxy({}, { get: () => noop });
  return loadWithStubs<Card>(
    new URL(
      "../src/features/hub/catalog/gguf-download-card.tsx",
      import.meta.url,
    ),
    {
      react: {
        memo: (component: unknown) => component,
        useCallback: (fn: unknown) => fn,
        useEffect: effects.useEffect,
        useMemo: (read: () => unknown) => read(),
        useState: (initial: unknown) => [
          typeof initial === "function" ? initial() : initial,
          noop,
        ],
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/hooks/use-system": {
        refreshSystemMemory: (options: unknown) => {
          calls.push(options);
          return Promise.resolve(null);
        },
      },
      "@/lib/gguf-fit": ggufFit,
      "../lib/gguf-variant-sort": variantSort,
      "../lib/model-identity": modelIdentity,
      "../lib/gguf-filename": ggufFilename,
      "../lib/format": format,
      "./use-gguf-variant-fetch-state": {
        useGgufVariantFetchState: () => ({
          variants: VARIANTS,
          loading: false,
          error: null,
          refreshError: null,
          refresh: noop,
        }),
      },
      "../download-manager": {
        downloadManager: { probeAndAdopt: noop },
        pendingDrafterPresentation: () => null,
        useDownloadManagerStore: () => new Map(),
        useHttpPartialsResumable: () => false,
        useRepoDownload: () => ({ progress: null, cancelling: false }),
      },
      "../download-manager/use-hub-download-queue": {
        useHubDownloadPlan: () => null,
      },
      "./gguf-live-variant-states": {
        activeDownloadState: () => false,
        applyLiveGgufVariantStates: (variants: unknown) => variants,
        createLiveGgufVariantStatesSelector: noop,
        createScopedLiveGgufFilesSelector: noop,
        isScopedLiveVariant: () => false,
      },
      "../hooks/use-media-companion-bytes": {
        awaitsCompanions: () => false,
        ggufVariantFootprint: () => null,
        useMediaCompanionBytes: () => new Map(),
      },
      "./use-download-card-state": {
        useDownloadCardState: () => ({
          starting: false,
          ariaLabel: "Download",
          onClick: noop,
        }),
      },
      "./delete-impact": { DeleteImpactSummary: noop, useDeleteImpact: noop },
      "./use-delete-confirm-action": {
        useDeleteConfirmAction: () => ({ deleting: false, runDelete: noop }),
      },
      "@/hooks/use-vram-budget-fraction": { useVramBudgetFraction: () => null },
      "../hooks/use-online-status": { useOnlineStatus: () => true },
      "../stores/hf-token-store": { useHfTokenStore: () => null },
      "./use-required-assets-download": {
        useRequiredAssetsDownload: () => ({ checking: false, request: noop }),
      },
      "../download-manager/hub-plan-progress": anyExport,
      "@/components/model-memory-bar": anyExport,
      "@/components/ui/dropdown-menu": anyExport,
      "@/components/ui/popover": anyExport,
      "@/components/ui/spinner": anyExport,
      "@/components/ui/tooltip": anyExport,
      "@/config/env": anyExport,
      "@/i18n": { useT: () => (key: string) => key },
      "@/features/chat": anyExport,
      "@/features/model-picker": anyExport,
      "@/lib/chevron-icons": anyExport,
      "@/lib/copy-to-clipboard": anyExport,
      "@/lib/toast": anyExport,
      "@/lib/utils": anyExport,
      "@hugeicons/core-free-icons": anyExport,
      "@/lib/hugeicons-derived": anyExport,
      "@/lib/copy-icon": anyExport,
      "@/lib/folder-icons": anyExport,
      "@hugeicons/react": anyExport,
      "../inventory": anyExport,
      "./dot-tag": anyExport,
      "./download-cancel-indicator": anyExport,
      "./download-card": anyExport,
      "./gguf-status-cards": anyExport,
    },
  );
}

const BASE = {
  isActive: false,
  activeQuant: null,
  isLoadingThisModel: false,
  gpuGb: 24,
  gpuCount: 1,
  systemRamGb: 9,
  unifiedLoadBudgetGb: 7.5,
};

test("opening a GGUF model in the Hub takes one fresh memory reading per repo", () => {
  const calls: unknown[] = [];
  const effects = effectRuntime();
  const { GgufDownloadCard } = loadCard(effects, calls);
  const render = (repoId: string) => {
    effects.begin();
    GgufDownloadCard({ ...BASE, repoId });
    effects.commit();
  };
  render("unsloth/Qwen3-8B-GGUF");
  assert.deepEqual(calls, [{ maxAgeMs: 10_000 }]);
  render("unsloth/Qwen3-8B-GGUF");
  assert.equal(
    calls.length,
    1,
    "a re-render of the same repo is not a new open",
  );
  render("unsloth/Llama-3.2-3B-GGUF");
  assert.equal(calls.length, 2, "another model is");
  effects.unmount();
});

test("a media GGUF shows no llama.cpp verdict, so opening it takes no reading", () => {
  const calls: unknown[] = [];
  const effects = effectRuntime();
  const { GgufDownloadCard } = loadCard(effects, calls);
  effects.begin();
  GgufDownloadCard({
    ...BASE,
    repoId: "unsloth/FLUX-GGUF",
    showMemoryBar: false,
    mediaPage: "images",
  });
  effects.commit();
  assert.deepEqual(calls, []);
  effects.unmount();
});
