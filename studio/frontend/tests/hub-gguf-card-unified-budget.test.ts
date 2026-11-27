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

const [ggufFit, variantSort, variantOrder, modelIdentity, ggufFilename, format] =
  await Promise.all([
    import("../src/lib/gguf-fit.ts"),
    import("../src/features/hub/lib/gguf-variant-sort.ts"),
    import("../src/features/model-picker/components/model-selector/variant-order.ts"),
    import("../src/features/hub/lib/model-identity.ts"),
    import("../src/features/hub/lib/gguf-filename.ts"),
    import("../src/features/hub/lib/format.ts"),
  ]);

const GiB = 1024 ** 3;
const VARIANTS = [
  { quant: "Q4_K_M", filename: "Qwen3-8B-Q4_K_M.gguf", size_bytes: 4.68 * GiB },
  { quant: "Q8_0", filename: "Qwen3-8B-Q8_0.gguf", size_bytes: 8.11 * GiB },
  { quant: "BF16", filename: "Qwen3-8B-BF16.gguf", size_bytes: 15.26 * GiB },
];

type Card = {
  GgufDownloadCard: (props: Record<string, unknown>) => StubElement;
};

function loadCard(
  useState?: (initial: unknown) => [unknown, (next: unknown) => void],
  {
    variants = VARIANTS,
    defaultVariant = "Q4_K_M",
  }: {
    variants?: typeof VARIANTS;
    defaultVariant?: string | null;
  } = {},
): Card {
  const noop = () => undefined;
  const anyExport = new Proxy({}, { get: () => noop });
  const modelPickerExport = new Proxy(
    { pickerOrderedGgufVariants: variantOrder.pickerOrderedGgufVariants },
    {
      get: (target, property) =>
        property in target
          ? target[property as keyof typeof target]
          : noop,
    },
  );
  return loadWithStubs<Card>(
    new URL(
      "../src/features/hub/catalog/gguf-download-card.tsx",
      import.meta.url,
    ),
    {
      react: {
        memo: (component: unknown) => component,
        useCallback: (fn: unknown) => fn,
        useEffect: noop,
        useMemo: (read: () => unknown) => read(),
        useState:
          useState ??
          ((initial: unknown) => [
            typeof initial === "function" ? initial() : initial,
            noop,
          ]),
      },
      "react/jsx-runtime": stubJsxRuntime(),
      "@/lib/gguf-fit": ggufFit,
      "../lib/gguf-variant-sort": variantSort,
      "../lib/model-identity": modelIdentity,
      "../lib/gguf-filename": ggufFilename,
      "../lib/format": format,
      "./use-gguf-variant-fetch-state": {
        useGgufVariantFetchState: () => ({
          variants,
          defaultVariant,
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
      "@/features/model-picker": modelPickerExport,
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

function elements(node: unknown, out: StubElement[] = []): StubElement[] {
  if (Array.isArray(node)) {
    for (const child of node) elements(child, out);
  } else if (node && typeof node === "object" && "props" in node) {
    const element = node as StubElement;
    out.push(element);
    for (const value of Object.values(element.props)) elements(value, out);
  }
  return out;
}

function textContent(node: unknown, out: string[] = []): string[] {
  if (typeof node === "string") {
    out.push(node);
  } else if (Array.isArray(node)) {
    for (const child of node) textContent(child, out);
  } else if (node && typeof node === "object" && "props" in node) {
    for (const value of Object.values((node as StubElement).props)) {
      textContent(value, out);
    }
  }
  return out;
}

function renderOnMac(
  unifiedLoadBudgetGb: number | null,
  options?: {
    variants?: typeof VARIANTS;
    defaultVariant?: string | null;
  },
) {
  const tree = loadCard(undefined, options).GgufDownloadCard({
    repoId: "unsloth/Qwen3-8B-GGUF",
    isActive: false,
    activeQuant: null,
    isLoadingThisModel: false,
    gpuGb: 24,
    gpuCount: 1,
    systemRamGb: 9,
    unifiedLoadBudgetGb,
  });
  const nodes = elements(tree);
  const trigger = nodes.find(
    (node) => "fit" in node.props && "lookupTableBytes" in node.props,
  );
  const rows = nodes
    .filter((node) => node.props.item !== undefined)
    .map((node) => {
      const rendered = (node.type as (props: unknown) => unknown)(node.props);
      return {
        item: node.props.item as { quant: string; fit: string },
        recommended: node.props.recommended === true,
        text: textContent(rendered).join(" "),
      };
    });
  const triggerButton = nodes.find(
    (node) =>
      node.type === "button" &&
      String(node.props.className).includes("hub-menu-trigger"),
  );
  const badge = trigger
    ? (trigger.type as (props: unknown) => unknown)(trigger.props)
    : null;
  return {
    recommended: trigger?.props.quant,
    recommendedFit: trigger?.props.fit,
    fits: Object.fromEntries(rows.map((row) => [row.item.quant, row.item.fit])),
    rowRecommended: Object.fromEntries(
      rows.map((row) => [row.item.quant, row.recommended]),
    ),
    rowText: Object.fromEntries(rows.map((row) => [row.item.quant, row.text])),
    triggerText: triggerButton ? textContent(triggerButton).join(" ") : "",
    tooltip: elements(badge)
      .flatMap((node) => [node.props.children].flat())
      .filter((child) => typeof child === "string"),
  };
}

test("the Hub GGUF card recommends a quant inside the Mac's Metal load budget", () => {
  const card = renderOnMac(7.5);
  assert.equal(card.recommended, "Q4_K_M");
  assert.equal(card.recommendedFit, "fits");
  assert.deepEqual(card.fits, {
    Q4_K_M: "fits",
    BF16: "marginal",
    Q8_0: "marginal",
  });
});

test("without a Metal budget the Hub GGUF card keeps ranking by memory", () => {
  const card = renderOnMac(null);
  assert.equal(card.recommended, "BF16");
  assert.deepEqual(card.fits, { BF16: "fits", Q8_0: "fits", Q4_K_M: "fits" });
});

test("the Hub GGUF card labels the recommended quant on the trigger and row", () => {
  const card = renderOnMac(7.5);
  assert.equal(card.recommended, "Q4_K_M");
  assert.ok(card.triggerText.includes("recommended"));
  assert.deepEqual(card.rowRecommended, {
    Q4_K_M: true,
    Q8_0: false,
    BF16: false,
  });
  assert.ok(card.rowText.Q4_K_M.includes("recommended"));
  assert.ok(!card.rowText.Q8_0.includes("recommended"));
});

test("the Hub GGUF recommendation follows the backend default when it fits", () => {
  const card = renderOnMac(null);
  assert.equal(card.recommended, "BF16");
  assert.equal(card.rowRecommended.Q4_K_M, true);
  assert.ok(!card.triggerText.includes("recommended"));
});

test("downloaded and partial variants do not show the recommended label", () => {
  const downloaded = renderOnMac(7.5, {
    variants: VARIANTS.map((variant) =>
      variant.quant === "Q4_K_M"
        ? { ...variant, downloaded: true }
        : variant,
    ),
  });
  assert.equal(downloaded.rowRecommended.Q4_K_M, true);
  assert.ok(!downloaded.rowText.Q4_K_M.includes("recommended"));
  assert.ok(!downloaded.triggerText.includes("recommended"));

  const partial = renderOnMac(7.5, {
    variants: VARIANTS.map((variant) =>
      variant.quant === "Q4_K_M" ? { ...variant, partial: true } : variant,
    ),
  });
  assert.equal(partial.rowRecommended.Q4_K_M, true);
  assert.ok(!partial.rowText.Q4_K_M.includes("recommended"));
});

test("an over-budget quant on a Mac points at the Metal budget, not the VRAM Budget", () => {
  const mac = renderOnMac(3);
  assert.equal(mac.recommendedFit, "marginal");
  assert.ok(mac.tooltip.includes("picker.unifiedTightFitHint"));
  const discrete = loadCard().GgufDownloadCard({
    repoId: "unsloth/Qwen3-8B-GGUF",
    isActive: false,
    activeQuant: null,
    isLoadingThisModel: false,
    gpuGb: 6.5,
    gpuCount: 1,
    systemRamGb: 32,
  });
  const trigger = elements(discrete).find(
    (node) => "lookupTableBytes" in node.props,
  );
  assert.equal(trigger?.props.fit, "marginal");
  const tooltip = elements(
    (trigger?.type as (props: unknown) => unknown)(trigger?.props),
  ).flatMap((node) => [node.props.children].flat());
  assert.ok(tooltip.some((child) => String(child).includes("VRAM Budget")));
});

test("starting a download pins the recommended quant so a moving Metal budget keeps the transfer on the card", () => {
  const writes: unknown[] = [];
  const tree = loadCard((initial) => [
    typeof initial === "function" ? initial() : initial,
    (next) => writes.push(next),
  ]).GgufDownloadCard({
    repoId: "unsloth/Qwen3-8B-GGUF",
    isActive: false,
    activeQuant: null,
    isLoadingThisModel: false,
    gpuGb: 24,
    gpuCount: 1,
    systemRamGb: 9,
    unifiedLoadBudgetGb: 7.5,
  });
  const download = elements(tree).find(
    (node) => node.type === "button" && node.props["aria-label"] === "Download",
  );
  assert.ok(download);
  (download.props.onClick as () => void)();
  assert.ok(
    writes.some(
      (write) =>
        typeof write === "object" &&
        write !== null &&
        (write as { quant?: string }).quant === "Q4_K_M" &&
        (write as { userPicked?: boolean }).userPicked === true,
    ),
  );
});
