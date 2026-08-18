// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness page for tests/studio/playwright_model_picker_scale.py: the REAL ModelSelector holding
// a large local model list, so what is timed is the app's own picker.
//
// The axis is NUMBER OF MODELS the picker is handed. ModelSelector takes `models: ModelOption[]`
// as a prop, so the fixture is a prop and not a stubbed fetch: that is the one input the picker
// cannot get from anywhere else, and it is the one a user with a full model cache grows.
//
// The Hugging Face LISTING half of the picker is deliberately not in the axis. It is paged behind
// an IntersectionObserver sentinel (pickers.tsx:4445-4490), so its row count is bounded by how
// far the user scrolled rather than by how much they own, and sweeping it would be measuring the
// harness's own scrolling. The DOWNLOADED half has no such bound and that is what this varies.
//
// Same shape as smoke-heavy-thread.html: a vite entry, no backend, no auth.

// The row type the app itself parses /api/hub/cached-models into, from the feature index. Shaping
// the fixture against a hand-written local type instead would let it drift from what the picker
// really reads, and the drift would show up as "the picker is empty", not as a type error.
import type { CachedModelRepo } from "@/features/hub";
import type { ModelOption } from "@/features/model-picker";
import { ModelSelector } from "@/features/model-picker";
import { TooltipProvider } from "@/components/ui/tooltip";
import { initializeLocale } from "@/i18n";
import {
  RouterProvider,
  createMemoryHistory,
  createRootRoute,
  createRoute,
  createRouter,
} from "@tanstack/react-router";
import { type ReactNode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./src/index.css";

// Deterministic and NOT Math.random(): model name length decides how much text layout a row pays
// for, so an unseeded fixture would hand two runs different work.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const ORGS = ["unsloth", "meta-llama", "Qwen", "mistralai", "google", "deepseek-ai"];
const FAMILIES = ["Llama", "Qwen", "Mistral", "Gemma", "Phi", "DeepSeek", "Yi", "Falcon"];
const PARAMS = ["1B", "3B", "7B", "8B", "13B", "14B", "32B", "70B"];
const SUFFIX = ["Instruct", "Chat", "Base", "Coder", "Math", "Vision"];

function buildModels(count: number): ModelOption[] {
  const rand = mulberry32(0x9e37_79b9);
  const out: ModelOption[] = [];
  for (let i = 0; i < count; i += 1) {
    const org = ORGS[Math.floor(rand() * ORGS.length)] as string;
    const family = FAMILIES[Math.floor(rand() * FAMILIES.length)] as string;
    const params = PARAMS[Math.floor(rand() * PARAMS.length)] as string;
    const suffix = SUFFIX[Math.floor(rand() * SUFFIX.length)] as string;
    // The index is in the id so the ids are unique however the random words land; two rows with
    // the same id would be deduped by optionById and the fixture would be smaller than it says.
    const id = `${org}/${family}-${params}-${suffix}-${i}`;
    const isGguf = i % 3 === 0;
    out.push({
      id,
      name: id,
      description: isGguf ? "GGUF" : "safetensors",
      isGguf,
      deviceQuant: isGguf ? "Q4_K_M" : undefined,
      deviceSize: `${1 + (i % 40)}.2 GB`,
      deviceSizeBytes: (1 + (i % 40)) * 1_000_000_000,
      deviceLoaded: false,
    });
  }
  return out;
}

// An explicit allowlist, never a blanket `/api/` match: a blanket match resolves every request the
// measured interactions make before Playwright emits it, so the stray counter stays at zero and
// the fan-out the harness exists to detect becomes invisible to it.
const realFetch = window.fetch.bind(window);
const stubbedApiCalls: string[] = [];

function jsonResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

// The DOWNLOADED half of the picker is fed by the hub INVENTORY, not by the `models` prop:
// use-device-inventory pulls /api/hub/cached-models, /api/hub/cached-gguf and /api/hub/local, and
// the picker's own rows come from those. Seeding the prop alone opened a panel with zero rows,
// measured. So the size axis is served here, from the same in-memory fixture.
let cachedRepos: CachedModelRepo[] = [];

const STUBBED: ReadonlyArray<readonly [RegExp, () => string]> = [
  // Studio polls this on a timer. Left to Playwright it is a round trip to another process
  // landing inside a timed region, once per tick.
  // chat_only:false and a device_type matter, not just a 200. `chatOnly` defaults TRUE until
  // /api/health answers with a verdict, and pickers.tsx:3763 reads
  // `chatOnly && !task ? [] : visibleCachedModels` -- so a bare {"status":"ok"} empties the whole
  // downloaded list and the On Device tab renders zero rows however many models are seeded.
  // Measured exactly that way: 200 models in, 0 rows out.
  [
    /\/api\/health$/,
    () =>
      JSON.stringify({
        status: "ok",
        version: "0.0.0-smoke",
        device_type: "nvidia",
        chat_only: false,
        chat_only_reason: null,
        hardware_detecting: false,
      }),
  ],
  [/\/api\/hub\/cached-models$/, () => JSON.stringify({ cached: cachedRepos })],
  [/\/api\/hub\/cached-gguf$/, () => '{"cached":[]}'],
  [/\/api\/hub\/local$/, () => '{"models":[],"folders":[]}'],
  [/\/api\/hub\/datasets\//, () => '{"datasets":[],"cached":[]}'],
  [/\/api\/hub\/scan-folders$/, () => '{"folders":[]}'],
  [/\/api\/hub\/hidden-models$/, () => '{"hidden":[]}'],
  [/\/api\/hub\//, () => "{}"],
  [/\/api\/inference\//, () => "{}"],
  [/\/api\/models\/scan-folders$/, () => '{"folders":[]}'],
  [/\/api\/models\/recommended-folders$/, () => '{"folders":[]}'],
  [/\/api\/models/, () => '{"models":[]}'],
  [/\/api\/system$/, () => "{}"],
];

window.fetch = (input, init) => {
  const url =
    typeof input === "string" ? input : ((input as Request).url ?? String(input));
  const path = url.split("?")[0] ?? url;
  for (const [pattern, body] of STUBBED) {
    if (pattern.test(path)) {
      stubbedApiCalls.push(url);
      return Promise.resolve(jsonResponse(body()));
    }
  }
  return realFetch(input, init);
};

declare global {
  interface Window {
    __pickerScale: {
      seed: (count: number) => { models: number };
      setOpen: (open: boolean) => void;
      isOpen: () => boolean;
      trigger: () => HTMLElement | null;
      panel: () => HTMLElement | null;
      searchInput: () => HTMLInputElement | null;
      query: () => string;
      counts: () => Record<string, number>;
      onDeviceTab: () => HTMLElement | null;
      modelCount: () => number;
      stubbedApi: () => string[];
    };
    __stubbedApi: string[];
  }
}

window.__stubbedApi = stubbedApiCalls;

let externalSeed: ((models: ModelOption[]) => void) | null = null;
let externalOpen: ((open: boolean) => void) | null = null;
let seededCount = 0;
let openNow = false;

function Harness(): ReactNode {
  const [models, setModels] = useState<ModelOption[]>([]);
  const [open, setOpen] = useState(false);
  // Published from an EFFECT, not during render. Assigning a module-level binding in the render
  // body is a side effect and `react-hooks/globals` rejects it; more to the point, a harness whose
  // control surface is installed by rendering is one whose control surface can be reinstalled by a
  // re-render in the middle of a measurement.
  useEffect(() => {
    externalSeed = setModels;
    externalOpen = (next: boolean) => {
      openNow = next;
      setOpen(next);
    };
    return () => {
      externalSeed = null;
      externalOpen = null;
    };
  }, []);
  return (
    <TooltipProvider>
      <div data-testid="harness-root" style={{ padding: 24 }}>
        <ModelSelector
          models={models}
          open={open}
          onOpenChange={(next) => {
            openNow = next;
            setOpen(next);
          }}
        />
      </div>
    </TooltipProvider>
  );
}

// The picker's own panel, portaled to document.body by Radix.
const PANEL = ".unsloth-model-selector-menu";
const SEARCH = "[data-model-picker-search-input]";

window.__pickerScale = {
  seed: (count: number) => {
    const models = buildModels(count);
    seededCount = models.length;
    // Both halves from ONE fixture: the prop the picker takes, and the inventory rows the
    // downloaded list is really built from. Seeding only one of them produced a panel with the
    // right model count in its props and zero rows on screen.
    cachedRepos = models.map((model, index) => ({
      repo_id: model.id,
      size_bytes: (1 + (index % 40)) * 1_000_000_000,
      cache_path: `/models/${model.id}`,
      last_modified: 1_700_000_000 - index * 60,
      model_format: "safetensors",
      partial: false,
      tags: [],
    })) as CachedModelRepo[];
    externalSeed?.(models);
    return { models: seededCount };
  },
  setOpen: (open: boolean) => externalOpen?.(open),
  isOpen: () => openNow,
  trigger: () =>
    document.querySelector<HTMLElement>(".unsloth-model-selector-trigger"),
  panel: () => document.querySelector<HTMLElement>(PANEL),
  searchInput: () => document.querySelector<HTMLInputElement>(SEARCH),
  query: () => document.querySelector<HTMLInputElement>(SEARCH)?.value ?? "",
  counts: () => {
    const panel = document.querySelector(PANEL);
    return {
      // `[data-model-picker-option]` is what a row really carries. `[role="option"]` was tried
      // first and counted ZERO against a panel holding 200 rows, which reads as "the picker
      // rendered nothing" rather than as "wrong selector"; the attribute census that found this
      // also showed 604 tooltip triggers and 200 dropdown triggers for those 200 rows.
      rows: panel ? panel.querySelectorAll("[data-model-picker-option]").length : 0,
      rowMenuTriggers: panel
        ? panel.querySelectorAll('[data-slot="dropdown-menu-trigger"]').length
        : 0,
      tooltipTriggers: panel
        ? panel.querySelectorAll('[data-slot="tooltip-trigger"]').length
        : 0,
      panelNodes: panel ? panel.getElementsByTagName("*").length : 0,
      domNodes: document.getElementsByTagName("*").length,
      searchInputs: document.querySelectorAll(SEARCH).length,
      panelScrollHeight: (() => {
        const scroller = panel
          ? panel.querySelector<HTMLElement>("[data-model-picker-option]")?.parentElement
          : null;
        return scroller ? scroller.scrollHeight : 0;
      })(),
      panels: document.querySelectorAll(PANEL).length,
    };
  },
  // Found by its label text once, here, rather than in the probe: the probe then drives an
  // element rather than a string, and a locale change breaks this one line instead of the run.
  onDeviceTab: () => {
    const panel = document.querySelector(PANEL);
    if (!panel) return null;
    for (const el of Array.from(panel.querySelectorAll<HTMLElement>("button"))) {
      if ((el.textContent || "").trim() === "On Device") return el;
    }
    return null;
  },
  modelCount: () => seededCount,
  stubbedApi: () => [...stubbedApiCalls],
};

// The picker navigates to the hub on "Browse", so it needs a router in scope. A memory router with
// one route keeps the harness backend-free while those hooks resolve.
const harnessRootRoute = createRootRoute({ component: Harness });
const harnessIndexRoute = createRoute({
  getParentRoute: () => harnessRootRoute,
  path: "/",
  component: () => null,
});
const harnessRouter = createRouter({
  routeTree: harnessRootRoute.addChildren([harnessIndexRoute]),
  history: createMemoryHistory({ initialEntries: ["/"] }),
});

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element not found");
const root = createRoot(rootElement);

function render(): void {
  // No StrictMode: it double-invokes render, which is the quantity being measured.
  root.render(<RouterProvider router={harnessRouter} />);
}

const localeInitialization = initializeLocale();
if (typeof localeInitialization !== "string") {
  void localeInitialization.then(render);
} else {
  render();
}
