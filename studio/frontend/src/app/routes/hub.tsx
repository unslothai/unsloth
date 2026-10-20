// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { HUB_CAPABILITY_FILTERS, type HubCapabilityFilter } from "@/features/hub";
import { hubAudioTaskSearch } from "@/lib/hub-audio-task";
import type { PipelineType } from "@huggingface/hub";
import { createRoute, lazyRouteComponent } from "@tanstack/react-router";
import { requireAuth } from "../auth-guards";
import { Route as rootRoute } from "./__root";

const ModelsPage = lazyRouteComponent(
  () => import("@/features/hub/hub-page"),
  "ModelsPage",
);


export interface ModelsSearch {
  tab?: "discover" | "downloaded";
  model?: string;
  file?: string;

  intent?: number;
  section?: "trending" | "latest" | "finetune";
  kind?: "models" | "datasets";
  capability?: HubCapabilityFilter;
  audioTask?: PipelineType[];
}

export const Route = createRoute({
  getParentRoute: () => rootRoute,
  path: "/hub",
  beforeLoad: () => requireAuth(),
  component: ModelsPage,
  validateSearch: (search: Record<string, unknown>): ModelsSearch => {
    const next: ModelsSearch = {};
    const raw = search.tab;
    if (raw === "discover" || raw === "downloaded") next.tab = raw;
    const model = search.model;
    if (typeof model === "string" && model.length > 0) next.model = model;
    const file = search.file;
    if (next.model && typeof file === "string" && file.length > 0)
      next.file = file;

    const intent = search.intent;
    if (
      next.file &&
      typeof intent === "number" &&
      Number.isSafeInteger(intent)
    ) {
      next.intent = intent;
    }
    const section = search.section;
    if (
      section === "trending" ||
      section === "latest" ||
      section === "finetune"
    ) {
      next.section = section;
    }
    const kind = search.kind;
    if (kind === "models" || kind === "datasets") {
      next.kind = kind;
    }
    // "diffusion" is the single image/video filter older links carry.
    const capability = search.capability === "diffusion" ? "image-gen" : search.capability;
    if (HUB_CAPABILITY_FILTERS.includes(capability as HubCapabilityFilter)) {
      next.capability = capability as HubCapabilityFilter;
    }
    const audioTask = hubAudioTaskSearch(search.audioTask, next.capability);
    if (audioTask) next.audioTask = audioTask;
    return next;
  },
});
