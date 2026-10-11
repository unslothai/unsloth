// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TourStep } from "@/features/tour";

const catalogStep: TourStep = {
  id: "catalog",
  target: "hub-catalog",
  title: "Results",
  body: (
    <>
      Open any model for its files, quant variants and size. From there you
      download it, run it in Chat, or delete it to free up space.
    </>
  ),
};

const detailStep: TourStep = {
  id: "detail",
  target: "hub-detail",
  title: "Model details",
  body: (
    <>
      Files, quant variants and sizes, with the README below. Download it here,
      run it in Chat, or use Back to return to the results.
    </>
  ),
};

/** Outside split view an open model detail covers the catalog, so describe what is on top. */
export function buildHubTourSteps({
  catalogCovered,
}: {
  catalogCovered: boolean;
}): TourStep[] {
  return [
    {
      id: "tabs",
      target: "hub-tabs",
      title: "Discover or On Device",
      body: (
        <>
          Discover browses Hugging Face. On Device lists what is downloaded
          here, including your own finetunes, with the disk each one uses.
        </>
      ),
    },
    {
      id: "search",
      target: "hub-search",
      title: "Search the Hub",
      body: (
        <>
          Search by name, or paste <span className="font-mono">org/model</span>{" "}
          to jump straight to a repo. Unsloth's own quants rank first.
        </>
      ),
    },
    {
      id: "device",
      target: "hub-device",
      title: "Your hardware",
      body: (
        <>
          VRAM, RAM and cache size, so you know what will run before you
          download. Add a Hugging Face token here for gated repos.
        </>
      ),
    },
    catalogCovered ? detailStep : catalogStep,
  ];
}
