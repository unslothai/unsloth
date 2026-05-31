// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioDatasetStep: TourStep = {
  id: "dataset",
  target: "studio-dataset",
  title: "Dataset",
  body: (
    <>
      Search Hub or paste <span className="font-mono">user/dataset</span>. Preview
      a few rows: formatting matters more than size. We’ll try to auto-convert
      your dataset into a supported training format. If we can’t infer it
      cleanly, we’ll prompt you to map the fields manually. If outputs look off
      in Chat later, dataset formatting/template is the first thing to check.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide" />
    </>
  ),
};
