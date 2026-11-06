// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ReadMore, type TourStep } from "@/features/tour";

export const studioDatasetStep: TourStep = {
  id: "dataset",
  target: "studio-dataset",
  title: "Dataset",
  body: (
    <>
      Search the Hub, or upload PDF, DOCX, JSONL, JSON, CSV or Parquet. Preview
      a few rows before you start, since formatting matters more than size. We
      map the columns for you and ask only when the format is unclear. No
      dataset yet? Build one on the Recipes page.{" "}
      <ReadMore href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide" />
    </>
  ),
};
