// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Suspense, lazy } from "react";
import { datasetSummary } from "../dataset/dataset-value";
import type { DatasetValue } from "../types";

const viewer = () => import("../dataset/dataset-viewer");
const Panel = lazy(() => viewer().then((m) => ({ default: m.DatasetPanel })));
const Preview = lazy(() => viewer().then((m) => ({ default: m.DatasetPreview })));

export function DatasetOutput({
  value,
  compact = false,
}: { value: DatasetValue; compact?: boolean }) {
  const View = compact ? Preview : Panel;
  return (
    <Suspense
      fallback={
        <span className="text-ui-11 text-muted-foreground">{datasetSummary(value)}</span>
      }
    >
      <View value={value} />
    </Suspense>
  );
}
