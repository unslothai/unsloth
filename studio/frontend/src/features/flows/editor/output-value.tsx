// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { DatasetRows, isDatasetPreview } from "./dataset-rows";
import { MediaThumb, isMediaValue } from "./media";
import { StructuredOutput, isStructuredValue } from "./value-viewers";

export function OutputValue({ value }: { value: unknown }) {
  if (value === null || value === undefined)
    return <span className="text-muted-foreground/70">-</span>;
  if (isMediaValue(value)) {
    return (
      <MediaThumb media={value} className="w-full rounded-md object-contain" />
    );
  }
  if (Array.isArray(value) && value.length > 0 && value.every(isMediaValue)) {
    return (
      <div className="grid grid-cols-2 gap-1.5">
        {value.slice(0, 24).map((item, i) => (
          <MediaThumb
            key={`${(item as { ref: string }).ref}-${i}`}
            media={item}
            className="w-full rounded object-cover"
          />
        ))}
      </div>
    );
  }
  if (isStructuredValue(value)) return <StructuredOutput value={value} />;
  if (isDatasetPreview(value)) return <DatasetRows value={value} />;
  if (typeof value === "string") {
    return (
      <p className="max-h-72 overflow-y-auto whitespace-pre-wrap text-xs leading-relaxed text-foreground">
        {value}
      </p>
    );
  }
  return (
    <pre className="max-h-72 overflow-auto rounded-lg border border-border/60 bg-muted/40 p-2 font-mono text-ui-11 leading-snug text-foreground">
      {JSON.stringify(value, null, 2).slice(0, 20_000)}
    </pre>
  );
}
