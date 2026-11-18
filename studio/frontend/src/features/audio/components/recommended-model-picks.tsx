// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useId } from "react";
import { Button } from "@/components/ui/button";

/** An empty page's shortcut to a model that runs it: up to three catalog picks, one click to load. */
export function RecommendedModelPicks({
  models,
  onPick,
}: {
  models: readonly { id: string; name: string }[];
  onPick?: (id: string) => void;
}) {
  const captionId = useId();
  const picks = onPick ? models.slice(0, 3) : [];
  if (picks.length === 0) return null;
  return (
    <div className="grid justify-items-center gap-2">
      <p id={captionId} className="text-ui-11p5 text-muted-foreground">
        Recommended models
      </p>
      <div
        role="group"
        className="flex flex-wrap justify-center gap-2"
        // Named by the visible caption, so the two cannot drift apart.
        aria-labelledby={captionId}
      >
        {picks.map((model) => (
          <Button
            key={model.id}
            type="button"
            variant="outline"
            size="sm"
            title={`Load ${model.name}`}
            translate="no"
            onClick={() => onPick?.(model.id)}
          >
            {model.name.replace(/-GGUF$/i, "")}
          </Button>
        ))}
      </div>
    </div>
  );
}
