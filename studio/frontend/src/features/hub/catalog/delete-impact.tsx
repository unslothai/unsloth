// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { formatBytes } from "@/features/hub/lib/format";
import { fetchDeleteImpact, type DeleteImpact } from "../inventory";

/** Image GGUFs share a large companion base repo, so show what a delete actually frees.
 *  Null until loaded or on failure, so the dialog opens either way. */
export function useDeleteImpact(
  open: boolean,
  repoId: string,
  variant?: string | null,
  cachePath?: string | null,
): DeleteImpact | null {
  const [impact, setImpact] = useState<DeleteImpact | null>(null);
  useEffect(() => {
    if (!open) {
      setImpact(null);
      return;
    }
    let cancelled = false;
    void fetchDeleteImpact(repoId, variant ?? undefined, cachePath ?? undefined).then((result) => {
      if (!cancelled) setImpact(result);
    });
    return () => {
      cancelled = true;
    };
  }, [open, repoId, variant, cachePath]);
  return impact;
}

function joinNames(names: string[]): string {
  if (names.length <= 2) return names.join(" and ");
  return `${names.slice(0, 2).join(", ")} and ${names.length - 2} more`;
}

/** States the retained size even when large; says nothing if the preview failed. */
export function DeleteImpactSummary({ impact }: { impact: DeleteImpact | null }) {
  if (!impact) return null;
  if (impact.blocked_by.length > 0) {
    return (
      <span className="mt-2 block text-ui-12p5 text-destructive">
        These are shared assets that {joinNames(impact.blocked_by)} still needs, so they cannot
        be removed yet. Delete those models first.
      </span>
    );
  }
  const retained = impact.retained_companions.reduce((sum, c) => sum + c.size_bytes, 0);
  const freeable = impact.freeable_companions.reduce((sum, c) => sum + c.size_bytes, 0);
  return (
    <span className="mt-2 block space-y-1 text-ui-12p5">
      {impact.cache_path && (
        <span className="block break-all text-muted-foreground" data-testid="delete-impact-folder">
          Folder: {impact.cache_path}
        </span>
      )}
      <span className="block text-foreground" data-testid="delete-impact-reclaimed">
        Frees {formatBytes(impact.reclaimed_bytes)} of disk space.
      </span>
      {retained > 0 ? (
        <span className="block text-muted-foreground" data-testid="delete-impact-retained">
          {formatBytes(retained)} of shared assets stay on disk:{" "}
          {joinNames(impact.retained_companions.map((c) => c.repo_id))} is still needed by{" "}
          {joinNames(
            Array.from(new Set(impact.retained_companions.flatMap((c) => c.needed_by))),
          )}
          .
        </span>
      ) : null}
      {freeable > 0 ? (
        <span className="block text-muted-foreground" data-testid="delete-impact-freeable">
          This also leaves {formatBytes(freeable)} of shared assets (
          {joinNames(impact.freeable_companions.map((c) => c.repo_id))}) that nothing else
          needs. Remove them with Free up space on the On Device tab.
        </span>
      ) : null}
    </span>
  );
}
