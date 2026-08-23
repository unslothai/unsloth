// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import { fetchDeleteImpact, type DeleteImpact } from "../inventory";
import { formatBytes } from "../lib/format";
import {
  isDeleteBlocked,
  isUnverifiable,
  repollDelayMs,
  shouldRefreshDeleteImpactOnWake,
} from "./delete-impact-state";

/**
 * Load the delete preview for a confirm dialog while it is open.
 *
 * An image GGUF is a small checkpoint plus a much larger companion base repo (text encoders,
 * VAE, tokenizer) shared by every quant of its family, so "removes it from disk" was never the
 * whole story: it could free 2.6 GB and silently leave 8.2 GB behind.
 *
 * The preview is advisory, so it reports `checking` and `unavailable` for the dialog to say out
 * loud and leaves `blocked` to what it positively knows; see delete-impact-state for why, and
 * for when a standing or unavailable preview is worth re-reading.
 */
export function useDeleteImpact(
  open: boolean,
  repoId: string,
  variant?: string | null,
  cachePath?: string | null,
): {
  impact: DeleteImpact | null;
  checking: boolean;
  unavailable: boolean;
  blocked: boolean;
} {
  const requestKey = open
    ? JSON.stringify([repoId, variant ?? null, cachePath ?? null])
    : null;
  const [settled, setSettled] = useState<{
    requestKey: string;
    impact: DeleteImpact | null;
  } | null>(null);
  useEffect(() => {
    if (requestKey === null) return;
    let cancelled = false;
    let repollTimer: number | undefined;
    let delay: number | null = null;
    let latest: DeleteImpact | null = null;
    let inFlight: "initial" | "timer" | "focus" | null = null;
    let focusRefreshPending = false;
    const controller = new AbortController();
    const check = (reason: "initial" | "timer" | "focus") => {
      if (inFlight !== null) {
        if (reason === "focus" && inFlight !== "focus") {
          focusRefreshPending = true;
        }
        return;
      }
      inFlight = reason;
      void fetchDeleteImpact(
        repoId,
        variant ?? undefined,
        cachePath ?? undefined,
        controller.signal,
      )
        .catch(() => null)
        .then((result) => {
          inFlight = null;
          if (cancelled) return;
          latest = result;
          setSettled({ requestKey, impact: result });
          if (focusRefreshPending) {
            focusRefreshPending = false;
            if (shouldRefreshDeleteImpactOnWake(result)) {
              check("focus");
              return;
            }
          }
          delay = repollDelayMs(result, delay);
          if (delay !== null) {
            repollTimer = window.setTimeout(() => check("timer"), delay);
          }
        });
    };
    const refreshBlockedOnWake = () => {
      if (document.hidden) return;
      if (inFlight !== null) {
        if (inFlight !== "focus") focusRefreshPending = true;
        return;
      }
      if (!shouldRefreshDeleteImpactOnWake(latest)) return;
      window.clearTimeout(repollTimer);
      check("focus");
    };
    window.addEventListener("focus", refreshBlockedOnWake);
    document.addEventListener("visibilitychange", refreshBlockedOnWake);
    check("initial");
    return () => {
      cancelled = true;
      controller.abort();
      window.clearTimeout(repollTimer);
      window.removeEventListener("focus", refreshBlockedOnWake);
      document.removeEventListener("visibilitychange", refreshBlockedOnWake);
      setSettled(null);
    };
  }, [cachePath, repoId, requestKey, variant]);
  const impact = settled?.requestKey === requestKey ? settled.impact : null;
  const checking = requestKey !== null && settled?.requestKey !== requestKey;
  const unavailable =
    requestKey !== null &&
    settled?.requestKey === requestKey &&
    settled.impact === null;
  return { impact, checking, unavailable, blocked: isDeleteBlocked(impact) };
}

function joinNames(names: string[]): string {
  if (names.length <= 2) return names.join(" and ");
  return `${names.slice(0, 2).join(", ")} and ${names.length - 2} more`;
}

/**
 * The truthful half of a delete confirmation: what comes back, and what does not.
 *
 * Deliberately says the retained number out loud even when it dwarfs the reclaimed one, and
 * refuses to guess when the preview is unavailable.
 */
export function DeleteImpactSummary({
  impact,
  checking = false,
  unavailable = false,
}: {
  impact: DeleteImpact | null;
  checking?: boolean;
  unavailable?: boolean;
}) {
  if (checking) {
    return (
      <span
        aria-live="polite"
        className="mt-2 block text-ui-12p5 text-muted-foreground"
      >
        Checking whether this model can be deleted…
      </span>
    );
  }
  if (unavailable) {
    return (
      <span
        aria-live="polite"
        className="mt-2 block text-ui-12p5 text-muted-foreground"
      >
        Couldn't check what this frees up. Deleting still refuses a model that
        is loaded or that another model needs.
      </span>
    );
  }
  if (!impact) return null;
  if (isUnverifiable(impact) && impact.blocked_by.length === 0) {
    return (
      <span
        aria-live="polite"
        className="mt-2 block text-ui-12p5 text-muted-foreground"
      >
        {impact.delete_block?.detail}
      </span>
    );
  }
  if (impact.delete_block || impact.blocked_by.length > 0) {
    return (
      <span
        aria-live="polite"
        className="mt-2 block space-y-1 text-ui-12p5 text-destructive"
      >
        {impact.delete_block && (
          <span
            className={
              isUnverifiable(impact) ? "block text-muted-foreground" : "block"
            }
          >
            {impact.delete_block.detail}
          </span>
        )}
        {impact.blocked_by.length > 0 && (
          <span className="block">
            These are shared assets that {joinNames(impact.blocked_by)} still
            needs, so they cannot be removed yet. Delete those models first.
          </span>
        )}
      </span>
    );
  }
  const retained = impact.retained_companions.reduce((sum, c) => sum + c.size_bytes, 0);
  const freeable = impact.freeable_companions.reduce((sum, c) => sum + c.size_bytes, 0);
  return (
    <span className="mt-2 block space-y-1 text-ui-12p5">
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
