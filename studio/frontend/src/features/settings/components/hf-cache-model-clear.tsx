// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  DeleteImpactSummary,
  useDeleteImpact,
} from "@/features/hub/catalog/delete-impact";
import { DeleteConfirmDialog } from "@/features/hub/catalog/download-card";
import { deleteCachedModel } from "@/features/hub/inventory";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";

/** Share the Hub's exact-copy deletion, dependency checks and inventory events. */
export function HfCacheModelClear({
  repoId,
  cachePath,
  disabled = false,
}: {
  repoId: string;
  cachePath: string;
  disabled?: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const hfToken = useHfTokenStore((state) => state.token);
  const impact = useDeleteImpact(open, repoId, null, cachePath);
  const clear = async () => {
    if (deleting || disabled || (impact?.blocked_by.length ?? 0) > 0) return;
    setDeleting(true);
    try {
      await deleteCachedModel(
        repoId,
        undefined,
        hfToken || undefined,
        cachePath,
      );
      setOpen(false);
    } catch (error) {
      toast.error(t("settings.resources.storage.caches.clearFailed"), {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setDeleting(false);
    }
  };
  return (
    <>
      <Button
        variant="ghost"
        size="xs"
        disabled={disabled || deleting}
        aria-label={t("settings.resources.storage.caches.confirmOneTitle", {
          name: repoId,
        })}
        title={cachePath}
        onClick={() => setOpen(true)}
      >
        {t("settings.resources.storage.caches.clearOneAction")}
      </Button>
      <DeleteConfirmDialog
        open={open}
        onOpenChange={(next) => {
          if (!deleting) setOpen(next);
        }}
        title={t("settings.resources.storage.caches.confirmOneTitle", {
          name: repoId,
        })}
        description={
          <>
            {t("settings.resources.storage.caches.clearModelDescription")}
            <span className="mt-2 block break-all font-mono text-ui-11">
              {cachePath}
            </span>
            <DeleteImpactSummary impact={impact} />
          </>
        }
        deleting={deleting}
        blocked={disabled || (impact?.blocked_by.length ?? 0) > 0}
        onConfirm={() => void clear()}
      />
    </>
  );
}
