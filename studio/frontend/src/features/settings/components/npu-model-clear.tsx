// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { DeleteConfirmDialog } from "@/features/hub/catalog/download-card";
import { bumpInventoryVersion } from "@/features/hub/stores/inventory-events";
import { deleteNpuModelByCheckpoint } from "@/features/npu/api";
import {
  refreshNpuModels,
  useNpuCatalogStore,
} from "@/features/npu/npu-catalog-store";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";

/** Keep the Hub's NPU list in step once it has been listed; listing before that would start lemond. */
export function refreshNpuCatalogIfListed(): void {
  if (useNpuCatalogStore.getState().models !== null) void refreshNpuModels();
}

/** Deletes through lemond, which keeps its own record of downloaded models, never from the disk. */
export function NpuModelClear({
  name,
  checkpoint,
  path,
  disabled = false,
}: {
  name: string;
  checkpoint: string;
  path: string;
  disabled?: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const clear = async () => {
    if (deleting || disabled) return;
    setDeleting(true);
    try {
      await deleteNpuModelByCheckpoint(checkpoint);
      setOpen(false);
      bumpInventoryVersion();
      refreshNpuCatalogIfListed();
    } catch (error) {
      toast.error(t("settings.resources.storage.caches.clearFailed"), {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setDeleting(false);
    }
  };
  const title = t("settings.resources.storage.caches.confirmOneTitle", {
    name,
  });
  return (
    <>
      <Button
        variant="ghost"
        size="xs"
        disabled={disabled || deleting}
        aria-label={title}
        title={path}
        onClick={() => setOpen(true)}
      >
        {t("settings.resources.storage.caches.clearOneAction")}
      </Button>
      <DeleteConfirmDialog
        open={open}
        onOpenChange={(next) => {
          if (!deleting) setOpen(next);
        }}
        title={title}
        description={
          <>
            {t("settings.resources.storage.caches.npuCost")}
            <span className="mt-2 block break-all font-mono text-ui-11">
              {path}
            </span>
          </>
        }
        deleting={deleting}
        blocked={disabled}
        onConfirm={() => void clear()}
      />
    </>
  );
}
