// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { formatBytes } from "@/features/hub/lib/format";
import {
  type OrphanCompanion,
  deleteCachedModel,
  fetchOrphanCompanions,
} from "../inventory";
import { useHfTokenStore } from "@/features/hub/stores/hf-token-store";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";

/**
 * Remove companion assets that no installed model needs any more.
 *
 * An image GGUF loads its text encoders, VAE and tokenizer from a separate base repo that every
 * quant of the family shares, and that is usually the larger half of the footprint. Deleting the
 * last quant cannot take those with it, because the delete is scoped to one repo and nothing on
 * screen knows the sharing has ended. Without this dialog the only way to recover the space was
 * to hand-edit the Hugging Face cache.
 *
 * Nothing here runs on its own. The list is recomputed from what is installed each time the
 * dialog opens, every row is opt-in, and removal goes through the ordinary delete endpoint, so
 * the shared-asset guard applies to this path exactly as it does to a delete from a card.
 */
export function FreeUpSpaceDialog({
  open,
  onOpenChange,
  onChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChange?: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [companions, setCompanions] = useState<OrphanCompanion[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [deleting, setDeleting] = useState(false);
  const hfToken = useHfTokenStore((s) => s.token);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchOrphanCompanions();
      setCompanions(result.companions);
      setSelected(new Set(result.companions.map((c) => c.repo_id)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setCompanions([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (open) void refresh();
  }, [open, refresh]);

  const selectedCompanions = companions.filter((c) => selected.has(c.repo_id));
  const selectedBytes = selectedCompanions.reduce(
    (sum, c) => sum + c.size_bytes,
    0,
  );

  const runDelete = useCallback(async () => {
    setDeleting(true);
    let removed = 0;
    let freed = 0;
    try {
      for (const companion of selectedCompanions) {
        try {
          await deleteCachedModel(
            companion.repo_id,
            undefined,
            hfToken || undefined,
            companion.cache_path ?? undefined,
          );
          removed += 1;
          freed += companion.size_bytes;
        } catch (e) {
          toast.error(`Could not remove ${companion.repo_id}`, {
            description: e instanceof Error ? e.message : String(e),
          });
        }
      }
    } finally {
      setDeleting(false);
    }
    if (removed > 0) {
      toast.success(
        `Removed ${removed} unused asset${removed === 1 ? "" : "s"}, freeing ${formatBytes(freed)}`,
      );
      onChange?.();
    }
    onOpenChange(false);
  }, [hfToken, onChange, onOpenChange, selectedCompanions]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-[520px]">
        <DialogHeader>
          <DialogTitle>Free up space</DialogTitle>
          <DialogDescription>
            Shared model assets, such as text encoders, VAE and tokenizers, that no installed
            model needs any more. Anything still in use by a model on this device is not listed
            and cannot be removed here.
          </DialogDescription>
        </DialogHeader>
        {loading ? (
          <div className="flex items-center gap-2 py-6 text-ui-13 text-muted-foreground">
            <Spinner /> Checking what is still in use…
          </div>
        ) : error ? (
          <p className="py-6 text-ui-13 text-destructive">{error}</p>
        ) : companions.length === 0 ? (
          <p className="py-6 text-ui-13 text-muted-foreground" data-testid="free-up-space-empty">
            Nothing to clean up. Every cached asset is still needed by a model you have
            installed.
          </p>
        ) : (
          <ul className="flex max-h-[260px] flex-col gap-1 overflow-y-auto py-1">
            {companions.map((companion) => (
              <li
                key={companion.repo_id}
                className="flex items-center gap-3 rounded-[10px] px-2 py-2 hover:bg-muted/50"
              >
                <Checkbox
                  id={`orphan-${companion.repo_id}`}
                  checked={selected.has(companion.repo_id)}
                  onCheckedChange={(checked) =>
                    setSelected((prev) => {
                      const next = new Set(prev);
                      if (checked) next.add(companion.repo_id);
                      else next.delete(companion.repo_id);
                      return next;
                    })
                  }
                />
                <label
                  htmlFor={`orphan-${companion.repo_id}`}
                  className="min-w-0 flex-1 cursor-pointer truncate text-ui-13"
                >
                  {companion.repo_id}
                </label>
                <span className="shrink-0 text-ui-12p5 text-muted-foreground">
                  {formatBytes(companion.size_bytes)}
                </span>
              </li>
            ))}
          </ul>
        )}
        <DialogFooter>
          <Button
            variant="ghost"
            disabled={deleting}
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            disabled={deleting || selectedCompanions.length === 0}
            onClick={() => void runDelete()}
            data-testid="free-up-space-confirm"
          >
            {deleting
              ? "Removing…"
              : `Remove ${formatBytes(selectedBytes)}`}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
