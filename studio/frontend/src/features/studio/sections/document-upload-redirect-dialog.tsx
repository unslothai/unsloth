// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import {
  ArrowRight01Icon,
  DocumentAttachmentIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";

type DocumentUploadRedirectDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  fileName: string | null;
  onOpenLearningRecipes: () => void;
};

export function DocumentUploadRedirectDialog({
  open,
  onOpenChange,
  fileName,
  onOpenLearningRecipes,
}: DocumentUploadRedirectDialogProps): ReactElement {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-lg"
        overlayClassName="bg-background/45 supports-backdrop-filter:backdrop-blur-[1px]"
      >
        <DialogHeader className="gap-3">
          <div className="flex items-center gap-2">
            <div className="flex size-10 items-center justify-center rounded-2xl border border-border/70 bg-muted/30">
              <HugeiconsIcon
                icon={DocumentAttachmentIcon}
                className="size-5 text-foreground/90"
              />
            </div>
            <Badge variant="outline">Recipe Studio</Badge>
          </div>
          <div className="space-y-1">
            <DialogTitle>This file needs conversion first</DialogTitle>
            <DialogDescription>
              {fileName ? (
                <>
                  <span className="font-medium text-foreground">{fileName}</span>{" "}
                  is source material, not a ready-to-train dataset.
                </>
              ) : (
                "This file is source material, not a ready-to-train dataset."
              )}{" "}
              Use Data Recipes to turn documents into a dataset, then bring the
              result back here for fine-tuning.
            </DialogDescription>
          </div>
        </DialogHeader>

        <div className="corner-squircle rounded-2xl border border-border/70 bg-muted/20 p-4">
          <p className="text-sm font-medium text-foreground">
            Best next step
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            Open Learning Recipes and start from a document-based recipe like PDF
            grounded QA.
          </p>
        </div>

        <DialogFooter className="sm:justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Cancel
          </Button>
          <Button type="button" onClick={onOpenLearningRecipes}>
            Open Learning Recipes
            <HugeiconsIcon icon={ArrowRight01Icon} className="size-4" />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
