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
import { useI18n } from "@/features/i18n";
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
  const { t } = useI18n();
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
            <Badge variant="outline">{t("studio.documentRedirect.recipeStudio")}</Badge>
          </div>
          <div className="space-y-1">
            <DialogTitle>{t("studio.documentRedirect.title")}</DialogTitle>
            <DialogDescription>
              {fileName ? (
                <>
                  <span className="font-medium text-foreground">{fileName}</span>{" "}
                  {t("studio.documentRedirect.fileNotReady")}
                </>
              ) : (
                t("studio.documentRedirect.fileNotReadyNoName")
              )}{" "}
              {t("studio.documentRedirect.descriptionSuffix")}
            </DialogDescription>
          </div>
        </DialogHeader>

        <div className="corner-squircle rounded-2xl border border-border/70 bg-muted/20 p-4">
          <p className="text-sm font-medium text-foreground">
            {t("studio.documentRedirect.bestNextStep")}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("studio.documentRedirect.bestNextStepHint")}
          </p>
        </div>

        <DialogFooter className="sm:justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            {t("studio.documentRedirect.cancel")}
          </Button>
          <Button type="button" onClick={onOpenLearningRecipes}>
            {t("studio.documentRedirect.openLearningRecipes")}
            <HugeiconsIcon icon={ArrowRight01Icon} className="size-4" />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
