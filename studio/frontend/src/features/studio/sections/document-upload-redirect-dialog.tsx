


import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useT } from "@/i18n";
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
  const t = useT();
  const fileLabel =
    fileName ?? t("studio.dataset.documentRedirect.genericFile");

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
            <DialogTitle>
              {t("studio.dataset.documentRedirect.title")}
            </DialogTitle>
            <DialogDescription>
              {t("studio.dataset.documentRedirect.description", {
                file: fileLabel,
              })}
            </DialogDescription>
          </div>
        </DialogHeader>

        <div className="corner-squircle rounded-2xl border border-border/70 bg-muted/20 p-4">
          <p className="text-sm font-medium text-foreground">
            {t("studio.dataset.documentRedirect.nextStepTitle")}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {t("studio.dataset.documentRedirect.nextStepDescription")}
          </p>
        </div>

        <DialogFooter className="sm:justify-between">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            {t("common.cancel")}
          </Button>
          <Button type="button" onClick={onOpenLearningRecipes}>
            {t("studio.dataset.documentRedirect.openAction")}
            <HugeiconsIcon icon={ArrowRight01Icon} className="size-4" />
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
