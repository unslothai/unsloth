import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter } from "@/components/ui/dialog";
import type { ReactElement } from "react";
import type { NodeConfig, SamplerConfig } from "../types";
import { renderBlockDialog } from "../blocks/registry";
import { DialogShell } from "./shared/dialog-shell";
import { ValidationBanner } from "./shared/validation-banner";

type ConfigDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  config: NodeConfig | null;
  categoryOptions: SamplerConfig[];
  onUpdate: (id: string, patch: Partial<NodeConfig>) => void;
};

export function ConfigDialog({
  open,
  onOpenChange,
  config,
  categoryOptions,
  onUpdate,
}: ConfigDialogProps): ReactElement {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogShell />
        {!config && (
          <div className="text-sm text-muted-foreground">
            Select a node to edit.
          </div>
        )}
        {config && (
          <div className="space-y-4">
            <ValidationBanner config={config} />
            {renderBlockDialog(config, categoryOptions, onUpdate)}
          </div>
        )}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
          >
            Done
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
