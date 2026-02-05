import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter } from "@/components/ui/dialog";
import { Switch } from "@/components/ui/switch";
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
  container?: HTMLDivElement | null;
};

export function ConfigDialog({
  open,
  onOpenChange,
  config,
  categoryOptions,
  onUpdate,
  container,
}: ConfigDialogProps): ReactElement {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="sm:max-w-2xl shadow-border"
      >
        <DialogShell />
        {!config && (
          <div className="text-sm text-muted-foreground">
            Select a node to edit.
          </div>
        )}
        {config && (
          <div className="space-y-4">
            <ValidationBanner config={config} />
            {(config.kind === "sampler" ||
              config.kind === "llm" ||
              config.kind === "expression") && (
              <div className="flex items-center justify-between gap-3 rounded-2xl border border-border/60 px-3 py-2">
                <div>
                  <p className="text-sm font-semibold">Drop from final dataset</p>
                  <p className="text-xs text-muted-foreground">
                    Keep for generation but omit from exported rows.
                  </p>
                </div>
                <Switch
                  checked={config.drop ?? false}
                  onCheckedChange={(value) => onUpdate(config.id, { drop: value })}
                />
              </div>
            )}
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
