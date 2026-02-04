import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter } from "@/components/ui/dialog";
import type { ReactElement } from "react";
import type { NodeConfig, SamplerConfig } from "../types";
import { LlmDialog } from "./llm/llm-dialog";
import { CategoryDialog } from "./samplers/category-dialog";
import { DatetimeDialog } from "./samplers/datetime-dialog";
import { GaussianDialog } from "./samplers/gaussian-dialog";
import { PersonDialog } from "./samplers/person-dialog";
import { SubcategoryDialog } from "./samplers/subcategory-dialog";
import { UniformDialog } from "./samplers/uniform-dialog";
import { UuidDialog } from "./samplers/uuid-dialog";
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
            {config.kind === "sampler" &&
              config.sampler_type === "category" && (
                <CategoryDialog
                  config={config}
                  onUpdate={(patch) => onUpdate(config.id, patch)}
                />
              )}
            {config.kind === "sampler" &&
              config.sampler_type === "subcategory" && (
                <SubcategoryDialog
                  config={config}
                  categoryOptions={categoryOptions}
                  onUpdate={(patch) => onUpdate(config.id, patch)}
                />
              )}
            {config.kind === "sampler" && config.sampler_type === "uniform" && (
              <UniformDialog
                config={config}
                onUpdate={(patch) => onUpdate(config.id, patch)}
              />
            )}
            {config.kind === "sampler" &&
              config.sampler_type === "gaussian" && (
                <GaussianDialog
                  config={config}
                  onUpdate={(patch) => onUpdate(config.id, patch)}
                />
              )}
            {config.kind === "sampler" &&
              config.sampler_type === "datetime" && (
                <DatetimeDialog
                  config={config}
                  onUpdate={(patch) => onUpdate(config.id, patch)}
                />
              )}
            {config.kind === "sampler" && config.sampler_type === "uuid" && (
              <UuidDialog
                config={config}
                onUpdate={(patch) => onUpdate(config.id, patch)}
              />
            )}
            {config.kind === "sampler" && config.sampler_type === "person" && (
              <PersonDialog
                config={config}
                onUpdate={(patch) => onUpdate(config.id, patch)}
              />
            )}
            {config.kind === "llm" && (
              <LlmDialog
                config={config}
                onUpdate={(patch) => onUpdate(config.id, patch)}
              />
            )}
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
