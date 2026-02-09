import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { VisuallyHidden } from "radix-ui";
import { type ReactElement, useMemo } from "react";
import type { CanvasProcessorConfig } from "../types";
import { buildDefaultSchemaTransform } from "../utils/processors";
type ProcessorsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  processors: CanvasProcessorConfig[];
  onProcessorsChange: (processors: CanvasProcessorConfig[]) => void;
  container?: HTMLDivElement | null;
};

export function ProcessorsDialog({
  open,
  onOpenChange,
  processors,
  onProcessorsChange,
  container,
}: ProcessorsDialogProps): ReactElement {
  const schemaIndex = useMemo(
    () =>
      processors.findIndex(
        (processor) => processor.processor_type === "schema_transform",
      ),
    [processors],
  );
  const schemaProcessor = schemaIndex >= 0 ? processors[schemaIndex] : null;
  const nameId = schemaProcessor ? `${schemaProcessor.id}-name` : "schema-transform-name";
  const templateId = schemaProcessor
    ? `${schemaProcessor.id}-template`
    : "schema-transform-template";

  const setSchemaEnabled = (enabled: boolean) => {
    if (enabled) {
      if (schemaProcessor) {
        return;
      }
      onProcessorsChange([...processors, buildDefaultSchemaTransform()]);
      return;
    }
    onProcessorsChange(
      processors.filter(
        (processor) => processor.processor_type !== "schema_transform",
      ),
    );
  };

  const updateSchema = (patch: Partial<CanvasProcessorConfig>) => {
    if (!schemaProcessor) {
      return;
    }
    const next = [...processors];
    next[schemaIndex] = { ...schemaProcessor, ...patch };
    onProcessorsChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle max-h-[650px] overflow-auto sm:max-w-2xl"
      >
        <VisuallyHidden.Root>
          <DialogTitle>Processors</DialogTitle>
        </VisuallyHidden.Root>
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-3 rounded-2xl border border-border/60 px-3 py-2">
            <div>
              <p className="text-sm font-semibold">Schema transform</p>
              <p className="text-xs text-muted-foreground">
                Transform final rows to target schema (post-batch).
              </p>
            </div>
            <Switch
              checked={Boolean(schemaProcessor)}
              onCheckedChange={setSchemaEnabled}
            />
          </div>

          {schemaProcessor && (
            <div className="space-y-3">
              <div className="grid gap-2">
                <label
                  className="text-xs font-semibold uppercase text-muted-foreground"
                  htmlFor={nameId}
                >
                  Name
                </label>
                <Input
                  id={nameId}
                  className="nodrag"
                  value={schemaProcessor.name}
                  onChange={(event) => updateSchema({ name: event.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <label
                  className="text-xs font-semibold uppercase text-muted-foreground"
                  htmlFor={templateId}
                >
                  Template (JSON)
                </label>
                <Textarea
                  id={templateId}
                  className="corner-squircle nodrag min-h-[220px]"
                  value={schemaProcessor.template}
                  onChange={(event) =>
                    updateSchema({ template: event.target.value })
                  }
                />
                <p className="text-xs text-muted-foreground">
                  Use Jinja refs like {"{{ customer_review }}"} in values.
                </p>
              </div>
            </div>
          )}
        </div>
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
