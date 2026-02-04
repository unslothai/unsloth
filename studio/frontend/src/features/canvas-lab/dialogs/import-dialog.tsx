import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useEffect, useState } from "react";

type ImportDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImport: (value: string) => string | null;
};

export function ImportDialog({
  open,
  onOpenChange,
  onImport,
}: ImportDialogProps): ReactElement {
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const payloadId = "canvas-import-payload";

  useEffect(() => {
    if (!open) {
      setValue("");
      setError(null);
    }
  }, [open]);

  const handleImport = () => {
    const message = onImport(value);
    if (message) {
      setError(message);
      return;
    }
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Import recipe</DialogTitle>
        </DialogHeader>
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor={payloadId}
          >
            JSON payload
          </label>
          <Textarea
            id={payloadId}
            className="nodrag min-h-[220px]"
            placeholder='{"recipe": { "columns": [] }}'
            value={value}
            onChange={(event) => setValue(event.target.value)}
          />
          {error && (
            <p className="text-xs text-rose-600" role="alert">
              {error}
            </p>
          )}
        </div>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={handleImport}>
            Import
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
