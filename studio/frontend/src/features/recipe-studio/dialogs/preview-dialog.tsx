import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { type ReactElement } from "react";

type PreviewDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  rows: number;
  onRowsChange: (rows: number) => void;
  loading: boolean;
  errors: string[];
  summary: {
    totalColumns: number;
    llmColumns: number;
    samplerColumns: number;
    expressionColumns: number;
    toolConfigs: number;
    mcpProviders: number;
  };
  onPreview: () => void;
  container?: HTMLDivElement | null;
};

export function PreviewDialog({
  open,
  onOpenChange,
  rows,
  onRowsChange,
  loading,
  errors,
  summary,
  onPreview,
  container,
}: PreviewDialogProps): ReactElement {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle sm:max-w-md"
      >
        <DialogHeader>
          <DialogTitle>Preview data</DialogTitle>
        </DialogHeader>
        <div className="grid grid-cols-2 gap-2">
          <div className="rounded-xl border border-border/60 p-2">
            <p className="text-[11px] uppercase text-muted-foreground">Columns</p>
            <p className="text-sm font-semibold">{summary.totalColumns}</p>
          </div>
          <div className="rounded-xl border border-border/60 p-2">
            <p className="text-[11px] uppercase text-muted-foreground">LLM</p>
            <p className="text-sm font-semibold">{summary.llmColumns}</p>
          </div>
          <div className="rounded-xl border border-border/60 p-2">
            <p className="text-[11px] uppercase text-muted-foreground">Samplers</p>
            <p className="text-sm font-semibold">{summary.samplerColumns}</p>
          </div>
          <div className="rounded-xl border border-border/60 p-2">
            <p className="text-[11px] uppercase text-muted-foreground">Expressions</p>
            <p className="text-sm font-semibold">{summary.expressionColumns}</p>
          </div>
          <div className="rounded-xl border border-border/60 p-2">
            <p className="text-[11px] uppercase text-muted-foreground">Tool configs</p>
            <p className="text-sm font-semibold">{summary.toolConfigs}</p>
          </div>
          <div className="rounded-xl border border-border/60 p-2">
            <p className="text-[11px] uppercase text-muted-foreground">MCP servers</p>
            <p className="text-sm font-semibold">{summary.mcpProviders}</p>
          </div>
        </div>
        <div className="grid gap-2">
          <label
            className="text-xs font-semibold uppercase text-muted-foreground"
            htmlFor="preview-rows"
          >
            Number of records
          </label>
          <Input
            id="preview-rows"
            type="number"
            min={1}
            max={1000}
            value={String(rows)}
            onChange={(event) => {
              const parsed = Number(event.target.value);
              if (Number.isFinite(parsed) && parsed > 0) {
                onRowsChange(Math.min(1000, Math.floor(parsed)));
              }
            }}
          />
        </div>
        {errors.length > 0 && (
          <div className="max-h-44 space-y-1 overflow-y-auto rounded-xl border border-destructive/30 bg-destructive/5 p-3">
            <p className="text-xs font-semibold uppercase text-destructive">
              Validation errors
            </p>
            {errors.map((error) => (
              <p key={error} className="text-xs text-destructive">
                {error}
              </p>
            ))}
          </div>
        )}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={loading}
          >
            Cancel
          </Button>
          <Button type="button" onClick={onPreview} disabled={loading}>
            {loading ? "Running..." : "Run preview"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
