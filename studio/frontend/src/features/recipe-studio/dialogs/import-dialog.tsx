// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { type ReactElement, useState } from "react";
import { FieldLabel } from "./shared/field-label";

type ImportDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImport: (value: string) => string | null;
  container?: HTMLDivElement | null;
};

export function ImportDialog({
  open,
  onOpenChange,
  onImport,
  container,
}: ImportDialogProps): ReactElement {
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);
  const payloadId = "recipe-import-payload";
  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      setValue("");
      setError(null);
    }
    onOpenChange(nextOpen);
  };

  const handleImport = () => {
    const message = onImport(value);
    if (message) {
      setError(message);
      return;
    }
    handleOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent
        container={container}
        position="absolute"
        overlayPosition="absolute"
        overlayClassName="bg-transparent"
        className="corner-squircle max-h-[650px] overflow-auto sm:max-w-2xl shadow-border"
      >
        <DialogHeader>
          <DialogTitle>Import recipe</DialogTitle>
        </DialogHeader>
        <div className="grid gap-2">
          <FieldLabel
            label="JSON payload"
            htmlFor={payloadId}
            hint="Paste exported recipe payload JSON."
          />
          <Textarea
            id={payloadId}
            className="corner-squircle nodrag min-h-[220px] max-h-[450px]"
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
