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
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { useRef, useState } from "react";
import { validateChatTemplate } from "../api/templates";
import {
  MAX_CHAT_TEMPLATE_BYTES,
  chatTemplateByteLength,
  isChatTemplateWithinLimit,
} from "../model-config/per-model-config";

interface ChatTemplateEditorDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  value: string | null;
  defaultTemplate: string | null;
  defaultLoading: boolean;
  onSave: (override: string | null) => void;
  readOnly?: boolean;
}

export function ChatTemplateEditorDialog({
  open,
  onOpenChange,
  value,
  defaultTemplate,
  defaultLoading,
  onSave,
  readOnly = false,
}: ChatTemplateEditorDialogProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [validating, setValidating] = useState(false);
  // Bumped whenever the dialog closes so a validation still in flight cannot
  // apply a template the user has already dismissed.
  const validationToken = useRef(0);
  const renderedDraft = draft ?? value ?? defaultTemplate ?? "";

  const byteLength = chatTemplateByteLength(renderedDraft);
  const overLimit = !isChatTemplateWithinLimit(renderedDraft);
  const matchesDefault =
    defaultTemplate != null && renderedDraft === defaultTemplate;

  const handleClose = () => {
    validationToken.current += 1;
    setDraft(null);
    setError(null);
    setValidating(false);
    onOpenChange(false);
  };

  const handleSave = async () => {
    if (renderedDraft.trim().length === 0 || matchesDefault) {
      onSave(null);
      handleClose();
      return;
    }
    if (overLimit) {
      setError("Template exceeds the size limit.");
      return;
    }
    setValidating(true);
    const token = validationToken.current;
    try {
      const result = await validateChatTemplate(renderedDraft);
      // Dialog was closed (or reopened) while validating; drop the result so a
      // discarded template is never applied.
      if (token !== validationToken.current) {
        return;
      }
      if (!result.valid) {
        setError(result.error ?? "Invalid Jinja template.");
        return;
      }
      onSave(renderedDraft);
      handleClose();
    } catch {
      if (token === validationToken.current) {
        setError("Could not validate the template.");
      }
    } finally {
      if (token === validationToken.current) {
        setValidating(false);
      }
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(nextOpen) => {
        if (nextOpen) {
          onOpenChange(true);
          return;
        }
        handleClose();
      }}
    >
      <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle>
            {readOnly ? "Chat Template" : "Edit Chat Template"}
          </DialogTitle>
          <DialogDescription>
            {readOnly
              ? "This is the model's chat template. Custom templates apply to GGUF models for now, so it is view only for safetensors models."
              : "Override the model's chat template with custom Jinja. The change applies when the model loads. Saving an empty template or one that matches the default clears the override."}
          </DialogDescription>
        </DialogHeader>
        <Textarea
          value={renderedDraft}
          onChange={(event) => {
            if (readOnly) return;
            setDraft(event.target.value);
            setError(null);
          }}
          readOnly={readOnly}
          className="min-h-[20rem] max-h-[50vh] overflow-y-auto border-0 font-mono text-xs leading-5 corner-squircle focus-visible:ring-0"
          rows={14}
          spellCheck={false}
          placeholder={defaultLoading ? "Loading model default..." : ""}
        />
        {readOnly ? null : (
          <div className="flex items-center justify-between gap-3 px-0.5 text-ui-11">
            <span
              className={overLimit ? "text-amber-500" : "text-muted-foreground"}
            >
              {byteLength.toLocaleString()} /{" "}
              {MAX_CHAT_TEMPLATE_BYTES.toLocaleString()} bytes
            </span>
            {error ? (
              <span className="truncate text-red-500" title={error}>
                {error}
              </span>
            ) : null}
          </div>
        )}
        <DialogFooter className="flex-wrap gap-2 sm:justify-between">
          {readOnly ? (
            <div className="flex w-full justify-end">
              <Button type="button" onClick={handleClose}>
                Close
              </Button>
            </div>
          ) : (
            <>
              <Button
                type="button"
                variant="ghost"
                onClick={() => setDraft(defaultTemplate ?? "")}
                disabled={
                  defaultLoading || renderedDraft === (defaultTemplate ?? "")
                }
                className="text-muted-foreground"
              >
                {defaultLoading ? (
                  <Spinner className="size-3.5" />
                ) : (
                  "Reset to default"
                )}
              </Button>
              <div className="flex gap-2">
                <Button type="button" variant="ghost" onClick={handleClose}>
                  Cancel
                </Button>
                <Button
                  type="button"
                  onClick={handleSave}
                  disabled={validating || overLimit}
                >
                  {validating ? "Validating..." : "Save"}
                </Button>
              </div>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
