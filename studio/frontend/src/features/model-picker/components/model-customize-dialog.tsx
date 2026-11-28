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
import { Input } from "@/components/ui/input";
import {
  MODEL_CUSTOM_NAME_MAX,
  imageFileToIconDataUrl,
  useModelCustomization,
  useModelCustomizationsStore,
} from "@/features/chat";
import { useRef, useState } from "react";

export function ModelCustomizeDialog({
  modelId,
  defaultName,
  onClose,
}: {
  modelId: string;
  defaultName: string;
  onClose: () => void;
}) {
  return (
    <Dialog open={true} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-md">
        <ModelCustomizeForm
          modelId={modelId}
          defaultName={defaultName}
          onClose={onClose}
        />
      </DialogContent>
    </Dialog>
  );
}

function ModelCustomizeForm({
  modelId,
  defaultName,
  onClose,
}: {
  modelId: string;
  defaultName: string;
  onClose: () => void;
}) {
  const saved = useModelCustomization(modelId);
  const setCustomization = useModelCustomizationsStore(
    (state) => state.setCustomization,
  );
  const [name, setName] = useState(saved?.name ?? "");
  const [icon, setIcon] = useState(saved?.icon);
  const [error, setError] = useState<string | null>(null);
  const [decoding, setDecoding] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  // Only the newest pick may land: a slow decode must not undo a later pick or Remove.
  const pick = useRef(0);

  const chooseIcon = async (file: File | undefined) => {
    if (!file) return;
    const current = ++pick.current;
    setError(null);
    setDecoding(true);
    try {
      const next = await imageFileToIconDataUrl(file);
      if (current === pick.current) setIcon(next);
    } catch (reason) {
      if (current === pick.current) {
        setError(
          reason instanceof Error
            ? reason.message
            : "Could not read this image.",
        );
      }
    } finally {
      if (current === pick.current) setDecoding(false);
    }
  };

  return (
    <form
      className="grid gap-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (decoding) return;
        setCustomization(modelId, { name, icon });
        onClose();
      }}
    >
      <DialogHeader>
        <DialogTitle className="text-ui-21">Customize model</DialogTitle>
        <DialogDescription>
          Only changes how {defaultName} is shown to you.
        </DialogDescription>
      </DialogHeader>
      <div className="flex items-center gap-3">
        <span className="grid size-12 shrink-0 place-items-center overflow-hidden rounded-lg bg-muted">
          {icon ? (
            <img src={icon} alt="" className="size-full object-cover" />
          ) : null}
        </span>
        <div className="flex flex-wrap gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => fileInput.current?.click()}
          >
            {icon ? "Change icon" : "Upload icon"}
          </Button>
          {icon ? (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                pick.current += 1;
                setDecoding(false);
                setError(null);
                setIcon(undefined);
              }}
            >
              Remove icon
            </Button>
          ) : null}
        </div>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => {
            void chooseIcon(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </div>
      <label className="grid gap-1.5 text-ui-13" htmlFor="model-custom-name">
        <span className="text-muted-foreground">Display name</span>
        <Input
          id="model-custom-name"
          value={name}
          maxLength={MODEL_CUSTOM_NAME_MAX}
          onChange={(event) => setName(event.target.value)}
          placeholder={defaultName}
        />
      </label>
      {error ? (
        <p role="alert" className="text-ui-11p5 leading-snug text-destructive">
          {error}
        </p>
      ) : null}
      <DialogFooter className="flex-wrap gap-2 sm:justify-end">
        <Button type="button" variant="ghost" onClick={onClose}>
          Cancel
        </Button>
        <Button type="submit" disabled={decoding}>
          Save
        </Button>
      </DialogFooter>
    </form>
  );
}
