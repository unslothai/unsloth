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
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useEffect, useId, useState } from "react";
import {
  closeModelAnnotationEditor,
  loadModelAnnotations,
  saveModelAnnotation,
  useModelAnnotationsStore,
  type ModelAnnotationTarget,
} from "../stores/model-annotations-store";

/** Hydrate only the model On Device view, report read failures, and host its single persistent editor. */
export function ModelAnnotationTools({ enabled }: { enabled: boolean }) {
  const loaded = useModelAnnotationsStore((state) => state.loaded);
  const error = useModelAnnotationsStore((state) => state.error);
  const editing = useModelAnnotationsStore((state) => state.editing);
  useEffect(() => {
    if (enabled && !loaded && !error) void loadModelAnnotations();
  }, [enabled, loaded, error]);
  if (!enabled) return null;
  return (
    <>
      {error && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 px-4 py-2 text-ui-12 text-destructive"
        >
          <span>{error}</span>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              void loadModelAnnotations();
            }}
          >
            Retry
          </Button>
        </div>
      )}
      {editing && <ModelAnnotationEditor key={editing.id} target={editing} />}
    </>
  );
}

/** Edit a snapshot of one row's details; failed saves keep the draft open and unchanged. */
function ModelAnnotationEditor({ target }: { target: ModelAnnotationTarget }) {
  const id = useId();
  const [name, setName] = useState(target.annotation?.display_name ?? "");
  const [labels, setLabels] = useState(
    target.annotation?.labels.join(", ") ?? "",
  );
  const [notes, setNotes] = useState(target.annotation?.notes ?? "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  /** Submit personal fields only; the inventory id remains the immutable update key. */
  async function save(): Promise<void> {
    setSaving(true);
    setError(null);
    try {
      await saveModelAnnotation(target.id, {
        display_name: name,
        labels: labels
          .split(",")
          .map((label) => label.trim())
          .filter(Boolean),
        notes,
      });
    } catch (failure) {
      setError(
        failure instanceof Error
          ? failure.message
          : "Could not save model details",
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open && !saving) closeModelAnnotationEditor();
      }}
    >
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Edit model details</DialogTitle>
          <DialogDescription>
            Personal details for {target.title}. Saved to your Studio account.
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault();
            void save();
          }}
        >
          <div className="space-y-2">
            <Label htmlFor={id + "-name"}>Display name</Label>
            <Input
              id={id + "-name"}
              value={name}
              maxLength={160}
              disabled={saving}
              onChange={(event) => setName(event.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={id + "-labels"}>Labels, separated by commas</Label>
            <Input
              id={id + "-labels"}
              value={labels}
              disabled={saving}
              onChange={(event) => setLabels(event.target.value)}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor={id + "-notes"}>Notes</Label>
            <Textarea
              id={id + "-notes"}
              value={notes}
              maxLength={8000}
              rows={5}
              disabled={saving}
              onChange={(event) => setNotes(event.target.value)}
            />
          </div>
          {error && (
            <p role="alert" className="text-ui-12 text-destructive">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button
              type="button"
              variant="ghost"
              disabled={saving}
              onClick={() => {
                setName("");
                setLabels("");
                setNotes("");
              }}
            >
              Clear fields
            </Button>
            <Button
              type="button"
              variant="outline"
              disabled={saving}
              onClick={closeModelAnnotationEditor}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={saving}>
              {saving ? "Saving…" : "Save details"}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
