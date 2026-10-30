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
import { Textarea } from "@/components/ui/textarea";
import { useEffect, useState } from "react";
import { useAudioVoicesStore } from "../stores/audio-voices-store";
import { Field } from "./field";
import { LanguageSelect } from "./language-select";

export interface VoiceDetails {
  name: string;
  transcript: string;
  language: string;
}

export function SaveVoiceDialog({
  open,
  onOpenChange,
  mode,
  initial,
  voiceId,
  onSubmit,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  mode: "create" | "edit";
  initial: VoiceDetails;
  voiceId?: string;
  onSubmit: (details: VoiceDetails) => Promise<void>;
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="corner-squircle dialog-soft-surface gap-5 sm:max-w-md">
        {/* Content unmounts on close, so each open starts from its own values. */}
        <SaveVoiceForm
          mode={mode}
          initial={initial}
          voiceId={voiceId}
          onCancel={() => onOpenChange(false)}
          onSubmit={async (details) => {
            await onSubmit(details);
            onOpenChange(false);
          }}
        />
      </DialogContent>
    </Dialog>
  );
}

function SaveVoiceForm({
  mode,
  initial,
  voiceId,
  onCancel,
  onSubmit,
}: {
  mode: "create" | "edit";
  initial: VoiceDetails;
  voiceId?: string;
  onCancel: () => void;
  onSubmit: (details: VoiceDetails) => Promise<void>;
}) {
  const [name, setName] = useState(initial.name.slice(0, 80));
  const [transcript, setTranscript] = useState(initial.transcript);
  const [language, setLanguage] = useState(initial.language);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const clean = name.trim();
  // Voices are picked by name, so names must be unique; the list may not be loaded yet.
  useEffect(() => {
    const voices = useAudioVoicesStore.getState();
    if (!voices.loaded) void voices.refresh();
  }, []);
  const taken = useAudioVoicesStore((state) =>
    state.voices.some(
      (voice) =>
        voice.id !== voiceId &&
        voice.name.trim().toLowerCase() === clean.toLowerCase(),
    ),
  );

  const submit = async () => {
    if (!clean || taken || saving) return;
    setSaving(true);
    setError(null);
    try {
      await onSubmit({ name: clean, transcript: transcript.trim(), language });
    } catch (reason) {
      setError(
        reason instanceof Error ? reason.message : "Could not save the voice.",
      );
      setSaving(false);
    }
  };

  return (
    <form
      className="grid gap-5"
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <DialogHeader>
        <DialogTitle className="text-ui-21">
          {mode === "create" ? "Save voice" : "Edit voice"}
        </DialogTitle>
        <DialogDescription>
          {mode === "create"
            ? "Keep this reference to use again on Clone and Speak. Its first 30 seconds are saved."
            : "Change how this voice is listed and what its clip says."}
        </DialogDescription>
      </DialogHeader>
      <Field label="Name" htmlFor="save-voice-name">
        <Input
          id="save-voice-name"
          value={name}
          maxLength={80}
          // biome-ignore lint/a11y/noAutofocus: the dialog opens to name the voice.
          autoFocus={true}
          onFocus={(event) => event.currentTarget.select()}
          onChange={(event) => setName(event.target.value)}
          placeholder="Narrator"
        />
      </Field>
      <Field
        label="What's said in the clip"
        htmlFor="save-voice-transcript"
        hint="Optional. Saves transcribing the clip again next time."
      >
        <Textarea
          id="save-voice-transcript"
          value={transcript}
          maxLength={4000}
          onChange={(event) => setTranscript(event.target.value)}
          className="min-h-20"
        />
      </Field>
      <LanguageSelect
        id="save-voice-language"
        label="Language"
        value={language}
        onChange={setLanguage}
        emptyLabel="Not set"
      />
      {taken ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          You already have a voice named {clean}. Pick another name.
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="text-ui-11p5 leading-snug text-destructive">
          {error}
        </p>
      ) : null}
      <DialogFooter className="flex-wrap gap-2 sm:justify-end">
        <Button type="button" variant="ghost" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" disabled={!clean || taken || saving}>
          {saving ? "Saving…" : mode === "create" ? "Save voice" : "Save"}
        </Button>
      </DialogFooter>
    </form>
  );
}
