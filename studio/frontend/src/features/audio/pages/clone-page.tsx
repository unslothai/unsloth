// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { toast } from "@/lib/toast";
import { ClosedCaptionIcon, FloppyDiskIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useState } from "react";
import type { AudioGalleryClip } from "../api";
import type { AudioSourceSelection } from "../audio-run-request";
import { SPEECH_TEXT_EXAMPLES } from "../audio-workspace-constants";
import { PromptExamples } from "../components/prompt-examples";
import {
  AudioHistoryProvider,
  AudioSourceInput,
} from "../components/audio-source-input";
import { FieldLabel, FieldTag } from "../components/field";
import { LanguageSelect } from "../components/language-select";
import { SaveVoiceDialog } from "../components/save-voice-dialog";
import {
  CLONE_REFERENCE_TEXT_FIELD_ID,
  CLONE_TEXT_FIELD_ID,
  type CloneGeneration,
} from "../hooks/use-clone-generation";
import { useAudioCloneStore } from "../stores/audio-clone-store";
import { useAudioVoicesStore } from "../stores/audio-voices-store";
import { AudioToolPanels } from "../tools/tool-panel-host";
import { RecommendedModelPicks } from "../components/recommended-model-picks";
import { TtsFooter, TtsOutput, TtsRailFields } from "./tts-workspace";

type RailProps = Omit<
  ComponentProps<typeof TtsRailFields>,
  | "musicGeneration"
  | "prompt"
  | "setPrompt"
  | "toolPanels"
  | "inputs"
  | "claimedOptions"
>;

export function adoptReference(next: AudioSourceSelection | null) {
  useAudioCloneStore.getState().adoptReference(next);
}

function CloneInputs({
  clone,
  historyClips,
  disabled,
}: {
  clone: CloneGeneration;
  historyClips: readonly AudioGalleryClip[];
  disabled: boolean;
}) {
  const setReferenceText = useAudioCloneStore(
    (state) => state.setReferenceText,
  );
  const setText = useAudioCloneStore((state) => state.setText);
  const setLanguage = useAudioCloneStore((state) => state.setLanguage);
  const { transcriber, transcriptField, reference } = clone;
  const referenceUsable =
    reference !== null &&
    !clone.referenceExpired &&
    clone.referenceStatus.phase !== "uploading";
  return (
    <AudioHistoryProvider value={historyClips}>
      <div data-tour="audio-clone-reference">
        <AudioSourceInput
          id="clone-reference"
          label="Reference"
          hint="5 to 15 s of one clear voice works best."
          idealSeconds={[5, 15]}
          value={reference}
          onChange={adoptReference}
          disabled={disabled}
          handleRef={clone.referenceHandle}
          onStatusChange={clone.setReferenceStatus}
          headerAction={<SaveReferenceVoice clone={clone} disabled={disabled} />}
        />
      </div>

      {transcriptField === "hidden" ? null : (
        <div className="grid gap-2" data-tour="audio-clone-transcript">
          <div className="flex items-center justify-between gap-2">
            <FieldLabel
              htmlFor={CLONE_REFERENCE_TEXT_FIELD_ID}
              className="text-ui-13 font-medium text-foreground"
            >
              What's said in the clip
              <FieldTag>
                {transcriptField === "required" ? "Required" : "Optional"}
              </FieldTag>
            </FieldLabel>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-auto gap-1 p-0 text-ui-12 text-muted-foreground hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"
              disabled={
                !referenceUsable || transcriber.transcribing || disabled
              }
              onClick={() => void transcriber.transcribe(reference)}
            >
              <HugeiconsIcon icon={ClosedCaptionIcon} className="size-[1em]" />
              {transcriber.transcribing ? "Transcribing…" : "Transcribe"}
            </Button>
          </div>
          <Textarea
            id={CLONE_REFERENCE_TEXT_FIELD_ID}
            value={clone.referenceText}
            disabled={transcriber.transcribing}
            aria-busy={transcriber.transcribing}
            onChange={(event) => setReferenceText(event.target.value)}
            placeholder="Type the words in the clip exactly, or press Transcribe."
            className="min-h-16"
          />
          {transcriber.error ? (
            <p
              role="alert"
              className="text-ui-11p5/snug text-destructive"
            >
              {transcriber.error}{" "}
              <Button
                type="button"
                variant="link"
                className="h-auto p-0 text-ui-11p5 font-medium text-foreground"
                onClick={() => void transcriber.transcribe(reference)}
              >
                Try again
              </Button>
            </p>
          ) : (
            <p className="text-ui-12/normal text-muted-foreground">
              {transcriptField === "required"
                ? "This model matches the voice using the clip's words."
                : "Helps the model match the voice."}
            </p>
          )}
        </div>
      )}

      <div
        className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-1.5"
        data-tour="audio-clone-text"
      >
        <FieldLabel
          htmlFor={CLONE_TEXT_FIELD_ID}
          className="text-ui-13 font-medium text-foreground"
        >
          Text to speak
        </FieldLabel>
        <Textarea
          data-type-to-activate="prompt"
          id={CLONE_TEXT_FIELD_ID}
          value={clone.text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Type what the voice should say…"
          className="min-h-28"
        />
        <PromptExamples
          label="Example sentences"
          examples={SPEECH_TEXT_EXAMPLES}
          empty={!clone.text.trim()}
          onPick={setText}
        />
      </div>
      {clone.languageShown ? (
        <LanguageSelect
          id="clone-language"
          label="Language"
          value={clone.language}
          onChange={setLanguage}
          hint="The language of the text to speak. Auto lets the model detect it."
        />
      ) : null}
    </AudioHistoryProvider>
  );
}

export function CloneRail({
  clone,
  historyClips,
  ...props
}: RailProps & {
  clone: CloneGeneration;
  historyClips: readonly AudioGalleryClip[];
}) {
  const disabled = props.busy === "generating";
  return (
    <TtsRailFields
      {...props}
      musicGeneration={false}
      prompt={clone.text}
      setPrompt={useAudioCloneStore.getState().setText}
      claimedOptions={clone.claimedOptions}
      inputs={
        <CloneInputs
          clone={clone}
          historyClips={historyClips}
          disabled={disabled}
        />
      }
      toolPanels={
        <AudioHistoryProvider value={historyClips}>
          <AudioToolPanels
            workflow="clone"
            ctx={clone.toolContext}
            values={clone.toolValues}
            onChange={clone.handleToolValueChange}
            specs={props.audioOptionSpecs}
            disabled={disabled}
            core={{
              text: clone.text,
              referenceText: clone.referenceText,
              hasReference: clone.reference !== null,
            }}
          />
        </AudioHistoryProvider>
      }
    />
  );
}

export function CloneFooter({
  clone,
  ...props
}: Omit<
  ComponentProps<typeof TtsFooter>,
  | "prompt"
  | "lyricsOptional"
  | "musicNeedsDescription"
  | "audioInstructions"
  | "handleGenerate"
  | "secondaryAction"
> & { clone: CloneGeneration }) {
  return (
    <TtsFooter
      {...props}
      prompt={clone.text}
      lyricsOptional={false}
      musicNeedsDescription={false}
      audioInstructions=""
      handleGenerate={clone.handleGenerate}
    />
  );
}

function SaveReferenceVoice({
  clone,
  disabled,
}: {
  clone: CloneGeneration;
  disabled: boolean;
}) {
  const [saving, setSaving] = useState(false);
  const save = useAudioVoicesStore((state) => state.save);
  const { reference } = clone;
  if (reference === null || reference.kind === "voice") return null;
  const canSave =
    !clone.referenceExpired &&
    clone.referenceStatus.phase !== "uploading" &&
    !disabled;
  return (
    <>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className="h-auto gap-1 p-0 text-ui-12 text-muted-foreground hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"
        disabled={!canSave}
        onClick={() => setSaving(true)}
      >
        <HugeiconsIcon icon={FloppyDiskIcon} className="size-3" />
        Save voice
      </Button>
      <SaveVoiceDialog
        open={saving}
        onOpenChange={setSaving}
        mode="create"
        initial={{
          name: (reference?.name ?? "").replace(/\.[a-z0-9]{2,4}$/i, ""),
          transcript: clone.referenceText,
          language: clone.language,
        }}
        onSubmit={async (details) => {
          if (!reference || reference.kind === "voice") return;
          const voice = await save({
            source:
              reference.kind === "input"
                ? { input_id: reference.id }
                : { clip_id: reference.id },
            name: details.name,
            transcript: details.transcript || null,
            language: details.language || null,
          });
          toast.success(
            `Saved ${voice.name}. Pick it under Saved voice next time.`,
          );
        }}
      />
    </>
  );
}

export function CloneOutput({
  modelReady,
  recommendedModels = [],
  onPickModel,
  ...props
}: Omit<
  ComponentProps<typeof TtsOutput>,
  "emptyText" | "clipBadge" | "emptyActions"
> & {
  modelReady: boolean;
  recommendedModels?: readonly ModelOption[];
  onPickModel?: (id: string) => void;
}) {
  return (
    <TtsOutput
      {...props}
      clipBadge={(clip, place) =>
        place === "history" ? (clip.reference_name ?? null) : null
      }
      emptyActions={
        modelReady ? null : (
          <RecommendedModelPicks models={recommendedModels} onPick={onPickModel} />
        )
      }
      emptyTitle="Cloned speech will appear here"
      emptyText={
        modelReady
          ? "Add a reference recording, type the text, and press Generate."
          : "Load a model that can clone a voice, add a reference recording, and press Generate."
      }
    />
  );
}
