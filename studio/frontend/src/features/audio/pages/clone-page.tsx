// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { toast } from "@/lib/toast";
import { FloppyDiskIcon, SpeechToTextIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useState } from "react";
import type { AudioGalleryClip } from "../api";
import { audioCppModelFor, audioCppWorkflowsFor } from "../audio-cpp-catalog";
import type { AudioSourceSelection } from "../audio-run-request";
import { macTtsCatalogChoiceIsRunnable } from "../catalog";
import { CLONE_TEXT_EXAMPLES } from "../clone-policy";
import {
  AudioHistoryProvider,
  AudioSourceInput,
} from "../components/audio-source-input";
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
import { TtsFooter, TtsOutput, TtsRailFields } from "./tts-workspace";

// Downloaded first-choice clone models, in the order the picker offers them.
const CLONE_MODEL_ORDER = [
  "Qwen3-TTS-12Hz-0.6B-Base-GGUF",
  "VoxCPM2-GGUF",
  "Chatterbox-GGUF",
];

function cloneRank(id: string): number {
  const index = CLONE_MODEL_ORDER.findIndex((name) => id.endsWith(`/${name}`));
  return index === -1 ? CLONE_MODEL_ORDER.length : index;
}

/** Clone's picker rows: catalog models that can clone, the recommended ones first. */
export function clonePageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models
    .filter((model) => {
      const entry = audioCppModelFor(model.id);
      return (
        entry !== null &&
        audioCppWorkflowsFor(entry).includes("clone") &&
        (!isMac || macTtsCatalogChoiceIsRunnable(model.id))
      );
    })
    .map((model, index) => ({ model, index }))
    .sort(
      (a, b) =>
        cloneRank(a.model.id) - cloneRank(b.model.id) || a.index - b.index,
    )
    .map(({ model }) => model);
}

type RailProps = Omit<
  ComponentProps<typeof TtsRailFields>,
  | "musicGeneration"
  | "prompt"
  | "setPrompt"
  | "toolPanels"
  | "inputs"
  | "claimedOptions"
>;

/** Picking a source fills in what it already knows: a history clip's text, a voice's transcript
 *  and language. Another clip says other words, so its text replaces whatever was there. */
function adoptReference(next: AudioSourceSelection | null) {
  const store = useAudioCloneStore.getState();
  const previous = store.reference;
  store.setReference(next);
  if (!next) return;
  if (previous?.kind !== next.kind || previous?.id !== next.id) {
    store.setReferenceText(next.transcript ?? "");
  }
  if (next.language && !store.language) store.setLanguage(next.language);
}

/** Clone's own inputs: the reference, what it says, and the text to speak. */
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
          value={reference}
          onChange={adoptReference}
          disabled={disabled}
          handleRef={clone.referenceHandle}
          onStatusChange={clone.setReferenceStatus}
        />
      </div>

      {transcriptField === "hidden" ? null : (
        <div className="grid gap-1.5" data-tour="audio-clone-transcript">
          <div className="flex items-center justify-between gap-2">
            <label
              htmlFor={CLONE_REFERENCE_TEXT_FIELD_ID}
              className="text-ui-13 font-medium text-foreground"
            >
              What's said in the clip
              <span className="ml-1.5 font-normal text-muted-foreground">
                {transcriptField === "required" ? "Required" : "Optional"}
              </span>
            </label>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="h-auto px-2 py-1 text-ui-11p5"
              disabled={
                !referenceUsable || transcriber.transcribing || disabled
              }
              onClick={() => void transcriber.transcribe(reference)}
            >
              <HugeiconsIcon icon={SpeechToTextIcon} className="size-3.5" />
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
              className="text-ui-11p5 leading-snug text-destructive"
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
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
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
        <label
          htmlFor={CLONE_TEXT_FIELD_ID}
          className="text-ui-13 font-medium text-foreground"
        >
          Text to speak
        </label>
        <Textarea
          id={CLONE_TEXT_FIELD_ID}
          value={clone.text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Type what the voice should say…"
          className="min-h-28"
        />
        {clone.text.trim() ? null : (
          <div
            className="flex flex-wrap gap-1.5"
            aria-label="Example sentences"
          >
            {CLONE_TEXT_EXAMPLES.map((example) => (
              <Button
                key={example}
                type="button"
                variant="outline"
                size="sm"
                className="h-auto max-w-full px-3 py-1 text-ui-11p5 font-normal"
                title={example}
                onClick={() => setText(example)}
              >
                <span className="truncate">{example}</span>
              </Button>
            ))}
          </div>
        )}
      </div>
      <LanguageSelect
        id="clone-language"
        label="Language"
        value={clone.language}
        onChange={setLanguage}
        hint="The language of the text to speak. Auto lets the model tell."
      />
    </AudioHistoryProvider>
  );
}

/** Clone's rail: its inputs, then the model's tools, the device and Advanced, as on Speak. */
export function CloneRail({
  clone,
  historyClips,
  ...props
}: RailProps & {
  clone: CloneGeneration;
  /** Gallery clips offered under From history. */
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

/** Generate, with Save voice… beside it for a reference worth keeping. */
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
  const [saving, setSaving] = useState(false);
  const save = useAudioVoicesStore((state) => state.save);
  const { reference } = clone;
  const canSave =
    reference !== null &&
    reference.kind !== "voice" &&
    !clone.referenceExpired &&
    clone.referenceStatus.phase !== "uploading" &&
    props.busy === null;
  return (
    <>
      <TtsFooter
        {...props}
        prompt={clone.text}
        lyricsOptional={false}
        musicNeedsDescription={false}
        audioInstructions=""
        handleGenerate={clone.handleGenerate}
        secondaryAction={
          reference?.kind === "voice" ? null : (
            <Button
              type="button"
              variant="outline"
              className="h-11 px-5"
              disabled={!canSave}
              onClick={() => setSaving(true)}
            >
              <HugeiconsIcon icon={FloppyDiskIcon} className="size-4" />
              Save voice…
            </Button>
          )
        }
      />
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

/** Clone's output: the selected clip, then Clone's history with the voice each one used. */
export function CloneOutput({
  modelReady,
  recommendedModels = [],
  onPickModel,
  ...props
}: Omit<
  ComponentProps<typeof TtsOutput>,
  "emptyText" | "clipBadge" | "emptyActions"
> & {
  /** Whether a clone model is loaded, so the empty copy asks only for what is missing. */
  modelReady: boolean;
  /** Models offered as one-click picks while none that clones is loaded. */
  recommendedModels?: readonly ModelOption[];
  onPickModel?: (id: string) => void;
}) {
  const picks = modelReady || !onPickModel ? [] : recommendedModels.slice(0, 3);
  return (
    <TtsOutput
      {...props}
      clipBadge={(clip) => clip.reference_name ?? null}
      emptyActions={
        picks.length > 0 ? (
          <div
            className="flex flex-wrap justify-center gap-2"
            aria-label="Recommended models"
          >
            {picks.map((model) => (
              <Button
                key={model.id}
                type="button"
                variant="outline"
                size="sm"
                onClick={() => onPickModel?.(model.id)}
              >
                Use {model.name}
              </Button>
            ))}
          </div>
        ) : null
      }
      emptyText={
        modelReady
          ? "Cloned speech lands here. Add a reference, type the text, and press Generate."
          : "Cloned speech lands here. Load a model that can clone, add a reference, and press Generate."
      }
    />
  );
}
