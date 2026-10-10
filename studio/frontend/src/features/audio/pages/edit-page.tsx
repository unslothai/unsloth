// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { SpeechToTextIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useEffect, useState } from "react";
import { type AudioGalleryClip, fetchAudioBlob } from "../api";
import { audioCppModelFor, audioCppWorkflowsFor } from "../audio-cpp-catalog";
import { EDIT_SOURCE_MAX_SECONDS } from "../audio-run-request";
import { macTtsCatalogChoiceIsRunnable } from "../catalog";
import { ABCompare } from "../components/ab-compare";
import {
  AudioHistoryProvider,
  AudioSourceInput,
} from "../components/audio-source-input";
import { TranscriptDiffEditor } from "../components/transcript-diff-editor";
import { EDIT_COPY } from "../edit-policy";
import {
  EDIT_CHANGES_FIELD_ID,
  EDIT_TRANSCRIPT_FIELD_ID,
  type EditGeneration,
  adoptEditSource,
} from "../hooks/use-edit-generation";
import { useAudioEditStore } from "../stores/audio-edit-store";
import { AudioToolPanels } from "../tools/tool-panel-host";
import { TtsFooter, TtsOutput, TtsRailFields } from "./tts-workspace";

const EDIT_MODEL_ORDER = [
  "DotTTS-Edit-GGUF",
  "Vevo2-GGUF",
  "FireRedAudio-GGUF",
];

function editRank(id: string): number {
  const index = EDIT_MODEL_ORDER.findIndex((name) => id.endsWith(`/${name}`));
  return index === -1 ? EDIT_MODEL_ORDER.length : index;
}

export function editPageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models
    .filter((model) => {
      const entry = audioCppModelFor(model.id);
      return (
        entry !== null &&
        audioCppWorkflowsFor(entry).includes("edit") &&
        (!isMac || macTtsCatalogChoiceIsRunnable(model.id))
      );
    })
    .map((model, index) => ({ model, index }))
    .sort(
      (a, b) =>
        editRank(a.model.id) - editRank(b.model.id) || a.index - b.index,
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

/** Selects ②'s last word so typing replaces it. */
function selectLastWord(field: HTMLTextAreaElement | null) {
  if (!field) return;
  const match = /([\p{L}\p{N}'’-]+)[^\p{L}\p{N}]*$/u.exec(field.value);
  field.focus();
  if (match)
    field.setSelectionRange(match.index, match.index + match[1].length);
}

function EditInputs({
  edit,
  historyClips,
  disabled,
}: {
  edit: EditGeneration;
  historyClips: readonly AudioGalleryClip[];
  disabled: boolean;
}) {
  const setTranscript = useAudioEditStore((state) => state.setTranscript);
  const setEdited = useAudioEditStore((state) => state.setEdited);
  const resetEdited = useAudioEditStore((state) => state.resetEdited);
  const setMode = useAudioEditStore((state) => state.setMode);
  const setDelivery = useAudioEditStore((state) => state.setDelivery);
  const { transcriber, source, adapter, mode, delivery } = edit;
  const sourceUsable =
    source !== null &&
    !edit.sourceExpired &&
    edit.sourceStatus.phase !== "uploading";
  const deliveryRange = adapter?.delivery ?? null;
  return (
    <AudioHistoryProvider value={historyClips}>
      <div data-tour="audio-edit-recording">
        <AudioSourceInput
          id="edit-recording"
          label={EDIT_COPY.recordingLabel}
          hint={EDIT_COPY.recordingHint}
          value={source}
          onChange={adoptEditSource}
          disabled={disabled}
          allowSavedVoice={false}
          usesFirstSeconds={null}
          maxRecordSeconds={EDIT_SOURCE_MAX_SECONDS}
          handleRef={edit.sourceHandle}
          onStatusChange={edit.setSourceStatus}
        />
      </div>

      <div className="grid gap-1.5" data-tour="audio-edit-transcript">
        <div className="flex items-center justify-between gap-2">
          <label
            htmlFor={EDIT_TRANSCRIPT_FIELD_ID}
            className="text-ui-13 font-medium text-foreground"
          >
            {EDIT_COPY.transcriptLabel}
            {mode === "delivery" ? (
              <span className="ml-1.5 font-normal text-muted-foreground">
                Optional
              </span>
            ) : null}
          </label>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="h-auto px-2 py-1 text-ui-11p5"
            disabled={!sourceUsable || transcriber.transcribing || disabled}
            onClick={() => void transcriber.transcribe(source)}
          >
            <HugeiconsIcon icon={SpeechToTextIcon} className="size-3.5" />
            {transcriber.transcribing ? "Transcribing…" : "Transcribe"}
          </Button>
        </div>
        <Textarea
          id={EDIT_TRANSCRIPT_FIELD_ID}
          value={edit.transcript}
          disabled={transcriber.transcribing || disabled}
          aria-busy={transcriber.transcribing}
          onChange={(event) =>
            setTranscript(event.target.value, source?.id ?? null)
          }
          placeholder={
            transcriber.transcribing
              ? "Listening to the recording…"
              : "What the recording says. It fills in once you add one."
          }
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
              onClick={() => void transcriber.transcribe(source)}
            >
              Try again
            </Button>
          </p>
        ) : (
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {EDIT_COPY.transcriptHint}
          </p>
        )}
      </div>

      {deliveryRange ? (
        <PillTabs
          ariaLabel="What to change"
          value={mode}
          onValueChange={(next) => setMode(next as "words" | "delivery")}
          disabled={disabled}
          fit={true}
          compact={true}
          className="[&>button]:px-3"
          tabs={[
            { value: "words", label: EDIT_COPY.wordsTab },
            { value: "delivery", label: EDIT_COPY.deliveryTab },
          ]}
        />
      ) : null}

      {mode === "words" ? (
        <div className="grid min-w-0 gap-1.5">
          <TranscriptDiffEditor
            id={EDIT_CHANGES_FIELD_ID}
            value={edit.edited}
            onChange={setEdited}
            transcript={edit.transcript}
            onReset={resetEdited}
            disabled={disabled || !edit.transcript.trim()}
            textareaRef={edit.changesRef}
            dataTour="audio-edit-changes"
          />
          {edit.transcript.trim() && edit.edited === edit.transcript ? (
            <div className="flex flex-wrap gap-1.5" aria-label="Examples">
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="h-auto px-3 py-1 text-ui-11p5 font-normal"
                disabled={disabled}
                onClick={() => selectLastWord(edit.changesRef.current)}
              >
                Change one word
              </Button>
            </div>
          ) : null}
        </div>
      ) : deliveryRange ? (
        <div className="grid gap-3" data-tour="audio-edit-changes">
          <p className="text-ui-13 font-medium text-foreground">
            {EDIT_COPY.deliveryLabel}
          </p>
          <ParamSlider
            label={EDIT_COPY.speedLabel}
            value={delivery.speed}
            min={deliveryRange.speed[0]}
            max={deliveryRange.speed[1]}
            step={deliveryRange.speed[2]}
            displayValue={`${Number(delivery.speed.toFixed(2))}×`}
            disabled={disabled}
            onChange={(speed) => setDelivery({ speed })}
          />
          <ParamSlider
            label={EDIT_COPY.pitchLabel}
            value={delivery.pitchSteps}
            min={0}
            max={deliveryRange.pitchSteps[1]}
            step={1}
            displayValue={
              delivery.pitchSteps > 0 ? `+${delivery.pitchSteps} steps` : "Off"
            }
            disabled={disabled}
            onChange={(pitchSteps) => setDelivery({ pitchSteps })}
          />
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {EDIT_COPY.pitchHint}
          </p>
        </div>
      ) : null}
      {adapter?.family === "vevo2" ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {EDIT_COPY.vevo2SwitchNote}
        </p>
      ) : null}
    </AudioHistoryProvider>
  );
}

export function EditRail({
  edit,
  historyClips,
  ...props
}: RailProps & {
  edit: EditGeneration;
  historyClips: readonly AudioGalleryClip[];
}) {
  const disabled = props.busy === "generating";
  return (
    <TtsRailFields
      {...props}
      musicGeneration={false}
      prompt={edit.edited}
      setPrompt={useAudioEditStore.getState().setEdited}
      claimedOptions={edit.claimedOptions}
      inputs={
        <EditInputs
          edit={edit}
          historyClips={historyClips}
          disabled={disabled}
        />
      }
      toolPanels={
        <AudioToolPanels
          workflow="edit"
          ctx={edit.toolContext}
          values={edit.toolValues}
          onChange={edit.handleToolValueChange}
          specs={props.audioOptionSpecs}
          disabled={disabled}
          core={{ text: edit.edited, edit: edit.core }}
        />
      }
    />
  );
}

export function EditFooter({
  edit,
  ...props
}: Omit<
  ComponentProps<typeof TtsFooter>,
  | "prompt"
  | "lyricsOptional"
  | "musicNeedsDescription"
  | "audioInstructions"
  | "handleGenerate"
> & { edit: EditGeneration }) {
  return (
    <TtsFooter
      {...props}
      prompt={edit.inputsReady ? "ready" : ""}
      lyricsOptional={false}
      musicNeedsDescription={false}
      audioInstructions=""
      handleGenerate={edit.handleGenerate}
    />
  );
}

function galleryFileUrl(id: string): string {
  return `/api/inference/audio/gallery/${encodeURIComponent(id)}/file`;
}

function EditComparePlayer({
  clip,
  src,
  focusRef,
}: {
  clip: AudioGalleryClip;
  src: string;
  focusRef: ((element: HTMLElement | null) => void) | undefined;
}) {
  const sourceId = clip.source_clip_id ?? null;
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  useEffect(() => {
    if (!sourceId) return;
    let url: string | null = null;
    let cancelled = false;
    setOriginalSrc(null);
    fetchAudioBlob(galleryFileUrl(sourceId))
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setOriginalSrc(url);
      })
      .catch(() => {
        // Original deleted: the compare still plays the edit.
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [sourceId]);
  return (
    <ABCompare
      original={{
        src: originalSrc,
        fileUrl: sourceId ? galleryFileUrl(sourceId) : null,
        label: "Original",
        durationS: null,
      }}
      edited={{
        src,
        fileUrl: galleryFileUrl(clip.id),
        label: "Edited",
        durationS: clip.duration_s ?? null,
      }}
      autoFocusRef={focusRef}
    />
  );
}

export function EditOutput({
  modelReady,
  recommendedModels = [],
  onPickModel,
  ...props
}: Omit<
  ComponentProps<typeof TtsOutput>,
  "emptyText" | "clipBadge" | "emptyActions" | "selectedPlayer"
> & {
  modelReady: boolean;
  recommendedModels?: readonly ModelOption[];
  onPickModel?: (id: string) => void;
}) {
  const picks = modelReady || !onPickModel ? [] : recommendedModels.slice(0, 3);
  return (
    <TtsOutput
      {...props}
      clipBadge={() => null}
      selectedPlayer={(clip, src, focusRef) =>
        clip.source_clip_id ? (
          <EditComparePlayer clip={clip} src={src} focusRef={focusRef} />
        ) : null
      }
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
          ? EDIT_COPY.emptyText
          : "Edited speech lands here. Load a model that can edit speech, add a recording, and press Generate."
      }
    />
  );
}
