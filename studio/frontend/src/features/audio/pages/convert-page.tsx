// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { SpeechToTextIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useEffect, useState } from "react";
import { type AudioGalleryClip, fetchAudioBlob } from "../api";
import { audioModelLabel } from "../audio-workspace-utils";
import { audioCppModelFor, audioCppWorkflowsFor } from "../audio-cpp-catalog";
import { sourceFileUrl } from "../audio-run-request";
import { macTtsCatalogChoiceIsRunnable } from "../catalog";
import { ABCompare } from "../components/ab-compare";
import {
  AudioHistoryProvider,
  AudioSourceInput,
} from "../components/audio-source-input";
import { CONVERT_MODEL_ORDER } from "../convert-policy";
import {
  CONVERT_SOURCE_TEXT_FIELD_ID,
  type ConvertGeneration,
} from "../hooks/use-convert-generation";
import { useAudioConvertStore } from "../stores/audio-convert-store";
import { AudioToolPanels } from "../tools/tool-panel-host";
import { TtsFooter, TtsOutput, TtsRailFields } from "./tts-workspace";

function convertRank(id: string): number {
  const index = CONVERT_MODEL_ORDER.findIndex((name) =>
    id.endsWith(`/${name}`),
  );
  return index === -1 ? CONVERT_MODEL_ORDER.length : index;
}

/** Convert's picker rows: catalog models that can convert a voice, the recommended ones first. */
export function convertPageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models
    .filter((model) => {
      const entry = audioCppModelFor(model.id);
      return (
        entry !== null &&
        audioCppWorkflowsFor(entry).includes("convert") &&
        (!isMac || macTtsCatalogChoiceIsRunnable(model.id))
      );
    })
    .map((model, index) => ({ model, index }))
    .sort(
      (a, b) =>
        convertRank(a.model.id) - convertRank(b.model.id) || a.index - b.index,
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
  | "audioOptionSpecs"
>;

function formatSemitones(value: number): string {
  return `${value > 0 ? "+" : ""}${value} st`;
}

/** Convert's own inputs: the recording, the voice it should take, the mode and the pitch. */
function ConvertInputs({
  convert,
  historyClips,
  disabled,
}: {
  convert: ConvertGeneration;
  historyClips: readonly AudioGalleryClip[];
  disabled: boolean;
}) {
  const store = useAudioConvertStore.getState();
  const { caps, source, pitchSupport } = convert;
  const maxMinutes = Math.round((caps?.source_max_seconds ?? 300) / 60);
  return (
    <AudioHistoryProvider value={historyClips}>
      <div data-tour="audio-convert-source">
        <AudioSourceInput
          id="convert-source"
          label="Recording"
          hint={`The speech or singing to convert. Converts the first ${maxMinutes} minutes.`}
          value={source}
          onChange={store.setSource}
          disabled={disabled}
          allowSavedVoice={false}
          handleRef={convert.sourceHandle}
          onStatusChange={convert.setSourceStatus}
        />
      </div>

      <div data-tour="audio-convert-target">
        {caps?.target === "builtin" ? (
          <div className="grid gap-1.5">
            <label
              htmlFor="convert-builtin-voice"
              className="text-ui-13 font-medium text-foreground"
            >
              Target voice
            </label>
            <Select
              value={convert.builtinVoice}
              onValueChange={store.setBuiltinVoice}
              disabled={disabled}
            >
              <SelectTrigger
                id="convert-builtin-voice"
                size="sm"
                className="w-full"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {caps.builtin_voices.map((voice) => (
                  <SelectItem key={voice.id} value={voice.id}>
                    {voice.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              This model converts to its built-in voices only.
            </p>
          </div>
        ) : (
          <AudioSourceInput
            id="convert-target"
            label="Target voice"
            hint="A few seconds of the voice the recording should take."
            value={convert.target}
            onChange={store.setTarget}
            disabled={disabled}
            handleRef={convert.targetHandle}
            onStatusChange={convert.setTargetStatus}
          />
        )}
      </div>

      {caps && caps.modes.length > 1 ? (
        <div className="grid gap-1.5">
          <span className="text-ui-13 font-medium text-foreground">
            The recording is
          </span>
          <PillTabs
            ariaLabel="Speech or singing"
            value={convert.mode}
            onValueChange={(mode) =>
              store.setMode(mode === "singing" ? "singing" : "speech")
            }
            disabled={disabled}
            fit={true}
            compact={true}
            className="[&>button]:px-3"
            tabs={[
              { value: "speech", label: "Speech" },
              { value: "singing", label: "Singing" },
            ]}
          />
        </div>
      ) : null}

      {pitchSupport.show ? (
        <div className="grid gap-1.5">
          {pitchSupport.auto ? (
            <label
              htmlFor="convert-pitch-auto"
              className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
            >
              Match the target's pitch
              <Switch
                id="convert-pitch-auto"
                checked={convert.pitchAuto}
                disabled={disabled}
                onCheckedChange={store.setPitchAuto}
              />
            </label>
          ) : null}
          {pitchSupport.auto && convert.pitchAuto ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              Turn off to set the pitch shift yourself.
            </p>
          ) : (
            <ParamSlider
              label="Pitch"
              value={convert.pitch}
              min={-12}
              max={12}
              step={1}
              disabled={disabled}
              displayValue={formatSemitones(convert.pitch)}
              info="Semitones. +12 is an octave up; try -12 to +12 when the target voice is much lower or higher."
              onChange={(next) => store.setPitch(Math.round(next))}
            />
          )}
        </div>
      ) : null}
    </AudioHistoryProvider>
  );
}

/** Take target style rebuilds the delivery from the words, so it asks for them under the style choice. */
function SourceTextField({
  convert,
  disabled,
}: {
  convert: ConvertGeneration;
  disabled: boolean;
}) {
  const store = useAudioConvertStore.getState();
  const { transcriber, source } = convert;
  const sourceUsable =
    source !== null &&
    !convert.sourceExpired &&
    convert.sourceStatus.phase !== "uploading";
  if (convert.style !== "target") return null;
  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <label
          htmlFor={CONVERT_SOURCE_TEXT_FIELD_ID}
          className="text-ui-13 font-medium text-foreground"
        >
          What's said in the recording
          <span className="ml-1.5 font-normal text-muted-foreground">
            Required
          </span>
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
        id={CONVERT_SOURCE_TEXT_FIELD_ID}
        value={convert.sourceText}
        disabled={transcriber.transcribing}
        aria-busy={transcriber.transcribing}
        onChange={(event) => store.setSourceText(event.target.value)}
        placeholder="Type the words in the recording, or press Transcribe."
        className="min-h-16"
      />
      {transcriber.error ? (
        <p role="alert" className="text-ui-11p5 leading-snug text-destructive">
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
          Taking the target's style rebuilds the delivery from these words.
        </p>
      )}
    </div>
  );
}

/** Convert's rail: its inputs, then the model's tools, the device and Advanced, as on Clone. */
export function ConvertRail({
  convert,
  historyClips,
  ...props
}: RailProps & {
  convert: ConvertGeneration;
  /** Gallery clips offered under From history. */
  historyClips: readonly AudioGalleryClip[];
}) {
  const disabled = props.busy === "generating";
  const ctx = { ...convert.toolContext, convertMode: convert.mode };
  return (
    <TtsRailFields
      {...props}
      musicGeneration={false}
      prompt=""
      setPrompt={() => {}}
      audioOptionSpecs={convert.advancedOptionSpecs}
      claimedOptions={convert.claimedOptions}
      inputs={
        <ConvertInputs
          convert={convert}
          historyClips={historyClips}
          disabled={disabled}
        />
      }
      toolPanels={
        <>
          <AudioToolPanels
            workflow="convert"
            ctx={ctx}
            values={convert.toolValues}
            onChange={convert.handleToolValueChange}
            specs={convert.advancedOptionSpecs}
            disabled={disabled}
            core={{
              text: "",
              referenceText: convert.sourceText,
              hasReference: convert.source !== null,
            }}
          />
          <SourceTextField convert={convert} disabled={disabled} />
        </>
      }
    />
  );
}

/** Generate, with a line under it when the run will reload the model first. */
export function ConvertFooter({
  convert,
  ...props
}: Omit<
  ComponentProps<typeof TtsFooter>,
  | "prompt"
  | "lyricsOptional"
  | "musicNeedsDescription"
  | "audioInstructions"
  | "handleGenerate"
> & { convert: ConvertGeneration }) {
  const notice =
    props.busy === null && props.blocker === null && !props.error
      ? (convert.switchNotice?.before ?? null)
      : null;
  return (
    <div className="flex w-full flex-col items-center gap-1.5">
      <TtsFooter
        {...props}
        prompt="convert"
        lyricsOptional={false}
        musicNeedsDescription={false}
        audioInstructions=""
        handleGenerate={convert.handleGenerate}
      />
      {notice ? (
        <p className="text-center text-ui-11p5 leading-snug text-muted-foreground">
          {notice}
        </p>
      ) : null}
    </div>
  );
}

/** The selected conversion's source, fetched once its clip is selected; a 404 means it is gone. */
function useSourceSide(clip: AudioGalleryClip | null): {
  src?: string;
  unavailable?: string;
} {
  // The copy saved with the clip outlives the upload; a history clip source is a clip of its own.
  const savedUrl = clip?.source_saved
    ? `/api/inference/audio/gallery/${encodeURIComponent(clip.id)}/source/file`
    : null;
  const sourceId = clip?.source_clip_id ?? clip?.source_input_id ?? null;
  const kind = clip?.source_clip_id ? "clip" : "input";
  const [side, setSide] = useState<{ src?: string; unavailable?: string }>({});
  useEffect(() => {
    if (!savedUrl && !sourceId) {
      setSide({ unavailable: "The source is not saved with this clip." });
      return;
    }
    let url: string | null = null;
    let cancelled = false;
    setSide({});
    fetchAudioBlob(
      savedUrl ??
        sourceFileUrl({ kind, id: sourceId ?? "", name: "", durationS: null }),
    )
      .then((blob) => {
        if (cancelled) return;
        url = URL.createObjectURL(blob);
        setSide({ src: url });
      })
      .catch(() => {
        if (!cancelled)
          setSide({
            unavailable: "The source recording is no longer available.",
          });
      });
    return () => {
      cancelled = true;
      if (url) URL.revokeObjectURL(url);
    };
  }, [savedUrl, sourceId, kind]);
  return side;
}

/** Convert's output: the selected conversion with a Source | Converted switch, then history. */
export function ConvertOutput({
  modelReady,
  recommendedModels = [],
  onPickModel,
  ...props
}: Omit<
  ComponentProps<typeof TtsOutput>,
  "emptyText" | "clipBadge" | "emptyActions" | "renderPlayer"
> & {
  /** Whether a model that converts is loaded, so the empty copy asks only for what is missing. */
  modelReady: boolean;
  /** Models offered as one-click picks while none that converts is loaded. */
  recommendedModels?: readonly ModelOption[];
  onPickModel?: (id: string) => void;
}) {
  const picks = modelReady || !onPickModel ? [] : recommendedModels.slice(0, 3);
  const compareSide = useAudioConvertStore((state) => state.compareSide);
  const setCompareSide = useAudioConvertStore((state) => state.setCompareSide);
  const sourceSide = useSourceSide(props.selectedClip);
  return (
    <TtsOutput
      {...props}
      // The title already names the target; in history the badge tells runs apart by model (the
      // selected clip's line names it already).
      clipBadge={(clip, place) =>
        place === "history" ? audioModelLabel(clip.model) : null
      }
      showCopyText={false}
      renderPlayer={(clip, src, focusRef) => (
        <ABCompare
          key={clip.id}
          ariaLabel="Compare the source and the converted voice"
          a={{ label: "Source", ...sourceSide }}
          b={{ label: "Converted", src }}
          side={compareSide === "source" ? "a" : "b"}
          onSideChange={(side) =>
            setCompareSide(side === "a" ? "source" : "converted")
          }
          playerRef={focusRef}
        />
      )}
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
          ? "Converted recordings land here. Add a recording and the target voice, and press Generate."
          : "Converted recordings land here. Load a model that can convert a voice, add a recording, and press Generate."
      }
    />
  );
}
