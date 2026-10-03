// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";
import { AdvancedDisclosure } from "@/components/advanced-disclosure";
import {
  GalleryItemMenu,
  GalleryPinBadge,
} from "@/components/gallery-item-menu";
import { StripDropLine } from "@/components/gallery-strip-reorder";
import { Button } from "@/components/ui/button";
import { DropdownMenuItem } from "@/components/ui/dropdown-menu";
import { Progress } from "@/components/ui/progress";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import {
  PillTabs,
} from "@/features/model-picker/components/model-selector/pill-tabs";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  AudioWave01Icon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  SparklesIcon,
  StopIcon,
  UserSwitchIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { addAudioClipToProject, type AudioGalleryClip } from "../api";
import { AudioOptionFields } from "../audio-options-fields";
import {
  type audioGenerationPresentation,
  MINIMAX_MUSIC_DEFAULT_SECONDS,
  MINIMAX_MUSIC_FRAMES_PER_SECOND,
  MOSS_TTS_DEFAULT_SECONDS,
  MOSS_TTS_FRAMES_PER_SECOND,
} from "../audio-page-policy";
import { TTS_MAX_TOKENS } from "../audio-workspace-constants";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
import { Field } from "../components/field";
import type { AudioHostState } from "../hooks/audio-host-state";
import type { AudioGallery } from "../hooks/use-audio-gallery";
import type { AudioModelSlot } from "../hooks/use-audio-model-slot";
import type { SpeechGeneration } from "../hooks/use-speech-generation";

export interface GenerateAction {
  label: string;
  onClick: () => void;
}

export interface GenerateBlocker {
  reason: string;
  /** The fix first, then any alternative. */
  actions?: GenerateAction[];
}

function GenerateActions({ actions }: { actions?: GenerateAction[] }) {
  if (!actions?.length) return null;
  return (
    <>
      {actions.map((action, index) => (
        <span key={action.label}>
          {index === 0 ? " " : " or "}
          <Button
            type="button"
            variant="link"
            className="h-auto p-0 text-ui-11p5 font-medium text-foreground"
            onClick={action.onClick}
          >
            {action.label}
          </Button>
        </span>
      ))}
    </>
  );
}

/** Speak and Music share the main-slot rail: the text, the model's tools, the device and Advanced. */
export function TtsRailFields({
  musicGeneration,
  prompt,
  setPrompt,
  toolPanels,
  audioDevice,
  busy,
  isRecording,
  status,
  setAudioDeviceState,
  ttsLoaded,
  handleEject,
  samplingControls,
  audioOptionSpecs: allAudioOptionSpecs,
  advancedOpen,
  setAdvancedOpen,
  temperature,
  mossFrameLimit,
  handleTemperatureChange,
  musicSeconds,
  musicRange,
  setMinimaxMaxSeconds,
  cudaMusicGeneration,
  mossMaxSeconds,
  mossMaxSecondsLimit,
  setMossMaxSeconds,
  maxTokens,
  setMaxTokens,
  audioOptionValues,
  handleAudioOptionChange,
  handleAudioOptionsReset,
  inputs,
  claimedOptions,
}: Pick<
  SpeechGeneration,
  | "prompt"
  | "setPrompt"
  | "samplingControls"
  | "audioOptionSpecs"
  | "temperature"
  | "mossFrameLimit"
  | "handleTemperatureChange"
  | "musicSeconds"
  | "musicRange"
  | "setMinimaxMaxSeconds"
  | "cudaMusicGeneration"
  | "mossMaxSeconds"
  | "mossMaxSecondsLimit"
  | "setMossMaxSeconds"
  | "maxTokens"
  | "setMaxTokens"
  | "audioOptionValues"
  | "handleAudioOptionChange"
  | "handleAudioOptionsReset"
  | "ttsLoaded"
> &
  Pick<AudioHostState, "audioDevice" | "busy" | "status" | "setAdvancedOpen"> &
  Pick<AudioModelSlot, "handleEject"> & {
    /** Music copy and controls, for the Music page. */
    musicGeneration: boolean;
    toolPanels: ReactNode;
    isRecording: boolean;
    setAudioDeviceState: (next: string) => void;
    advancedOpen: boolean;
    /** The page's own inputs in place of the Text field (Clone's reference, transcript and text). */
    inputs?: ReactNode;
    /** Spec options a shown tool panel renders itself, which Advanced leaves out. */
    claimedOptions?: ReadonlySet<string>;
  }) {
  // Advanced lists only what no shown tool panel renders itself.
  const audioOptionSpecs = claimedOptions?.size
    ? allAudioOptionSpecs.filter((spec) => !claimedOptions.has(spec.name))
    : allAudioOptionSpecs;
  return (
    <>
      {inputs ?? (
        <Field
          label={musicGeneration ? "Lyrics" : "Text"}
          htmlFor="audio-prompt"
          hint={
            musicGeneration
              ? "Lyrics may use sections such as [verse] and [chorus]. The completed song lands in the gallery."
              : "What the model should say. Generation runs on the loaded TTS model and lands in the gallery."
          }
        >
          <Textarea
            id="audio-prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder={
              musicGeneration
                ? "[verse]\nMorning light through the pines…\n\n[chorus]\n…"
                : "Type the sentence to speak…"
            }
            className="min-h-28"
          />
        </Field>
      )}
      {toolPanels}
      {/* Field inlined: its label needs a form control to point
          at, and PillTabs is a tablist with its own name. */}
      <div className="grid gap-1.5">
        <span className="text-ui-13 font-medium text-foreground">
          Load model into
        </span>
        <PillTabs
          ariaLabel="Load model into"
          value={audioDevice === "cpu" ? "cpu" : "auto"}
          // The eject below applies the change and cannot interrupt a load.
          disabled={busy !== null || isRecording}
          onValueChange={(value) => {
            const next = value === "cpu" ? "cpu" : "auto";
            if (next === audioDevice) return;
            // MiniMax needs CUDA, and the backend's refusal cannot save a
            // model already ejected here.
            if (next === "cpu" && status?.audio_type === "minimax_music3") {
              toast.info(
                "MiniMax Music 3 needs a GPU, so it cannot be held in CPU RAM.",
              );
              return;
            }
            setAudioDeviceState(next);
            if (ttsLoaded) handleEject();
          }}
          fit={true}
          className="h-[calc(30px*var(--ui-space-scale,1))] self-start [&>button]:h-[calc(30px*var(--ui-space-scale,1))] [&>button]:px-6"
          tabs={[
            { value: "auto", label: "GPU when available" },
            { value: "cpu", label: "CPU RAM" },
          ]}
        />
        {/* Phrased as what the next load will do, not as the resident
            model's state: a model loaded by another tab or client can
            be on the other device, and status does not report it. */}
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {audioDevice === "cpu"
            ? "New loads go into system RAM instead of the GPU. Slower to generate, and no GPU memory is used."
            : "New loads use the GPU when there is one, and the CPU otherwise."}
        </p>
      </div>
      {/* GGUF runtime speech keeps its own sampling and length; its options come from the model. */}
      {musicGeneration ||
      samplingControls ||
      audioOptionSpecs.length > 0 ? (
        <AdvancedDisclosure
          open={advancedOpen}
          onOpenChange={setAdvancedOpen}
          description={
            musicGeneration
              ? "Generation length and model options. Changes apply to the next audio clip."
              : samplingControls
                ? "Generation sampling. Changes apply to the next audio clip."
                : "Model options. Changes apply to the next audio clip."
          }
        >
          {!musicGeneration && samplingControls ? (
            <ParamSlider
              label="Temperature"
              value={temperature}
              min={0}
              max={mossFrameLimit !== null ? 2 : 1.5}
              step={0.05}
              onChange={handleTemperatureChange}
            />
          ) : null}
          {musicGeneration ? (
            <ParamSlider
              label="Max duration (seconds)"
              value={musicSeconds}
              min={musicRange.min}
              max={musicRange.max}
              step={1 / MINIMAX_MUSIC_FRAMES_PER_SECOND}
              onChange={setMinimaxMaxSeconds}
              valueSize={8}
              info={
                cudaMusicGeneration
                  ? `Starts at ${MINIMAX_MUSIC_DEFAULT_SECONDS} seconds. MiniMax Music 3 generates ${MINIMAX_MUSIC_FRAMES_PER_SECOND} frames per second, up to ${musicRange.max} seconds.`
                  : `Starts at ${MINIMAX_MUSIC_DEFAULT_SECONDS} seconds. This model generates between ${musicRange.min} and ${musicRange.max} seconds.`
              }
            />
          ) : mossFrameLimit !== null ? (
            <ParamSlider
              label="Max duration (seconds)"
              value={mossMaxSeconds}
              min={1}
              max={mossMaxSecondsLimit}
              step={1 / MOSS_TTS_FRAMES_PER_SECOND}
              onChange={setMossMaxSeconds}
              valueSize={8}
              info={`Starts at ${MOSS_TTS_DEFAULT_SECONDS} seconds. This model reports ${mossFrameLimit?.toLocaleString()} frames (${mossMaxSecondsLimit.toLocaleString(undefined, { maximumFractionDigits: 2 })} seconds); the prompt uses part of that context.`}
            />
          ) : samplingControls ? (
            <ParamSlider
              label="Max tokens"
              value={maxTokens}
              min={256}
              max={TTS_MAX_TOKENS}
              step={256}
              onChange={setMaxTokens}
            />
          ) : null}
          {audioOptionSpecs.length > 0 ? (
            <>
              <AudioOptionFields
                specs={audioOptionSpecs}
                values={audioOptionValues}
                onChange={handleAudioOptionChange}
                disabled={busy === "generating"}
                family={status?.audio_family}
              />
              {Object.keys(audioOptionValues).length > 0 ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="self-start"
                  onClick={handleAudioOptionsReset}
                >
                  Reset model options
                </Button>
              ) : null}
            </>
          ) : null}
        </AdvancedDisclosure>
      ) : null}
    </>
  );
}

/** The rail footer: generation progress and the one primary action, Generate or Stop. */
export function TtsFooter({
  busy,
  generationPresentation,
  handleStopGeneration,
  handleGenerate,
  ttsLoaded,
  prompt,
  lyricsOptional,
  musicNeedsDescription,
  audioInstructions,
  blocker,
  shortcutLabel,
  error,
  elapsedSeconds,
  secondaryAction,
}: Pick<
  SpeechGeneration,
  | "handleGenerate"
  | "ttsLoaded"
  | "prompt"
  | "lyricsOptional"
  | "musicNeedsDescription"
  | "audioInstructions"
> &
  Pick<AudioHostState, "busy" | "handleStopGeneration"> & {
    generationPresentation: ReturnType<typeof audioGenerationPresentation>;
    /** Why Generate is off, said under it, with the fix when there is one. */
    blocker: GenerateBlocker | null;
    /** Mod+Enter, as the platform spells it. */
    shortcutLabel: string;
    /** The last run's failure, kept under the button until the next run. */
    error: GenerateBlocker | null;
    /** Seconds since the run started, while it runs. */
    elapsedSeconds: number | null;
    /** A quieter action beside Generate (Clone's Save voice…). */
    secondaryAction?: ReactNode;
  }) {
  return (
    <div className="flex w-full max-w-sm flex-col gap-2">
      {busy === "generating" && generationPresentation ? (
        <>
          <output
            aria-live="polite"
            aria-atomic="true"
            className="text-center text-ui-12 text-muted-foreground"
          >
            {generationPresentation.status}
            {elapsedSeconds !== null ? (
              <span className="ml-1.5 font-mono tabular-nums">
                {formatClipDuration(elapsedSeconds)}
              </span>
            ) : null}
          </output>
          <Progress
            indeterminate
            aria-label="Audio task in progress"
            className="h-1.5"
          />
        </>
      ) : null}
      <div className="flex flex-wrap items-center justify-center gap-2">
        {secondaryAction}
        <Button
          className="relative z-10 mx-auto h-11 px-8 disabled:bg-muted disabled:text-muted-foreground disabled:opacity-100"
          onClick={
            generationPresentation?.canStop
              ? handleStopGeneration
              : handleGenerate
          }
          disabled={
            generationPresentation
              ? !generationPresentation.canStop
              : busy !== null ||
                !ttsLoaded ||
                (!prompt.trim() && !lyricsOptional) ||
                (musicNeedsDescription && !audioInstructions.trim()) ||
                blocker !== null
          }
          variant={generationPresentation?.canStop ? "destructive" : "default"}
          aria-describedby={blocker ? "audio-generate-blocker" : undefined}
          aria-keyshortcuts="Control+Enter Meta+Enter"
          title={
            generationPresentation ? undefined : `Generate (${shortcutLabel})`
          }
        >
          {generationPresentation?.canStop ? (
            <>
              <HugeiconsIcon icon={StopIcon} className="mr-2 size-4" />
              Stop
            </>
          ) : (
            (generationPresentation?.actionLabel ?? "Generate")
          )}
        </Button>
      </div>
      {blocker && !generationPresentation ? (
        <p
          id="audio-generate-blocker"
          className="text-center text-ui-11p5 leading-snug text-muted-foreground"
        >
          {blocker.reason}
          <GenerateActions actions={blocker.actions} />
        </p>
      ) : error && !generationPresentation ? (
        <p
          role="alert"
          className="text-center text-ui-11p5 leading-snug text-destructive"
        >
          {error.reason}
          <GenerateActions actions={error.actions} />
        </p>
      ) : null}
    </div>
  );
}

function ClipBadge({ text }: { text: string }) {
  return (
    <span
      title={text}
      className="max-w-[40%] shrink-0 truncate rounded-4xl bg-muted px-2 py-0.5 text-ui-11 text-muted-foreground"
    >
      {text}
    </span>
  );
}

/** The Speak and Music output pane: the selected clip's player, then this page's history. */
export function TtsOutput({
  clips,
  selectedClip,
  selectedClipSrc,
  srcById,
  handleDownloadClip,
  handleDeleteClip,
  fallbackClip,
  handleDownloadFallbackClip,
  emptyText,
  emptyActions,
  handleClearGallery,
  historyReorder,
  hasMore,
  loadMore,
  selectedId,
  selectClip,
  handleTogglePin,
  active,
  handleArchiveClip,
  handleDownloadClipById,
  onUseTextAgain,
  handleCopyPrompt,
  freshClipId,
  onFreshClipFocused,
  announcement,
  clipBadge,
  renderPlayer,
  useAgainLabel = "Use text again",
  showCopyText = true,
  onSendToConvert,
}: Pick<
  AudioGallery,
  | "srcById"
  | "handleDownloadClip"
  | "handleDeleteClip"
  | "fallbackClip"
  | "handleDownloadFallbackClip"
  | "historyReorder"
  | "hasMore"
  | "loadMore"
  | "selectedId"
  | "selectClip"
  | "handleTogglePin"
  | "handleArchiveClip"
  | "handleDownloadClipById"
  | "handleCopyPrompt"
> &
  Pick<AudioHostState, "active"> & {
    /** This page's clips only. */
    clips: AudioGalleryClip[];
    selectedClip: AudioGalleryClip | null;
    selectedClipSrc: string | undefined;
    handleClearGallery: () => Promise<void>;
    onUseTextAgain: (clip: AudioGalleryClip) => void;
    emptyText: string;
    /** What to do from an empty page, under its text (Clone's recommended models). */
    emptyActions?: ReactNode;
    /** The clip a run just made: its player takes focus once, and the change is announced. */
    freshClipId: string | null;
    onFreshClipFocused: () => void;
    announcement: string;
    /** A short tag after a clip's text, such as the voice a clone used. */
    clipBadge?: (
      clip: AudioGalleryClip,
      place: "selected" | "history",
    ) => string | null;
    renderPlayer?: (
      clip: AudioGalleryClip,
      src: string,
      focusRef: ((element: HTMLAudioElement | null) => void) | undefined,
    ) => ReactNode;
    useAgainLabel?: string;
    showCopyText?: boolean;
    onSendToConvert?: (clip: AudioGalleryClip) => void;
  }) {
  const focusFreshClip = (element: HTMLAudioElement | null) => {
    if (!element) return;
    element.focus();
    onFreshClipFocused();
  };
  return (
    <>
      <output aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </output>
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4">
        {selectedClip ? (
          <div className="flex w-full max-w-xl flex-col gap-3">
            <p className="line-clamp-2 text-ui-13 text-muted-foreground">
              {selectedClip.prompt}
            </p>
            {/* Auth-protected bytes, so mount a fresh player only once this clip's object URL exists: reusing
                one media element while src is changing left History switches showing broken controls. */}
            {selectedClipSrc && renderPlayer ? (
              renderPlayer(
                selectedClip,
                selectedClipSrc,
                selectedClip.id === freshClipId ? focusFreshClip : undefined,
              )
            ) : selectedClipSrc ? (
              <audio
                key={selectedClip.id}
                ref={selectedClip.id === freshClipId ? focusFreshClip : undefined}
                controls={true}
                src={selectedClipSrc}
                // The focus ring follows the player's pill instead of boxing it.
                className="w-full rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            ) : (
              <div
                role="status"
                className="flex h-12 w-full items-center justify-center rounded-md border border-border text-ui-12 text-muted-foreground"
              >
                Loading audio…
              </div>
            )}
            <div className="flex items-center gap-2 text-ui-11p5 text-muted-foreground">
              <span title={selectedClip.model}>{audioModelLabel(selectedClip.model)}</span>
              <span>·</span>
              <span>{formatClipDuration(selectedClip.duration_s)}</span>
              {clipBadge?.(selectedClip, "selected") ? (
                <ClipBadge text={clipBadge(selectedClip, "selected") ?? ""} />
              ) : null}
              <span className="flex-1" />
              <Button
                variant="ghost"
                size="sm"
                aria-label="Download audio clip"
                disabled={!srcById[selectedClip.id]}
                onClick={() => handleDownloadClip(selectedClip)}
              >
                <HugeiconsIcon
                  icon={Download01Icon}
                  className="size-3.5"
                />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                aria-label="Delete audio clip"
                onClick={() => void handleDeleteClip(selectedClip.id)}
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  className="size-3.5"
                />
              </Button>
            </div>
          </div>
        ) : fallbackClip ? (
          <div className="flex w-full max-w-xl flex-col gap-3">
            <p className="line-clamp-2 text-ui-13 text-muted-foreground">
              {fallbackClip.prompt}
            </p>
            <audio
              controls={true}
              src={fallbackClip.url}
              className="w-full"
            />
            <div className="flex items-center gap-2 text-ui-11p5 text-muted-foreground">
              <span>
                {audioModelLabel(fallbackClip.model)}
                {fallbackClip.saved
                  ? " · saved, waiting for the gallery"
                  : " · not saved to the gallery"}
              </span>
              <span className="flex-1" />
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownloadFallbackClip}
              >
                <HugeiconsIcon
                  icon={Download01Icon}
                  className="size-3.5"
                />
                Download WAV
              </Button>
            </div>
          </div>
        ) : (
          <div className="grid justify-items-center gap-3">
            <p className="text-ui-13 text-muted-foreground">{emptyText}</p>
            {emptyActions}
          </div>
        )}
      </div>
      {clips.length > 0 ? (
        <div className="flex shrink-0 flex-col gap-2">
          <div className="flex items-center justify-between">
            <span className="text-ui-11p5 font-medium text-muted-foreground">
              History
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => void handleClearGallery()}
            >
              Clear all
            </Button>
          </div>
          <div
            {...historyReorder.stripProps}
            // py-1 keeps the first and last drop lines inside the scroller.
            className="hover-scrollbar flex max-h-40 flex-col gap-1 overflow-y-auto py-1"
            onScroll={(event) => {
              const el = event.currentTarget;
              if (
                hasMore &&
                el.scrollTop + el.clientHeight >= el.scrollHeight - 40
              ) {
                void loadMore();
              }
            }}
          >
            {clips.map((clip) => (
              // Shell, not a button: the pin badge and dots menu are buttons and cannot nest.
              <div
                key={clip.id}
                {...historyReorder.tileProps(clip.id)}
                className={cn(
                  "group relative flex items-center gap-1 rounded-md pr-1 transition-colors hover:bg-muted",
                  clip.id === selectedId && "bg-muted",
                  historyReorder.draggingId === clip.id && "opacity-40",
                )}
              >
                {historyReorder.cue?.id === clip.id && (
                  <StripDropLine
                    axis="y"
                    edge={historyReorder.cue.edge}
                  />
                )}
                <button
                  type="button"
                  onClick={() => selectClip(clip.id)}
                  aria-current={
                    clip.id === selectedId ? "true" : undefined
                  }
                  className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-ui-13"
                >
                  <HugeiconsIcon
                    icon={AudioWave01Icon}
                    className="size-3.5 shrink-0 text-muted-foreground"
                  />
                  <span className="min-w-0 flex-1 truncate">{clip.prompt}</span>
                  {clipBadge?.(clip, "history") ? (
                    <ClipBadge text={clipBadge(clip, "history") ?? ""} />
                  ) : null}
                  <span className="shrink-0 text-ui-11p5 text-muted-foreground">
                    {formatClipDuration(clip.duration_s)}
                  </span>
                </button>
                {clip.pinned && (
                  <GalleryPinBadge
                    noun="clip"
                    className="static shrink-0"
                    onUnpin={() => void handleTogglePin(clip.id, false)}
                  />
                )}
                <GalleryItemMenu
                  variant="row"
                  noun="clip"
                  active={active}
                  pinned={Boolean(clip.pinned)}
                  archived={false}
                  onTogglePin={() =>
                    void handleTogglePin(clip.id, !clip.pinned)
                  }
                  onToggleArchive={() => void handleArchiveClip(clip.id)}
                  onDelete={() => void handleDeleteClip(clip.id)}
                  onDownload={() => void handleDownloadClipById(clip)}
                  onAddToProject={(projectId) =>
                    addAudioClipToProject(clip.id, projectId)
                  }
                  leadingItems={
                    <>
                      <DropdownMenuItem
                        onClick={() => onUseTextAgain(clip)}
                      >
                        <HugeiconsIcon
                          icon={SparklesIcon}
                          strokeWidth={1.75}
                          className="size-icon"
                        />
                        {useAgainLabel}
                      </DropdownMenuItem>
                      {onSendToConvert ? (
                        <DropdownMenuItem
                          onClick={() => onSendToConvert(clip)}
                        >
                          <HugeiconsIcon
                            icon={UserSwitchIcon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          Convert this voice
                        </DropdownMenuItem>
                      ) : null}
                      {showCopyText ? (
                        <DropdownMenuItem
                          onClick={() => void handleCopyPrompt(clip.prompt)}
                        >
                          <HugeiconsIcon
                            icon={Copy01Icon}
                            strokeWidth={1.75}
                            className="size-icon"
                          />
                          Copy text
                        </DropdownMenuItem>
                      ) : null}
                    </>
                  }
                />
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </>
  );
}
