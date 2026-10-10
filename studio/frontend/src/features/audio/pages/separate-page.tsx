// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  GalleryItemMenu,
  GalleryPinBadge,
} from "@/components/gallery-item-menu";
import { Button } from "@/components/ui/button";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { AudioWave01Icon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useEffect, useMemo, useState } from "react";
import {
  type AudioGalleryClip,
  addAudioClipToProject,
  fetchAudioBlob,
  setAudioClipFlags,
} from "../api";
import { audioCppModelFor, audioCppWorkflowsFor } from "../audio-cpp-catalog";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
import { macTtsCatalogChoiceIsRunnable } from "../catalog";
import {
  AudioHistoryProvider,
  AudioSourceInput,
} from "../components/audio-source-input";
import { StemMixer } from "../components/stem-mixer";
import type { SendTarget } from "../components/stem-mixer-types";
import { stemFileName, stemZipName, zipStems } from "../components/stem-zip";
import type { AudioHostState } from "../hooks/audio-host-state";
import type { AudioGallery } from "../hooks/use-audio-gallery";
import {
  SEPARATE_TRACK_EXPIRED_MESSAGE,
  type SeparateGeneration,
} from "../hooks/use-separate-generation";
import { useStemSources } from "../hooks/use-stem-sources";
import { STEM_SEND_TARGETS } from "../send-targets";
import { saveAudio } from "../save-audio";
import {
  SEPARATE_MAX_SECONDS,
  separatePresentation,
} from "../separate-policy";
import {
  SEPARATION_STEMS_BY_FAMILY,
  type SeparationGroup,
  groupSeparationClips,
  stemLabel,
} from "../separation-stems";
import { useAudioSeparateStore } from "../stores/audio-separate-store";
import { AudioToolPanels } from "../tools/tool-panel-host";
import { TtsFooter, TtsRailFields } from "./tts-workspace";

const SEPARATE_MODEL_ORDER = [
  "HTDemucs-GGUF",
  "BS-RoFormer-ep368-GGUF",
  "HTDemucs-6stems-GGUF",
  "Mel-Band-RoFormer-GGUF",
];

function separateRank(id: string): number {
  const index = SEPARATE_MODEL_ORDER.findIndex((name) =>
    id.endsWith(`/${name}`),
  );
  return index === -1 ? SEPARATE_MODEL_ORDER.length : index;
}

export function separatePageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models
    .filter((model) => {
      const entry = audioCppModelFor(model.id);
      return (
        entry !== null &&
        audioCppWorkflowsFor(entry).includes("separate") &&
        (!isMac || macTtsCatalogChoiceIsRunnable(model.id))
      );
    })
    .map((model, index) => ({ model, index }))
    .sort(
      (a, b) =>
        separateRank(a.model.id) - separateRank(b.model.id) ||
        a.index - b.index,
    )
    .map(({ model }) => model);
}

export function separateOutputStems(
  family: string | null | undefined,
  modelId: string | null | undefined,
): readonly string[] | null {
  const known = family ? SEPARATION_STEMS_BY_FAMILY[family] : undefined;
  if (known) return known.stems;
  const stems = audioCppModelFor(modelId)?.stems;
  return stems ?? null;
}

type RailProps = Omit<
  ComponentProps<typeof TtsRailFields>,
  | "musicGeneration"
  | "prompt"
  | "setPrompt"
  | "toolPanels"
  | "inputs"
  | "claimedOptions"
  | "samplingControls"
  | "audioOptionSpecs"
>;

function StemChips({ stems }: { stems: readonly string[] }) {
  return (
    <div className="grid gap-1">
      <span className="text-ui-13 font-medium text-foreground">Outputs</span>
      <p className="text-ui-11p5 leading-snug text-muted-foreground">
        {stems.map(stemLabel).join(" · ")}
      </p>
    </div>
  );
}

function SeparateInputs({
  separate,
  historyClips,
  disabled,
  pageModelLoaded,
  pageModelId,
}: {
  separate: SeparateGeneration;
  historyClips: readonly AudioGalleryClip[];
  disabled: boolean;
  pageModelLoaded: boolean;
  pageModelId: string | null;
}) {
  const setSource = useAudioSeparateStore((state) => state.setSource);
  const outputs = separateOutputStems(
    pageModelLoaded ? separate.family : null,
    pageModelId,
  );
  return (
    <AudioHistoryProvider value={historyClips}>
      <div data-tour="audio-separate-source">
        <AudioSourceInput
          id="separate-source"
          label="Track"
          hint="Converted to 44.1 kHz automatically. Up to 10 minutes."
          value={separate.source}
          onChange={setSource}
          disabled={disabled}
          allowSavedVoice={false}
          usesFirstSeconds={null}
          expiredMessage={SEPARATE_TRACK_EXPIRED_MESSAGE}
          recordHint="Play the track near the mic, up to 10 minutes."
          // A second under the cap: the timer fires late, and the server allows no overrun.
          maxRecordSeconds={SEPARATE_MAX_SECONDS - 1}
          handleRef={separate.sourceHandle}
          onStatusChange={separate.setSourceStatus}
        />
      </div>
      {outputs ? <StemChips stems={outputs} /> : null}
      {pageModelLoaded ? null : (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          Loading a separation model replaces the model in the main slot.
        </p>
      )}
      {pageModelLoaded && separate.cpuWarning ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {separate.cpuWarning}
        </p>
      ) : null}
    </AudioHistoryProvider>
  );
}

export function SeparateRail({
  separate,
  historyClips,
  pageModelLoaded,
  pageModelId,
  ...props
}: RailProps & {
  separate: SeparateGeneration;
  historyClips: readonly AudioGalleryClip[];
  pageModelLoaded: boolean;
  pageModelId: string | null;
}) {
  const disabled = props.busy === "generating";
  return (
    <TtsRailFields
      {...props}
      musicGeneration={false}
      prompt=""
      setPrompt={() => {}}
      samplingControls={false}
      audioOptionSpecs={[]}
      inputs={
        <SeparateInputs
          separate={separate}
          historyClips={historyClips}
          disabled={disabled}
          pageModelLoaded={pageModelLoaded}
          pageModelId={pageModelId}
        />
      }
      toolPanels={
        <AudioToolPanels
          workflow="separate"
          ctx={separate.toolContext}
          values={separate.toolValues}
          onChange={separate.handleToolValueChange}
          specs={[]}
          disabled={disabled}
          core={{ text: "" }}
        />
      }
    />
  );
}

export function SeparateFooter({
  separate,
  generationPresentation,
  generationPhase,
  ...props
}: Omit<
  ComponentProps<typeof TtsFooter>,
  | "prompt"
  | "lyricsOptional"
  | "musicNeedsDescription"
  | "audioInstructions"
  | "handleGenerate"
  | "secondaryAction"
> & {
  separate: SeparateGeneration;
  generationPhase: string | null;
}) {
  const presentation = separatePresentation(
    generationPresentation,
    generationPhase,
    separate.reloading,
  );
  const ready =
    props.ttsLoaded && props.blocker === null && props.busy === null;
  return (
    <div className="flex w-full max-w-sm flex-col items-center gap-1.5">
      <TtsFooter
        {...props}
        generationPresentation={presentation}
        prompt=""
        lyricsOptional={true}
        musicNeedsDescription={false}
        audioInstructions=""
        handleGenerate={separate.handleGenerate}
      />
      {ready && !presentation && !props.error ? (
        <p className="text-center text-ui-11p5 leading-snug text-muted-foreground">
          {separate.willReload
            ? "Reloads the model with the new overlap first (a few seconds). "
            : ""}
          {separate.estimateSeconds !== null && !separate.cpuWarning ? (
            <>
              About{" "}
              <span className="font-mono tabular-nums">
                {separate.estimateSeconds}
              </span>{" "}
              s on the GPU.
            </>
          ) : null}
        </p>
      ) : null}
    </div>
  );
}

function downloadGroup(group: SeparationGroup) {
  // Each stem is fetched when the zip reaches it, so a long song never holds every stem at once.
  const files = group.stems.map((clip) => ({
    name: stemFileName(group.title, stemLabel(clip.role ?? "")),
    blob: () => fetchAudioBlob(clip.url),
  }));
  return saveAudio(stemZipName(group.title), null, () => zipStems(files));
}

/** keyed by group so each mixer starts from its own clock. */
function SelectedSeparation({
  group,
  autoFocus,
  active,
  onSendStem,
}: {
  group: SeparationGroup;
  autoFocus: boolean;
  active: boolean;
  onSendStem: (
    target: SendTarget,
    clip: AudioGalleryClip,
    name: string,
  ) => void;
}) {
  const inputs = useMemo(
    () => group.stems.map((clip) => ({ clipId: clip.id, url: clip.url })),
    [group.stems],
  );
  const [attempt, setAttempt] = useState(0);
  const sources = useStemSources(inputs, attempt);
  // Stable while the group and its fetched audio are, so the transport keeps its players.
  const stems = useMemo(
    () =>
      group.stems.map((clip) => ({
        clipId: clip.id,
        role: clip.role ?? clip.id,
        label: stemLabel(clip.role ?? ""),
        src: sources.srcById[clip.id] ?? null,
        failed: sources.failedIds.includes(clip.id),
        durationS: clip.duration_s,
        peaks: sources.peaksById[clip.id] ?? null,
      })),
    [group.stems, sources.srcById, sources.peaksById, sources.failedIds],
  );
  return (
    <>
      <StemMixer
        groupId={group.groupId}
        title={group.title}
        subtitle={audioModelLabel(group.model)}
        stems={stems}
        autoFocus={autoFocus}
        active={active}
        sendTargets={STEM_SEND_TARGETS}
        onDownloadStem={(clipId) => {
          const clip = group.stems.find((item) => item.id === clipId);
          const src = sources.srcById[clipId];
          if (!clip || !src) return;
          void saveAudio(
            stemFileName(group.title, stemLabel(clip.role ?? "")),
            src,
            () => fetchAudioBlob(clip.url),
          );
        }}
        onDownloadAll={() => void downloadGroup(group)}
        onSend={(target, clipId) => {
          const clip = group.stems.find((item) => item.id === clipId);
          if (!clip) return;
          onSendStem(
            target,
            clip,
            `${group.title.replace(/\.[a-z0-9]{2,4}$/i, "")} - ${stemLabel(clip.role ?? "")}`,
          );
        }}
      />
      {sources.failedIds.length > 0 ? (
        <p role="alert" className="mt-2 text-ui-11p5 text-muted-foreground">
          Could not load {sources.failedIds.length} of {group.stems.length}{" "}
          stems.{" "}
          <Button
            variant="link"
            size="sm"
            className="h-auto p-0"
            onClick={() => setAttempt((n) => n + 1)}
          >
            Try again
          </Button>
        </p>
      ) : null}
    </>
  );
}

export function SeparateOutput({
  clips,
  selectedId,
  selectClip,
  fallbackClip,
  handleDownloadFallbackClip,
  handleDeleteClip,
  handleDeleteGroup,
  handleArchiveClip,
  handleTogglePin,
  handleClearGallery,
  hasMore,
  loadMore,
  active,
  modelReady,
  recommendedModels = [],
  onPickModel,
  separate,
  onSendStem,
}: Pick<
  AudioGallery,
  | "selectedId"
  | "selectClip"
  | "fallbackClip"
  | "handleDownloadFallbackClip"
  | "handleDeleteClip"
  | "handleDeleteGroup"
  | "handleArchiveClip"
  | "handleTogglePin"
  | "hasMore"
  | "loadMore"
> &
  Pick<AudioHostState, "active"> & {
    clips: AudioGalleryClip[];
    handleClearGallery: () => Promise<void>;
    modelReady: boolean;
    recommendedModels?: readonly ModelOption[];
    onPickModel?: (id: string) => void;
    separate: SeparateGeneration;
    onSendStem: (
      target: SendTarget,
      clip: AudioGalleryClip,
      name: string,
    ) => void;
  }) {
  const groups = useMemo(
    () => groupSeparationClips(clips, hasMore),
    [clips, hasMore],
  );
  // The hidden oldest run waits on the next page; a short list never scrolls, so fetch it here.
  const tailHidden = useMemo(
    () => hasMore && groupSeparationClips(clips).length > groups.length,
    [clips, hasMore, groups],
  );
  // biome-ignore lint/correctness/useExhaustiveDependencies: each loaded page re-checks, in case the run is still cut.
  useEffect(() => {
    if (tailHidden) void loadMore();
  }, [tailHidden, clips, loadMore]);
  const selected =
    groups.find((group) =>
      group.stems.some((clip) => clip.id === selectedId),
    ) ?? null;
  const { lastResult, clearLastResult, pendingRun } = separate;
  const fresh =
    lastResult !== null &&
    selected !== null &&
    (lastResult.groupId === null || selected.groupId === lastResult.groupId);
  const [focusedGroup, setFocusedGroup] = useState<string | null>(null);
  useEffect(() => {
    if (fresh && selected) setFocusedGroup(selected.groupId);
  }, [fresh, selected]);
  const announcement = lastResult
    ? `Separated into ${lastResult.stems} stem${lastResult.stems === 1 ? "" : "s"}.`
    : "";
  const picks = modelReady || !onPickModel ? [] : recommendedModels.slice(0, 3);

  const pinGroup = (group: SeparationGroup, pinned: boolean) => {
    for (const clip of group.stems) void handleTogglePin(clip.id, pinned);
  };
  const archiveGroup = async (group: SeparationGroup) => {
    const [last, ...rest] = [...group.stems].reverse();
    try {
      await Promise.all(
        rest.map((clip) => setAudioClipFlags(clip.id, { archived: true })),
      );
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not archive the stems.",
      );
      return;
    }
    if (last) await handleArchiveClip(last.id);
  };
  const deleteGroup = async (group: SeparationGroup) => {
    // A clip saved without a run is a group of one.
    const groupId = group.stems[0]?.group_id;
    if (!groupId) {
      for (const clip of group.stems) await handleDeleteClip(clip.id);
      return;
    }
    await handleDeleteGroup(
      groupId,
      group.stems.map((clip) => clip.id),
    );
  };

  return (
    <>
      <output aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </output>
      <div
        className={cn(
          "flex min-h-0 flex-1 flex-col items-center gap-4",
          selected || pendingRun ? "justify-start" : "justify-center",
        )}
      >
        {pendingRun ? (
          <output className="corner-squircle grid w-full gap-1 rounded-4xl bg-card p-4 ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]">
            <span className="block truncate text-ui-13 font-medium text-foreground">
              {pendingRun.title}
            </span>
            <span className="block text-ui-11p5 text-muted-foreground">
              {pendingRun.model
                ? `Separating with ${audioModelLabel(pendingRun.model)}…`
                : "Separating…"}{" "}
              The stems appear here when they are ready.
            </span>
          </output>
        ) : selected ? (
          <div className="w-full">
            <SelectedSeparation
              key={selected.groupId}
              group={selected}
              autoFocus={focusedGroup === selected.groupId}
              active={active}
              onSendStem={(target, clip, name) => {
                clearLastResult();
                onSendStem(target, clip, name);
              }}
            />
            {selected.complete ? null : (
              <p className="mt-2 text-ui-11p5 text-muted-foreground">
                {selected.stems.length} of {selected.expectedStems} stems; the
                others were deleted.
              </p>
            )}
          </div>
        ) : fallbackClip ? (
          <div className="flex w-full max-w-xl flex-col gap-3">
            <p className="text-ui-13 text-muted-foreground">
              Stems saved, waiting for the history to list them.
            </p>
            <Button
              variant="outline"
              size="sm"
              className="self-start"
              onClick={handleDownloadFallbackClip}
            >
              <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
              Download the first stem
            </Button>
          </div>
        ) : (
          <div className="grid justify-items-center gap-3">
            <p className="text-ui-13 text-muted-foreground">
              {modelReady
                ? "Stems land here. Add a track and press Generate."
                : "Stems land here. Load a separation model, add a track, and press Generate."}
            </p>
            {picks.length > 0 ? (
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
            ) : null}
          </div>
        )}
      </div>
      {groups.length > 0 ? (
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
            {groups.map((group) => {
              const current = group.groupId === selected?.groupId;
              return (
                // Shell, not a button: the pin badge and dots menu are buttons and cannot nest.
                <div
                  key={group.groupId}
                  className={cn(
                    "group relative flex items-center gap-1 rounded-md pr-1 transition-colors hover:bg-muted",
                    current && "bg-muted",
                  )}
                >
                  <button
                    type="button"
                    onClick={() => {
                      const first = group.stems[0];
                      if (first) selectClip(first.id);
                    }}
                    aria-current={current ? "true" : undefined}
                    className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-ui-13"
                  >
                    <HugeiconsIcon
                      icon={AudioWave01Icon}
                      className="size-3.5 shrink-0 text-muted-foreground"
                    />
                    <span className="min-w-0 flex-1 truncate">
                      {group.title}
                    </span>
                    <span
                      title={group.model}
                      className="max-w-[30%] shrink-0 truncate text-ui-11p5 text-muted-foreground"
                    >
                      {audioModelLabel(group.model)}
                    </span>
                    <span className="shrink-0 rounded-4xl bg-muted px-2 py-0.5 text-ui-11 text-muted-foreground group-hover:bg-background">
                      {group.complete
                        ? `${group.stems.length} stems`
                        : `${group.stems.length} of ${group.expectedStems} stems`}
                    </span>
                    <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
                      {formatClipDuration(group.durationS)}
                    </span>
                  </button>
                  {group.pinned && (
                    <GalleryPinBadge
                      noun="clip"
                      className="static shrink-0"
                      onUnpin={() => pinGroup(group, false)}
                    />
                  )}
                  <GalleryItemMenu
                    variant="row"
                    noun="clip"
                    active={active}
                    pinned={group.pinned}
                    archived={false}
                    onTogglePin={() => pinGroup(group, !group.pinned)}
                    onToggleArchive={() => void archiveGroup(group)}
                    onDelete={() => void deleteGroup(group)}
                    onDownload={() => void downloadGroup(group)}
                    onAddToProject={async (projectId) => {
                      let already = true;
                      for (const clip of group.stems) {
                        const result = await addAudioClipToProject(
                          clip.id,
                          projectId,
                        );
                        already = already && result.already;
                      }
                      return { already };
                    }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </>
  );
}
