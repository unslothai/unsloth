// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  GalleryItemMenu,
  GalleryPinBadge,
} from "@/components/gallery-item-menu";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { AiSpeechIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, useEffect, useMemo, useState } from "react";
import {
  type AudioGalleryClip,
  addAudioClipToProject,
  fetchAudioBlob,
  setAudioClipFlags,
} from "../api";
import { audioCppModelFor } from "../audio-cpp-catalog";
import { audioModelLabel, formatClipDuration } from "../audio-workspace-utils";
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
import {
  CLIP_ICON_COLUMN,
  CLIPS_HEADER_LINK,
  CLIPS_LIST,
  CLIPS_TITLE_INSET,
  CLIPS_TOP_ROW,
  ClipsCard,
  ClipsCardEmpty,
} from "../components/clips-card";
import { TtsFooter, TtsRailFields } from "./tts-workspace";

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
      <p className="text-ui-12 leading-normal text-muted-foreground">
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
        <p className="text-ui-12 leading-normal text-muted-foreground">
          Loading a separation model replaces the model in the main slot.
        </p>
      )}
      {pageModelLoaded && separate.cpuWarning ? (
        <p className="text-ui-12 leading-normal text-muted-foreground">
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
    <div className="flex w-full flex-col items-center gap-1.5">
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
        <p className="text-center text-ui-12 leading-normal text-muted-foreground">
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
      <ClipsCard
        label={groups.length > 0 ? "Separations" : "Separate audio"}
        title={groups.length > 0 ? "Separations" : "Separate audio"}
        count={groups.length}
        countMore={hasMore}
        action={
          groups.length > 0 ? (
            <button
              type="button"
              onClick={() => void handleClearGallery()}
              className={CLIPS_HEADER_LINK}
            >
              Clear all
            </button>
          ) : null
        }
      >
        {pendingRun ? (
          <output className={cn(CLIPS_TOP_ROW, "flex items-center gap-3")}>
            <span
              className={cn(
                "grid shrink-0 place-items-center rounded-full bg-wash-6 text-muted-foreground",
                CLIP_ICON_COLUMN,
              )}
            >
              <Spinner className="size-4" />
            </span>
            <span className="grid min-w-0 flex-1 gap-0.5">
              <span className="truncate text-ui-15 font-medium leading-snug text-foreground">
                {pendingRun.title}
              </span>
              <span className="text-ui-12 text-muted-foreground">
                {pendingRun.model
                  ? `Separating with ${audioModelLabel(pendingRun.model)}…`
                  : "Separating…"}{" "}
                The stems appear here when they are ready.
              </span>
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
              <p className={cn("mt-2 text-ui-12 text-muted-foreground", CLIPS_TITLE_INSET)}>
                {selected.stems.length} of {selected.expectedStems} stems; the
                others were deleted.
              </p>
            )}
          </div>
        ) : fallbackClip ? (
          <ClipsCardEmpty
            text="Stems saved, waiting for the history to list them."
            actions={
              <div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownloadFallbackClip}
                >
                  <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                  Download the first stem
                </Button>
              </div>
            }
          />
        ) : (
          <ClipsCardEmpty
            text={
              modelReady
                ? "Stems land here. Add a track and press Generate."
                : "Stems land here. Load a separation model, add a track, and press Generate."
            }
            actions={
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
          />
        )}
        {/* The selected run is the top row, so the list leaves it out. */}
        {groups.some((group) => group.groupId !== selected?.groupId) ? (
          <div
            className={CLIPS_LIST}
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
            {groups
              .filter((group) => group.groupId !== selected?.groupId)
              .map((group) => {
              return (
                // Shell, not a button: the pin badge and dots menu are buttons and cannot nest.
                <div
                  key={group.groupId}
                  className="group relative flex items-center gap-1 rounded-md ps-3 pe-1 transition-colors hover:bg-wash-3"
                >
                  {/* Same 40px column and title edge as the card's top row. */}
                  <span
                    aria-hidden="true"
                    className="grid w-[calc(40px*var(--ui-space-scale,1))] shrink-0 place-items-center text-muted-foreground"
                  >
                    <HugeiconsIcon icon={AiSpeechIcon} className="size-3.5" />
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      const first = group.stems[0];
                      if (first) selectClip(first.id);
                    }}
                    className="flex min-w-0 flex-1 items-center gap-3 py-2.5 pr-3 ps-2 text-left"
                  >
                    <span className="grid min-w-0 flex-1 gap-0.5">
                      <span className="truncate text-ui-13" title={group.title}>
                        {group.title}
                      </span>
                      <span
                        className="truncate text-ui-11p5 text-muted-foreground"
                        title={group.model}
                      >
                        {audioModelLabel(group.model)} ·{" "}
                        {group.complete
                          ? `${group.stems.length} stems`
                          : `${group.stems.length} of ${group.expectedStems} stems`}
                      </span>
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
        ) : null}
      </ClipsCard>
    </>
  );
}
