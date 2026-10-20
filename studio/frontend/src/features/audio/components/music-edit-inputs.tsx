// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import { type JSX, type Ref, useEffect, useState } from "react";
import type { AudioSourceStatus } from "../hooks/audio-source-state";
import {
  EXTEND_MAX_S,
  EXTEND_MIN_S,
  MUSIC_EDIT_ACTION_HINT,
  MUSIC_EDIT_ACTION_LABEL,
  MUSIC_EDIT_DEFAULT_STRENGTH,
  editActions,
  editBeyondEndS,
  editMaxRanges,
  editRangeProblem,
  editSourceTooLong,
  editUsesRanges,
  editUsesStrength,
  formatEditLimit,
} from "../music/music-edit-rules";
import type {
  MusicEditAction,
  MusicEditDraft,
  MusicModeRule,
} from "../music/music-types";
import {
  AudioSourceInput,
  type AudioSourceInputHandle,
} from "./audio-source-input";
import { Waveform } from "./waveform";
import { secondsBeyondEnd } from "./waveform-range";
import { WaveformRangeSelect } from "./waveform-range-select";

export { musicEditProblem } from "../music/music-edit-rules";

const SOURCE_ID = "music-edit-source";
const PROMPT_ID = "music-edit-prompt";

export interface MusicEditInputsProps {
  rule: MusicModeRule;
  draft: MusicEditDraft;
  onChange: (patch: Partial<MusicEditDraft>) => void;
  disabled: boolean;
  onSourceStatusChange?: (status: AudioSourceStatus) => void;
  sourceHandleRef?: Ref<AudioSourceInputHandle>;
}

export function MusicEditInputs({
  rule,
  draft,
  onChange,
  disabled,
  onSourceStatusChange,
  sourceHandleRef,
}: MusicEditInputsProps): JSX.Element {
  const actions = editActions(rule);
  const action: MusicEditAction | null =
    draft.action && actions.includes(draft.action) ? draft.action : null;
  const sourceDurationS = draft.source?.durationS ?? null;
  const [draggedExtendS, setDraggedExtendS] = useState<number | null>(null);

  // The tabs always show a choice, so a missing or stale action takes the first one.
  const firstAction = actions[0] ?? null;
  useEffect(() => {
    if (action === null && firstAction !== null)
      onChange({ action: firstAction });
  }, [action, firstAction, onChange]);

  const usesRanges = editUsesRanges(action);
  const maxRanges = editMaxRanges(rule, action);
  const beyondEndS = editBeyondEndS(action);
  const tooLong = editSourceTooLong(rule, sourceDurationS);
  const rangeProblem =
    draft.source && !tooLong
      ? editRangeProblem(
          rule,
          { action, ranges: draft.ranges },
          sourceDurationS,
        )
      : null;
  const extendedBy =
    action === "repaint" && sourceDurationS !== null && draft.ranges[0]
      ? secondsBeyondEnd(draft.ranges[0], sourceDurationS)
      : 0;
  const defaultStrength = action
    ? MUSIC_EDIT_DEFAULT_STRENGTH[action]
    : undefined;

  return (
    <div className="grid gap-4">
      <AudioSourceInput
        id={SOURCE_ID}
        label="Clip to edit"
        hint={
          rule.max_source_s
            ? `Up to ${formatEditLimit(rule.max_source_s)} of music.`
            : undefined
        }
        value={draft.source}
        onChange={(next) =>
          onChange({
            source: next,
            // Parts picked on another clip point at the wrong audio.
            ranges: next && next.id === draft.source?.id ? draft.ranges : [],
          })
        }
        disabled={disabled}
        allowHistory={true}
        allowSavedVoice={false}
        usesFirstSeconds={null}
        expiredMessage="This clip expired. Add it again."
        recordHint="Play or sing the part you want to change."
        handleRef={sourceHandleRef}
        onStatusChange={onSourceStatusChange}
        renderWaveform={(preview) =>
          usesRanges ? (
            <WaveformRangeSelect
              peaks={preview.peaks}
              durationS={preview.durationS}
              src={preview.src}
              label={preview.label}
              ranges={draft.ranges}
              onChange={(ranges) => onChange({ ranges })}
              maxRanges={maxRanges}
              beyondEndS={beyondEndS}
              disabled={disabled}
            />
          ) : (
            <Waveform
              peaks={preview.peaks}
              durationS={preview.durationS}
              src={preview.src}
              label={preview.label}
              tailS={action === "extend" ? (draggedExtendS ?? draft.extendS) : 0}
            />
          )
        }
      />
      {tooLong ? (
        <p
          role="alert"
          className="-mt-2 text-ui-11p5 leading-snug text-foreground"
        >
          {tooLong}
        </p>
      ) : null}

      {actions.length > 0 ? (
        <div className="grid gap-1.5">
          <span className="text-ui-13 font-medium text-foreground">
            What to do
          </span>
          <PillTabs
            ariaLabel="What to do with the clip"
            value={action ?? ""}
            onValueChange={(next) => {
              const nextAction = next as MusicEditAction;
              if (nextAction === action) return;
              // Repaint keeps one part; inpaint can keep several.
              const keep = editMaxRanges(rule, nextAction);
              onChange({
                action: nextAction,
                ranges: draft.ranges.slice(0, keep),
              });
            }}
            disabled={disabled}
            fit={true}
            compact={true}
            className="[&>button]:px-3"
            tabs={actions.map((value) => ({
              value,
              label: MUSIC_EDIT_ACTION_LABEL[value],
            }))}
          />
          {action ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              {MUSIC_EDIT_ACTION_HINT[action]}
            </p>
          ) : null}
          {usesRanges && rangeProblem ? (
            <p className="text-ui-11p5 leading-snug text-foreground">
              {draft.ranges.length === 0
                ? `${rangeProblem} Drag across it, or use Select a part.`
                : rangeProblem}
            </p>
          ) : extendedBy > 0 ? (
            <p className="text-ui-11p5 leading-snug text-muted-foreground">
              Extends the clip by{" "}
              <span className="font-mono tabular-nums">
                {extendedBy.toFixed(1)}
              </span>{" "}
              s.
            </p>
          ) : null}
        </div>
      ) : (
        <p className="text-ui-12 leading-snug text-muted-foreground">
          This model cannot edit clips. Load ACE-Step or Stable Audio to edit.
        </p>
      )}

      {editUsesStrength(action) && defaultStrength !== undefined ? (
        <ParamSlider
          label="How much to change"
          value={draft.strength ?? defaultStrength}
          min={0}
          max={1}
          step={0.05}
          onChange={(strength) => onChange({ strength })}
          format={(v) => v.toFixed(2)}
          disabled={disabled}
          info="Lower keeps more of the original."
        />
      ) : null}

      {action === "extend" ? (
        <ParamSlider
          label="Add seconds"
          value={Math.min(EXTEND_MAX_S, Math.max(EXTEND_MIN_S, draft.extendS))}
          min={EXTEND_MIN_S}
          max={EXTEND_MAX_S}
          step={1}
          onChange={(extendS) => onChange({ extendS })}
          onDraft={setDraggedExtendS}
          disabled={disabled}
          info="New music is added after the clip's end."
        />
      ) : null}

      {action ? (
        <div className="grid gap-1.5">
          <label
            htmlFor={PROMPT_ID}
            className="text-ui-13 font-medium text-foreground"
          >
            {editUsesStrength(action)
              ? "What should it sound like?"
              : "What should the new part sound like?"}
          </label>
          <Textarea
            id={PROMPT_ID}
            value={draft.prompt}
            onChange={(event) => onChange({ prompt: event.target.value })}
            disabled={disabled}
            placeholder={
              editUsesStrength(action)
                ? "Lo-fi jazz, warm piano, brushed drums…"
                : "A soaring guitar solo over the same groove…"
            }
            className="min-h-20 rounded-xl"
          />
        </div>
      ) : null}
    </div>
  );
}
