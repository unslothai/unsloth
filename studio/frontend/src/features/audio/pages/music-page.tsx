// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { ParamSlider } from "@/features/chat";
import { PillTabs } from "@/features/model-picker/components/model-selector/pill-tabs";
import type { ModelOption } from "@/features/model-picker/components/model-selector/types";
import type { ComponentProps } from "react";
import type { AudioGalleryClip } from "../api";
import {
  isMusicGenerationModel,
  macTtsCatalogChoiceIsRunnable,
} from "../catalog";
import { AudioHistoryProvider } from "../components/audio-source-input";
import { Field } from "../components/field";
import { LyricsEditor } from "../components/lyrics-editor";
import { MusicEditInputs } from "../components/music-edit-inputs";
import type { MusicGeneration } from "../hooks/use-music-generation";
import {
  MUSIC_EXAMPLES,
  durationLabel,
  exampleLyrics,
  instrumentalChoice,
  lyricsShown,
  musicDurationFor,
  musicModeHint,
  musicModeLabel,
  variationsMax,
} from "../music/music-policy";
import type { MusicMode, MusicModeRule } from "../music/music-types";
import { useAudioMusicStore } from "../stores/audio-music-store";
import { TtsOutput, TtsRailFields } from "./tts-workspace";

export const MUSIC_DESCRIPTION_FIELD_ID = "music-description";
export const MUSIC_LYRICS_FIELD_ID = "music-lyrics";
export const MUSIC_SFX_FIELD_ID = "music-sfx-prompt";

export function musicPageModels(
  models: ModelOption[],
  isMac: boolean,
): ModelOption[] {
  return models.filter(
    (model) =>
      isMusicGenerationModel(model.id, model.audioType) &&
      (!isMac || macTtsCatalogChoiceIsRunnable(model.id)),
  );
}

/** One line under a prompt: example picks while it is empty, the hint once it has text. */
function ExamplesOrHint({
  label,
  examples,
  empty,
  hint,
  onPick,
  disabled,
}: {
  label: string;
  examples: readonly { label: string; text: string }[];
  empty: boolean;
  hint: string;
  onPick: (text: string) => void;
  disabled: boolean;
}) {
  if (!empty) {
    return (
      <p className="text-ui-11p5 leading-snug text-muted-foreground">{hint}</p>
    );
  }
  return (
    <fieldset
      className="m-0 flex min-w-0 flex-wrap items-center gap-1 border-0 p-0"
      aria-label={label}
    >
      <span className="me-0.5 text-ui-11p5 text-muted-foreground">Try</span>
      {examples.map((example) => (
        <Button
          key={example.label}
          type="button"
          variant="muted"
          size="sm"
          disabled={disabled}
          className="h-[calc(24px*var(--ui-space-scale,1))] px-2.5 text-ui-11p5 font-normal"
          title={example.text}
          onClick={() => onPick(example.text)}
        >
          {example.label}
        </Button>
      ))}
    </fieldset>
  );
}

function SecondsSlider({
  rule,
  value,
  onChange,
  disabled,
}: {
  rule: MusicModeRule;
  value: number | null;
  onChange: (next: number) => void;
  disabled: boolean;
}) {
  const duration = rule.duration;
  if (!duration) return null;
  const seconds = musicDurationFor(rule, value) ?? duration.default;
  return (
    <ParamSlider
      label={durationLabel(rule)}
      value={seconds}
      min={duration.min}
      max={duration.max}
      step={1}
      valueSize={6}
      disabled={disabled}
      onChange={onChange}
      info={
        duration.approximate
          ? "This model aims for the length; the song may end a little earlier or later."
          : `Between ${duration.min} and ${duration.max} seconds.`
      }
    />
  );
}

function VariationsSlider({
  rule,
  value,
  onChange,
  disabled,
  notice,
}: {
  rule: MusicModeRule;
  value: number;
  onChange: (next: number) => void;
  disabled: boolean;
  notice: string | null;
}) {
  const max = variationsMax(rule);
  if (max === null) return null;
  return (
    <div className="grid gap-1.5">
      <ParamSlider
        label="Variations"
        value={Math.min(max, Math.max(1, value))}
        min={1}
        max={max}
        step={1}
        valueSize={2}
        disabled={disabled}
        onChange={onChange}
        commitWhileDragging
        info="Several takes of the same request, kept together in history."
      />
      {notice ? (
        <p className="text-ui-11p5 leading-snug text-muted-foreground">
          {notice}
        </p>
      ) : null}
    </div>
  );
}

function SongInputs({
  rule,
  music,
  lyrics,
  setLyrics,
  description,
  setDescription,
  disabled,
}: {
  rule: MusicModeRule;
  music: MusicGeneration;
  lyrics: string;
  setLyrics: (next: string) => void;
  description: string;
  setDescription: (next: string) => void;
  disabled: boolean;
}) {
  const patch = useAudioMusicStore.getState().patchDraft;
  const instrumental = instrumentalChoice(rule, music.song.instrumental);
  const showLyrics = lyricsShown(rule, instrumental.value);
  return (
    <>
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-1.5">
        <Field
          label={
            rule.description === "required"
              ? "Description"
              : "Description (optional)"
          }
          htmlFor={MUSIC_DESCRIPTION_FIELD_ID}
        >
          <Textarea
            data-type-to-activate="prompt"
            id={MUSIC_DESCRIPTION_FIELD_ID}
            value={description}
            disabled={disabled}
            onChange={(event) => setDescription(event.target.value)}
            placeholder={
              instrumental.value
                ? "Genre, mood, instruments and tempo…"
                : "Genre, mood, instruments, voice and tempo…"
            }
            className="min-h-16"
          />
        </Field>
        <ExamplesOrHint
          label="Example descriptions"
          examples={
            instrumental.value
              ? MUSIC_EXAMPLES.instrumental
              : MUSIC_EXAMPLES.description
          }
          empty={!description.trim()}
          hint={
            instrumental.value
              ? "Genre, mood, instruments and tempo."
              : "Genre, mood, instruments, voice and tempo."
          }
          onPick={setDescription}
          disabled={disabled}
        />
      </div>
      {instrumental.shown ? (
        <div className="grid gap-1">
          <label
            htmlFor="music-instrumental"
            className="flex items-center justify-between gap-3 text-ui-13 font-medium text-foreground"
          >
            Instrumental
            <Switch
              id="music-instrumental"
              checked={instrumental.value}
              disabled={disabled || !instrumental.enabled}
              onCheckedChange={(checked) =>
                patch("song", { instrumental: checked })
              }
            />
          </label>
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {instrumental.reason ?? "No vocals; the lyrics are kept for later."}
          </p>
        </div>
      ) : null}
      {showLyrics ? (
        <div className="grid gap-1.5">
          <LyricsEditor
            id={MUSIC_LYRICS_FIELD_ID}
            label={rule.lyrics === "required" ? "Lyrics" : "Lyrics (optional)"}
            value={lyrics}
            onChange={setLyrics}
            sectionCase={rule.section_case}
            disabled={disabled}
            describedBy="music-lyrics-hint"
            placeholder={exampleLyrics(
              rule.section_case,
              "Morning light through the pines…\n\n…",
            )}
          />
          {lyrics.trim() ? (
            <p
              id="music-lyrics-hint"
              className="text-ui-11p5 leading-snug text-muted-foreground"
            >
              Section tags shape the song.
            </p>
          ) : (
            <p
              id="music-lyrics-hint"
              className="text-ui-11p5 leading-snug text-muted-foreground"
            >
              Section tags shape the song.{" "}
              <Button
                type="button"
                variant="link"
                disabled={disabled}
                className="h-auto p-0 text-ui-11p5 font-medium text-foreground"
                onClick={() =>
                  setLyrics(
                    exampleLyrics(
                      rule.section_case,
                      MUSIC_EXAMPLES.lyrics[0].text,
                    ),
                  )
                }
              >
                Try example lyrics
              </Button>
            </p>
          )}
        </div>
      ) : null}
      <SecondsSlider
        rule={rule}
        value={music.song.durationS}
        onChange={(durationS) => patch("song", { durationS })}
        disabled={disabled}
      />
      <VariationsSlider
        rule={rule}
        value={music.song.variations}
        onChange={(variations) => patch("song", { variations })}
        disabled={disabled}
        notice={music.reloadNotice}
      />
    </>
  );
}

function SfxInputs({
  rule,
  music,
  disabled,
}: {
  rule: MusicModeRule;
  music: MusicGeneration;
  disabled: boolean;
}) {
  const patch = useAudioMusicStore.getState().patchDraft;
  return (
    <>
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-1.5">
        <Field label="Sound" htmlFor={MUSIC_SFX_FIELD_ID}>
          <Textarea
            id={MUSIC_SFX_FIELD_ID}
            value={music.sfx.prompt}
            disabled={disabled}
            onChange={(event) => patch("sfx", { prompt: event.target.value })}
            placeholder="What you want to hear, and where…"
            className="min-h-16"
          />
        </Field>
        <ExamplesOrHint
          label="Example sounds"
          examples={MUSIC_EXAMPLES.sfx}
          empty={!music.sfx.prompt.trim()}
          hint="What you want to hear, and where."
          onPick={(prompt) => patch("sfx", { prompt })}
          disabled={disabled}
        />
      </div>
      <SecondsSlider
        rule={rule}
        value={music.sfx.durationS}
        onChange={(durationS) => patch("sfx", { durationS })}
        disabled={disabled}
      />
      <VariationsSlider
        rule={rule}
        value={music.sfx.variations}
        onChange={(variations) => patch("sfx", { variations })}
        disabled={disabled}
        notice={music.reloadNotice}
      />
    </>
  );
}

export function MusicStudioInputs({
  music,
  lyrics,
  setLyrics,
  description,
  setDescription,
  historyClips,
  disabled,
}: {
  music: MusicGeneration;
  lyrics: string;
  setLyrics: (next: string) => void;
  description: string;
  setDescription: (next: string) => void;
  historyClips: readonly AudioGalleryClip[];
  disabled: boolean;
}) {
  const { capabilities, rule } = music;
  if (!capabilities || !rule) return null;
  const modes = capabilities.modes.map((mode) => mode.id);
  return (
    <AudioHistoryProvider value={historyClips}>
      {modes.length > 1 ? (
        <div className="grid gap-1.5">
          <PillTabs
            ariaLabel="What to make"
            value={rule.id}
            onValueChange={(mode) => music.setMusicMode(mode as MusicMode)}
            disabled={disabled}
            fit={true}
            compact={true}
            className="self-start [&>button]:px-3"
            tabs={capabilities.modes.map((mode) => ({
              value: mode.id,
              label: musicModeLabel(mode),
            }))}
          />
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {musicModeHint(rule)}
          </p>
        </div>
      ) : (
        <div className="grid gap-0.5">
          <span className="text-ui-13 font-medium text-foreground">
            {musicModeLabel(rule)}
          </span>
          <p className="text-ui-11p5 leading-snug text-muted-foreground">
            {musicModeHint(rule)}
          </p>
        </div>
      )}
      {rule.id === "song" ? (
        <SongInputs
          rule={rule}
          music={music}
          lyrics={lyrics}
          setLyrics={setLyrics}
          description={description}
          setDescription={setDescription}
          disabled={disabled}
        />
      ) : rule.id === "sfx" ? (
        <SfxInputs rule={rule} music={music} disabled={disabled} />
      ) : (
        <MusicEditInputs
          rule={rule}
          draft={music.edit}
          onChange={(patch) =>
            useAudioMusicStore.getState().patchDraft("edit", patch)
          }
          disabled={disabled}
          onSourceStatusChange={music.setSourceStatus}
          sourceHandleRef={music.sourceHandle}
        />
      )}
    </AudioHistoryProvider>
  );
}

export function MusicRail({
  music,
  historyClips,
  description,
  setDescription,
  ...props
}: Omit<ComponentProps<typeof TtsRailFields>, "musicGeneration"> & {
  music?: MusicGeneration;
  historyClips?: readonly AudioGalleryClip[];
  description?: string;
  setDescription?: (next: string) => void;
}) {
  if (!music?.studio || !setDescription) {
    return <TtsRailFields {...props} musicGeneration={true} />;
  }
  return (
    <TtsRailFields
      {...props}
      // Length lives in the page's fields, so Advanced keeps only the model's options.
      musicGeneration={false}
      samplingControls={false}
      inputs={
        <MusicStudioInputs
          music={music}
          lyrics={props.prompt}
          setLyrics={props.setPrompt}
          description={description ?? ""}
          setDescription={setDescription}
          historyClips={historyClips ?? []}
          disabled={props.busy === "generating"}
        />
      }
    />
  );
}

export function MusicOutput({
  modelReady,
  ...props
}: Omit<ComponentProps<typeof TtsOutput>, "emptyText"> & {
  modelReady: boolean;
}) {
  return (
    <TtsOutput
      {...props}
      emptyText={
        modelReady
          ? "Generated music lands here. Write lyrics or a description, then press Generate."
          : "Generated music lands here. Load a music model, write lyrics or a description, and press Generate."
      }
    />
  );
}
