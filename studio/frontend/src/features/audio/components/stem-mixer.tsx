// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Slider } from "@/components/ui/slider";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  Download01Icon,
  HeadphonesIcon,
  PauseIcon,
  PlayIcon,
  SentIcon,
  VolumeHighIcon,
  VolumeOffIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type KeyboardEvent,
  useEffect,
  useMemo,
  useReducer,
  useRef,
} from "react";
import { useStemTransport } from "../hooks/use-stem-transport";
import {
  INITIAL_STEM_MIXER_STATE,
  effectiveGain,
  stemMix,
  stemMixerReducer,
} from "./stem-mixer-state";
import type { StemMixerProps } from "./stem-mixer-types";
import { WaveformBars } from "./waveform";
import { formatSeconds } from "./waveform-peaks";

const SEEK_STEP_SECONDS = 5;

export function StemMixer({
  groupId,
  title,
  subtitle,
  stems,
  autoFocus = false,
  onDownloadStem,
  onDownloadAll,
  sendTargets,
  onSend,
  active,
}: StemMixerProps) {
  const [mix, dispatch] = useReducer(
    stemMixerReducer,
    INITIAL_STEM_MIXER_STATE,
  );
  const playRef = useRef<HTMLButtonElement | null>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: groupId is the trigger, not a value read inside.
  useEffect(() => {
    dispatch({ type: "reset" });
  }, [groupId]);

  // A stem that failed to load would hold the shared clock forever; the rest play without it.
  const playable = useMemo(() => stems.filter((stem) => !stem.failed), [stems]);
  const gains = playable.map((stem) => effectiveGain(mix, stem.role));
  const sources = useMemo(
    () => playable.map((stem) => ({ id: stem.clipId, src: stem.src })),
    [playable],
  );
  const durationHint = Math.max(0, ...stems.map((stem) => stem.durationS || 0));
  const transport = useStemTransport({
    sources,
    gains,
    active,
    durationHint,
  });
  const { ready, playing, position, duration, toggle, seek } = transport;
  const fraction = duration > 0 ? Math.min(1, position / duration) : 0;
  const loadingCount = playable.filter((stem) => !stem.src).length;

  const focusedGroup = useRef<string | null>(null);
  useEffect(() => {
    if (!(autoFocus && ready) || focusedGroup.current === groupId) return;
    focusedGroup.current = groupId;
    playRef.current?.focus();
  }, [autoFocus, ready, groupId]);

  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as HTMLElement;
    const ours =
      target === event.currentTarget ||
      target.closest("[data-stem-bars]") !== null;
    if (!ours) return;
    if (event.key === " " || event.key === "k" || event.key === "K") {
      event.preventDefault();
      toggle();
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      seek(position + SEEK_STEP_SECONDS);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      seek(position - SEEK_STEP_SECONDS);
    } else if (event.key === "Home") {
      event.preventDefault();
      seek(0);
    }
  };

  const status = loadingCount > 0 ? "Loading stems…" : "";

  return (
    <section
      // biome-ignore lint/a11y/noNoninteractiveTabindex: the mixer is a keyboard target for Space/K and the arrows.
      tabIndex={0}
      aria-label="Stem mixer"
      onKeyDown={handleKeyDown}
      className="@container grid gap-3 rounded-4xl bg-card p-4 ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <output aria-live="polite" aria-atomic="true" className="sr-only">
        {status}
      </output>
      <header className="flex flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-ui-13 font-medium text-foreground">
            {title}
          </h3>
          <p className="truncate text-ui-11p5 text-muted-foreground">
            {subtitle ? `${subtitle} · ` : ""}
            <span className="font-mono tabular-nums">
              {formatSeconds(duration)}
            </span>
          </p>
        </div>
        <Button
          ref={playRef}
          type="button"
          variant="muted"
          size="icon"
          className="size-[calc(32px*var(--ui-space-scale,1))] shrink-0"
          aria-label={playing ? "Pause stems" : "Play stems"}
          disabled={!ready}
          onClick={toggle}
        >
          <HugeiconsIcon
            icon={playing ? PauseIcon : PlayIcon}
            className="size-3.5"
          />
        </Button>
        <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
          {formatSeconds(position)} / {formatSeconds(duration)}
        </span>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={stems.some((stem) => !stem.src)}
          onClick={onDownloadAll}
        >
          <HugeiconsIcon
            icon={Download01Icon}
            strokeWidth={1.75}
            data-icon="inline-start"
          />
          Download all
        </Button>
        {sendTargets.length > 0 ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild={true}>
              <Button type="button" variant="outline" size="sm">
                <HugeiconsIcon
                  icon={SentIcon}
                  strokeWidth={1.75}
                  data-icon="inline-start"
                />
                Send to
                <HugeiconsIcon
                  icon={ArrowDown01Icon}
                  strokeWidth={1.75}
                  data-icon="inline-end"
                />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              {stems.map((stem) => (
                <DropdownMenuSub key={stem.clipId}>
                  <DropdownMenuSubTrigger disabled={!stem.src}>
                    <span>{stem.label}</span>
                  </DropdownMenuSubTrigger>
                  <DropdownMenuSubContent>
                    {sendTargets.map((target) => (
                      <DropdownMenuItem
                        key={target.id}
                        onSelect={() => onSend(target, stem.clipId)}
                      >
                        {target.label}
                      </DropdownMenuItem>
                    ))}
                  </DropdownMenuSubContent>
                </DropdownMenuSub>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
        ) : null}
      </header>
      <ul className="grid gap-2">
        {stems.map((stem) => {
          const level = stemMix(mix, stem.role);
          const silent = effectiveGain(mix, stem.role) === 0;
          const percent = Math.round(level.volume * 100);
          return (
            <li
              key={stem.clipId}
              className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 @[30rem]:grid-cols-[calc(88px*var(--ui-space-scale,1))_minmax(0,1fr)_auto]"
            >
              <span
                className={cn(
                  "col-start-1 row-start-1 truncate text-ui-13 font-medium",
                  silent ? "text-muted-foreground" : "text-foreground",
                )}
              >
                {stem.label}
              </span>
              <div
                data-stem-bars={true}
                aria-busy={!(stem.src || stem.failed)}
                className={cn(
                  "col-span-2 row-start-2 flex min-w-0 @[30rem]:col-span-1 @[30rem]:col-start-2 @[30rem]:row-start-1",
                  !(stem.src || stem.failed) &&
                    "animate-pulse motion-reduce:animate-none",
                  silent && "opacity-35",
                )}
              >
                <WaveformBars
                  peaks={stem.src ? stem.peaks : null}
                  fraction={fraction}
                  label={stem.label}
                  valueNow={position}
                  valueMax={duration}
                  disabled={!ready}
                  onSeek={(next) => seek(next * duration)}
                  className="h-[calc(28px*var(--ui-space-scale,1))]"
                />
              </div>
              <div className="col-start-2 row-start-1 flex items-center gap-1 @[30rem]:col-start-3">
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Solo ${stem.label}`}
                  aria-pressed={level.solo}
                  className="size-[calc(24px*var(--ui-space-scale,1))] text-muted-foreground aria-pressed:bg-foreground aria-pressed:text-background"
                  onClick={() =>
                    dispatch({ type: "toggleSolo", role: stem.role })
                  }
                >
                  <HugeiconsIcon icon={HeadphonesIcon} strokeWidth={1.75} />
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Mute ${stem.label}`}
                  aria-pressed={level.muted}
                  className="size-[calc(24px*var(--ui-space-scale,1))] text-muted-foreground aria-pressed:bg-foreground aria-pressed:text-background"
                  onClick={() =>
                    dispatch({ type: "toggleMute", role: stem.role })
                  }
                >
                  <HugeiconsIcon
                    icon={level.muted ? VolumeOffIcon : VolumeHighIcon}
                    strokeWidth={1.75}
                  />
                </Button>
                <Slider
                  className="panel-slider w-[calc(88px*var(--ui-space-scale,1))] px-1"
                  min={0}
                  max={100}
                  step={1}
                  value={[percent]}
                  aria-label={`${stem.label} volume`}
                  thumbValueText={(value) => `${stem.label} volume ${value} %`}
                  onValueChange={([value]) =>
                    dispatch({
                      type: "setVolume",
                      role: stem.role,
                      volume: (value ?? percent) / 100,
                    })
                  }
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-xs"
                  aria-label={`Download ${stem.label}`}
                  className="size-[calc(24px*var(--ui-space-scale,1))] text-muted-foreground"
                  disabled={!stem.src}
                  onClick={() => onDownloadStem(stem.clipId)}
                >
                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} />
                </Button>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
