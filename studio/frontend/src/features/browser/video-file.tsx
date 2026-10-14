// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  ArrowExpand01Icon,
  ArrowShrink01Icon,
  MoreHorizontalIcon,
  PauseIcon,
  PlayIcon,
  VolumeHighIcon,
  VolumeMute02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type KeyboardEvent, useEffect, useRef, useState } from "react";
import { canCopyVideoFrame, copyVideoFrame, registerTabVideo } from "./video-registry";

const RATES = [0.5, 0.75, 1, 1.25, 1.5, 2] as const;
const SEEK_SECONDS = 5;

function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = String(whole % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${s}` : `${m}:${s}`;
}

const CONTROL =
  "flex size-8 shrink-0 cursor-pointer items-center justify-center rounded-full text-foreground transition-colors hover:bg-[color-mix(in_oklab,var(--foreground)_7%,transparent)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-40";

/** A video tab: the clip on a soft band, with its own controls along the bottom. */
export function VideoFile({
  src,
  name,
  tabId,
  onError,
}: {
  src: string;
  name: string;
  tabId: string | undefined;
  onError: () => void;
}) {
  const t = useT();
  const rootRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [rate, setRate] = useState(1);
  const [loop, setLoop] = useState(false);
  const [fullscreened, setFullscreened] = useState(false);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !tabId) return;
    return registerTabVideo(tabId, video);
  }, [tabId]);

  // Esc and the browser's own controls leave fullscreen too, so follow the document, not the button.
  useEffect(() => {
    const sync = () => setFullscreened(document.fullscreenElement === rootRef.current);
    document.addEventListener("fullscreenchange", sync);
    return () => document.removeEventListener("fullscreenchange", sync);
  }, []);

  const video = () => videoRef.current;
  const togglePlay = () => {
    const el = video();
    if (!el) return;
    if (el.paused || el.ended) void el.play().catch(() => undefined);
    else el.pause();
  };
  const seekBy = (delta: number) => {
    const el = video();
    if (!el) return;
    const end = Number.isFinite(el.duration) ? el.duration : el.currentTime + delta;
    el.currentTime = Math.min(end, Math.max(0, el.currentTime + delta));
  };
  const toggleMute = () => {
    const el = video();
    if (!el) return;
    el.muted = !el.muted;
    if (!el.muted && el.volume === 0) el.volume = 1;
  };
  // The whole player, so its controls stay on screen.
  const fullscreen = () => {
    const root = rootRef.current;
    if (!root) return;
    const request = document.fullscreenElement === root ? document.exitFullscreen() : root.requestFullscreen?.();
    void request?.catch(() => undefined);
  };
  const pictureInPicture = () => {
    const el = video();
    if (!el) return;
    const request =
      document.pictureInPictureElement === el ? document.exitPictureInPicture() : el.requestPictureInPicture();
    void request.catch(() => undefined);
  };
  const copyFrame = () => {
    const el = video();
    if (!el) return;
    void copyVideoFrame(el).then((ok) =>
      ok ? toast.success(t("browser.video.frameCopied")) : toast.error(t("browser.video.copyFrameFailed")),
    );
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    // Keys from the portaled menu bubble here too; inputs and a focused button's Space are theirs.
    if (!event.currentTarget.contains(target) || target instanceof HTMLInputElement) return;
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === " " && target !== event.currentTarget && target.closest("button")) return;
    const handlers: Record<string, () => void> = {
      " ": togglePlay,
      k: togglePlay,
      ArrowLeft: () => seekBy(-SEEK_SECONDS),
      ArrowRight: () => seekBy(SEEK_SECONDS),
      m: toggleMute,
      f: fullscreen,
    };
    const handler = handlers[event.key];
    if (!handler) return;
    event.preventDefault();
    handler();
  };

  const progress = duration > 0 ? Math.min(100, (time / duration) * 100) : 0;
  const level = muted ? 0 : volume;
  const pipSupported = typeof document !== "undefined" && document.pictureInPictureEnabled;

  return (
    // biome-ignore lint/a11y/noNoninteractiveTabindex: the player takes its keyboard shortcuts while focused
    <div
      ref={rootRef}
      tabIndex={0}
      role="group"
      aria-label={name}
      onKeyDown={onKeyDown}
      className="browser-video flex size-full min-h-0 flex-col bg-background outline-none"
    >
      <div className="flex min-h-0 flex-1 items-center" style={{ containerType: "size" }}>
        {/* Brand-tinted band, so any clip reads as framed rather than a black box. */}
        <div className="browser-video-band flex w-full items-center justify-center px-[7%] py-[clamp(1.5rem,6cqh,3.5rem)]">
          {/* biome-ignore lint/a11y/useMediaCaption: user video has no captions */}
          <video
            ref={videoRef}
            src={src}
            playsInline={true}
            loop={loop}
            onClick={togglePlay}
            onDoubleClick={fullscreen}
            onPlay={() => setPlaying(true)}
            onPause={() => setPlaying(false)}
            onEnded={() => setPlaying(false)}
            onTimeUpdate={(event) => setTime(event.currentTarget.currentTime)}
            onDurationChange={(event) => {
              const value = event.currentTarget.duration;
              if (Number.isFinite(value)) setDuration(value);
            }}
            onLoadedMetadata={(event) => {
              const el = event.currentTarget;
              if (Number.isFinite(el.duration)) return;
              // Recorded WebM leaves the duration out until the end has been read: seek there once.
              const settle = () => {
                el.removeEventListener("timeupdate", settle);
                el.currentTime = 0;
              };
              el.addEventListener("timeupdate", settle);
              el.currentTime = Number.MAX_SAFE_INTEGER;
            }}
            onVolumeChange={(event) => {
              setMuted(event.currentTarget.muted);
              setVolume(event.currentTarget.volume);
            }}
            onRateChange={(event) => setRate(event.currentTarget.playbackRate)}
            onError={onError}
            className="block max-h-[calc(100cqh-clamp(3rem,12cqh,7rem))] max-w-full cursor-pointer rounded-md bg-black shadow-[0_18px_50px_-12px_rgb(0_0_0/0.35)]"
          />
        </div>
      </div>
      <div className="browser-video-controls flex h-13 shrink-0 items-center gap-2 border-t border-border/60 px-3">
        <button
          type="button"
          aria-label={playing ? t("browser.video.pause") : t("browser.video.play")}
          onClick={togglePlay}
          className={CONTROL}
        >
          {playing ? (
            <HugeiconsIcon icon={PauseIcon} strokeWidth={1.75} className="size-4.5" />
          ) : (
            <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-4.5" />
          )}
        </button>
        <input
          type="range"
          aria-label={t("browser.video.seek")}
          min={0}
          max={duration || 0}
          step="any"
          value={Math.min(time, duration || 0)}
          disabled={duration === 0}
          onChange={(event) => {
            const el = video();
            if (el) el.currentTime = Number(event.currentTarget.value);
          }}
          className="browser-video-range mx-1 min-w-0 flex-1"
          style={{ ["--range-fill" as string]: `${progress}%` }}
        />
        <span className="shrink-0 px-1 text-ui-13 text-muted-foreground tabular-nums">
          {clock(time)} / {clock(duration)}
        </span>
        <div className="group/volume flex shrink-0 items-center">
          <input
            type="range"
            aria-label={t("browser.video.volume")}
            min={0}
            max={1}
            step={0.05}
            value={level}
            onChange={(event) => {
              const el = video();
              if (!el) return;
              el.volume = Number(event.currentTarget.value);
              el.muted = el.volume === 0;
            }}
            className="browser-video-range w-0 opacity-0 transition-[width,opacity] duration-150 group-focus-within/volume:mr-1 group-focus-within/volume:w-16 group-focus-within/volume:opacity-100 group-hover/volume:mr-1 group-hover/volume:w-16 group-hover/volume:opacity-100"
            style={{ ["--range-fill" as string]: `${level * 100}%` }}
          />
          <button
            type="button"
            aria-label={muted || volume === 0 ? t("browser.video.unmute") : t("browser.video.mute")}
            onClick={toggleMute}
            className={CONTROL}
          >
            {muted || volume === 0 ? (
              <HugeiconsIcon icon={VolumeMute02Icon} strokeWidth={1.75} className="size-4.5" />
            ) : (
              <HugeiconsIcon icon={VolumeHighIcon} strokeWidth={1.75} className="size-4.5" />
            )}
          </button>
        </div>
        <DropdownMenu>
          <DropdownMenuTrigger asChild={true}>
            <button type="button" aria-label={t("browser.video.more")} className={cn(CONTROL, "aria-expanded:bg-[color-mix(in_oklab,var(--foreground)_8%,transparent)]")}>
              <HugeiconsIcon icon={MoreHorizontalIcon} strokeWidth={1.75} className="size-4.5" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            side="top"
            align="end"
            sideOffset={8}
            className="browser-menu min-w-52 rounded-[20px] p-1.5"
          >
            <DropdownMenuLabel className="text-ui-12 text-muted-foreground">
              {t("browser.video.speed")}
            </DropdownMenuLabel>
            <DropdownMenuRadioGroup
              value={String(rate)}
              onValueChange={(value) => {
                const el = video();
                if (el) el.playbackRate = Number(value);
              }}
            >
              {RATES.map((value) => (
                <DropdownMenuRadioItem key={value} value={String(value)} onSelect={(event) => event.preventDefault()}>
                  {value === 1 ? t("browser.video.normalSpeed") : `${value}×`}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
            <DropdownMenuSeparator />
            <DropdownMenuCheckboxItem checked={loop} onCheckedChange={setLoop}>
              {t("browser.video.loop")}
            </DropdownMenuCheckboxItem>
            {pipSupported ? (
              <DropdownMenuItem onSelect={pictureInPicture}>{t("browser.video.pictureInPicture")}</DropdownMenuItem>
            ) : null}
            {canCopyVideoFrame() ? (
              <DropdownMenuItem onSelect={copyFrame}>{t("browser.video.copyFrame")}</DropdownMenuItem>
            ) : null}
          </DropdownMenuContent>
        </DropdownMenu>
        <button
          type="button"
          aria-label={fullscreened ? t("browser.video.exitFullscreen") : t("browser.video.fullscreen")}
          onClick={fullscreen}
          className={CONTROL}
        >
          <HugeiconsIcon
            icon={fullscreened ? ArrowShrink01Icon : ArrowExpand01Icon}
            strokeWidth={1.75}
            className="size-4.5"
          />
        </button>
      </div>
    </div>
  );
}
