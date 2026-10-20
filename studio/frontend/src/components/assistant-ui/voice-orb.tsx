// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { decayBands, easeToward, isFlat } from "@/features/chat/voice/blob-decay";
import { BLOB_AMP_PX, blobBezierPath, createBlobControls, foldControls } from "@/features/chat/voice/blob-shape";
import {
  ORB_IDLE_GRADIENT,
  ORB_IDLE_SHADOW,
  type OrbStateName,
  SPECTRUM_SIDE,
  type SpectrumReader,
  orbConfig,
} from "@/features/chat/voice/orb-config";
import { type FadeLayer, pushGradient, settleGradient } from "@/features/chat/voice/orb-crossfade";
import {
  ORB_CHIP_HEIGHT,
  ORB_STACK_GAP,
  ORB_STATUS_HEIGHT,
  orbStackLayout,
} from "@/features/chat/voice/orb-stack-layout";
import { pendingApproval } from "@/features/chat/voice/orb-widget";
import { type VoiceDecisions, useVoiceDecisions } from "@/features/chat/voice/voice-decisions";
import { describeToolCall } from "@/features/chat/voice/voice-loop-bridge";
import { SPECTRUM_BANDS, readSpectrum } from "@/features/chat/voice/voice-spectrum";
import { useVoiceSettingsStore } from "@/features/settings";
import { type TranslationKey, useT } from "@/i18n";
import { EditResponseIcon, ReadAloudIcon } from "@/lib/action-bar-icons";
import { isTauri } from "@/lib/api-base";
import { BulbIcon } from "@/lib/bulb-icon";
import { MicIcon as DictateMicIcon } from "@/lib/mic-icon";
import { cn } from "@/lib/utils";
import { HugeiconsIcon } from "@hugeicons/react";
import { AudioLinesIcon, AudioWaveformIcon, InfoIcon, Loader2Icon, MicOffIcon } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { Component, type FC, createRef, useEffect, useRef, useState } from "react";
import { OrbApprovalCard } from "./orb-approval-card";

export type { OrbStateName, SpectrumReader };

// The full-overlay orb is a dark 3D glass sphere. Each state carves an icon into
// it as a recessed "black hole" (near-black fill + a light bottom rim so the
// shape reads as pressed into the surface) and shows a one-word caption below.
const orbMeta: Record<OrbStateName, { label: TranslationKey; icon: IconKind }> = {
  waiting: { label: "voiceOrb.states.waiting", icon: "mic-off" },
  listening: { label: "voiceOrb.states.listening", icon: "mic" },
  hearing: { label: "voiceOrb.states.hearing", icon: "wave" },
  loading: { label: "voiceOrb.states.loading", icon: "spinner" },
  transcribing: { label: "voiceOrb.states.transcribing", icon: "pencil" },
  thinking: { label: "voiceOrb.states.thinking", icon: "brain" },
  generating: { label: "voiceOrb.states.generating", icon: "dots" },
  synthesizing: { label: "voiceOrb.states.synthesizing", icon: "synth" },
  speaking: { label: "voiceOrb.states.speaking", icon: "waveform" },
  paused: { label: "voiceOrb.states.paused", icon: "pause" },
};

type IconKind = "mic" | "mic-off" | "wave" | "dots" | "brain" | "spinner" | "pencil" | "waveform" | "synth" | "pause";

// Deboss: near-black fill with a faint highlight below and a dark cut above, so
// the glyph looks carved into the sphere rather than sitting on top of it.
const HOLE = "#08080b";
const DEBOSS = "drop-shadow(0 1px 0.5px rgba(255,255,255,0.22)) drop-shadow(0 -1px 1.5px rgba(0,0,0,0.6))";
// The same deboss as a box-shadow, for glyphs that move every frame. WebKit keeps a
// filtered element's raster when only its transform changes, so a drop-shadow under
// a per-frame scaleY leaves ghost bands at the unscaled edges (seen on macOS).
const DEBOSS_BOX = "0 1px 0.5px rgba(255,255,255,0.22), 0 -1px 1.5px rgba(0,0,0,0.6)";

// Three dots that jump left-to-right in sequence (thinking). This is a vertical
// translate, not a scale, so the dots never grow or shrink independently -- their
// size stays locked to the ball's breathe.
// A CSS keyframe, not a scripted tween: three scripted dots over the pulsing
// amber body dropped frames in the desktop webview.
const Dots: FC = () => (
  <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
    {[0, 1, 2].map((i) => (
      <span
        key={i}
        style={{
          width: 11,
          height: 11,
          borderRadius: "50%",
          background: HOLE,
          boxShadow: DEBOSS_BOX,
          willChange: "transform",
          animation: `voice-orb-dot 0.7s ease-in-out ${i * 0.15}s infinite`,
        }}
      />
    ))}
  </div>
);

// The sphere's edge is the live spectrum of whoever is talking: your mic while the
// loop listens or hears you, the reply while it speaks. Log bands from low to high and
// back round the ring, mirrored and slowly turning so a held vowel still moves
// (blob-shape.ts). A quiet orb is a plain circle; nothing reaches the ring.
// Wide enough for the edge's full reach (70 + 3 + BLOB_AMP_PX), or the peaks are cut flat.
const ORB_BOX = 196;
const ORB_R = 70;
const BLOB_AMP = BLOB_AMP_PX;
// The speech detector's ring keeps the size it had before the edge grew.
const VAD_RIPPLE_BOX = 184;
// The bezier spans are cheap to rebuild; the frame rate is the display's.
const BLOB_FRAME_MS = 1000 / 61;
// Attack and release on each band per 30 fps frame, on top of the analyser's own
// smoothing, so a burst swells the edge and lets it sink back instead of jittering.
// Scaled by the real frame time, so the feel holds at any frame rate.
const BLOB_ATTACK = 0.45;
const BLOB_RELEASE = 0.2;
const BLOB_EASE_FRAME_MS = 1000 / 30;
const BLOB_KERNEL = Float32Array.from({ length: 7 }, (_, k) => Math.exp(-((k - 3) ** 2) / 4));
const BLOB_KERNEL_SUM = BLOB_KERNEL.reduce((a, b) => a + b, 0);
// One orb on screen at a time, so the per-frame buffers live for the module.
const blobRaw = new Float32Array(SPECTRUM_BANDS);
const blobEased = new Float32Array(SPECTRUM_BANDS);
const blobControls = createBlobControls();

function blobPath(bands: Float32Array, level: number, k = 1): string {
  const c = (ORB_BOX * k) / 2;
  // Held still with the mirror axis horizontal: lows on the right, highs on the left.
  foldControls(bands, 0, blobControls);
  return `path("${blobBezierPath(blobControls, ORB_R + level * 3, BLOB_AMP, c, c, k)}")`;
}

/** Blur round the ring so bands blend into a contour, then ease each band toward it. */
function easeBands(raw: Float32Array, eased: Float32Array, dtMs: number): void {
  const frames = dtMs / BLOB_EASE_FRAME_MS;
  const attack = 1 - Math.pow(1 - BLOB_ATTACK, frames);
  const release = 1 - Math.pow(1 - BLOB_RELEASE, frames);
  for (let i = 0; i < SPECTRUM_BANDS; i++) {
    let sum = 0;
    for (let k = -3; k <= 3; k++) {
      const j = Math.min(SPECTRUM_BANDS - 1, Math.max(0, i + k));
      sum += raw[j] * BLOB_KERNEL[k + 3];
    }
    const target = sum / BLOB_KERNEL_SUM;
    const prev = eased[i];
    eased[i] = prev + (target - prev) * (target > prev ? attack : release);
  }
}

let blobLevel = 0;

/**
 * Drives one element's clip-path from the live spectrum of `side`; a circle when idle.
 * When the voice stops the edge decays flat before it becomes the circle again, and a
 * starting voice eases in, so neither end snaps.
 */
function useBlobClip(
  ref: React.RefObject<HTMLDivElement | null>,
  side: "input" | "output" | undefined,
  active: boolean,
  read: SpectrumReader = readSpectrum,
  k = 1,
): void {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // The body is drawn bigger than the sphere so the spectrum has room; with no
    // voice to show it is clipped back to the sphere's own size.
    const plain = `circle(${ORB_R * k}px at 50% 50%)`;
    const live = side && active ? side : null;
    if (!live && isFlat(blobEased, blobLevel)) {
      el.style.clipPath = plain;
      return;
    }
    let raf = 0;
    let last = Number.NEGATIVE_INFINITY;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < BLOB_FRAME_MS) return;
      const dt = last === Number.NEGATIVE_INFINITY ? BLOB_FRAME_MS : now - last;
      last = now;
      if (live) {
        const level = read(live, blobRaw);
        easeBands(blobRaw, blobEased, dt);
        blobLevel = easeToward(blobLevel, level, dt);
      } else {
        blobLevel = decayBands(blobEased, blobLevel, dt);
        if (isFlat(blobEased, blobLevel)) {
          blobEased.fill(0);
          blobLevel = 0;
          cancelAnimationFrame(raf);
          el.style.clipPath = plain;
          return;
        }
      }
      el.style.clipPath = blobPath(blobEased, blobLevel, k);
    };
    raf = requestAnimationFrame(tick);
    // The clip is left where it is: the next run carries on from it.
    return () => cancelAnimationFrame(raf);
  }, [ref, side, active, read, k]);
}

function scaleOf(el: Element): number {
  try {
    return new DOMMatrixReadOnly(getComputedStyle(el).transform).a || 1;
  } catch {
    return 1;
  }
}

/**
 * Glides the pulse from the scale it had at a state change to rest, instead of the new
 * keyframe snapping it to 1. A class for getSnapshotBeforeUpdate: the only hook that
 * still sees the old pulse before the DOM changes.
 */
class ScaleHandoff extends Component<{
  animationKey: string;
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}> {
  private wrap = createRef<HTMLDivElement>();

  getSnapshotBeforeUpdate(prev: { animationKey: string }): number | null {
    if (prev.animationKey === this.props.animationKey) return null;
    const wrap = this.wrap.current;
    const el = wrap?.querySelector("[data-orb-pulse]");
    if (!wrap || !el) return null;
    // Times the glide still running from the last change, when states flip quickly.
    return scaleOf(el) * scaleOf(wrap);
  }

  componentDidUpdate(_: unknown, __: unknown, from: number | null): void {
    if (from === null) return;
    // A new duration on the same keyframe (listening's slow breathe into hearing's quick
    // one) keeps the old start time, so the pulse jumped to another point of its cycle.
    // Start it over at rest instead; the glide below carries the old scale into it.
    const pulse = this.wrap.current?.querySelector("[data-orb-pulse]");
    for (const a of pulse?.getAnimations?.() ?? []) {
      if ("animationName" in a) a.currentTime = 0;
    }
    for (const a of this.wrap.current?.getAnimations?.() ?? []) a.cancel();
    if (Math.abs(from - 1) > 0.002) {
      this.wrap.current?.animate?.([{ transform: `scale(${from})` }, { transform: "scale(1)" }], {
        duration: 300,
        easing: "ease-out",
      });
    }
  }

  render() {
    return (
      <div ref={this.wrap} className={this.props.className} style={this.props.style}>
        {this.props.children}
      </div>
    );
  }
}

/** The body's gradient; a new state's colour fades in over whatever is showing. */
const GradientCrossfade: FC<{ gradient: string }> = ({ gradient }) => {
  const [layers, setLayers] = useState<FadeLayer[]>(() => [{ g: gradient, id: 0, fading: false }]);
  const next = pushGradient(layers, gradient);
  if (next !== layers) setLayers(next);
  return (
    <>
      {layers.map((layer) => (
        <div
          key={layer.id}
          className="absolute inset-0"
          style={{
            background: layer.g,
            animation: layer.fading ? "voice-orb-fade-in 300ms ease-out both" : undefined,
          }}
          onAnimationEnd={layer.fading ? () => setLayers((prev) => settleGradient(prev, layer.id)) : undefined}
        />
      ))}
    </>
  );
};

// A plain circle that leaves the orb's edge once per speech-detector verdict: green
// when the clip was kept, grey when it was dropped. A circle on purpose, so it reads
// as a call on your audio whichever voice the blob is showing.
const VadRipple: FC = () => {
  const d = useVoiceDecisions();
  if (!d.vad) return null;
  return (
    <span
      key={d.vad.at}
      aria-hidden
      className="pointer-events-none absolute rounded-full"
      style={{
        width: VAD_RIPPLE_BOX,
        height: VAD_RIPPLE_BOX,
        border: `3px solid ${d.vad.kept ? "#34d399" : "#8a8a8a"}`,
        animation: "voice-vad-ripple 700ms ease-out forwards",
      }}
    />
  );
};

const OrbIcon: FC<{ icon: IconKind }> = ({ icon }) => {
  // Soundwave (hearing): a static glyph; the spectrum edge supplies the motion.
  if (icon === "wave") return <AudioLinesIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />;
  if (icon === "dots") return <Dots />;
  // The orb borrows the app's own glyphs where one exists for the same idea, so a
  // state reads the same here as in the composer and the message action bar.
  if (icon === "brain")
    // Thinking: the composer's Thinking toggle bulb.
    return (
      // The glyph's own viewBox runs to the bulb's edge, and the filtered raster is
      // clipped to the box, so the top of the bulb was lost some frames: inset it.
      <span
        style={{
          color: HOLE,
          filter: DEBOSS,
          display: "grid",
          width: 46,
          height: 46,
        }}
      >
        <BulbIcon className="h-full w-full" viewBox="-80 -80 1184 1184" />
      </span>
    );
  if (icon === "spinner")
    return (
      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          duration: 1.1,
          repeat: Number.POSITIVE_INFINITY,
          ease: "linear",
        }}
        style={{ display: "grid", placeItems: "center" }}
      >
        <Loader2Icon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
      </motion.div>
    );
  if (icon === "pencil")
    // Transcribing: a pencil writing down what you said. Static, rides the breathe.
    // Transcribing: the action bar's Edit pencil.
    return (
      <span style={{ color: HOLE, filter: DEBOSS, display: "grid" }}>
        <HugeiconsIcon icon={EditResponseIcon} size={44} strokeWidth={2.2} />
      </span>
    );
  if (icon === "synth")
    // Generating voice: a waveform being built, not the speaker, which means
    // audio is actually playing.
    return <AudioWaveformIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />;
  if (icon === "waveform")
    // Speaking: the action bar's Read aloud speaker.
    return (
      <span style={{ color: HOLE, filter: DEBOSS, display: "grid" }}>
        <HugeiconsIcon icon={ReadAloudIcon} size={46} strokeWidth={2.2} />
      </span>
    );
  if (icon === "mic-off")
    // Waiting on a chat model: the struck-through mic says the loop is open but
    // not capturing, without needing the caption to be read first.
    return <MicOffIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />;
  if (icon === "pause")
    return (
      // Two slim bars, matching the composer button, not lucide's filled blocks.
      <svg width={46} height={46} viewBox="0 0 24 24" fill={HOLE} style={{ filter: DEBOSS }} aria-hidden={true}>
        <rect x="7" y="5" width="2.6" height="14" rx="1.3" />
        <rect x="14.4" y="5" width="2.6" height="14" rx="1.3" />
      </svg>
    );
  // mic (listening): a resting state, so the icon gets NO independent animation.
  // It rides the ball's breathe scale only, so it grows and shrinks in perfect
  // sync with the sphere -- reading as carved into the surface, not floating.
  // The same mic the composer's Dictate button uses, so the two read as one control.
  return (
    <span
      style={{
        color: HOLE,
        filter: DEBOSS,
        display: "grid",
        width: 46,
        height: 46,
      }}
    >
      <DictateMicIcon className="h-full w-full" />
    </span>
  );
};

// What the three turn models last decided, drawn as three arcs around the orb,
// each in the colour of the orb state it belongs to: the turn judge on the left
// in "hearing you" salmon (how sure the turn is over), the filler across the top
// in "speaking" sky (lit while a line was chosen), the mid-think call on the
// right in "thinking" indigo (how sure of its call). Faint tracks until a model
// has spoken; a filled arc never moves on its own.
const RING_R = 110;
const RING_W = 6.5;
const RING = Math.ceil(2 * (RING_R + RING_W));
// Three thirds with rounded ends and a small gap at each seam; the judge owns the
// left third, the filler the top-right, the mid-think call the bottom-right.
const ARC_DEG = 120;
const GAP_DEG = 7;
const ARCS: Array<{
  key: keyof VoiceDecisions;
  start: number;
  color: string;
  label: string;
}> = [
  { key: "turn", start: 180, color: "#fb7185", label: "turn" },
  { key: "filler", start: 300, color: "#38bdf8", label: "filler" },
  { key: "interject", start: 60, color: "#818cf8", label: "interrupt" },
];

/** Short value for the label beside each arc. */
function arcValue(d: VoiceDecisions, key: keyof VoiceDecisions): string {
  if (key === "turn") return d.turn ? `${Math.round(d.turn.p * 100)}% ${d.turn.sent ? "sent" : "wait"}` : "";
  if (key === "filler")
    return d.filler ? `${d.filler.phrase ? "spoke" : "quiet"} ${Math.round(d.filler.p * 100)}%` : "";
  return d.interject ? `${d.interject.action} ${Math.round(d.interject.confidence * 100)}%` : "";
}

const VoiceDecisionRing: FC = () => {
  const d = useVoiceDecisions();
  // The canvas is wider than the ring so the labels outside it are never clipped.
  const PAD = 110;
  const circ = 2 * Math.PI * RING_R;
  // The gap is split between both ends, and a round cap adds half the width on each side.
  const arcLen = (circ * (ARC_DEG - GAP_DEG)) / 360 - RING_W;
  const SIZE = RING + PAD * 2;
  const C = SIZE / 2;
  const fill = (key: keyof VoiceDecisions): number => {
    if (key === "turn") return d.turn ? d.turn.p : 0;
    if (key === "filler") return d.filler ? d.filler.p : 0;
    return d.interject ? d.interject.confidence : 0;
  };
  return (
    <svg
      className="pointer-events-none absolute"
      style={{ left: -PAD, top: -PAD }}
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      aria-hidden
    >
      {ARCS.map((arc) => {
        const v = fill(arc.key);
        // Label at the arc's midpoint, just outside the ring; left arc reads right-aligned.
        const mid = ((arc.start + ARC_DEG / 2) * Math.PI) / 180;
        const lx = C + Math.sin(mid) * (RING_R + 18);
        const ly = C - Math.cos(mid) * (RING_R + 18);
        const anchor = Math.sin(mid) < -0.1 ? "end" : Math.sin(mid) > 0.1 ? "start" : "middle";
        const value = arcValue(d, arc.key);
        // The top arc's two lines stack upward, away from the ring; the side arcs stack down.
        const above = Math.cos(mid) > 0.5;
        const labelY = above ? ly - 14 : ly;
        const valueY = above ? ly : ly + 14;
        return (
          <g key={arc.key}>
            <text
              x={lx}
              y={labelY}
              textAnchor={anchor}
              dominantBaseline="central"
              fill={arc.color}
              fillOpacity={0.85}
              letterSpacing="0.14em"
              style={{
                fontFamily: "var(--font-heading)",
                fontSize: "calc(11px * var(--ui-font-scale, 1))",
                textTransform: "uppercase",
              }}
            >
              {arc.label}
            </text>
            {/* A new value fades in over the old one rather than swapping. */}
            <AnimatePresence initial={false}>
              {value && (
                <motion.text
                  key={value}
                  x={lx}
                  y={valueY}
                  textAnchor={anchor}
                  dominantBaseline="central"
                  fill={arc.color}
                  style={{
                    fontFamily: "var(--font-heading)",
                    fontSize: "calc(11px * var(--ui-font-scale, 1))",
                  }}
                  initial={{ fillOpacity: 0 }}
                  animate={{ fillOpacity: 0.7 }}
                  exit={{ fillOpacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  {value}
                </motion.text>
              )}
            </AnimatePresence>
            <g transform={`rotate(${arc.start + GAP_DEG / 2 - 90} ${C} ${C})`}>
              <circle
                cx={C}
                cy={C}
                r={RING_R}
                fill="none"
                stroke={arc.color}
                strokeOpacity={0.18}
                strokeWidth={RING_W}
                strokeLinecap="round"
                strokeDasharray={`${arcLen} ${circ}`}
              />
              {/* The dash is the full arc; the offset hides the unfilled part, so a
                  new value slides the end along the track instead of jumping. */}
              <circle
                cx={C}
                cy={C}
                r={RING_R}
                fill="none"
                stroke={arc.color}
                strokeOpacity={v > 0 ? 1 : 0}
                strokeWidth={RING_W}
                strokeLinecap="round"
                strokeDasharray={`${arcLen} ${circ}`}
                strokeDashoffset={arcLen * (1 - Math.max(1 / arcLen, v))}
                style={{
                  transition: "stroke-dashoffset 450ms ease-out, stroke-opacity 200ms ease-out",
                }}
              />
            </g>
          </g>
        );
      })}
    </svg>
  );
};

// A small "?" beside the state caption: what the ring and the orb colours mean.
const HELP_ROWS: { color: string; name?: TranslationKey; text: TranslationKey }[] = [
  { color: "#fb7185", name: "voiceOrb.help.turnName", text: "voiceOrb.help.turn" },
  { color: "#38bdf8", name: "voiceOrb.help.fillerName", text: "voiceOrb.help.filler" },
  { color: "#818cf8", name: "voiceOrb.help.interruptName", text: "voiceOrb.help.interrupt" },
  { color: "#34d399", text: "voiceOrb.help.vad" },
  { color: "", text: "voiceOrb.help.button" },
];

const VoiceHelpButton: FC = () => {
  const t = useT();
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={t("voiceOrb.help.aria")}
          className="grid size-6 place-items-center rounded-full text-foreground/40 transition-colors hover:text-foreground/80 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <InfoIcon className="size-4" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="center"
        sideOffset={8}
        className="menu-soft-surface w-80 border-0 p-3 text-ui-13 ring-0"
      >
        <div className="pb-2 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground">
          {t("voiceOrb.help.title")}
        </div>
        <ul className="flex flex-col gap-2">
          {HELP_ROWS.map((row) => (
            <li key={row.text} className="flex gap-2">
              <span
                className={cn("mt-1.5 size-2 shrink-0 rounded-full", !row.color && "bg-muted-foreground")}
                style={row.color ? { background: row.color } : undefined}
              />
              <span>
                {row.name ? <span className="font-medium">{t(row.name)}</span> : null}
                {row.name ? ": " : null}
                {t(row.text)}
              </span>
            </li>
          ))}
        </ul>
      </PopoverContent>
    </Popover>
  );
};

// One line under the state caption naming the newest call, so the arcs never
// need a legend. Empties when the loop stops.
type LatestDecision = {
  at: number;
  tag: string;
  color: string;
  text: string;
  quote: boolean;
};

function latestDecision(d: VoiceDecisions): LatestDecision | undefined {
  const interjectKind = d.interject?.kind ?? d.interject?.action;
  return [
    d.turn && {
      at: d.turn.at,
      tag: "turn",
      color: "#fb7185",
      text: `${d.turn.sent ? "Sent" : "Waiting"}, ${Math.round(d.turn.p * 100)}% done`,
      quote: false,
    },
    d.filler && {
      at: d.filler.at,
      tag: "filler",
      color: "#38bdf8",
      text: d.filler.phrase || "Quiet",
      quote: Boolean(d.filler.phrase),
    },
    d.interject && {
      at: d.interject.at,
      tag: "interrupt",
      color: "#818cf8",
      text: `${interjectKind?.[0]?.toUpperCase() ?? ""}${interjectKind?.slice(1) ?? ""}, ${Math.round(d.interject.confidence * 100)}%`,
      quote: false,
    },
    d.vad && {
      at: d.vad.at,
      tag: "vad",
      color: "#34d399",
      text: d.vad.kept ? "Kept" : "Dropped",
      quote: false,
    },
    d.stt && {
      at: d.stt.at,
      tag: "stt",
      color: "#94a3b8",
      text: d.stt.kept ? "Kept" : `Dropped, ${d.stt.why}`,
      quote: false,
    },
  ]
    .filter((x): x is LatestDecision => Boolean(x))
    .sort((a, b) => b.at - a.at)[0];
}

const VoiceDecisionCaption: FC = () => {
  const latest = latestDecision(useVoiceDecisions());
  return (
    <div className="flex h-9 items-center justify-center">
      <AnimatePresence mode="wait">
        {latest && (
          <motion.div
            key={latest.tag + latest.text}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2 }}
            className="flex max-w-[calc(26rem*var(--ui-space-scale,1))] items-center gap-2.5 rounded-full border border-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] py-1.5 pr-4 pl-3"
          >
            <span className="size-2 shrink-0 rounded-full" style={{ background: latest.color }} />
            <span
              className="shrink-0 text-ui-11 font-medium uppercase tracking-[0.2em]"
              style={{ color: latest.color, fontFamily: "var(--font-heading)" }}
            >
              {latest.tag}
            </span>
            <span className={cn("truncate text-ui-15 text-foreground/85", latest.quote && "italic")}>
              {latest.quote ? `\u201c${latest.text}\u201d` : latest.text}
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const ORB_KEYFRAMES = `
        @keyframes voice-orb-breathe {
          0%, 100% { transform: scale(1);    }
          50%       { transform: scale(1.05); }
        }
        @keyframes voice-orb-pulse {
          0%, 100% { transform: scale(1);    }
          50%       { transform: scale(1.06); }
        }
        @keyframes voice-orb-speak {
          0%, 100% { transform: scale(1);    }
          50%       { transform: scale(1.1);  }
        }
        @keyframes voice-orb-dot {
          0%, 100% { transform: translateY(0);    }
          50%       { transform: translateY(-9px); }
        }
        @keyframes voice-orb-fade-in {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes voice-vad-ripple {
          0%   { transform: scale(1);    opacity: 0.9; }
          100% { transform: scale(1.14); opacity: 0;   }
        }
`;

/** Largest distance of the spectrum edge from the centre, at the full orb's size. */
const ORB_EDGE_MAX_R = ORB_R + 3 + BLOB_AMP;

const scalePx = (shadow: string, k: number) =>
  shadow.replace(/(\d+(?:\.\d+)?)px/g, (_, n: string) => `${Number(n) * k}px`);

/**
 * The bare sphere for the floating orb window: the full orb's body, glow, motion,
 * spectrum edge and carved icon at any size, without the decision ring or caption.
 * `read` supplies the bands, which the widget gets streamed from the main window.
 * Sized natively rather than through a scale transform, so the edge's clip and the
 * glow are laid out at their real extent.
 */
export const OrbBall: FC<{
  orbState: OrbStateName | null;
  muted: boolean;
  size: number;
  read?: SpectrumReader;
}> = ({ orbState, muted, size, read }) => {
  const k = size / 140;
  const cfg = orbState ? orbConfig[orbState] : null;
  // Only the glyph: the ball has no caption.
  const icon =
    orbState && muted && (orbState === "listening" || orbState === "hearing")
      ? ("mic-off" as const)
      : orbState
        ? orbMeta[orbState].icon
        : null;
  const blobRef = useRef<HTMLDivElement | null>(null);
  // The same edge as the in-app orb; the widget keeps its control bar below the edge's reach.
  useBlobClip(blobRef, orbState ? SPECTRUM_SIDE[orbState] : undefined, Boolean(read), read, k);
  return (
    <ScaleHandoff animationKey={cfg ? `${cfg.animation} ${cfg.duration}` : ""} style={{ width: size, height: size }}>
      <style>{ORB_KEYFRAMES}</style>
      <div
        data-orb-pulse
        aria-hidden
        className="relative grid size-full place-items-center transition-[box-shadow] duration-300 ease-out"
        style={{
          borderRadius: "50%",
          // Its own layer, so the pulse is a compositor transform and never a repaint.
          willChange: "transform",
          boxShadow: scalePx(cfg?.shadow ?? ORB_IDLE_SHADOW, k),
          animation: cfg ? `${cfg.animation} ${cfg.duration} ease-in-out infinite` : undefined,
        }}
      >
        <div
          ref={blobRef}
          className="absolute"
          style={{
            width: ORB_BOX * k,
            height: ORB_BOX * k,
            left: (size - ORB_BOX * k) / 2,
            top: (size - ORB_BOX * k) / 2,
            borderRadius: "50%",
            clipPath: `circle(${ORB_R * k}px at 50% 50%)`,
            // Its own layer: the edge repaints every frame, and inside the pulse's layer it
            // made the glow's blurred shadow repaint with it.
            transform: "translateZ(0)",
          }}
        >
          <GradientCrossfade gradient={cfg?.gradient ?? ORB_IDLE_GRADIENT} />
        </div>
        {/* Only the glyph is scaled: it was drawn for the 140 px sphere. */}
        <div
          style={{
            gridArea: "1 / 1",
            display: "grid",
            placeItems: "center",
            transform: `scale(${k})`,
          }}
        >
          {icon && <OrbIcon icon={icon} />}
        </div>
      </div>
    </ScaleHandoff>
  );
};

/** Room kept between the approval card and the composer under the orb. */
const COMPOSER_CLEARANCE = 24;

/**
 * How far to lift the orb's column so the approval card's bottom stays COMPOSER_CLEARANCE
 * above the composer, read from the layout when the card shows and whenever the window,
 * the composer or the card change size: events only, nothing per frame. The column never
 * lifts past the top of the overlay.
 */
function useComposerClearance(
  rootRef: React.RefObject<HTMLDivElement | null>,
  columnRef: React.RefObject<HTMLDivElement | null>,
  ringRef: React.RefObject<HTMLDivElement | null>,
  /** The card's top below the ball's centre (orbStackLayout), or null with no card. */
  cardTop: number | null,
  /** Which call the card shows: a new one is a new element to watch. */
  cardKey: string,
): number {
  const [lift, setLift] = useState(0);
  useEffect(() => {
    const root = rootRef.current;
    const column = columnRef.current;
    const ring = ringRef.current;
    if (cardTop === null || !root || !column || !ring) {
      setLift(0);
      return;
    }
    const composerOf = () =>
      (root.parentElement?.querySelector(".aui-composer-root") ??
        document.querySelector(".aui-thread-composer-dock .aui-composer-root")) as HTMLElement | null;
    const measure = () => {
      const card = column.querySelector("[data-orb-approval]");
      const composer = composerOf();
      if (!card || !composer) return setLift(0);
      // From the layout, not the screen: offsetTop and offsetHeight ignore the lift and the
      // stack's glide, so this is where the card comes to rest without any lift.
      const orbCentre = root.getBoundingClientRect().top + ring.offsetTop + ring.offsetHeight / 2;
      const cardBottom = orbCentre + cardTop + (card as HTMLElement).offsetHeight;
      const need = composer.getBoundingClientRect().top - COMPOSER_CLEARANCE - cardBottom;
      const next = need >= 0 ? 0 : Math.max(need, 8 - column.offsetTop);
      setLift((prev) => (Math.abs(prev - next) < 0.5 ? prev : next));
    };
    const frame = requestAnimationFrame(measure);
    const watch = new ResizeObserver(() => measure());
    watch.observe(root);
    const card = column.querySelector("[data-orb-approval]");
    if (card) watch.observe(card);
    const composer = composerOf();
    if (composer) watch.observe(composer);
    window.addEventListener("resize", measure);
    return () => {
      cancelAnimationFrame(frame);
      watch.disconnect();
      window.removeEventListener("resize", measure);
    };
  }, [rootRef, columnRef, ringRef, cardTop, cardKey]);
  return lift;
}

/** One item of the stack under the ball, gliding to its offset from the ball's centre. */
const StackItem: FC<{ y: number; children: React.ReactNode }> = ({ y, children }) => (
  <div
    className="pointer-events-auto absolute top-0 left-0 w-max"
    style={{
      transform: `translate(-50%, ${y}px)`,
      transition: "transform 150ms ease-out",
    }}
  >
    {children}
  </div>
);

export const VoiceOrb: FC = () => {
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  // Minimized: the loop keeps running (orbState stays set) but the full-screen
  // overlay is hidden so the chat is visible underneath.
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);
  const voiceOff = useChatRuntimeStore((s) => s.voiceMode === "off");
  const showOverlay = Boolean(orbState) && !collapsed && !voiceOff;

  // Desktop: build the floating orb's window while voice is up, so holding the mini
  // orb only has to show it.
  useEffect(() => {
    if (!isTauri || voiceOff) return;
    void import("@/features/chat/voice/orb-widget-host").then((m) => m.prewarmOrbWidget());
  }, [voiceOff]);

  const showArcs = useVoiceSettingsStore((s) => s.showVoiceArcs);
  const showCaption = useVoiceSettingsStore((s) => s.showVoiceCaption);
  const hasDecision = latestDecision(useVoiceDecisions()) !== undefined;
  // Only the voiced chat's calls: the store also holds Compare panes' and background chats'.
  const confirmations = useChatRuntimeStore((s) => s.toolConfirmations);
  const voiceThreadId = useChatRuntimeStore((s) => s.activeThreadId);
  const approval = pendingApproval(confirmations, voiceThreadId, (id) => describeToolCall(id) !== null);
  const stack = orbStackLayout({
    arcs: showOverlay && showArcs,
    chip: showOverlay && showCaption && hasDecision,
    card: showOverlay && approval !== null,
  });
  const rootRef = useRef<HTMLDivElement | null>(null);
  const columnRef = useRef<HTMLDivElement | null>(null);
  const ringRef = useRef<HTMLDivElement | null>(null);
  const lift = useComposerClearance(rootRef, columnRef, ringRef, stack.card, approval?.id ?? "");
  const muted = useChatRuntimeStore((s) => s.voiceMuted);
  const heldForEdit = useChatRuntimeStore((s) => s.voiceHeldForEdit);
  const t = useT();
  const cfg = orbState ? orbConfig[orbState] : null;
  // Muted while it would be listening: the struck mic and a plain caption.
  const meta =
    orbState && muted && (orbState === "listening" || orbState === "hearing")
      ? { label: t("voiceOrb.states.muted"), icon: "mic-off" as const }
      : orbState === "listening" && heldForEdit
        ? { label: t("voiceOrb.waitingForSend"), icon: "mic" as const }
        : orbState
          ? { label: t(orbMeta[orbState].label), icon: orbMeta[orbState].icon }
          : null;

  // Esc only minimizes the orb back to chat -- it does NOT stop voice mode. The
  // single place to turn voice off is the + menu's Voice toggle, so exiting the
  // orb view is never confused with ending the session. Listener attached only
  // while the overlay is showing.
  useEffect(() => {
    if (!showOverlay) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setVoiceOrbCollapsed(true);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [showOverlay, setVoiceOrbCollapsed]);

  const blobRef = useRef<HTMLDivElement | null>(null);
  useBlobClip(blobRef, orbState ? SPECTRUM_SIDE[orbState] : undefined, showOverlay);

  return (
    <>
      <style>{ORB_KEYFRAMES}</style>

      <div
        className={cn(
          // When active, the backdrop must intercept pointer events so hovers
          // don't fall through to hidden chat history underneath (tooltips,
          // hover menus, side panels). The composer dock sits at z-40 above this
          // and keeps its own pointer-events-auto, so it stays interactive. When
          // inactive the backdrop is still mounted (opacity-0), so it must let
          // clicks pass through.
          "absolute inset-0 z-30 flex flex-col items-center justify-center gap-8 bg-background",
          // Fast, and on its own compositor layer: this is a full-viewport
          // cross-fade over a chat that keeps rendering underneath, and at 500ms
          // the swap read as lag rather than a transition.
          "transition-opacity duration-200 ease-out will-change-[opacity]",
          showOverlay ? "pointer-events-auto opacity-100" : "pointer-events-none opacity-0",
        )}
        ref={rootRef}
      >
        <div
          ref={columnRef}
          className="flex flex-col items-center gap-8"
          style={{
            transform: lift ? `translateY(${lift}px)` : undefined,
            transition: "transform 150ms ease-out",
          }}
        >
          {/* Original sphere: a single radial-gradient ball with a soft glow that
            breathes/pulses via the CSS keyframes. The state icon is a near-black
            deboss ("door hole") carved into the surface. It lives INSIDE the ball,
            so it grows and shrinks with the sphere -- reading as part of the
            surface rather than floating on top. */}
          <div ref={ringRef} className="relative grid place-items-center" style={{ width: RING, height: RING }}>
            {/* The stack under the ball: status, decision chip, approval card. Each keeps a fixed
            gap to the one above; the arcs off or the chip hidden pull the rest up. */}
            <div className="pointer-events-none absolute top-1/2 left-1/2 size-0">
              <StackItem y={stack.status}>
                {/* Live caption: names the current state so the color never needs a
                legend. Studio heading font (Hellix / Space Grotesk) to match chrome. */}
                <div className="flex h-5 items-center gap-2">
                  <AnimatePresence mode="wait">
                    {meta && (
                      <motion.span
                        key={meta.label}
                        initial={{ opacity: 0, y: 3 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -3 }}
                        // mode="wait" plays exit then enter, so the swap costs twice this.
                        transition={{ duration: 0.09 }}
                        className="text-ui-13 font-medium uppercase tracking-[0.28em] text-foreground/55"
                        style={{ fontFamily: "var(--font-heading)" }}
                      >
                        {meta.label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                  {showOverlay && <VoiceHelpButton />}
                </div>
              </StackItem>
              {stack.chip !== null && (
                <StackItem y={stack.chip}>
                  <VoiceDecisionCaption />
                </StackItem>
              )}
              {stack.card !== null && approval && (
                <StackItem y={stack.card}>
                  <OrbApprovalCard key={approval.id} id={approval.id} more={approval.more} />
                </StackItem>
              )}
            </div>
            {showOverlay && showArcs && <VoiceDecisionRing />}
            <VadRipple />
            {/* Glides the pulse from its old scale into the new state's. */}
            <ScaleHandoff
              animationKey={cfg && showOverlay ? `${cfg.animation} ${cfg.duration}` : ""}
              className="relative grid place-items-center"
            >
              <div
                data-orb-pulse
                // Transition the two things that change between states, not `all`:
                // that was also animating transform, which fights the breathe keyframe
                // on every state swap.
                className="relative grid place-items-center transition-[box-shadow] duration-300 ease-out"
                aria-hidden
                style={{
                  width: 140,
                  height: 140,
                  borderRadius: "50%",
                  boxShadow: cfg?.shadow ?? "none",
                  // The keyframe and its 32px blurred shadow only run while the orb is
                  // on screen. Minimized, the loop keeps going and orbState stays set,
                  // so without this the sphere kept repainting under an invisible
                  // layer for the whole conversation, and that is what made the swap
                  // back to chat stutter.
                  animation: cfg && showOverlay ? `${cfg.animation} ${cfg.duration} ease-in-out infinite` : undefined,
                }}
              >
                {/* The coloured body, bigger than the shadow disc so the spectrum can push
                past it; clipped to the live blob (a circle when nothing is talking). */}
                <div
                  ref={blobRef}
                  className="absolute grid place-items-center"
                  style={{
                    width: ORB_BOX,
                    height: ORB_BOX,
                    borderRadius: "50%",
                    clipPath: `circle(${ORB_R}px at 50% 50%)`,
                    // Its own layer, so the per-frame edge never repaints the glow's shadow.
                    transform: "translateZ(0)",
                  }}
                >
                  {/* No state yet falls back to the resting tones, not to listening's
                  green: a green orb with no loop behind it reads as live. */}
                  <GradientCrossfade gradient={cfg?.gradient ?? ORB_IDLE_GRADIENT} />
                </div>
                {/* Keyed on visibility too, so the icon unmounts when minimized: the
                speaking bars run a requestAnimationFrame loop that has no business
                ticking under a hidden overlay. */}
                {/* Both glyphs share one grid cell, so the old one fades out under the
                new one instead of leaving a blank frame between them. */}
                <AnimatePresence initial={false}>
                  <motion.div
                    key={showOverlay ? (meta?.icon ?? "none") : "hidden"}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    // Speaking and hearing trade places on every barge-in; a slow swap
                    // read as lag, so this is a quick fade and nothing else.
                    transition={{ duration: 0.09 }}
                    style={{
                      display: "grid",
                      placeItems: "center",
                      gridArea: "1 / 1",
                    }}
                  >
                    {meta && <OrbIcon icon={meta.icon} />}
                  </motion.div>
                </AnimatePresence>
              </div>
            </ScaleHandoff>
          </div>

          {/* Holds the stack's default room in the column, so the ball sits where it always
            has whatever is showing; the stack itself hangs from the ball's centre. */}
          <div
            aria-hidden
            style={{
              height: ORB_STATUS_HEIGHT + ORB_STACK_GAP + ORB_CHIP_HEIGHT,
            }}
          />
        </div>
      </div>
    </>
  );
};
