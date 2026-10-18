// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  useVoiceDecisions,
  type VoiceDecisions,
} from "@/features/chat/voice/voice-decisions";
import { EditResponseIcon, ReadAloudIcon } from "@/lib/action-bar-icons";
import { BulbIcon } from "@/lib/bulb-icon";
import { MicIcon as DictateMicIcon } from "@/lib/mic-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { cn } from "@/lib/utils";
import {
  AudioLinesIcon,
  AudioWaveformIcon,
  Loader2Icon,
  MicOffIcon,
  InfoIcon,
  PauseIcon,
} from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, type FC } from "react";
import { SPECTRUM_BANDS, readSpectrum } from "@/features/chat/voice/voice-spectrum";

const ORB_IDLE_GRADIENT_VALUE =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";

export const orbConfig = {
  // Voice is open but no chat model is loaded, so nothing can answer a turn. The
  // resting theme tones, a slow breathe, and a caption that names the block --
  // it is the one state that is waiting on the user, not on the machine.
  waiting: {
    gradient:
      "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))",
    animation: "voice-orb-breathe",
    duration: "4.5s",
    shadow: "0 0 26px 6px var(--voice-orb-idle-glow)",
  },
  listening: {
    gradient: "radial-gradient(circle at 40% 35%, #34d399, #059669)",
    animation: "voice-orb-breathe",
    duration: "3s",
    shadow: "0 0 32px 8px rgba(52,211,153,0.35)",
  },
  // Salmon pink: the mic is actively hearing your voice (distinct from the idle
  // green "listening"). A quicker breathe so it reads as responsive/live.
  hearing: {
    gradient: "radial-gradient(circle at 40% 35%, #ffb0a0, #fb7185)",
    animation: "voice-orb-breathe",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(251,113,133,0.4)",
  },
  // Grey-blue slow pulse: a model is loading / warming (the voice slot, or the
  // transcription model),
  // so it clearly reads as "not ready yet" -- distinct from the idle grey and the
  // ready green. Separate from lilac, which now means only TTS generating speech.
  loading: {
    gradient: "radial-gradient(circle at 40% 35%, #94a3b8, #475569)",
    animation: "voice-orb-pulse",
    duration: "2s",
    shadow: "0 0 26px 6px rgba(148,163,184,0.3)",
  },
  // Fuchsia quick pulse: the transcription engine is turning your captured
  // speech into text. A
  // magenta hue -- the farthest from the idle/listening green on the palette -- so
  // "transcribing" never reads as "ready", and its pencil icon plus this colour
  // keep it distinct from the amber "generating LLM response" below.
  transcribing: {
    gradient: "radial-gradient(circle at 40% 35%, #e879f9, #c026d3)",
    animation: "voice-orb-pulse",
    duration: "1.1s",
    shadow: "0 0 30px 7px rgba(217,70,239,0.4)",
  },
  // Indigo slow breathe: a reasoning model is thinking, no answer text yet. Slower
  // than generating on purpose; this is the long wait of a thinking turn.
  thinking: {
    gradient: "radial-gradient(circle at 40% 35%, #a5b4fc, #4f46e5)",
    animation: "voice-orb-breathe",
    duration: "2.4s",
    shadow: "0 0 32px 8px rgba(129,140,248,0.35)",
  },
  // Amber pulse: the LLM is writing its reply, nothing playing yet. The typing
  // dots are the classic "assistant is generating a response" indicator.
  generating: {
    gradient: "radial-gradient(circle at 40% 35%, #fbbf24, #d97706)",
    animation: "voice-orb-pulse",
    duration: "1s",
    shadow: "0 0 32px 8px rgba(251,191,36,0.35)",
  },
  synthesizing: {
    gradient: "radial-gradient(circle at 40% 35%, #c4b5fd, #8b5cf6)",
    animation: "voice-orb-pulse",
    duration: "1.4s",
    shadow: "0 0 32px 8px rgba(167,139,250,0.35)",
  },
  speaking: {
    gradient: "radial-gradient(circle at 40% 35%, #38bdf8, #0284c7)",
    animation: "voice-orb-speak",
    duration: "0.8s",
    shadow: "0 0 32px 8px rgba(56,189,248,0.35)",
  },
  // Paused by the user: the resting tones, barely breathing, so it reads as held
  // rather than busy or broken.
  paused: {
    gradient: ORB_IDLE_GRADIENT_VALUE,
    animation: "voice-orb-breathe",
    duration: "6s",
    shadow: "0 0 26px 6px var(--voice-orb-idle-glow)",
  },
} as const;

type OrbStateName = keyof typeof orbConfig;

// The orb at rest, before the loop starts. Theme tokens rather than fixed greys
// (index.css, :root and .dark): a flat mid-grey reads as disabled chrome on the
// light page and as a hole on the dark one.
export const ORB_IDLE_GRADIENT =
  "radial-gradient(circle at 40% 35%, var(--voice-orb-idle-from), var(--voice-orb-idle-to))";
export const ORB_IDLE_SHADOW = "0 0 26px 6px var(--voice-orb-idle-glow)";

// The full-overlay orb is a dark 3D glass sphere. Each state carves an icon into
// it as a recessed "black hole" (near-black fill + a light bottom rim so the
// shape reads as pressed into the surface) and shows a one-word caption below.
const orbMeta: Record<OrbStateName, { label: string; icon: IconKind }> = {
  waiting: { label: "Select a model", icon: "mic-off" },
  listening: { label: "Listening", icon: "mic" },
  hearing: { label: "Hearing you", icon: "wave" },
  loading: { label: "Warming up", icon: "spinner" },
  transcribing: { label: "Transcribing", icon: "pencil" },
  thinking: { label: "Thinking", icon: "brain" },
  generating: { label: "Writing the answer", icon: "dots" },
  synthesizing: { label: "Generating voice", icon: "synth" },
  speaking: { label: "Speaking", icon: "waveform" },
  paused: { label: "Paused", icon: "pause" },
};

type IconKind =
  | "mic"
  | "mic-off"
  | "wave"
  | "dots"
  | "brain"
  | "spinner"
  | "pencil"
  | "waveform"
  | "synth"
  | "pause";

// Deboss: near-black fill with a faint highlight below and a dark cut above, so
// the glyph looks carved into the sphere rather than sitting on top of it.
const HOLE = "#08080b";
const DEBOSS =
  "drop-shadow(0 1px 0.5px rgba(255,255,255,0.22)) drop-shadow(0 -1px 1.5px rgba(0,0,0,0.6))";
// The same deboss as a box-shadow, for glyphs that move every frame. WebKit keeps a
// filtered element's raster when only its transform changes, so a drop-shadow under
// a per-frame scaleY leaves ghost bands at the unscaled edges (seen on macOS).
const DEBOSS_BOX =
  "0 1px 0.5px rgba(255,255,255,0.22), 0 -1px 1.5px rgba(0,0,0,0.6)";

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
// loop listens or hears you, the reply while it speaks. Log bands from low at the
// bottom to high at the top, mirrored, blurred round the ring and slowly turning so
// a held vowel still moves. A quiet orb is a plain circle; nothing reaches the ring.
const ORB_BOX = 184;
const ORB_R = 70;
const BLOB_AMP = 16;
const BLOB_POINTS = 96;
// Reassigning a 97-point clip-path is a repaint of the whole orb; 30 writes a
// second is plenty for a contour the eye reads as fluid, and halves that work.
const BLOB_FRAME_MS = 1000 / 30;
// Attack and release on each band, on top of the analyser's own smoothing, so a
// burst swells the edge and lets it sink back instead of jittering frame to frame.
const BLOB_ATTACK = 0.45;
const BLOB_RELEASE = 0.2;
const BLOB_KERNEL = Float32Array.from({ length: 7 }, (_, k) => Math.exp(-((k - 3) ** 2) / 4));
const BLOB_KERNEL_SUM = BLOB_KERNEL.reduce((a, b) => a + b, 0);
const BLOB_COS = new Float32Array(BLOB_POINTS + 1);
const BLOB_SIN = new Float32Array(BLOB_POINTS + 1);
for (let i = 0; i <= BLOB_POINTS; i++) {
  BLOB_COS[i] = Math.cos((i / BLOB_POINTS) * Math.PI * 2);
  BLOB_SIN[i] = Math.sin((i / BLOB_POINTS) * Math.PI * 2);
}
// One orb on screen at a time, so the per-frame buffers live for the module.
const blobRaw = new Float32Array(SPECTRUM_BANDS);
const blobEased = new Float32Array(SPECTRUM_BANDS);
const blobParts: string[] = new Array(BLOB_POINTS + 1);
const SPECTRUM_SIDE: Partial<Record<OrbStateName, "input" | "output">> = {
  listening: "input",
  hearing: "input",
  speaking: "output",
};

function blobPath(bands: Float32Array, t: number, level: number): string {
  const c = ORB_BOX / 2;
  const n = bands.length;
  const base = ORB_R + level * 3;
  const phase = -Math.PI / 2 + t * 0.15;
  for (let i = 0; i <= BLOB_POINTS; i++) {
    const a = (i / BLOB_POINTS) * Math.PI * 2;
    // Angle from the bottom, folded so left and right mirror; the slow turn keeps it alive.
    const fold = (1 - Math.cos(a + phase)) / 2;
    const x = fold * (n - 1);
    const lo = Math.floor(x);
    const hi = Math.min(n - 1, lo + 1);
    const v = bands[lo] + (bands[hi] - bands[lo]) * (x - lo);
    const r = base + v * BLOB_AMP;
    blobParts[i] = `${(c + BLOB_COS[i] * r).toFixed(1)} ${(c + BLOB_SIN[i] * r).toFixed(1)}`;
  }
  return `path("M${blobParts.join("L")}Z")`;
}

/** Blur round the ring so bands blend into a contour, then ease each band toward it. */
function easeBands(raw: Float32Array, eased: Float32Array): void {
  for (let i = 0; i < SPECTRUM_BANDS; i++) {
    let sum = 0;
    for (let k = -3; k <= 3; k++) {
      const j = Math.min(SPECTRUM_BANDS - 1, Math.max(0, i + k));
      sum += raw[j] * BLOB_KERNEL[k + 3];
    }
    const target = sum / BLOB_KERNEL_SUM;
    const prev = eased[i];
    eased[i] = prev + (target - prev) * (target > prev ? BLOB_ATTACK : BLOB_RELEASE);
  }
}

/** Drives one element's clip-path from the live spectrum of `side`; a circle when idle. */
function useBlobClip(
  ref: React.RefObject<HTMLDivElement | null>,
  side: "input" | "output" | undefined,
  active: boolean,
): void {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    // The body is drawn bigger than the sphere so the spectrum has room; with no
    // voice to show it is clipped back to the sphere's own size.
    const plain = `circle(${ORB_R}px at 50% 50%)`;
    if (!side || !active) {
      el.style.clipPath = plain;
      return;
    }
    blobEased.fill(0);
    el.style.willChange = "clip-path";
    let raf = 0;
    let last = -Infinity;
    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      if (now - last < BLOB_FRAME_MS) return;
      last = now;
      const level = readSpectrum(side, blobRaw);
      easeBands(blobRaw, blobEased);
      el.style.clipPath = blobPath(blobEased, now / 1000, level);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(raf);
      el.style.willChange = "";
      el.style.clipPath = plain;
    };
  }, [ref, side, active]);
}

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
        width: ORB_BOX,
        height: ORB_BOX,
        border: `3px solid ${d.vad.kept ? "#34d399" : "#8a8a8a"}`,
        animation: "voice-vad-ripple 700ms ease-out forwards",
      }}
    />
  );
};

const OrbIcon: FC<{ icon: IconKind }> = ({ icon }) => {
  // Soundwave (hearing): a static glyph -- the ball's pulse supplies the motion
  // and the icon rides it, growing/shrinking with the sphere, not on its own.
  if (icon === "wave")
    return (
      <AudioLinesIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  if (icon === "dots") return <Dots />;
  // The orb borrows the app's own glyphs where one exists for the same idea, so a
  // state reads the same here as in the composer and the message action bar.
  if (icon === "brain")
    // Thinking: the composer's Thinking toggle bulb.
    return (
      // The glyph's own viewBox runs to the bulb's edge, and the filtered raster is
      // clipped to the box, so the top of the bulb was lost some frames: inset it.
      <span style={{ color: HOLE, filter: DEBOSS, display: "grid", width: 46, height: 46 }}>
        <BulbIcon className="h-full w-full" viewBox="-80 -80 1184 1184" />
      </span>
    );
  if (icon === "spinner")
    return (
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1.1, repeat: Infinity, ease: "linear" }}
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
    return (
      <AudioWaveformIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
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
    return (
      <MicOffIcon size={46} strokeWidth={2.4} color={HOLE} style={{ filter: DEBOSS }} />
    );
  if (icon === "pause")
    return (
      <PauseIcon size={44} strokeWidth={2.4} color={HOLE} fill={HOLE} style={{ filter: DEBOSS }} />
    );
  // mic (listening): a resting state, so the icon gets NO independent animation.
  // It rides the ball's breathe scale only, so it grows and shrinks in perfect
  // sync with the sphere -- reading as carved into the surface, not floating.
  // The same mic the composer's Dictate button uses, so the two read as one control.
  return (
    <span style={{ color: HOLE, filter: DEBOSS, display: "grid", width: 46, height: 46 }}>
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
const ARCS: Array<{ key: keyof VoiceDecisions; start: number; color: string; label: string }> = [
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
              fontSize={11}
              letterSpacing="0.14em"
              style={{ fontFamily: "var(--font-heading)", textTransform: "uppercase" }}
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
                  fontSize={11}
                  style={{ fontFamily: "var(--font-heading)" }}
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
const VoiceHelpButton: FC = () => (
  <Popover>
    <PopoverTrigger asChild>
      <button
        type="button"
        aria-label="What voice mode is detecting"
        className="grid size-6 place-items-center rounded-full text-foreground/40 transition-colors hover:text-foreground/80"
      >
        <InfoIcon className="size-4" />
      </button>
    </PopoverTrigger>
    <PopoverContent
      align="center"
      sideOffset={8}
      className="menu-soft-surface w-80 rounded-lg border-0 p-3 text-ui-13 ring-0"
    >
      <div className="pb-2 text-ui-11 font-semibold uppercase tracking-wide text-muted-foreground">
        What the ring shows
      </div>
      <ul className="flex flex-col gap-2">
        <li className="flex gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: "#fb7185" }} />
          <span>
            <span className="font-medium">Turn</span>: how sure the turn judge is that you finished
            speaking. Low and it keeps listening with your words in the box.
          </span>
        </li>
        <li className="flex gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: "#38bdf8" }} />
          <span>
            <span className="font-medium">Filler</span>: whether it said a short line while the
            model thinks, like "let me check". Quiet on greetings and short answers.
          </span>
        </li>
        <li className="flex gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: "#818cf8" }} />
          <span>
            <span className="font-medium">Interrupt</span>: what it did with speech that landed while
            the model was thinking. Restart folds it into the request, queue holds it for after
            the reply, ignore drops it.
          </span>
        </li>
        <li className="flex gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full" style={{ background: "#34d399" }} />
          <span>
            Every clip and every interruption is checked for a real voice first, so a cough or a
            door never reaches the models.
          </span>
        </li>
        <li className="flex gap-2">
          <span className="mt-1.5 size-2 shrink-0 rounded-full bg-muted-foreground" />
          <span>
            The composer button: tap to pause (mic off, reply held where it is), tap again to
            resume. Hold it until the ring fills and let go to stop everything and listen again.
          </span>
        </li>
      </ul>
    </PopoverContent>
  </Popover>
);

// One line under the state caption naming the newest call, so the arcs never
// need a legend. Empties when the loop stops.
const VoiceDecisionCaption: FC = () => {
  const d = useVoiceDecisions();
  const latest = [
    d.turn && {
      at: d.turn.at,
      text: `${d.turn.sent ? "sent" : "waiting"} · ${Math.round(d.turn.p * 100)}% done`,
    },
    d.filler && { at: d.filler.at, text: d.filler.phrase ? `filler · "${d.filler.phrase}"` : "filler · quiet" },
    d.interject && {
      at: d.interject.at,
      text: `mid-think · ${d.interject.action} ${Math.round(d.interject.confidence * 100)}%`,
    },
    d.vad && { at: d.vad.at, text: `vad · ${d.vad.kept ? "kept" : "dropped"}` },
    d.stt && { at: d.stt.at, text: `stt · ${d.stt.kept ? "kept" : `dropped, ${d.stt.why}`}` },
  ]
    .filter((x): x is { at: number; text: string } => Boolean(x))
    .sort((a, b) => b.at - a.at)[0];
  return (
    <div className="h-5">
      <AnimatePresence mode="wait">
        {latest && (
          <motion.span
            key={latest.text}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.2 }}
            className="block max-w-[calc(22rem*var(--ui-space-scale,1))] truncate text-ui-13 text-foreground/70"
          >
            {latest.text}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
};

export const VoiceOrb: FC = () => {
  const orbState = useChatRuntimeStore((s) => s.voiceOrbState);
  // Minimized: the loop keeps running (orbState stays set) but the full-screen
  // overlay is hidden so the chat is visible underneath.
  const collapsed = useChatRuntimeStore((s) => s.voiceOrbCollapsed);
  const setVoiceOrbCollapsed = useChatRuntimeStore((s) => s.setVoiceOrbCollapsed);
  const voiceOff = useChatRuntimeStore((s) => s.voiceMode === "off");
  const showOverlay = Boolean(orbState) && !collapsed && !voiceOff;

  const muted = useChatRuntimeStore((s) => s.voiceMuted);
  const cfg = orbState ? orbConfig[orbState] : null;
  // Muted while it would be listening: the struck mic and a plain caption.
  const meta =
    orbState && muted && (orbState === "listening" || orbState === "hearing")
      ? { label: "Muted", icon: "mic-off" as const }
      : orbState
        ? orbMeta[orbState]
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
      <style>{`
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
        @keyframes voice-vad-ripple {
          0%   { transform: scale(1);    opacity: 0.9; }
          100% { transform: scale(1.14); opacity: 0;   }
        }
      `}</style>

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
          showOverlay
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0",
        )}
      >
        {/* Original sphere: a single radial-gradient ball with a soft glow that
            breathes/pulses via the CSS keyframes. The state icon is a near-black
            deboss ("door hole") carved into the surface. It lives INSIDE the ball,
            so it grows and shrinks with the sphere -- reading as part of the
            surface rather than floating on top. */}
        <div className="relative grid place-items-center" style={{ width: RING, height: RING }}>
        {showOverlay && <VoiceDecisionRing />}
        <VadRipple />
        <div
          // Transition the two things that change between states, not `all`:
          // that was also animating transform, which fights the breathe keyframe
          // on every state swap.
          className="relative grid place-items-center transition-[box-shadow] duration-100 ease-out"
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
            animation:
              cfg && showOverlay
                ? `${cfg.animation} ${cfg.duration} ease-in-out infinite`
                : undefined,
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
              // No state yet falls back to the resting tones, not to listening's
              // green: a green orb with no loop behind it reads as live.
              background: cfg?.gradient ?? ORB_IDLE_GRADIENT,
            }}
          />
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
              style={{ display: "grid", placeItems: "center", gridArea: "1 / 1" }}
            >
              {meta && <OrbIcon icon={meta.icon} />}
            </motion.div>
          </AnimatePresence>
        </div>

        </div>

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
        {showOverlay && <VoiceDecisionCaption />}
      </div>
    </>
  );
};
