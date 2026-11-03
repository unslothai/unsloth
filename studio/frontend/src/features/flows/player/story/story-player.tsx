// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowLeftIcon,
  FastForwardIcon,
  HistoryIcon,
  PlayIcon,
  RotateCcwIcon,
  SaveIcon,
  SettingsIcon,
  SparklesIcon,
  Volume2Icon,
  VolumeXIcon,
  XIcon,
} from "lucide-react";
import {
  memo,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import { cn } from "@/lib/utils";

import { seedFrom } from "../../expr";
import type {
  MediaValue,
  StoryCharacter,
  StoryManifest,
  StoryPosition,
  StoryScene,
  StoryTransition,
} from "../../types";
import {
  type PlayerApi,
  type SessionRecord,
  mediaObjectUrl,
} from "../player-api";
import { useMediaUrl } from "../use-media-url";
import {
  BackgroundStage,
  PlayerKeyframes,
  spriteAnimation,
  useReducedMotion,
} from "./backdrop";
import {
  Lanes,
  MAX_WRITE_AHEAD_DEPTH,
  WriteAhead,
  historyText,
  nextSceneRequest,
  planWriteAhead,
  storyContext,
  writeKey,
} from "./lookahead";
import {
  CHOICE_SPEAKER,
  type StoryAction,
  type StoryState,
  atSceneStart,
  backgroundFor,
  currentLine,
  initialState,
  interpolate,
  nextSceneIds,
  pauseOf,
  reduce,
  saveFits,
  sceneOf,
  stageAt,
  visibleChoices,
} from "./reducer";
import { playCues } from "./sound-cues";

const SPEED_KEY = "unsloth.flows.player.textSpeed";
const VOICE_KEY = "unsloth.flows.player.voice";
const WRITE_AHEAD_KEY = "unsloth.flows.player.writeAhead";
const SLOTS = ["1", "2", "3", "4", "5", "6"];
const AUTO_SLOT = "story-auto";

export interface StoryPlayerProps {
  flowId: string;
  api: PlayerApi;
  manifest: StoryManifest;
  seed?: number;
  canReroll?: boolean;
  onReroll?: () => void;
  onExit?: () => void;
}

function readNumber(key: string, fallback: number): number {
  try {
    const v = Number(localStorage.getItem(key));
    return Number.isFinite(v) && v > 0 ? v : fallback;
  } catch {
    return fallback;
  }
}

function readWriteAhead(): number {
  try {
    const v = localStorage.getItem(WRITE_AHEAD_KEY);
    if (v === null) return 1;
    const n = Number(v);
    return Number.isInteger(n) && n >= 0 && n <= MAX_WRITE_AHEAD_DEPTH ? n : 1;
  } catch {
    return 1;
  }
}

interface TypewriterHandle {
  complete: () => void;
  typing: () => boolean;
}

const Typewriter = memo(function Typewriter({
  text,
  cps,
  handle,
  onDone,
}: {
  text: string;
  cps: number;
  handle: React.MutableRefObject<TypewriterHandle | null>;
  onDone: () => void;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const chars = Array.from(text);
    let shown = cps >= 1000 ? chars.length : 0;
    let raf = 0;
    let finished = false;
    const start = performance.now();
    const finish = () => {
      if (finished) return;
      finished = true;
      cancelAnimationFrame(raf);
      el.textContent = text;
      doneRef.current();
    };
    const tick = (now: number) => {
      const want = Math.min(
        chars.length,
        Math.floor(((now - start) / 1000) * cps),
      );
      if (want !== shown) {
        shown = want;
        el.textContent = chars.slice(0, shown).join("");
      }
      if (shown >= chars.length) finish();
      else raf = requestAnimationFrame(tick);
    };
    el.textContent = chars.slice(0, shown).join("");
    if (shown >= chars.length) finish();
    else raf = requestAnimationFrame(tick);
    handle.current = { complete: finish, typing: () => !finished };
    return () => cancelAnimationFrame(raf);
  }, [text, cps, handle]);
  return <span ref={ref} />;
});

const AT: Record<StoryPosition, string> = {
  left: "22%",
  twoleft: "33%",
  center: "50%",
  tworight: "67%",
  right: "78%",
};

const Portrait = memo(function Portrait({
  api,
  character,
  speaking,
  expression,
  slot,
  at,
  enter,
  exit,
  instant,
}: {
  api: PlayerApi;
  character: StoryCharacter;
  speaking: boolean;
  expression: string | null;
  slot: number;
  at: StoryPosition | null;
  enter: StoryTransition | null;
  exit?: StoryTransition | null;
  instant: boolean;
}) {
  const media =
    (expression && character.expressions?.[expression]) ||
    character.portrait ||
    null;
  const url = useMediaUrl(api, media);
  const side = character.side ?? "left";
  const offset = at
    ? AT[at]
    : side === "center"
      ? "50%"
      : side === "left"
        ? `${14 + slot * 13}%`
        : `${86 - slot * 13}%`;
  const motion = instant
    ? undefined
    : exit
      ? spriteAnimation(exit, true)
      : enter
        ? spriteAnimation(enter, false)
        : undefined;
  return (
    <div
      className={cn(
        "absolute bottom-0 flex items-end justify-center ease-out",
        !instant && "transition-all duration-300",
      )}
      style={{
        left: offset,
        transform: `translateX(-50%) translateY(${speaking ? "0" : "6px"}) scale(${speaking ? 1.02 : 0.97})`,
        filter: speaking ? "none" : "brightness(0.55) saturate(0.8)",
        height: "78%",
        zIndex: speaking ? 3 : 2,
        pointerEvents: exit ? "none" : undefined,
      }}
    >
      <div className="flex h-full items-end" style={{ animation: motion }}>
        {url ? (
          <img
            key={url}
            src={url}
            alt={character.name}
            draggable={false}
            className="h-full w-auto max-w-[38vw] object-contain drop-shadow-[0_12px_28px_rgba(0,0,0,0.55)]"
            style={
              instant ? undefined : { animation: "flows-swap 200ms ease-out" }
            }
          />
        ) : (
          <div
            className="mb-[16vh] flex aspect-[3/4] h-[42%] min-h-40 items-center justify-center rounded-[2rem] text-6xl font-bold text-white/90 shadow-2xl ring-1 ring-white/20"
            style={{
              background: `linear-gradient(160deg, ${character.color ?? "#a78bfa"} 0%, rgba(20,20,30,0.85) 100%)`,
            }}
          >
            {character.name.slice(0, 1)}
          </div>
        )}
      </div>
    </div>
  );
});

function Overlay({
  children,
  onClose,
}: { children: React.ReactNode; onClose?: () => void }) {
  return (
    <div
      className="absolute inset-0 z-30 flex items-center justify-center bg-black/60 p-6 backdrop-blur-sm"
      onClick={(e) => {
        e.stopPropagation();
        if (e.target === e.currentTarget) onClose?.();
      }}
      onKeyDown={(e) => e.stopPropagation()}
      role="presentation"
    >
      <div
        className="relative max-h-[85%] w-full max-w-2xl overflow-auto rounded-2xl bg-zinc-950/90 p-6 text-white shadow-2xl ring-1 ring-white/10"
        style={{ animation: "flows-rise 240ms ease-out" }}
      >
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            className="absolute top-4 right-4 rounded-full p-1.5 text-white/60 hover:bg-white/10 hover:text-white"
            aria-label="Close"
            title="Close"
          >
            <XIcon className="size-4" />
          </button>
        ) : null}
        {children}
      </div>
    </div>
  );
}

function ToolbarButton({
  label,
  active,
  onClick,
  children,
}: {
  label: string;
  active?: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={cn(
        "rounded-full p-2 text-white/70 transition hover:bg-white/15 hover:text-white",
        active && "bg-white/20 text-white",
      )}
    >
      {children}
    </button>
  );
}

const ENDING_GLOW: Record<string, string> = {
  good: "from-amber-200 via-yellow-100 to-amber-300",
  bad: "from-rose-300 via-red-200 to-rose-400",
  secret: "from-fuchsia-300 via-violet-200 to-sky-300",
  neutral: "from-sky-200 via-white to-indigo-200",
};

export function StoryPlayer({
  flowId,
  api,
  manifest,
  seed = 0,
  canReroll,
  onReroll,
  onExit,
}: StoryPlayerProps) {
  const reducer = useCallback(
    (s: StoryState, a: StoryAction) => reduce(manifest, s, a),
    [manifest],
  );
  const [state, dispatch] = useReducer(reducer, manifest, (m) =>
    initialState(m, seed),
  );
  const [started, setStarted] = useState(false);
  const [typing, setTyping] = useState(true);
  const [auto, setAuto] = useState(false);
  const [skip, setSkip] = useState(false);
  const [panel, setPanel] = useState<
    null | "log" | "save" | "load" | "settings"
  >(null);
  const [cps, setCps] = useState(() => readNumber(SPEED_KEY, 45));
  const [voiceOn, setVoiceOn] = useState(() => readNumber(VOICE_KEY, 1) === 1);
  const [aheadDepth, setAheadDepth] = useState(readWriteAhead);
  const [painting, setPainting] = useState<Set<string>>(() => new Set());
  const [toast, setToast] = useState<string | null>(null);
  const [autosave, setAutosave] = useState<SessionRecord | null>(null);
  const handle = useRef<TypewriterHandle | null>(null);
  const stateRef = useRef(state);
  stateRef.current = state;

  const scene = sceneOf(manifest, state, state.sceneId);
  const line = currentLine(manifest, state);
  const chars = useMemo(
    () => new Map(manifest.characters.map((c) => [c.id, c])),
    [manifest.characters],
  );
  const speaker = line?.speaker ? (chars.get(line.speaker) ?? null) : null;
  const text = line ? interpolate(line.text, state.vars) : "";
  const choices =
    scene && state.status === "choosing"
      ? visibleChoices(scene, state.vars)
      : [];
  const bgMedia = backgroundFor(manifest, state, state.sceneId);
  const bgUrl = useMediaUrl(api, bgMedia);

  useEffect(() => {
    api
      .loadSession(flowId, AUTO_SLOT)
      .then(setAutosave)
      .catch(() => setAutosave(null));
  }, [api, flowId]);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    window.setTimeout(() => setToast((t) => (t === msg ? null : t)), 2200);
  }, []);

  const lanes = useRef(new Lanes());
  const requested = useRef<Set<string>>(new Set());
  const heldLines = useRef<Map<string, NonNullable<StoryScene["lines"]>>>(
    new Map(),
  );
  const artEntry = manifest.entry_points?.scene_art;
  const nextEntry = manifest.entry_points?.next_scene;
  const [writtenTick, setWrittenTick] = useState(0);
  const writeAhead = useMemo(
    () =>
      new WriteAhead(
        async (req, signal) => {
          if (!nextEntry) throw new Error("This story has no Next Scene");
          const res = await api.invoke(flowId, nextEntry, req, { signal });
          const written = res.outputs.scene as StoryScene | undefined;
          if (!written) throw new Error("The story writer returned nothing");
          return written;
        },
        () => setWrittenTick((t) => t + 1),
      ),
    [api, flowId, nextEntry],
  );
  useEffect(() => {
    writeAhead.open();
    return () => writeAhead.close();
  }, [writeAhead]);
  useEffect(() => {
    if (nextEntry && api.warm)
      void api.warm(flowId, nextEntry).catch(() => false);
  }, [api, flowId, nextEntry]);

  const paint = useCallback(
    (target: StoryScene, rev: number, current = false) => {
      if (!artEntry) return;
      const key = `${rev}:art:${target.id}`;
      if (requested.current.has(key)) return;
      requested.current.add(key);
      const prompt = [
        target.background_prompt || target.title || "",
        manifest.style,
      ]
        .filter(Boolean)
        .join(", ");
      if (!prompt) return;
      setPainting((p) => new Set(p).add(target.id));
      const job = () =>
        lanes.current.run("art", () =>
          api.invoke(flowId, artEntry, { prompt }, { rev }),
        );
      (current ? writeAhead.hold(job) : job())
        .then((res) => {
          const media = res.outputs.image as MediaValue | undefined;
          if (media)
            dispatch({ type: "setBackground", sceneId: target.id, media, rev });
        })
        .catch(() => undefined)
        .finally(() =>
          setPainting((p) => {
            const n = new Set(p);
            n.delete(target.id);
            return n;
          }),
        );
    },
    [api, artEntry, flowId, manifest.style, writeAhead],
  );

  // biome-ignore lint/correctness/useExhaustiveDependencies: only on a new playthrough
  useEffect(() => {
    writeAhead.reset();
  }, [writeAhead, state.rev]);

  useEffect(() => {
    if (state.status !== "generating" || !state.pending || !nextEntry) return;
    const { rev } = state;
    const key = `${rev}:next:${state.pending.sceneId}`;
    if (requested.current.has(key)) return;
    requested.current.add(key);
    const snapshot = stateRef.current;
    const req = nextSceneRequest(manifest, snapshot);
    if (!req || !snapshot.pending) return;
    const asId = snapshot.pending.sceneId;
    writeAhead
      .take(req, writeKey(snapshot, req))
      .then((written) =>
        dispatch({ type: "addScene", scene: written, asId, rev }),
      )
      .catch((e: unknown) =>
        dispatch({
          type: "generationFailed",
          message: e instanceof Error ? e.message : String(e),
          rev,
        }),
      );
  }, [state.status, state.pending, state.rev, manifest, nextEntry, writeAhead]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: re-planned when the scene, playthrough or a write changes
  useEffect(() => {
    if (!started || !nextEntry) return;
    if (state.status === "ended" || state.status === "error") {
      writeAhead.plan([]);
      return;
    }
    if (state.status !== "reading" && state.status !== "choosing") return;
    writeAhead.plan(
      planWriteAhead(manifest, stateRef.current, aheadDepth, (k) =>
        writeAhead.written(k),
      ),
    );
  }, [
    started,
    state.sceneId,
    state.rev,
    state.status,
    aheadDepth,
    writtenTick,
    manifest,
    nextEntry,
    writeAhead,
  ]);

  useEffect(() => {
    if (!started || !scene) return;
    const rev = state.rev;
    if (!backgroundFor(manifest, state, scene.id) && scene.generate_background)
      paint(scene, rev, true);
    for (const id of nextSceneIds(manifest, state, scene.id)) {
      const nxt = sceneOf(manifest, state, id);
      if (nxt?.generate_background && !backgroundFor(manifest, state, id))
        paint(nxt, rev);
    }
    if (scene.generate_lines && nextEntry) {
      const key = `${rev}:lines:${scene.id}`;
      const held = heldLines.current.get(key);
      if (held) {
        heldLines.current.delete(key);
        dispatch({ type: "setLines", sceneId: scene.id, lines: held, rev });
      } else if (!requested.current.has(key)) {
        requested.current.add(key);
        const snapshot = stateRef.current;
        const brief = [
          scene.title,
          scene.background_prompt,
          ...(scene.lines ?? []).map((l) => l.text),
        ]
          .filter(Boolean)
          .join(" / ");
        writeAhead
          .hold(() =>
            lanes.current.run("text", () =>
              api.invoke(
                flowId,
                nextEntry,
                {
                  state: {
                    story: storyContext(manifest, snapshot),
                    scene_id: scene.id,
                    vars: snapshot.vars,
                    history: historyText(manifest, snapshot),
                    choice: `(Write the dialogue for this scene: ${brief})`,
                  },
                },
                { rev },
              ),
            ),
          )
          .then((res) => {
            const written = res.outputs.scene as StoryScene | undefined;
            const now = stateRef.current;
            if (
              written?.lines?.length &&
              now.rev === rev &&
              now.sceneId === scene.id &&
              now.status === "reading" &&
              !atSceneStart(manifest, now)
            )
              heldLines.current.set(key, written.lines);
            else if (written?.lines?.length)
              dispatch({
                type: "setLines",
                sceneId: scene.id,
                lines: written.lines,
                rev,
              });
            else dispatch({ type: "linesFailed", sceneId: scene.id, rev });
          })
          .catch(() =>
            dispatch({ type: "linesFailed", sceneId: scene.id, rev }),
          );
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [started, state.sceneId, state.rev]);

  useEffect(() => {
    if (!started || state.status === "generating" || state.status === "error")
      return;
    const t = window.setTimeout(() => {
      api
        .saveSession(
          flowId,
          AUTO_SLOT,
          stateRef.current,
          scene?.title ?? manifest.title,
        )
        .catch(() => undefined);
    }, 800);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [started, state.sceneId, state.status]);

  const voiceRef = useRef<HTMLAudioElement | null>(null);
  const voiceUrl = useMediaUrl(api, voiceOn ? (line?.voice ?? null) : null);
  useEffect(() => {
    voiceRef.current?.pause();
    if (!voiceUrl || !started) return;
    const audio = new Audio(voiceUrl);
    voiceRef.current = audio;
    audio.play().catch(() => undefined);
    return () => audio.pause();
  }, [voiceUrl, started]);
  const musicUrl = useMediaUrl(api, scene?.music ?? null);

  const stage = useMemo(() => stageAt(manifest, state), [manifest, state]);
  const reduced = useReducedMotion();
  const instant = reduced || skip;
  const pause = state.status === "reading" ? pauseOf(line) : null;

  const cued = useRef("");
  const quietCue = useRef(false);
  useEffect(() => {
    if (!started) return;
    const key = `${state.rev}:${state.sceneId}:${state.line}:${state.status}`;
    if (cued.current === key) return;
    cued.current = key;
    if (quietCue.current) {
      quietCue.current = false;
      return;
    }
    if (skip) return;
    const rev = state.rev;
    playCues(
      stage.sounds,
      manifest.sounds,
      (media) => mediaObjectUrl(api, media),
      () => stateRef.current.rev === rev,
    );
  }, [api, manifest.sounds, stage, started, skip, state]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: two pauses in a row restart the timer
  useEffect(() => {
    if (pause === null || !started || panel || skip) return;
    const t = window.setTimeout(
      () => dispatch({ type: "advance" }),
      pause * 1000,
    );
    return () => window.clearTimeout(t);
  }, [pause, started, panel, skip, state.sceneId, state.line, state.rev]);

  const advance = useCallback(() => {
    if (panel) return;
    if (!started) {
      setStarted(true);
      return;
    }
    if (pause === null && handle.current?.typing()) {
      handle.current.complete();
      return;
    }
    if (stateRef.current.status === "reading") dispatch({ type: "advance" });
  }, [panel, started, pause]);

  const choose = useCallback((id: string) => {
    setSkip(false);
    dispatch({ type: "choose", choiceId: id });
  }, []);

  useEffect(() => {
    setTyping(true);
  }, [text, state.line, state.sceneId]);

  useEffect(() => {
    if (
      !auto ||
      typing ||
      pause !== null ||
      state.status !== "reading" ||
      panel
    )
      return;
    const t = window.setTimeout(
      () => dispatch({ type: "advance" }),
      900 + text.length * 28,
    );
    return () => window.clearTimeout(t);
  }, [
    auto,
    typing,
    pause,
    state.status,
    state.line,
    state.sceneId,
    text.length,
    panel,
  ]);

  useEffect(() => {
    if (!skip || state.status !== "reading" || panel) {
      if (skip && state.status !== "reading") setSkip(false);
      return;
    }
    const t = window.setTimeout(() => {
      handle.current?.complete();
      dispatch({ type: "advance" });
    }, 70);
    return () => window.clearTimeout(t);
  }, [skip, state.status, state.line, state.sceneId, panel]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      )
        return;
      if (e.key === "Escape") {
        setPanel(null);
        setSkip(false);
        return;
      }
      if (panel) return;
      if (e.key === " " || e.key === "Enter") {
        e.preventDefault();
        advance();
      } else if (
        /^[1-9]$/.test(e.key) &&
        stateRef.current.status === "choosing"
      ) {
        const c = choices[Number(e.key) - 1];
        if (c) choose(c.id);
      } else if (e.key === "a" || e.key === "A") setAuto((v) => !v);
      else if (e.key === "s" || e.key === "S") setSkip((v) => !v);
      else if (e.key === "b" || e.key === "B" || e.key === "l" || e.key === "L")
        setPanel("log");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [advance, choices, choose, panel]);

  const [sessions, setSessions] = useState<SessionRecord[] | null>(null);
  useEffect(() => {
    if (panel !== "save" && panel !== "load") return;
    setSessions(null);
    api
      .listSessions(flowId)
      .then(setSessions)
      .catch(() => setSessions([]));
  }, [panel, api, flowId]);

  const save = async (slot: string) => {
    const label = `${scene?.title || manifest.title} · ${new Date().toLocaleString()}`;
    try {
      await api.saveSession(flowId, slot, stateRef.current, label);
      showToast(`Saved to slot ${slot.replace("story-", "")}`);
      setPanel(null);
    } catch (e) {
      showToast(`Could not save: ${e instanceof Error ? e.message : e}`);
    }
  };

  const load = async (slot: string) => {
    try {
      const rec = await api.loadSession(flowId, slot);
      if (!rec) return;
      if (!saveFits(manifest, rec.state as StoryState)) {
        showToast("That save is from an earlier version of this story");
        return;
      }
      quietCue.current = true;
      dispatch({ type: "load", state: rec.state as StoryState });
      setStarted(true);
      setPanel(null);
      showToast("Loaded");
    } catch (e) {
      showToast(`Could not load: ${e instanceof Error ? e.message : e}`);
    }
  };

  const restart = () => {
    requested.current.clear();
    heldLines.current.clear();
    dispatch({ type: "start", seed: state.seed });
    setPanel(null);
    setStarted(true);
  };

  const slots = { left: 0, right: 0, center: 0 };
  const placed = stage.cast.flatMap((st) => {
    const c = chars.get(st.id);
    if (!c) return [];
    const slot = st.at ? 0 : slots[c.side ?? "left"]++;
    return [{ c, slot, st }];
  });
  const leaving = stage.leaving.flatMap((st) => {
    const c = chars.get(st.id);
    return c && st.exit && st.exit !== "none" && !instant ? [{ c, st }] : [];
  });
  const generating = state.status === "generating";
  const nameColor = speaker?.color ?? "#e5e7eb";

  return (
    <div
      className="relative h-full w-full select-none overflow-hidden bg-black font-sans text-white"
      onClick={advance}
      onKeyDown={() => undefined}
      role="presentation"
      data-testid="story-player"
    >
      <PlayerKeyframes />
      <BackgroundStage
        seedKey={state.sceneId}
        url={bgUrl}
        painting={painting.has(state.sceneId)}
        transition={instant ? "none" : (scene?.transition ?? "dissolve")}
      />
      {musicUrl ? <audio src={musicUrl} autoPlay={true} loop={true} /> : null}

      {started && state.status !== "ended" ? (
        <div className="absolute inset-0">
          {placed.map(({ c, slot, st }) => (
            <Portrait
              key={c.id}
              api={api}
              character={c}
              slot={slot}
              at={st.at}
              enter={st.enter}
              instant={instant}
              speaking={speaker?.id === c.id}
              expression={
                (speaker?.id === c.id ? line?.expression : null) ??
                st.expression
              }
            />
          ))}
          {leaving.map(({ c, st }) => (
            <Portrait
              key={c.id}
              api={api}
              character={c}
              slot={0}
              at={st.at}
              enter={null}
              exit={st.exit}
              instant={false}
              speaking={false}
              expression={st.expression}
            />
          ))}
        </div>
      ) : null}

      {/* toolbar */}
      <div className="absolute top-3 right-3 z-20 flex items-center gap-1 rounded-full bg-black/35 px-1.5 py-1 backdrop-blur-md ring-1 ring-white/10">
        <ToolbarButton
          label="Auto (A)"
          active={auto}
          onClick={() => setAuto((v) => !v)}
        >
          <PlayIcon className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label="Skip (S)"
          active={skip}
          onClick={() => setSkip((v) => !v)}
        >
          <FastForwardIcon className="size-4" />
        </ToolbarButton>
        <ToolbarButton label="History (B)" onClick={() => setPanel("log")}>
          <HistoryIcon className="size-4" />
        </ToolbarButton>
        <ToolbarButton label="Save" onClick={() => setPanel("save")}>
          <SaveIcon className="size-4" />
        </ToolbarButton>
        <ToolbarButton label="Load" onClick={() => setPanel("load")}>
          <RotateCcwIcon className="size-4" />
        </ToolbarButton>
        <ToolbarButton
          label={voiceOn ? "Voice on" : "Voice off"}
          onClick={() => {
            setVoiceOn((v) => {
              localStorage.setItem(VOICE_KEY, v ? "2" : "1");
              return !v;
            });
          }}
        >
          {voiceOn ? (
            <Volume2Icon className="size-4" />
          ) : (
            <VolumeXIcon className="size-4" />
          )}
        </ToolbarButton>
        <ToolbarButton label="Settings" onClick={() => setPanel("settings")}>
          <SettingsIcon className="size-4" />
        </ToolbarButton>
        {onExit ? (
          <ToolbarButton label="Back to the editor" onClick={onExit}>
            <ArrowLeftIcon className="size-4" />
          </ToolbarButton>
        ) : null}
      </div>

      {scene?.title && started && state.status !== "ended" ? (
        <div
          key={scene.id}
          className="absolute top-4 left-5 z-10 rounded-full bg-black/35 px-4 py-1.5 text-sm tracking-wide text-white/85 backdrop-blur-md ring-1 ring-white/10"
          style={{ animation: "flows-rise 500ms ease-out" }}
        >
          {scene.title}
        </div>
      ) : null}

      {/* title card */}
      {started ? null : (
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-8 bg-black/40 backdrop-blur-[2px]">
          <h1
            className="max-w-4xl px-6 text-center font-semibold text-5xl text-white tracking-tight drop-shadow-[0_6px_24px_rgba(0,0,0,0.7)] md:text-6xl"
            style={{ animation: "flows-rise 700ms ease-out" }}
          >
            {manifest.title}
          </h1>
          <div className="flex gap-3">
            <button
              type="button"
              data-testid="story-begin"
              onClick={(e) => {
                e.stopPropagation();
                setStarted(true);
              }}
              className="rounded-full bg-white px-8 py-3 font-semibold text-black shadow-xl transition hover:scale-105"
            >
              Begin
            </button>
            {autosave ? (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  void load(AUTO_SLOT);
                }}
                className="rounded-full bg-white/15 px-8 py-3 font-semibold text-white ring-1 ring-white/30 backdrop-blur transition hover:bg-white/25"
              >
                Continue
              </button>
            ) : null}
          </div>
          <p className="text-sm text-white/60">
            Click, Space or Enter to continue · 1-4 to choose · A auto · S skip
          </p>
        </div>
      )}

      {/* choices */}
      {started && choices.length > 0 ? (
        <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 pb-40">
          {choices.map((c, i) => (
            <button
              key={c.id}
              type="button"
              data-testid="story-choice"
              onClick={(e) => {
                e.stopPropagation();
                choose(c.id);
              }}
              className="group w-[min(560px,86vw)] rounded-2xl bg-black/55 px-6 py-3.5 text-left text-lg text-white shadow-xl ring-1 ring-white/15 backdrop-blur-md transition hover:-translate-y-0.5 hover:bg-white/90 hover:text-black"
              style={{ animation: `flows-rise ${260 + i * 80}ms ease-out` }}
            >
              <span className="mr-3 text-white/45 group-hover:text-black/50">
                {i + 1}
              </span>
              {interpolate(c.text, state.vars)}
            </button>
          ))}
        </div>
      ) : null}

      {/* dialogue */}
      {started && line && pause === null && state.status !== "ended" ? (
        <div
          className="absolute inset-x-0 bottom-0 z-10 p-4 md:p-8"
          data-testid="story-dialogue"
        >
          <div className="relative mx-auto min-h-38 max-w-4xl rounded-2xl bg-black/60 px-7 pt-7 pb-6 text-lg leading-relaxed text-white/95 shadow-2xl ring-1 ring-white/10 backdrop-blur-md md:text-xl">
            {speaker ? (
              <div
                className="absolute -top-4 left-6 rounded-full px-4 py-1 font-semibold text-sm text-black shadow-lg"
                style={{ background: nameColor }}
              >
                {speaker.name}
              </div>
            ) : null}
            <p className={cn(!speaker && "italic text-white/85")}>
              <Typewriter
                text={text}
                cps={cps}
                handle={handle}
                onDone={() => setTyping(false)}
              />
            </p>
            {!typing && state.status === "reading" ? (
              <span
                className="absolute right-6 bottom-4 text-white/70"
                style={{ animation: "flows-bob 1.2s ease-in-out infinite" }}
              >
                ▼
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      {generating ? (
        <div className="absolute inset-x-0 bottom-10 z-10 flex justify-center">
          <div className="flex items-center gap-3 rounded-full bg-black/60 px-6 py-3 text-white/90 backdrop-blur-md ring-1 ring-white/10">
            <SparklesIcon className="size-4 text-amber-200" />
            The story is writing what happens next
            <span className="flex gap-1">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  style={{ animation: `flows-dots 1.2s ${i * 0.2}s infinite` }}
                >
                  •
                </span>
              ))}
            </span>
          </div>
        </div>
      ) : null}

      {state.status === "error" ? (
        <Overlay>
          <h2 className="mb-2 font-semibold text-xl">The story got stuck</h2>
          <p className="mb-5 text-white/70">{state.error}</p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={restart}
              className="rounded-full bg-white px-5 py-2 font-medium text-black"
            >
              Start over
            </button>
            {onExit ? (
              <button
                type="button"
                onClick={onExit}
                className="rounded-full bg-white/10 px-5 py-2"
              >
                Back to the editor
              </button>
            ) : null}
          </div>
        </Overlay>
      ) : null}

      {state.status === "ended" ? (
        <div
          className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-6 bg-black/55 backdrop-blur-sm"
          data-testid="story-ending"
        >
          <p className="text-sm uppercase tracking-[0.4em] text-white/60">
            {state.ending?.kind === "secret" ? "Secret ending" : "Ending"}
          </p>
          <h2
            className={cn(
              "bg-gradient-to-r bg-clip-text px-6 text-center font-bold text-5xl text-transparent md:text-6xl",
              ENDING_GLOW[state.ending?.kind ?? "neutral"] ??
                ENDING_GLOW.neutral,
            )}
            style={{ animation: "flows-rise 900ms ease-out" }}
          >
            {state.ending?.title ?? "The End"}
          </h2>
          <p className="text-white/60">
            {state.visited.length} scenes visited ·{" "}
            {Object.keys(state.vars).length ? formatVars(state.vars) : ""}
          </p>
          <div className="flex flex-wrap justify-center gap-3">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                restart();
              }}
              className="rounded-full bg-white px-7 py-3 font-semibold text-black shadow-xl transition hover:scale-105"
            >
              Play again
            </button>
            {canReroll && onReroll ? (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onReroll();
                }}
                className="flex items-center gap-2 rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 px-7 py-3 font-semibold text-white shadow-xl transition hover:scale-105"
              >
                <SparklesIcon className="size-4" /> New story
              </button>
            ) : null}
            {onExit ? (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onExit();
                }}
                className="rounded-full bg-white/10 px-7 py-3 font-semibold text-white ring-1 ring-white/25"
              >
                Back to the editor
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      {panel === "log" ? (
        <Overlay onClose={() => setPanel(null)}>
          <h2 className="mb-4 font-semibold text-lg">History</h2>
          <div className="flex flex-col gap-3">
            {state.backlog.map((b, i) =>
              b.speaker === CHOICE_SPEAKER ? (
                <p key={i} className="text-sm text-amber-200/90">
                  ➜ {b.text}
                </p>
              ) : (
                <p key={i} className="text-white/85">
                  {b.speaker ? (
                    <span
                      className="mr-2 font-semibold"
                      style={{ color: chars.get(b.speaker)?.color ?? "#fff" }}
                    >
                      {chars.get(b.speaker)?.name ?? b.speaker}
                    </span>
                  ) : null}
                  <span className={cn(!b.speaker && "italic text-white/70")}>
                    {b.text}
                  </span>
                </p>
              ),
            )}
          </div>
        </Overlay>
      ) : null}

      {panel === "save" || panel === "load" ? (
        <Overlay onClose={() => setPanel(null)}>
          <h2 className="mb-4 font-semibold text-lg">
            {panel === "save" ? "Save" : "Load"}
          </h2>
          <div className="grid grid-cols-2 gap-3">
            {SLOTS.map((slot) => {
              const rec = sessions?.find((s) => s.slot === `story-${slot}`);
              const disabled = panel === "load" && !rec;
              return (
                <button
                  key={slot}
                  type="button"
                  disabled={disabled || sessions === null}
                  onClick={() =>
                    void (panel === "save"
                      ? save(`story-${slot}`)
                      : load(`story-${slot}`))
                  }
                  className="rounded-xl bg-white/5 p-4 text-left ring-1 ring-white/10 transition hover:bg-white/15 disabled:opacity-40"
                >
                  <div className="text-xs text-white/50">Slot {slot}</div>
                  <div className="truncate text-sm">
                    {rec?.label ?? "Empty"}
                  </div>
                </button>
              );
            })}
          </div>
          {panel === "load" && autosave ? (
            <button
              type="button"
              onClick={() => void load(AUTO_SLOT)}
              className="mt-3 w-full rounded-xl bg-white/5 p-3 text-left text-sm ring-1 ring-white/10 hover:bg-white/15"
            >
              Autosave · {autosave.label}
            </button>
          ) : null}
          <button
            type="button"
            onClick={restart}
            className="mt-5 text-sm text-white/60 underline-offset-4 hover:text-white hover:underline"
          >
            Restart from the beginning
          </button>
        </Overlay>
      ) : null}

      {panel === "settings" ? (
        <Overlay onClose={() => setPanel(null)}>
          <h2 className="mb-4 font-semibold text-lg">Settings</h2>
          <label className="flex items-center gap-4 text-sm">
            <span className="w-28 text-white/70">Text speed</span>
            <input
              type="range"
              min={10}
              max={160}
              value={Math.min(cps, 160)}
              onChange={(e) => {
                const v = Number(e.target.value);
                setCps(v >= 160 ? 1000 : v);
                localStorage.setItem(SPEED_KEY, String(v >= 160 ? 1000 : v));
              }}
              className="flex-1"
            />
            <span className="w-16 text-right text-white/60">
              {cps >= 1000 ? "Instant" : `${cps}/s`}
            </span>
          </label>
          {nextEntry ? (
            <label className="mt-4 flex items-center gap-4 text-sm">
              <span className="w-28 text-white/70">Write ahead</span>
              <select
                value={aheadDepth}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setAheadDepth(v);
                  localStorage.setItem(WRITE_AHEAD_KEY, String(v));
                }}
                className="flex-1 rounded bg-white/10 px-2 py-1 text-white"
              >
                <option value={0}>Off</option>
                <option value={1}>Next scene</option>
                <option value={2}>Two scenes</option>
              </select>
            </label>
          ) : null}
        </Overlay>
      ) : null}

      {toast ? (
        <div className="absolute top-16 left-1/2 z-40 -translate-x-1/2 rounded-full bg-black/75 px-5 py-2 text-sm backdrop-blur ring-1 ring-white/10">
          {toast}
        </div>
      ) : null}
    </div>
  );
}

function formatVars(vars: Record<string, unknown>): string {
  return Object.entries(vars)
    .filter(
      ([, v]) =>
        typeof v === "number" ||
        typeof v === "string" ||
        typeof v === "boolean",
    )
    .slice(0, 4)
    .map(([k, v]) => `${k} ${String(v)}`)
    .join(" · ");
}

export function storySeed(flowId: string): number {
  return seedFrom("play", flowId);
}
