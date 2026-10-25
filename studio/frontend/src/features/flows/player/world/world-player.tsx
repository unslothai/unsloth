// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowLeftIcon,
  BackpackIcon,
  BoxIcon,
  MapIcon,
  MapPinIcon,
  PencilRulerIcon,
  RotateCcwIcon,
  SaveIcon,
  ScrollTextIcon,
  SendIcon,
  SparklesIcon,
  XIcon,
} from "lucide-react";
import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { cn } from "@/lib/utils";

import type { MediaValue, WorldSpec } from "../../types";
import { type PlayerApi, mediaObjectUrl } from "../player-api";
import { PlayerKeyframes } from "../story/backdrop";
import { useMediaUrl } from "../use-media-url";
import { type LiveReply, appendReplyDelta } from "./live-reply";
import type { Interactable, WorldArt, WorldGameHandle } from "./phaser-game";
import { type TileKind, generateMap } from "./procgen";
import {
  type EditorState,
  GlbError,
  SaveQueue,
  type WorldEditor,
  inspectGlb,
  saveUntilCurrent,
} from "./props3d";
import {
  BackgroundQueue,
  type Note,
  type PlanBlock,
  activityEmoji,
  clockLabel,
  currentBlock,
  dayOf,
  emptyMind,
  findEncounter,
  markChat,
  placeAt,
  remember,
  simOf,
  touch,
} from "./world-sim";
import {
  type NpcReply,
  type WorldState,
  addTurn,
  applyReply,
  deserialize,
  initialWorldState,
  pickUp,
  serialize,
  startQuestsOf,
  withMind,
} from "./world-state";
import { webglAvailable } from "./world3d";

const WorldEditorPanel = lazy(() => import("./world-editor-panel"));

export interface WorldPlayerProps {
  flowId: string;
  api: PlayerApi;
  spec: WorldSpec;
  canReroll?: boolean;
  onReroll?: () => void;
  onExit?: () => void;
  /** Called with the running game (the smoke harness drives it). */
  onReady?: (game: WorldGameHandle) => void;
}

const SLOTS = ["world-1", "world-2", "world-3"];
const LINE_MS = 2400;

interface NpcChat {
  lines: { npc: string; text: string }[];
  a_learns: Note[];
  b_learns: Note[];
  a_recalled: string[];
  b_recalled: string[];
}

async function loadImage(url: string): Promise<HTMLImageElement> {
  const img = new Image();
  img.decoding = "async";
  img.src = url;
  await img.decode();
  return img;
}

async function loadArt(
  api: PlayerApi,
  spec: WorldSpec,
  onProgress: (f: number) => void,
  models: string[] = [],
): Promise<WorldArt> {
  const jobs: [string, MediaValue][] = [];
  for (const [kind, m] of Object.entries(spec.tileset ?? {}))
    jobs.push([`tile:${kind}`, m]);
  if (spec.player.sprite) jobs.push(["sprite:player", spec.player.sprite]);
  for (const n of spec.npcs)
    if (n.sprite) jobs.push([`sprite:${n.id}`, n.sprite]);
  for (const it of spec.items)
    if (it.sprite) jobs.push([`sprite:${it.id}`, it.sprite]);
  const art: WorldArt = { tiles: {}, sprites: {}, models: {} };
  await Promise.all(
    models.map(async (ref) => {
      try {
        const blob = await api.mediaBlob({ kind: "image", ref });
        if (art.models) art.models[ref] = await blob.arrayBuffer();
      } catch {
        // keep the placeholder
      }
    }),
  );
  let done = 0;
  await Promise.all(
    jobs.map(async ([key, media]) => {
      try {
        const img = await loadImage(await mediaObjectUrl(api, media));
        const [kind, id] = key.split(":");
        if (kind === "tile") art.tiles[id as TileKind] = img;
        else art.sprites[id] = img;
      } catch {
        // Missing art falls back to procedural tiles and sprites.
      } finally {
        onProgress(++done / Math.max(1, jobs.length));
      }
    }),
  );
  return art;
}

function Panel({
  title,
  onClose,
  children,
}: { title: string; onClose: () => void; children: React.ReactNode }) {
  return (
    <div
      className="absolute top-16 right-4 z-20 w-80 rounded-2xl bg-zinc-950/85 p-4 text-white shadow-2xl ring-1 ring-white/10 backdrop-blur-md"
      style={{ animation: "flows-rise 200ms ease-out" }}
    >
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-semibold">{title}</h3>
        <button
          type="button"
          onClick={onClose}
          className="rounded-full p-1 text-white/60 hover:bg-white/10"
          aria-label="Close"
        >
          <XIcon className="size-4" />
        </button>
      </div>
      {children}
    </div>
  );
}

function SpriteThumb({
  api,
  media,
  fallback,
}: { api: PlayerApi; media?: MediaValue | null; fallback: string }) {
  const url = useMediaUrl(api, media ?? null);
  return url ? (
    <img
      src={url}
      alt=""
      className="size-9 rounded-lg bg-white/5 object-contain"
    />
  ) : (
    <div className="flex size-9 items-center justify-center rounded-lg bg-gradient-to-br from-violet-400/60 to-sky-400/60 font-bold text-sm">
      {fallback.slice(0, 1)}
    </div>
  );
}

export function WorldPlayer({
  flowId,
  api,
  spec,
  canReroll,
  onReroll,
  onExit,
  onReady,
}: WorldPlayerProps) {
  const map = useMemo(() => generateMap(spec), [spec]);
  const host = useRef<HTMLDivElement>(null);
  const game = useRef<WorldGameHandle | null>(null);
  const [ws, setWs] = useState<WorldState>(() => initialWorldState(spec));
  const wsRef = useRef(ws);
  wsRef.current = ws;
  const [loadProgress, setLoadProgress] = useState(0);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [started, setStarted] = useState(false);
  const [nearby, setNearby] = useState<Interactable | null>(null);
  const [talking, setTalking] = useState<string | null>(null);
  const [panel, setPanel] = useState<null | "inventory" | "quests" | "save">(
    null,
  );
  const [toasts, setToasts] = useState<{ id: number; text: string }[]>([]);
  const [region, setRegion] = useState<string | null>(null);
  const [mount, setMount] = useState(0);
  const [editor, setEditor] = useState<WorldEditor | null>(null);
  const [edState, setEdState] = useState<EditorState | null>(null);
  const [edStatus, setEdStatus] = useState("");
  const [edError, setEdError] = useState<string | null>(null);
  const [savedProps, setSavedProps] = useState(() =>
    JSON.stringify(spec.props3d ?? []),
  );
  const editing = editor !== null;
  const can3d = useMemo(() => webglAvailable(), []);
  const [view, setView] = useState<"2d" | "3d">(() =>
    spec.render === "3d" && webglAvailable() ? "3d" : "2d",
  );
  const lastTile = useRef<{ x: number; y: number } | null>(null);
  const bootedView = useRef<"2d" | "3d" | null>(null);
  const ambientUrl = useMediaUrl(
    api,
    view === "3d" ? spec.view3d?.ambient : null,
  );
  const lockedRef = useRef(true);
  lockedRef.current =
    !started || talking !== null || panel === "save" || editor !== null;

  const toast = useCallback((text: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t.slice(-3), { id, text }]);
    window.setTimeout(
      () => setToasts((t) => t.filter((x) => x.id !== id)),
      3200,
    );
  }, []);

  const npcById = useMemo(
    () => new Map(spec.npcs.map((n) => [n.id, n])),
    [spec.npcs],
  );
  const itemById = useMemo(
    () => new Map(spec.items.map((i) => [i.id, i])),
    [spec.items],
  );

  const sim = useMemo(() => simOf(spec), [spec]);
  const planEntry = sim ? spec.entry_points?.npc_plan : undefined;
  const chatEntry = sim ? spec.entry_points?.npc_chat : undefined;
  const queue = useMemo(
    () => new BackgroundQueue(sim?.calls_per_minute ?? 0),
    [sim],
  );
  useEffect(() => () => queue.clear(), [queue]);
  const busy = useRef(new Set<string>());
  const [background, setBackground] = useState<string | null>(null);

  // NPCs within sight remember what the player does.
  const witness = useCallback(
    (s: WorldState, what: string, importance: number): WorldState => {
      const tiles = sim ? game.current?.npcTiles() : undefined;
      const me = game.current?.playerTile();
      if (!tiles || !me) return s;
      let next = s;
      for (const [id, at] of Object.entries(tiles)) {
        if (Math.abs(at.x - me.x) + Math.abs(at.y - me.y) <= 6)
          next = withMind(next, id, (m) =>
            remember(
              m,
              [{ text: `${spec.player.name} ${what}`, importance }],
              s.clock,
            ),
          );
      }
      return next;
    },
    [sim, spec.player.name],
  );

  const onInteract = useCallback(
    (t: Interactable) => {
      if (t.kind === "item") {
        setWs((s) => witness(pickUp(s, t.id), `picked up the ${t.name}`, 3));
        game.current?.removeItem(t.id);
        toast(`Picked up ${t.name}`);
        return;
      }
      const npc = npcById.get(t.id);
      if (!npc) return;
      setWs((s) => {
        let next = startQuestsOf(spec, s, npc.id);
        for (const q of spec.quests) {
          if (
            q.giver === npc.id &&
            next.activeQuests.includes(q.id) &&
            !s.activeQuests.includes(q.id)
          ) {
            toast(`New quest: ${q.title}`);
          }
        }
        if (!next.talks[npc.id]?.length)
          next = addTurn(next, npc.id, {
            role: "npc",
            text: npc.greeting || "Hello!",
          });
        return next;
      });
      game.current?.faceNpc(npc.id);
      setTalking(npc.id);
    },
    [npcById, spec, toast, witness],
  );

  useEffect(() => {
    let live = true;
    let handle: WorldGameHandle | null = null;
    const switched = bootedView.current !== null && bootedView.current !== view;
    bootedView.current = view;
    const startAt = switched ? lastTile.current : wsRef.current.player;
    setReady(false);
    (async () => {
      try {
        const [art, create] = await Promise.all([
          loadArt(
            api,
            spec,
            (f) => live && setLoadProgress(f),
            view === "3d"
              ? [
                  ...new Set(
                    (map.props ?? [])
                      .map((p) => p.asset?.ref)
                      .filter((r): r is string => Boolean(r)),
                  ),
                ]
              : [],
          ),
          view === "3d"
            ? import("./three-world").then((m) => m.createWorld3d)
            : import("./phaser-game").then((m) => m.createWorldGame),
        ]);
        if (!live || !host.current) return;
        handle = create(
          host.current,
          spec,
          map,
          art,
          wsRef.current.picked,
          startAt,
          {
            onNearby: (t) => setNearby(t),
            onInteract,
            onRegion: (id) => {
              setRegion(id);
              const r = spec.regions.find((x) => x.id === id);
              if (r)
                toast(
                  r.on_enter_text ? `${r.name} · ${r.on_enter_text}` : r.name,
                );
            },
            locked: () => lockedRef.current,
          },
        );
        game.current = handle;
        setReady(true);
        onReady?.(handle);
      } catch (e) {
        if (live) setFailed(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      live = false;
      if (handle) lastTile.current = handle.playerTile();
      handle?.destroy();
      game.current = null;
      setEditor(null);
      setEdState(null);
    };
    // `mount` re-creates the game after loading a save.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, spec, map, mount, view]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" || target.tagName === "TEXTAREA")
      ) {
        if (e.key === "Escape") {
          setTalking(null);
          game.current?.faceNpc(null);
        }
        return;
      }
      if (e.key === "Escape") {
        setPanel(null);
        setTalking(null);
        game.current?.faceNpc(null);
      } else if (editing) {
      } else if (e.key === "i" || e.key === "I")
        setPanel((p) => (p === "inventory" ? null : "inventory"));
      else if (e.key === "q" || e.key === "Q")
        setPanel((p) => (p === "quests" ? null : "quests"));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing]);

  const edDirty = edState
    ? JSON.stringify(edState.props) !== savedProps
    : false;
  const openEditor = useCallback(async () => {
    const open = game.current?.openEditor;
    if (!open || !game.current) return;
    setPanel(null);
    setEdError(null);
    try {
      const ed = await open.call(game.current);
      setEdState(ed.state());
      ed.subscribe(() => setEdState(ed.state()));
      setEdStatus(api.saveWorldProps ? "" : "Changes last until you leave");
      setEditor(ed);
    } catch (e) {
      toast(e instanceof Error ? e.message : String(e));
    }
  }, [api, toast]);
  // Saves run one at a time and each writes the props as they are when it starts, so an older save never lands
  // after a newer one; `lastSaved` is what the server holds.
  const saves = useRef(new SaveQueue());
  const lastSaved = useRef(savedProps);
  const saveScene = useCallback(
    (): Promise<boolean> =>
      saves.current.run(async () => {
        if (!editor || !api.saveWorldProps) return true;
        const props = editor.state().props;
        setEdStatus("Saving");
        setEdError(null);
        try {
          await api.saveWorldProps(flowId, props, spec.player_node);
          lastSaved.current = JSON.stringify(props);
          setSavedProps(lastSaved.current);
          setEdState(editor.state());
          setEdStatus("Saved");
          return true;
        } catch (e) {
          setEdStatus("");
          setEdError(
            `Not saved: ${e instanceof Error ? e.message : String(e)}`,
          );
          return false;
        }
      }),
    [api, editor, flowId, spec.player_node],
  );
  const imports = useRef(new Set<Promise<void>>());
  const closeEditor = useCallback(async () => {
    if (!editor) return;
    while (imports.current.size) await Promise.allSettled([...imports.current]);
    await saves.current.idle();
    const current = () => JSON.stringify(editor.state().props);
    if (!(await saveUntilCurrent(current, lastSaved.current, saveScene)))
      return;
    editor.close();
    setEditor(null);
    setEdState(null);
  }, [editor, saveScene]);
  const importModel = useCallback(
    async (file: File) => {
      if (!editor || !api.uploadModel) return;
      const upload = api.uploadModel;
      setEdError(null);
      const run = (async () => {
        try {
          const data = await file.arrayBuffer();
          inspectGlb(data);
          setEdStatus("Uploading model");
          const asset = await upload(file, file.name);
          await editor.addModel(asset, data, file.name.replace(/\.glb$/i, ""));
          setEdStatus("");
        } catch (e) {
          setEdStatus("");
          setEdError(
            e instanceof GlbError
              ? e.message
              : `Could not import the model: ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      })();
      imports.current.add(run);
      await run.finally(() => imports.current.delete(run));
    },
    [api, editor],
  );
  const exportScene = useCallback(async () => {
    if (!editor) return;
    try {
      const blob = await editor.exportGlb();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${(spec.title || "world").replace(/[^A-Za-z0-9]+/g, "-").toLowerCase()}.glb`;
      a.click();
      window.setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    } catch (e) {
      setEdError(
        `Export failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }, [editor, spec.title]);

  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  const [live, setLive] = useState<LiveReply | null>(null);
  const replyEntry = spec.entry_points?.npc_reply;
  // Load the villagers' chat model while the intro shows, so the first conversation does not wait for it.
  const [warmed, setWarmed] = useState(true);
  useEffect(() => {
    if (!replyEntry || !api.warm) return;
    let alive = true;
    setWarmed(false);
    void api
      .warm(flowId, replyEntry)
      .catch(() => false)
      .finally(() => {
        if (alive) setWarmed(true);
      });
    return () => {
      alive = false;
    };
  }, [api, flowId, replyEntry]);
  const talkNpc = talking ? npcById.get(talking) : undefined;
  const turns = talking ? (ws.talks[talking] ?? []) : [];
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scroller.current?.scrollTo({
      top: scroller.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns.length, thinking, live?.text]);

  // Bumped on every load, so a reply to a conversation from before the load is dropped.
  const loads = useRef(0);

  // Village life: no new background call while the player talks, and the one in flight is dropped.
  useEffect(() => {
    queue.setPaused(!ready || lockedRef.current);
    queue.pump();
  }, [queue, ready, started, talking, panel]);

  const planDay = useCallback(
    async (npcId: string, day: number, signal: AbortSignal) => {
      if (!planEntry || !sim) return;
      const generation = loads.current;
      const s = wsRef.current;
      setBackground(`${npcById.get(npcId)?.name ?? npcId} is planning the day`);
      try {
        const res = await api.invoke(
          flowId,
          planEntry,
          {
            state: {
              world: spec,
              npc_id: npcId,
              memory: s.npc[npcId]?.memory ?? [],
              clock: s.clock,
            },
          },
          { signal },
        );
        if (generation !== loads.current) return;
        const blocks =
          (res.outputs.plan as { plan?: PlanBlock[] } | undefined)?.plan ?? [];
        const names = new Map(sim.places.map((p) => [p.id, p.name]));
        const text = blocks
          .map(
            (b) =>
              `${clockLabel(b.start).split(" · ")[1]} ${b.activity} at ${names.get(b.place) ?? b.place}`,
          )
          .join("; ");
        setWs((st) =>
          withMind(st, npcId, (m) =>
            remember(
              { ...m, plan: blocks, planDay: day },
              text
                ? [
                    {
                      text: `My plan for day ${day + 1}: ${text}`,
                      importance: 4,
                    },
                  ]
                : [],
              st.clock,
              "plan",
            ),
          ),
        );
      } catch (e) {
        // A failed plan leaves the NPC wandering for the day instead of retrying on every tick.
        if (!signal.aborted && generation === loads.current)
          setWs((st) =>
            withMind(st, npcId, (m) => ({ ...m, plan: [], planDay: day })),
          );
        throw e;
      } finally {
        setBackground(null);
      }
    },
    [api, flowId, npcById, planEntry, sim, spec],
  );

  const chat = useCallback(
    async (a: string, b: string, signal: AbortSignal) => {
      if (!chatEntry || !sim) return;
      const generation = loads.current;
      const s = wsRef.current;
      const g = game.current;
      const nameA = npcById.get(a)?.name ?? a;
      const nameB = npcById.get(b)?.name ?? b;
      for (const id of [a, b]) {
        busy.current.add(id);
        g?.setNpcBusy(id, true);
      }
      setBackground(`${nameA} is chatting with ${nameB}`);
      let hold = 0;
      try {
        const place =
          placeAt(sim.places, g?.npcTile(a) ?? { x: -1, y: -1 })?.name ?? "";
        const res = await api.invoke(
          flowId,
          chatEntry,
          {
            state: {
              world: spec,
              a,
              b,
              clock: s.clock,
              place,
              memory_a: s.npc[a]?.memory ?? [],
              memory_b: s.npc[b]?.memory ?? [],
            },
          },
          { signal },
        );
        if (generation !== loads.current) return;
        const c = res.outputs.chat as NpcChat | undefined;
        if (!c?.lines?.length) return;
        const where = place ? ` at ${place}` : "";
        setWs((st) => {
          const learn =
            (other: string, notes: Note[], ids: string[]) =>
            (m: ReturnType<typeof emptyMind>) =>
              touch(
                remember(
                  m,
                  [
                    ...notes,
                    { text: `Chatted with ${other}${where}`, importance: 2 },
                  ],
                  st.clock,
                ),
                ids,
                st.clock,
              );
          return withMind(
            withMind(st, a, learn(nameB, c.a_learns ?? [], c.a_recalled ?? [])),
            b,
            learn(nameA, c.b_learns ?? [], c.b_recalled ?? []),
          );
        });
        c.lines.forEach((line, i) =>
          window.setTimeout(
            () => game.current?.say(line.npc, line.text, LINE_MS + 200),
            i * LINE_MS,
          ),
        );
        hold = c.lines.length * LINE_MS;
      } finally {
        window.setTimeout(() => {
          for (const id of [a, b]) {
            busy.current.delete(id);
            game.current?.setNpcBusy(id, false);
          }
          setBackground((t) =>
            t === `${nameA} is chatting with ${nameB}` ? null : t,
          );
        }, hold);
      }
    },
    [api, chatEntry, flowId, npcById, sim, spec],
  );

  useEffect(() => {
    if (!sim || !ready || !started) return;
    const timer = window.setInterval(() => {
      const g = game.current;
      if (!g || lockedRef.current) return;
      const s = wsRef.current;
      const clock = s.clock + sim.minutes_per_second;
      setWs((st) => ({ ...st, clock: st.clock + sim.minutes_per_second }));
      const day = dayOf(clock);
      const places = new Map(sim.places.map((p) => [p.id, p]));
      for (const n of spec.npcs) {
        const mind = s.npc[n.id] ?? emptyMind();
        if (planEntry && mind.planDay !== day)
          queue.push({
            key: `plan:${n.id}`,
            kind: "plan",
            run: (signal) => planDay(n.id, day, signal),
          });
        const block =
          mind.planDay === day ? currentBlock(mind.plan, clock) : null;
        g.setNpcPlace(n.id, block ? (places.get(block.place) ?? null) : null);
        g.setNpcStatus(n.id, block ? activityEmoji(block.activity) : "");
      }
      // Only look for a chat when nothing else waits, so an encounter is never stale by the time it runs.
      if (chatEntry && !queue.busy && !queue.pending && queue.allows()) {
        const pair = findEncounter(
          g.npcTiles(),
          busy.current,
          s.npc,
          clock,
          sim.chat_cooldown_min,
        );
        if (pair) {
          setWs((st) => ({
            ...st,
            npc: markChat(st.npc, pair[0], pair[1], st.clock),
          }));
          queue.push({
            key: `chat:${pair[0]}:${pair[1]}`,
            kind: "chat",
            run: (signal) => chat(pair[0], pair[1], signal),
          });
        }
      }
      queue.pump();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [
    sim,
    ready,
    started,
    spec.npcs,
    planEntry,
    chatEntry,
    queue,
    planDay,
    chat,
  ]);

  const send = async () => {
    const text = draft.trim();
    if (!text || !talking || thinking) return;
    setDraft("");
    const npcId = talking;
    const before = addTurn(wsRef.current, npcId, { role: "player", text });
    setWs(before);
    if (!replyEntry) {
      setWs((s) => addTurn(s, npcId, { role: "npc", text: "..." }));
      toast(
        "Add an NPC Reply (live) node to the flow so villagers can answer.",
      );
      return;
    }
    setThinking(true);
    const generation = loads.current;
    try {
      const res = await api.invoke(
        flowId,
        replyEntry,
        {
          state: {
            world: spec,
            npc_id: npcId,
            history: (before.talks[npcId] ?? []).slice(0, -1),
            player_message: text,
            vars: before.vars,
            inventory: before.inventory,
            completed_quests: before.completedQuests,
            ...(sim
              ? { memory: before.npc[npcId]?.memory ?? [], clock: before.clock }
              : {}),
          },
        },
        {
          // The line appears as it is written; items and quests apply once the whole reply is in.
          onDelta: (piece) =>
            setLive((l) =>
              appendReplyDelta(l, npcId, generation, loads.current, piece),
            ),
        },
      );
      if (generation !== loads.current) return;
      setLive(null);
      const reply = res.outputs.reply as NpcReply | undefined;
      if (!reply?.text) throw new Error("No reply");
      const prior = wsRef.current;
      const applied = applyReply(spec, prior, npcId, reply);
      let state = applied.state;
      const notes = applied.notes;
      if (sim) {
        const done = spec.quests.find(
          (q) =>
            state.completedQuests.includes(q.id) &&
            !prior.completedQuests.includes(q.id),
        );
        state = withMind(state, npcId, (m) =>
          touch(
            remember(
              m,
              [
                ...(reply.remember ?? []),
                ...(done
                  ? [
                      {
                        text: `${spec.player.name} completed "${done.title}" for me`,
                        importance: 6,
                      },
                    ]
                  : []),
              ],
              state.clock,
            ),
            reply.recalled ?? [],
            state.clock,
          ),
        );
      }
      setWs(state);
      // An item handed over in conversation leaves the map too, like one picked up by walking to it.
      for (const id of state.picked)
        if (!prior.picked.includes(id)) game.current?.removeItem(id);
      for (const n of notes) toast(n);
    } catch (e) {
      if (generation !== loads.current) return;
      setLive(null);
      setWs((s) =>
        addTurn(s, npcId, {
          role: "npc",
          text: "(They seem lost in thought. Try again.)",
        }),
      );
      toast(e instanceof Error ? e.message : String(e));
    } finally {
      // A reply from before a load must not unlock the newer request still in flight.
      if (generation === loads.current) setThinking(false);
    }
  };

  const save = async (slot: string) => {
    const state = serialize({
      ...wsRef.current,
      player: game.current?.playerTile() ?? null,
    });
    try {
      await api.saveSession(
        flowId,
        slot,
        state,
        `${spec.title} · ${new Date().toLocaleString()}`,
      );
    } catch (e) {
      toast(`Could not save: ${e instanceof Error ? e.message : e}`);
      return;
    }
    toast("Saved");
    setPanel(null);
  };
  const load = async (slot: string) => {
    let rec: Awaited<ReturnType<typeof api.loadSession>>;
    try {
      rec = await api.loadSession(flowId, slot);
    } catch (e) {
      toast(`Could not load: ${e instanceof Error ? e.message : e}`);
      return;
    }
    const state = rec ? deserialize(spec, rec.state) : null;
    if (!state) {
      toast("That save is empty or from another world");
      return;
    }
    loads.current += 1;
    queue.clear();
    busy.current.clear();
    setBackground(null);
    // The reply in flight belongs to the old game; let the loaded one talk at once.
    setThinking(false);
    setLive(null);
    setWs(state);
    setTalking(null);
    setPanel(null);
    setMount((m) => m + 1);
    toast("Loaded");
  };

  const activeQuests = spec.quests.filter((q) =>
    ws.activeQuests.includes(q.id),
  );
  const doneQuests = spec.quests.filter((q) =>
    ws.completedQuests.includes(q.id),
  );
  const regionName = spec.regions.find((r) => r.id === region)?.name;
  const allDone =
    spec.quests.length > 0 && doneQuests.length === spec.quests.length;

  return (
    <div
      className="relative h-full w-full overflow-hidden bg-[#0b0d12] text-white"
      data-testid="world-player"
    >
      <PlayerKeyframes />
      {ambientUrl && started ? (
        // biome-ignore lint/a11y/useMediaCaption: background sound with no speech
        <audio
          src={ambientUrl}
          autoPlay={true}
          loop={true}
          data-testid="world-ambient"
          ref={(a) => {
            if (a) a.volume = 0.35;
          }}
        />
      ) : null}
      <div ref={host} className="absolute inset-0" data-testid="world-canvas" />

      {!ready && !failed ? (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-4 bg-[#0b0d12]">
          <SparklesIcon className="size-6 animate-pulse text-amber-200" />
          <p className="text-white/70">Building {spec.title}...</p>
          <div className="h-1.5 w-64 overflow-hidden rounded-full bg-white/10">
            <div
              className="h-full bg-gradient-to-r from-emerald-300 to-sky-300 transition-all"
              style={{ width: `${Math.round(loadProgress * 100)}%` }}
            />
          </div>
        </div>
      ) : null}
      {failed ? (
        <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-3 p-6 text-center">
          <p className="font-semibold text-lg">The world could not start</p>
          <p className="max-w-md text-white/60">{failed}</p>
        </div>
      ) : null}

      {/* HUD */}
      {ready ? (
        <>
          <div className="pointer-events-none absolute top-3 left-3 z-10 flex flex-col gap-1">
            <div className="rounded-full bg-black/45 px-4 py-1.5 font-semibold text-sm backdrop-blur-md ring-1 ring-white/10">
              {spec.title}
            </div>
            {sim ? (
              <div
                className="self-start rounded-full bg-black/35 px-3 py-1 text-white/80 text-xs tabular-nums backdrop-blur-md"
                data-testid="world-clock"
              >
                {clockLabel(ws.clock)}
              </div>
            ) : null}
            {regionName ? (
              <div className="flex items-center gap-1.5 self-start rounded-full bg-black/35 px-3 py-1 text-white/80 text-xs backdrop-blur-md">
                <MapPinIcon className="size-3" /> {regionName}
              </div>
            ) : null}
          </div>
          <div className="absolute top-3 right-3 z-10 flex items-center gap-1 rounded-full bg-black/40 px-1.5 py-1 backdrop-blur-md ring-1 ring-white/10">
            {[
              {
                id: "inventory" as const,
                label: `Inventory (I) · ${ws.inventory.length}`,
                icon: BackpackIcon,
              },
              {
                id: "quests" as const,
                label: "Quests (Q)",
                icon: ScrollTextIcon,
              },
              { id: "save" as const, label: "Save / Load", icon: SaveIcon },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                title={label}
                aria-label={label}
                onClick={() => setPanel((p) => (p === id ? null : id))}
                className={cn(
                  "relative rounded-full p-2 text-white/75 hover:bg-white/15 hover:text-white",
                  panel === id && "bg-white/20 text-white",
                )}
              >
                <Icon className="size-4" />
                {id === "inventory" && ws.inventory.length ? (
                  <span className="absolute -top-0.5 -right-0.5 flex size-4 items-center justify-center rounded-full bg-amber-300 font-bold text-[10px] text-black">
                    {ws.inventory.length}
                  </span>
                ) : null}
                {id === "quests" && activeQuests.length ? (
                  <span className="absolute -top-0.5 -right-0.5 size-2.5 rounded-full bg-emerald-300" />
                ) : null}
              </button>
            ))}
            {view === "3d" && ready && started && !editor ? (
              <button
                type="button"
                title="Edit the scene: place, move and rotate props"
                aria-label="Edit scene"
                data-testid="world-edit-toggle"
                onClick={() => void openEditor()}
                className="flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-white/75 text-xs hover:bg-white/15 hover:text-white"
              >
                <PencilRulerIcon className="size-4" />
                Edit
              </button>
            ) : null}
            {can3d && !editor ? (
              <button
                type="button"
                title={view === "3d" ? "Map view" : "3D view"}
                aria-label={view === "3d" ? "Map view" : "3D view"}
                data-testid="world-view-toggle"
                onClick={() => setView((v) => (v === "3d" ? "2d" : "3d"))}
                className={cn(
                  "flex items-center gap-1.5 rounded-full px-2.5 py-1.5 text-white/75 text-xs hover:bg-white/15 hover:text-white",
                  view === "3d" && "bg-white/20 text-white",
                )}
              >
                {view === "3d" ? (
                  <MapIcon className="size-4" />
                ) : (
                  <BoxIcon className="size-4" />
                )}
                {view === "3d" ? "Map" : "3D"}
              </button>
            ) : null}
            {canReroll && onReroll ? (
              <button
                type="button"
                title="New world"
                aria-label="New world"
                onClick={onReroll}
                className="rounded-full p-2 text-white/75 hover:bg-white/15"
              >
                <RotateCcwIcon className="size-4" />
              </button>
            ) : null}
            {onExit ? (
              <button
                type="button"
                title="Back to the editor"
                aria-label="Back to the editor"
                onClick={onExit}
                className="rounded-full p-2 text-white/75 hover:bg-white/15"
              >
                <ArrowLeftIcon className="size-4" />
              </button>
            ) : null}
          </div>
          {background && !talking ? (
            <div
              className="pointer-events-none absolute top-3 left-1/2 z-10 flex -translate-x-1/2 items-center gap-2 rounded-full bg-black/45 px-3 py-1 text-white/75 text-xs backdrop-blur-md"
              data-testid="world-sim-activity"
            >
              <span className="size-1.5 animate-pulse rounded-full bg-emerald-300" />
              {background}
            </div>
          ) : null}
          {nearby && !talking && started ? (
            <div className="pointer-events-none absolute bottom-6 left-1/2 z-10 -translate-x-1/2 rounded-full bg-black/55 px-5 py-2 text-sm backdrop-blur-md ring-1 ring-white/10">
              <kbd className="mr-2 rounded bg-white px-1.5 py-0.5 font-bold text-black text-xs">
                E
              </kbd>
              {nearby.kind === "npc"
                ? `Talk to ${nearby.name}`
                : `Pick up ${nearby.name}`}
            </div>
          ) : null}
        </>
      ) : null}

      {panel === "inventory" ? (
        <Panel title="Inventory" onClose={() => setPanel(null)}>
          {ws.inventory.length === 0 ? (
            <p className="text-sm text-white/50">Nothing yet. Explore!</p>
          ) : null}
          <div className="flex flex-col gap-2">
            {ws.inventory.map((id) => {
              const it = itemById.get(id);
              return it ? (
                <div
                  key={id}
                  className="flex items-center gap-3 rounded-xl bg-white/5 p-2"
                >
                  <SpriteThumb api={api} media={it.sprite} fallback={it.name} />
                  <div>
                    <div className="font-medium text-sm">{it.name}</div>
                    {it.description ? (
                      <div className="text-white/55 text-xs">
                        {it.description}
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null;
            })}
          </div>
        </Panel>
      ) : null}
      {panel === "quests" ? (
        <Panel title="Quests" onClose={() => setPanel(null)}>
          {activeQuests.length + doneQuests.length === 0 ? (
            <p className="text-sm text-white/50">
              Talk to the villagers to find something to do.
            </p>
          ) : null}
          <div className="flex flex-col gap-2">
            {activeQuests.map((q) => (
              <div key={q.id} className="rounded-xl bg-white/5 p-3">
                <div className="font-medium text-sm">{q.title}</div>
                <div className="text-white/60 text-xs">{q.description}</div>
                <div className="mt-1 text-[11px] text-emerald-200/80">
                  From {npcById.get(q.giver)?.name}
                </div>
              </div>
            ))}
            {doneQuests.map((q) => (
              <div
                key={q.id}
                className="rounded-xl bg-emerald-400/10 p-3 text-emerald-100/80"
              >
                <div className="font-medium text-sm line-through decoration-emerald-200/50">
                  {q.title}
                </div>
                <div className="text-xs">{q.reward_text}</div>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}
      {editor && edState ? (
        <Suspense fallback={null}>
          <WorldEditorPanel
            editor={editor}
            state={edState}
            status={edStatus}
            error={edError}
            canSave={Boolean(api.saveWorldProps)}
            canImport={Boolean(api.uploadModel)}
            dirty={edDirty}
            onImport={(f) => void importModel(f)}
            onExport={() => void exportScene()}
            onSave={() => void saveScene()}
            onDone={() => void closeEditor()}
          />
        </Suspense>
      ) : null}

      {panel === "save" ? (
        <Panel title="Save / Load" onClose={() => setPanel(null)}>
          <div className="flex flex-col gap-2">
            {SLOTS.map((slot, i) => (
              <div
                key={slot}
                className="flex items-center justify-between rounded-xl bg-white/5 p-2 pl-3 text-sm"
              >
                Slot {i + 1}
                <div className="flex gap-1">
                  <button
                    type="button"
                    onClick={() => void save(slot)}
                    className="rounded-full bg-white/10 px-3 py-1 hover:bg-white/20"
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    onClick={() => void load(slot)}
                    className="rounded-full bg-white/10 px-3 py-1 hover:bg-white/20"
                  >
                    Load
                  </button>
                </div>
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      {/* dialogue */}
      {talkNpc ? (
        <div
          className="absolute inset-x-0 bottom-0 z-20 p-4 md:p-6"
          data-testid="world-dialogue"
        >
          <div
            className="mx-auto flex max-w-3xl gap-4 rounded-2xl bg-zinc-950/80 p-4 shadow-2xl ring-1 ring-white/10 backdrop-blur-md"
            style={{ animation: "flows-rise 220ms ease-out" }}
          >
            <div className="hidden w-24 shrink-0 flex-col items-center gap-2 md:flex">
              <div className="flex size-20 items-center justify-center overflow-hidden rounded-2xl bg-gradient-to-b from-white/10 to-white/0">
                <SpriteThumb
                  api={api}
                  media={talkNpc.sprite}
                  fallback={talkNpc.name}
                />
              </div>
              <div className="text-center font-semibold text-sm">
                {talkNpc.name}
              </div>
            </div>
            <div className="flex min-w-0 flex-1 flex-col gap-3">
              <div
                ref={scroller}
                className="flex max-h-52 flex-col gap-2 overflow-y-auto pr-1"
              >
                {turns.map((t, i) => (
                  <div
                    key={i}
                    className={cn(
                      "max-w-[85%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed",
                      t.role === "npc"
                        ? "self-start bg-white/10"
                        : "self-end bg-sky-400/25 text-sky-50",
                    )}
                  >
                    {t.text}
                  </div>
                ))}
                {thinking && live && live.npcId === talking ? (
                  <div
                    className="max-w-[85%] self-start rounded-2xl bg-white/10 px-3.5 py-2 text-sm leading-relaxed"
                    data-testid="world-npc-streaming"
                  >
                    {live.text}
                  </div>
                ) : thinking && !warmed ? (
                  <div className="self-start rounded-2xl bg-white/10 px-4 py-2 text-sm text-white/70">
                    Waking up the villagers...
                  </div>
                ) : thinking ? (
                  <div className="self-start rounded-2xl bg-white/10 px-4 py-2 text-sm">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        style={{
                          animation: `flows-dots 1.2s ${i * 0.2}s infinite`,
                        }}
                      >
                        •
                      </span>
                    ))}
                  </div>
                ) : null}
              </div>
              <form
                className="flex gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  void send();
                }}
              >
                <input
                  // biome-ignore lint/a11y/noAutofocus: the conversation box is the only thing to type into
                  autoFocus={true}
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  placeholder={`Say something to ${talkNpc.name}...`}
                  className="min-w-0 flex-1 rounded-full bg-white/10 px-4 py-2 text-sm outline-none ring-1 ring-white/10 placeholder:text-white/35 focus:ring-white/30"
                  data-testid="world-say"
                />
                <button
                  type="submit"
                  disabled={thinking || !draft.trim()}
                  className="rounded-full bg-white px-4 text-black disabled:opacity-40"
                  aria-label="Send"
                >
                  <SendIcon className="size-4" />
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setTalking(null);
                    game.current?.faceNpc(null);
                  }}
                  className="rounded-full bg-white/10 px-3 text-sm hover:bg-white/20"
                >
                  Bye
                </button>
              </form>
            </div>
          </div>
        </div>
      ) : null}

      {/* intro */}
      {ready && !started ? (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/55 p-6 backdrop-blur-sm">
          <div
            className="max-w-lg rounded-3xl bg-zinc-950/85 p-8 text-center shadow-2xl ring-1 ring-white/10"
            style={{ animation: "flows-rise 500ms ease-out" }}
          >
            <h1 className="mb-3 font-bold text-3xl">{spec.title}</h1>
            {spec.intro_text ? (
              <p className="mb-5 text-white/75 leading-relaxed">
                {spec.intro_text}
              </p>
            ) : null}
            <p className="mb-6 text-white/45 text-xs">
              {view === "3d"
                ? "Click to look with the mouse · WASD to walk, arrows to turn · E or Space to talk and pick up · I inventory · Q quests"
                : "WASD / arrows or click to walk · E or Space to talk and pick up · I inventory · Q quests"}
            </p>
            <button
              type="button"
              data-testid="world-start"
              onClick={() => setStarted(true)}
              className="rounded-full bg-white px-8 py-3 font-semibold text-black shadow-xl transition hover:scale-105"
            >
              Start exploring
            </button>
          </div>
        </div>
      ) : null}

      {allDone ? (
        <div className="pointer-events-none absolute top-16 left-1/2 z-10 -translate-x-1/2 rounded-full bg-gradient-to-r from-amber-300 to-pink-300 px-5 py-1.5 font-semibold text-black text-sm shadow-xl">
          Every quest complete!
        </div>
      ) : null}

      <div className="pointer-events-none absolute bottom-24 left-4 z-30 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="rounded-xl bg-black/70 px-4 py-2 text-sm shadow-lg backdrop-blur ring-1 ring-white/10"
            style={{ animation: "flows-rise 220ms ease-out" }}
          >
            {t.text}
          </div>
        ))}
      </div>
    </div>
  );
}
