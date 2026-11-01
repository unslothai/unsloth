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
  type RefObject,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";

import { cn } from "@/lib/utils";

import type { MediaValue, WorldSpec } from "../../types";
import { type PlayerApi, mediaObjectUrl } from "../player-api";
import { PlayerKeyframes } from "../story/backdrop";
import { useMediaUrl } from "../use-media-url";
import {
  type Cmd,
  GOODS,
  GOOD_IDS,
  type Good,
  type Ground,
  PLAYER,
  type Stand,
  autoStep,
  buyBack,
  buyBackPrice,
  homeMarker,
  playerBuy,
  playerSell,
  priceOf,
  resolveEncounter,
  safeAt,
} from "./autonomy";
import {
  type LiveReply,
  OFFLINE_NOTE,
  appendReplyDelta,
  isNoChatModelError,
  offlineLine,
} from "./live-reply";
import { cityQueueRate, mayorOn } from "./city-advisor";
import { MayorHud } from "./mayor-panel";
import { heldInChests, interactProp } from "./action-verbs";
import { ACTIONS, NEEDS, type Needs, advertisements } from "./needs";
import type { Interactable, WorldArt, WorldGameHandle } from "./phaser-game";
import {
  type Point,
  type TileKind,
  type WorldMap,
  generateMap,
} from "./procgen";
import { RIVALS_CALLS_PER_MINUTE } from "./rivals";
import { RivalsPanel } from "./rivals-panel";
import { autonomyOf, autonomyOn, roleOf, rivalFor } from "./roles";
import {
  placeAreas,
  steerActions,
  steeredNeeds,
  villagerStep,
} from "./world-actions";
import {
  type DirectorPatch,
  applyPatch,
  directorDue,
  directorRequest,
  noAdditions,
  withAdditions,
} from "./world-director";
import {
  type EditorState,
  GlbError,
  SaveQueue,
  type WorldEditor,
  inspectGlb,
  saveUntilCurrent,
} from "./props3d";
import * as Life from "./life-sim";
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
  needsOn,
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
  applyUse,
  deserialize,
  foldSummary,
  initialWorldState,
  pickUp,
  serialize,
  startQuestsOf,
  summaryDue,
  talkHistory,
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
  a_feels?: { affinity?: number; fact?: string };
  b_feels?: { affinity?: number; fact?: string };
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
      } catch {}
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
      } finally {
        onProgress(++done / Math.max(1, jobs.length));
      }
    }),
  );
  return art;
}

function run(g: WorldGameHandle, cmds: Cmd[]): void {
  for (const c of cmds) {
    if (c.do === "place") g.setNpcPlace(c.id, c.area);
    else if (c.do === "status") g.setNpcStatus(c.id, c.icon);
    else if (c.do === "say") g.say(c.id, c.text, 3000);
    else if (c.do === "remove") g.removeItem(c.id);
    else g.addMarker(c.id, c.at, c.icon, c.label);
  }
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
          title="Close"
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

interface LiveStore {
  get(): LiveReply | null;
  set(next: LiveReply | null): void;
  subscribe(fn: () => void): () => void;
}

function liveStore(): LiveStore {
  let value: LiveReply | null = null;
  const subs = new Set<() => void>();
  return {
    get: () => value,
    set(next) {
      if (next === value) return;
      value = next;
      for (const fn of subs) fn();
    },
    subscribe(fn) {
      subs.add(fn);
      return () => subs.delete(fn);
    },
  };
}

// Streamed tokens re-render only this bubble, not the whole player.
function ReplyStatus({
  live: store,
  talking,
  thinking,
  warmed,
  scroller,
}: {
  live: LiveStore;
  talking: string;
  thinking: boolean;
  warmed: boolean;
  scroller: RefObject<HTMLDivElement | null>;
}) {
  const live = useSyncExternalStore(store.subscribe, store.get);
  const shown = useRef(live?.text);
  useEffect(() => {
    if (shown.current === live?.text) return;
    shown.current = live?.text;
    scroller.current?.scrollTo({
      top: scroller.current.scrollHeight,
      behavior: "smooth",
    });
  }, [live?.text, scroller]);
  return thinking && live && live.npcId === talking ? (
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
  ) : null;
}

// Owns the draft so typing re-renders only the form.
function SayForm({
  npcName,
  thinking,
  draft: kept,
  onSend,
  onBye,
}: {
  npcName: string;
  thinking: boolean;
  draft: RefObject<string>;
  onSend: (text: string) => boolean;
  onBye: () => void;
}) {
  const [draft, setDraft] = useState(kept.current);
  const edit = (text: string) => {
    kept.current = text;
    setDraft(text);
  };
  return (
    <form
      className="flex gap-2"
      onSubmit={(e) => {
        e.preventDefault();
        if (onSend(draft)) edit("");
      }}
    >
      <input
        // biome-ignore lint/a11y/noAutofocus: the conversation box is the only thing to type into
        autoFocus={true}
        value={draft}
        onChange={(e) => edit(e.target.value)}
        placeholder={`Say something to ${npcName}...`}
        className="min-w-0 flex-1 rounded-full bg-white/10 px-4 py-2 text-sm outline-none ring-1 ring-white/10 placeholder:text-white/35 focus:ring-white/30"
        data-testid="world-say"
      />
      <button
        type="submit"
        disabled={thinking || !draft.trim()}
        className="rounded-full bg-white px-4 text-black disabled:opacity-40"
        aria-label="Send"
        title="Send"
      >
        <SendIcon className="size-4" />
      </button>
      <button
        type="button"
        onClick={onBye}
        className="rounded-full bg-white/10 px-3 text-sm hover:bg-white/20"
      >
        Bye
      </button>
    </form>
  );
}

export function WorldPlayer({
  flowId,
  api,
  spec: baseSpec,
  canReroll,
  onReroll,
  onExit,
  onReady,
}: WorldPlayerProps) {
  const map = useMemo(() => generateMap(baseSpec), [baseSpec]);
  const host = useRef<HTMLDivElement>(null);
  const game = useRef<WorldGameHandle | null>(null);
  const worldRef = useRef<WorldMap | null>(null);
  const [ws, setWs] = useState<WorldState>(() => initialWorldState(baseSpec));
  const wsRef = useRef(ws);
  wsRef.current = ws;
  const spec = useMemo(
    () => withAdditions(baseSpec, ws.additions),
    [baseSpec, ws.additions],
  );
  const specRef = useRef(spec);
  specRef.current = spec;
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
  const carried = useRef<{
    map: WorldMap;
    props: WorldMap["props"];
    blocked: WorldMap["blocked"];
    npcs?: Record<string, Point>;
  } | null>(null);
  const bootedView = useRef<"2d" | "3d" | null>(null);
  const ambientUrl = useMediaUrl(
    api,
    view === "3d" ? spec.view3d?.ambient : null,
  );
  const [envoyTalking, setEnvoyTalking] = useState(false);
  const lockedRef = useRef(true);
  lockedRef.current =
    !started ||
    talking !== null ||
    envoyTalking ||
    panel === "save" ||
    !!ws.auto?.encounter ||
    editor !== null;

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

  const sim = useMemo(() => simOf(baseSpec), [baseSpec]);
  const planEntry = sim ? spec.entry_points?.npc_plan : undefined;
  const chatEntry = sim ? spec.entry_points?.npc_chat : undefined;
  const queue = useMemo(
    () =>
      new BackgroundQueue(
        sim?.calls_per_minute ??
          (mayorOn(baseSpec)
            ? cityQueueRate(baseSpec)
            : baseSpec.rivals
              ? RIVALS_CALLS_PER_MINUTE
              : 0),
      ),
    [sim, baseSpec],
  );
  const cityTile = useRef<(p: Point, drag: boolean) => boolean>(() => false);
  useEffect(() => () => queue.clear(), [queue]);
  const busy = useRef(new Set<string>());
  const steered = useRef(new Set<string>());
  const withNeeds = useMemo(() => needsOn(baseSpec), [baseSpec]);
  const offers = useMemo(() => advertisements(sim?.places ?? []), [sim]);
  const areas = useMemo(() => placeAreas(baseSpec), [baseSpec]);
  const lifeCfg = useMemo(() => Life.lifeOf(baseSpec), [baseSpec]);
  const [background, setBackground] = useState<string | null>(null);

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
              [
                {
                  text: `${spec.player.name} ${what}`,
                  importance,
                  guess: lifeCfg.reflectAt > 0,
                },
              ],
              s.clock,
            ),
          );
      }
      return next;
    },
    [sim, spec.player.name, lifeCfg],
  );

  const onInteract = useCallback(
    (t: Interactable) => {
      if (t.kind === "prop") {
        const g = game.current;
        if (!t.prop || !g) return;
        const s = wsRef.current;
        const at = (p: { x: number; y: number }) =>
          `${Math.floor(p.x)},${Math.floor(p.y)}`;
        const out = interactProp(
          t.prop,
          worldRef.current?.props ?? [],
          s.props ?? {},
          s.inventory,
          {
            name: (id) => itemById.get(id)?.name ?? id,
            occupied: (p) =>
              g.blocksProp?.(p) === true ||
              [g.playerTile(), ...Object.values(g.npcTiles())].some(
                (q) => at(q) === at(p),
              ),
          },
        );
        if (out.changed.length || out.gained.length) {
          const next = applyUse(s, out);
          wsRef.current = next;
          setWs(next);
          g.setPropStates?.(out.states, out.changed, out.loud);
        }
        toast(out.text);
        return;
      }
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
          next = addTurn(
            next,
            npc.id,
            { role: "npc", text: npc.greeting || "Hello!" },
            summarising,
          );
        return next;
      });
      game.current?.faceNpc(npc.id);
      setTalking(npc.id);
    },
    [itemById, npcById, spec, toast, witness],
  );
  const onInteractRef = useRef(onInteract);
  onInteractRef.current = onInteract;

  useEffect(() => {
    let live = true;
    const bootedFor = loads.current;
    let handle: WorldGameHandle | null = null;
    let world: WorldMap | null = null;
    const switched = bootedView.current !== null && bootedView.current !== view;
    bootedView.current = view;
    const startAt = switched ? lastTile.current : wsRef.current.player;
    setReady(false);
    const kept = carried.current?.map === map ? carried.current : null;
    (async () => {
      try {
        const [art, create] = await Promise.all([
          loadArt(
            api,
            baseSpec,
            (f) => live && setLoadProgress(f),
            view === "3d"
              ? [
                  ...new Set(
                    ((kept ? kept.props : map.props) ?? [])
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
        const added = wsRef.current.additions ?? noAdditions();
        const saved = (switched && kept?.npcs) || wsRef.current.npcAt || {};
        world = {
          ...map,
          ...(kept ? { props: kept.props, blocked: kept.blocked } : {}),
          npcs: {
            ...map.npcs,
            ...Object.fromEntries(added.npcs.map((n) => [n.id, n])),
            ...saved,
          },
          items: {
            ...map.items,
            ...Object.fromEntries(added.items.map((i) => [i.id, i])),
          },
        };
        worldRef.current = world;
        handle = create(
          host.current,
          specRef.current,
          world,
          art,
          [
            ...wsRef.current.picked,
            ...heldInChests(world.props ?? [], wsRef.current.props ?? {}),
          ],
          startAt,
          {
            onNearby: (t) => setNearby(t),
            onInteract: (t) => onInteractRef.current(t),
            onRegion: (id) => {
              setRegion(id);
              const r = baseSpec.regions.find((x) => x.id === id);
              if (r)
                toast(
                  r.on_enter_text ? `${r.name} · ${r.on_enter_text}` : r.name,
                );
            },
            locked: () => lockedRef.current,
            action: wsRef.current.action,
            props: wsRef.current.props,
            onEvent: (text) => toast(text),
            safeAt: (p) => safeRef.current(p),
            onTile: (p, drag) => cityTile.current(p, drag),
          },
        );
        game.current = handle;
        for (const [id, h] of Object.entries(wsRef.current.auto?.homes ?? {}))
          if (h.at)
            run(handle, [
              homeMarker(
                id,
                specRef.current.npcs.find((n) => n.id === id)?.name ?? id,
                h.at,
              ),
            ]);
        setReady(true);
        onReady?.(handle);
      } catch (e) {
        if (live) setFailed(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      live = false;
      if (handle) {
        lastTile.current = handle.playerTile();
        if (world)
          carried.current = {
            map,
            props: world.props,
            blocked: world.blocked,
            npcs: loads.current === bootedFor ? handle.npcTiles() : undefined,
          };
        const action = handle.actionState?.() as WorldState["action"];
        if (action && loads.current === bootedFor) {
          wsRef.current = { ...wsRef.current, action };
          setWs((st) => ({ ...st, action }));
        }
      }
      handle?.destroy();
      game.current = null;
      setEditor(null);
      setEdState(null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, baseSpec, map, mount, view]);

  const dashKey =
    view === "3d" &&
    baseSpec.action?.movement === true &&
    baseSpec.action?.verbs === true;
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
      else if ((e.key === "q" || e.key === "Q") && !dashKey)
        setPanel((p) => (p === "quests" ? null : "quests"));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [editing, dashKey]);

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

  const draft = useRef("");
  const [thinking, setThinking] = useState(false);
  const live = useMemo(() => liveStore(), []);
  const replyEntry = spec.entry_points?.npc_reply;
  const placeEntry = spec.entry_points?.describe_place;
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
  const talkMind = talking ? ws.npc[talking] : undefined;
  const talkNeeds = talkMind?.needs;
  const talkDoing = talkMind?.doing
    ? talkMind.doing.kind === "go_to"
      ? (talkMind.doing.label ?? "")
      : `${ACTIONS[talkMind.doing.kind].icon} ${talkMind.doing.kind}`
    : "";
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scroller.current?.scrollTo({
      top: scroller.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns.length, thinking]);

  const loads = useRef(0);
  const messages = useRef(0);
  const offlineNoted = useRef(false);

  const summarising = spec.summarise !== false;
  const summaryRun = useRef<AbortController | null>(null);
  const summarise = useCallback(
    async (npcId: string, s: WorldState) => {
      const due = summarising && replyEntry ? summaryDue(s, npcId) : null;
      if (!due || summaryRun.current) return;
      const generation = loads.current;
      const run = new AbortController();
      summaryRun.current = run;
      try {
        const res = await api.invoke(
          flowId,
          replyEntry!,
          {
            state: {
              task: "summarise",
              world: spec,
              npc_id: npcId,
              summary: s.summaries[npcId],
              turns: due.turns,
            },
          },
          { signal: run.signal },
        );
        const text = (res.outputs.reply as { summary?: string } | undefined)
          ?.summary;
        if (text && !run.signal.aborted && generation === loads.current)
          setWs((st) => foldSummary(st, npcId, text, due.upto));
      } catch {
      } finally {
        if (summaryRun.current === run) summaryRun.current = null;
      }
    },
    [api, flowId, replyEntry, spec, summarising],
  );

  const growing = baseSpec.new_quests !== false;
  const directorRun = useRef<AbortController | null>(null);
  const events = useRef<string[]>([]);
  const direct = useCallback(
    async (justCompleted: boolean, from?: WorldState) => {
      const s = from ?? wsRef.current;
      const played = specRef.current;
      if (
        !growing ||
        !replyEntry ||
        directorRun.current ||
        !directorDue(played, s, justCompleted)
      )
        return;
      const generation = loads.current;
      const run = new AbortController();
      directorRun.current = run;
      try {
        const res = await api.invoke(
          flowId,
          replyEntry,
          { state: directorRequest(played, s, events.current) },
          { signal: run.signal },
        );
        const patch = (
          res.outputs.reply as { patch?: DirectorPatch } | undefined
        )?.patch;
        if (!patch || run.signal.aborted || generation !== loads.current)
          return;
        const h = game.current;
        const occupied = [
          ...Object.values(h?.npcTiles() ?? map.npcs),
          ...Object.values(map.items),
          ...(wsRef.current.additions?.items ?? []),
          ...(h ? [h.playerTile()] : []),
        ];
        const now = wsRef.current;
        const added = applyPatch(specRef.current, map, now, patch, occupied);
        if (!added) return;
        setWs(added.state);
        for (const n of added.npcs) h?.addNpc(n, n);
        for (const i of added.items) h?.addItem(i, i);
        const giver = withAdditions(baseSpec, added.state.additions).npcs.find(
          (n) => n.id === patch.quest.giver,
        );
        toast(
          `New quest: ${patch.quest.title}${giver ? ` · ask ${giver.name}` : ""}`,
        );
      } catch {
      } finally {
        if (directorRun.current === run) directorRun.current = null;
      }
    },
    [api, baseSpec, flowId, growing, map, replyEntry, toast],
  );
  useEffect(() => {
    if (ready && started) void direct(false);
  }, [ready, started, direct]);

  useEffect(() => {
    queue.setPaused(!ready || lockedRef.current);
    queue.pump();
  }, [queue, ready, started, talking, panel, envoyTalking]);

  const planDay = useCallback(
    async (npcId: string, day: number, signal: AbortSignal) => {
      if (!planEntry || !sim) return;
      const generation = loads.current;
      const s = wsRef.current;
      const here =
        placeAt(sim.places, game.current?.npcTile(npcId) ?? { x: -1, y: -1 })
          ?.id ?? "";
      const mind = s.npc[npcId] ?? emptyMind();
      if (mind.planDay === day && !Life.planDue(lifeCfg, mind, s.clock))
        return false;
      const hit = Life.cachedPlan(spec, lifeCfg, s, npcId, s.clock, here);
      if (hit) {
        setWs(hit);
        return false;
      }
      setBackground(`${npcById.get(npcId)?.name ?? npcId} is planning the day`);
      const key = Life.planCacheKey(
        lifeCfg,
        spec,
        npcId,
        s.npc[npcId],
        s.clock,
        here,
      );
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
              ...Life.planAsk(lifeCfg, s.npc[npcId]),
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
          Life.storePlan(
            withMind(st, npcId, (m) =>
              remember(
                {
                  ...m,
                  plan: blocks,
                  planDay: day,
                  ...Life.lifePlanned(m, res.outputs.plan, st.clock),
                },
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
            lifeCfg,
            key,
            res.outputs.plan,
            st.clock,
          ),
        );
      } catch (e) {
        if (!signal.aborted && generation === loads.current)
          setWs((st) =>
            withMind(st, npcId, (m) => ({
              ...m,
              plan: [],
              planDay: day,
              ...Life.lifePlanned(m, null, st.clock),
            })),
          );
        throw e;
      } finally {
        setBackground(null);
      }
    },
    [api, flowId, npcById, planEntry, sim, spec, lifeCfg],
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
              ...Life.chatAsk(lifeCfg, s, a, b),
            },
          },
          { signal },
        );
        if (generation !== loads.current) return;
        const c = res.outputs.chat as NpcChat | undefined;
        if (!c?.lines?.length) return;
        const where = place ? ` at ${place}` : "";
        setWs((st) => {
          const heard = (from: string, notes: Note[]) =>
            lifeCfg.relationships ? Life.rumours(st, from, notes) : notes;
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
          return Life.afterChat(
            lifeCfg,
            withMind(
              withMind(
                st,
                a,
                learn(nameB, heard(b, c.a_learns ?? []), c.a_recalled ?? []),
              ),
              b,
              learn(nameA, heard(a, c.b_learns ?? []), c.b_recalled ?? []),
            ),
            a,
            b,
            c,
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
    [api, chatEntry, flowId, npcById, sim, spec, lifeCfg],
  );

  // One 1 Hz timer for the sim, errand and autonomy ticks, so their updates land in one render.
  const ticks = useRef<((() => void) | null)[]>([null, null, null]);
  const tickWs = useCallback((fn: (s: WorldState) => WorldState) => {
    wsRef.current = fn(wsRef.current);
    setWs(fn);
  }, []);
  useEffect(() => {
    if (!ready || !started) return;
    const timer = window.setInterval(() => {
      for (const tick of ticks.current) tick?.();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [ready, started]);

  useEffect(() => {
    if (!sim || !ready || !started) return;
    ticks.current[0] = () => {
      const g = game.current;
      if (!g || lockedRef.current) return;
      const s = wsRef.current;
      const clock = s.clock + sim.minutes_per_second;
      tickWs((st) => ({ ...st, clock: st.clock + sim.minutes_per_second }));
      const day = dayOf(clock);
      const places = new Map(sim.places.map((p) => [p.id, p]));
      const lives: Record<
        string,
        NonNullable<ReturnType<typeof villagerStep>>
      > = {};
      const ticked: Record<string, Needs> = {};
      for (const n of specRef.current.npcs) {
        const mind = s.npc[n.id] ?? emptyMind();
        const cached =
          planEntry && Life.planDue(lifeCfg, mind, clock)
            ? Life.cachedPlan(
                specRef.current,
                lifeCfg,
                s,
                n.id,
                clock,
                placeAt(sim.places, g.npcTile(n.id) ?? { x: -1, y: -1 })?.id ??
                  "",
              )
            : undefined;
        if (cached) setWs(cached);
        else if (cached === null)
          queue.push({
            key: `plan:${n.id}`,
            kind: "plan",
            run: (signal) => planDay(n.id, day, signal),
          });
        if (steered.current.has(n.id) || autoSteered.current.has(n.id)) {
          const needs = withNeeds
            ? steeredNeeds(mind, sim.minutes_per_second)
            : null;
          if (needs) ticked[n.id] = needs;
          continue;
        }
        const life = withNeeds
          ? villagerStep({
              g,
              spec: specRef.current,
              sim,
              npcId: n.id,
              mind,
              clock,
              offers,
              areas,
              held: s.auto?.held,
            })
          : null;
        if (life) {
          lives[n.id] = life;
          continue;
        }
        const block = Life.lifeBlock(mind, clock);
        g.setNpcPlace(n.id, block ? (places.get(block.place) ?? null) : null);
        g.setNpcStatus(n.id, block ? activityEmoji(block.activity) : "");
      }
      if (Object.keys(lives).length || Object.keys(ticked).length)
        tickWs((st) => {
          const npc = { ...st.npc };
          for (const [id, life] of Object.entries(lives))
            if (npc[id]) npc[id] = { ...npc[id], ...life };
          for (const [id, needs] of Object.entries(ticked))
            if (npc[id]) npc[id] = { ...npc[id], needs };
          return { ...st, npc };
        });
      if (chatEntry && !queue.busy && !queue.pending && queue.allows()) {
        const pair = findEncounter(
          g.npcTiles(),
          busy.current,
          s.npc,
          clock,
          sim.chat_cooldown_min,
        );
        if (pair) {
          tickWs((st) => ({
            ...st,
            npc: markChat(st.npc, pair[0], pair[1], st.clock),
          }));
          if (
            !Life.smallTalk({
              cfg: lifeCfg,
              spec: specRef.current,
              s,
              a: pair[0],
              b: pair[1],
              clock,
              say: g.say,
              update: setWs,
            })
          )
            queue.push({
              key: `chat:${pair[0]}:${pair[1]}`,
              kind: "chat",
              run: (signal) => chat(pair[0], pair[1], signal),
            });
        }
      }
      if (planEntry) {
        const gen = loads.current;
        for (const job of Life.lifeJobs(
          specRef.current,
          lifeCfg,
          s,
          (state, signal) =>
            api
              .invoke(flowId, planEntry, { state }, { signal })
              .then((r) => r.outputs.plan),
          (fn) => {
            if (gen === loads.current) setWs(fn);
          },
        ))
          queue.push(job);
      }
      queue.pump();
    };
    return () => {
      ticks.current[0] = null;
    };
  }, [
    sim,
    ready,
    started,
    baseSpec,
    planEntry,
    chatEntry,
    queue,
    planDay,
    chat,
    withNeeds,
    offers,
    areas,
    lifeCfg,
    api,
    flowId,
    tickWs,
  ]);

  useEffect(() => {
    if (!ready || !started) return;
    ticks.current[1] = () => {
      const g = game.current;
      if (!g) return;
      const { steered: now, expired } = steerActions(
        g,
        areas,
        wsRef.current,
        steered.current,
      );
      steered.current = now;
      if (expired.length)
        tickWs((st) => {
          const errands = { ...(st.errands ?? {}) };
          for (const id of expired) delete errands[id];
          return { ...st, errands };
        });
    };
    return () => {
      ticks.current[1] = null;
    };
  }, [ready, started, areas, tickWs]);

  const flags = useMemo(() => autonomyOf(baseSpec), [baseSpec]);
  const autoOn = useMemo(() => autonomyOn(baseSpec), [baseSpec]);
  const shooterOn = baseSpec.action?.shooter === true && view === "3d";
  const safeRef = useRef<(p: Point) => boolean>(() => false);
  const autoSteered = useRef(new Set<string>());
  useEffect(() => {
    if (!autoOn || !ready || !started) return;
    ticks.current[2] = () => {
      const g = game.current;
      if (!g || lockedRef.current) return;
      const played = specRef.current;
      const added = wsRef.current.additions ?? noAdditions();
      const chestHeld = new Set(
        heldInChests(map.props ?? [], wsRef.current.props ?? {}),
      );
      const ground: Ground = {
        spec: played,
        map,
        areas,
        // Items still in an unopened chest are not on the ground to race for.
        items: Object.fromEntries(
          Object.entries({
            ...map.items,
            ...Object.fromEntries(
              added.items.map((i) => [i.id, { x: i.x, y: i.y }]),
            ),
          }).filter(([id]) => !chestHeld.has(id)),
        ),
        homes: {
          ...map.npcs,
          ...Object.fromEntries(
            added.npcs.map((n) => [n.id, { x: n.x, y: n.y }]),
          ),
        },
      };
      const base = wsRef.current;
      const out = autoStep(ground, base, {
        player: g.playerTile(),
        npcs: g.npcTiles(),
      });
      safeRef.current = (p) => safeAt(ground, out.state.auto ?? base.auto!, p);
      if (shooterOn && out.state.auto?.encounter && !base.auto?.encounter) {
        out.state = {
          ...out.state,
          auto: { ...out.state.auto, encounter: null },
        };
        out.notes = out.notes.filter((n) => !n.endsWith("blocks your way!"));
      }
      run(g, out.cmds);
      const now = new Set(out.steered);
      for (const id of autoSteered.current)
        if (!now.has(id) && !steered.current.has(id)) {
          g.setNpcPlace(id, null);
          g.setNpcStatus(id, "");
        }
      autoSteered.current = now;
      tickWs((st) => {
        const npc = { ...st.npc };
        for (const [id, m] of Object.entries(out.state.npc))
          if (m !== base.npc[id] && npc[id])
            npc[id] = { ...npc[id], needs: m.needs };
        return {
          ...st,
          auto: out.state.auto,
          picked: [...new Set([...st.picked, ...out.state.picked])],
          npc,
        };
      });
      for (const n of out.notes) toast(n);
      events.current = [...events.current, ...out.notes].slice(-6);
    };
    return () => {
      ticks.current[2] = null;
    };
  }, [autoOn, ready, started, map, areas, toast, shooterOn, tickWs]);

  const deal = (
    fn: (st: WorldState) => { state: WorldState; note: string } | null,
    refused: string,
  ) => {
    const out = fn(wsRef.current);
    const text = out?.note || refused;
    if (text) toast(text);
    if (out) setWs((st) => fn(st)?.state ?? st);
  };
  const stand = (choice: Stand) =>
    deal((st) => resolveEncounter(spec, st, choice), "");
  const [offer, setOffer] = useState<{ npc: string; good: Good } | null>(null);

  const replyRun = useRef<AbortController | null>(null);
  const send = (raw: string): boolean => {
    const text = raw.trim();
    if (!text || !talking || thinking) return false;
    void reply(text, talking);
    return true;
  };
  const reply = async (text: string, npcId: string) => {
    const turn = ++messages.current;
    summaryRun.current?.abort();
    summaryRun.current = null;
    directorRun.current?.abort();
    directorRun.current = null;
    const history = talkHistory(wsRef.current, npcId);
    const before = addTurn(
      wsRef.current,
      npcId,
      { role: "player", text },
      summarising,
    );
    setWs(before);
    if (!replyEntry) {
      setWs((s) =>
        addTurn(s, npcId, { role: "npc", text: "..." }, summarising),
      );
      toast(
        "Add an NPC Reply (live) node to the flow so villagers can answer.",
      );
      return;
    }
    setThinking(true);
    const generation = loads.current;
    const run = new AbortController();
    replyRun.current = run;
    try {
      const res = await api.invoke(
        flowId,
        replyEntry,
        {
          state: {
            world: spec,
            npc_id: npcId,
            history,
            summary: before.summaries[npcId],
            player_message: text,
            vars: before.vars,
            inventory: before.inventory,
            completed_quests: before.completedQuests,
            following: (before.following ?? []).includes(npcId),
            ...(autoOn
              ? {
                  role: roleOf(spec, npcId),
                  goods: flags.trade
                    ? Object.entries(before.auto?.goods[npcId] ?? {}).map(
                        ([good, n]) => ({
                          id: good,
                          name: GOODS[good as Good].name,
                          price: priceOf(good as Good, n ?? 0),
                        }),
                      )
                    : [],
                }
              : {}),
            ...(sim
              ? { memory: before.npc[npcId]?.memory ?? [], clock: before.clock }
              : {}),
          },
        },
        {
          signal: run.signal,
          onDelta: (piece) =>
            live.set(
              appendReplyDelta(
                live.get(),
                npcId,
                generation,
                loads.current,
                piece,
              ),
            ),
        },
      );
      if (generation !== loads.current) return;
      live.set(null);
      const reply = res.outputs.reply as NpcReply | undefined;
      if (!reply?.text) throw new Error("No reply");
      const prior = wsRef.current;
      const applied = applyReply(spec, prior, npcId, reply, summarising);
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
      for (const id of state.picked)
        if (!prior.picked.includes(id)) game.current?.removeItem(id);
      for (const n of notes) toast(n);
      const offered = reply.offer_trade as Good | undefined;
      if (
        flags.trade &&
        offered &&
        (state.auto?.goods[npcId]?.[offered] ?? 0) > 0
      ) {
        setOffer({ npc: npcId, good: offered });
        toast(
          `${npcById.get(npcId)?.name ?? npcId} offers ${GOODS[offered].name.toLowerCase()} · ${priceOf(offered, state.auto?.goods[npcId]?.[offered] ?? 0)} coins`,
        );
      }
      events.current = [...events.current, ...notes].slice(-6);
      const completed =
        state.completedQuests.length > prior.completedQuests.length;
      void (async () => {
        await summarise(npcId, state);
        if (turn === messages.current) await direct(completed, state);
      })();
    } catch (e) {
      if (generation !== loads.current) return;
      live.set(null);
      const message = e instanceof Error ? e.message : String(e);
      if (isNoChatModelError(message)) {
        const npc = npcById.get(npcId);
        const spoken = (wsRef.current.talks[npcId] ?? []).filter(
          (t) => t.role === "npc",
        ).length;
        setWs((s) =>
          addTurn(
            s,
            npcId,
            { role: "npc", text: offlineLine(npc?.greeting, spoken) },
            summarising,
          ),
        );
        if (!offlineNoted.current) {
          offlineNoted.current = true;
          toast(OFFLINE_NOTE);
        }
        return;
      }
      setWs((s) =>
        addTurn(
          s,
          npcId,
          { role: "npc", text: "(They seem lost in thought. Try again.)" },
          summarising,
        ),
      );
      toast(message);
    } finally {
      if (generation === loads.current) setThinking(false);
      if (replyRun.current === run) replyRun.current = null;
    }
  };

  const save = async (slot: string) => {
    const action = game.current?.actionState?.() as WorldState["action"];
    const state = serialize({
      ...wsRef.current,
      ...(action ? { action } : {}),
      player: game.current?.playerTile() ?? null,
      ...(autoOn && game.current ? { npcAt: game.current.npcTiles() } : {}),
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
    // Props edited in the scene editor this session are what the save's door and chest states refer to.
    const props = worldRef.current?.props;
    const state = rec
      ? deserialize(
          props ? { ...baseSpec, props3d: props } : baseSpec,
          rec.state,
        )
      : null;
    if (!state) {
      toast("That save is empty or from another world");
      return;
    }
    loads.current += 1;
    replyRun.current?.abort();
    replyRun.current = null;
    messages.current += 1;
    summaryRun.current?.abort();
    summaryRun.current = null;
    directorRun.current?.abort();
    directorRun.current = null;
    queue.clear();
    busy.current.clear();
    setBackground(null);
    setThinking(false);
    live.set(null);
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
  const stockOf = (id: string) =>
    GOOD_IDS.flatMap((g) => {
      const n = ws.auto?.goods[id]?.[g] ?? 0;
      return n > 0 ? [[g, n] as [Good, number]] : [];
    });
  const playerGoods = stockOf(PLAYER);
  const talkStock = talkNpc ? stockOf(talkNpc.id) : [];
  const talkHeld = talkNpc
    ? Object.entries(ws.auto?.held ?? {})
        .filter(([, by]) => by === talkNpc.id)
        .map(([id]) => id)
    : [];
  const talkRole = talkNpc && ws.auto ? roleOf(spec, talkNpc.id) : null;
  const encounterNpc = ws.auto?.encounter
    ? npcById.get(ws.auto.encounter)
    : undefined;
  const raceNote = (questId: string, item?: string | null) => {
    if (!ws.auto || !item) return "";
    const holder = ws.auto.held[item];
    if (holder)
      return `${npcById.get(holder)?.name} has the ${itemById.get(item)?.name}: buy it back from them`;
    const q = spec.quests.find((x) => x.id === questId);
    const rival = q && !ws.picked.includes(item) ? rivalFor(spec, q) : null;
    return rival && questId in ws.auto.race
      ? `Rival: ${npcById.get(rival)?.name} is after it too`
      : "";
  };
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
            {ws.auto ? (
              <div
                className="self-start rounded-full bg-black/35 px-3 py-1 text-amber-100 text-xs tabular-nums backdrop-blur-md"
                data-testid="world-coins"
              >
                🪙 {ws.auto.coins[PLAYER] ?? 0} coins
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
          {mayorOn(baseSpec) && started ? (
            <MayorHud
              spec={spec}
              map={map}
              city={ws.city}
              setWs={setWs}
              game={game}
              ready={ready}
              view={view}
              locked={lockedRef}
              queue={queue}
              api={api}
              flowId={flowId}
              replyEntry={replyEntry}
              tileRef={cityTile}
              toast={toast}
            />
          ) : null}
          {background && !talking ? (
            <div
              className="pointer-events-none absolute top-3 left-1/2 z-10 flex -translate-x-1/2 items-center gap-2 rounded-full bg-black/45 px-3 py-1 text-white/75 text-xs backdrop-blur-md"
              data-testid="world-sim-activity"
              title={queue.callsLabel()}
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
                : nearby.kind === "prop"
                  ? `Use ${nearby.name}`
                  : `Pick up ${nearby.name}`}
            </div>
          ) : null}
        </>
      ) : null}

      {panel === "inventory" ? (
        <Panel title="Inventory" onClose={() => setPanel(null)}>
          {ws.inventory.length === 0 && !playerGoods.length ? (
            <p className="text-sm text-white/50">Nothing yet. Explore!</p>
          ) : null}
          {playerGoods.length ? (
            <div
              className="mb-2 flex flex-wrap gap-1.5"
              data-testid="world-goods"
            >
              {playerGoods.map(([good, n]) => (
                <span
                  key={good}
                  className="rounded-full bg-white/10 px-2.5 py-1 text-xs"
                >
                  {GOODS[good].icon} {GOODS[good].name} × {n}
                </span>
              ))}
            </div>
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
                {raceNote(q.id, q.requires_item) ? (
                  <div
                    className="mt-1 text-amber-200/90 text-ui-11"
                    data-testid="world-quest-rival"
                  >
                    {raceNote(q.id, q.requires_item)}
                  </div>
                ) : null}
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
      {ready && started && !editor && spec.rivals && !mayorOn(spec) ? (
        <RivalsPanel
          api={api}
          flowId={flowId}
          entry={replyEntry}
          spec={baseSpec}
          map={map}
          state={ws.rivals}
          onChange={(r) => setWs((st) => ({ ...st, rivals: r }))}
          queue={queue}
          onTalking={setEnvoyTalking}
        />
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
            onDescribe={
              placeEntry || replyEntry
                ? async (req) => {
                    const res = await api.invoke(
                      flowId,
                      (placeEntry || replyEntry)!,
                      placeEntry
                        ? { request: req }
                        : { state: { task: "describe_place", ...req } },
                    );
                    const out = (
                      placeEntry
                        ? res.outputs.place
                        : (res.outputs.reply as { place?: unknown } | undefined)
                            ?.place
                    ) as { props3d?: unknown } | undefined;
                    return out?.props3d;
                  }
                : undefined
            }
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
              {talkRole ? (
                <div
                  className={cn(
                    "rounded-full px-2 py-0.5 text-ui-11",
                    talkRole === "hostile"
                      ? "bg-rose-400/20 text-rose-100"
                      : "bg-amber-300/20 text-amber-100",
                  )}
                  data-testid="world-npc-role"
                >
                  {talkRole === "hostile" ? "Hostile" : "Rival"}
                </div>
              ) : null}
              {ws.following?.includes(talkNpc.id) ? (
                <div className="text-[11px] text-emerald-200/80">
                  Following you
                </div>
              ) : null}
              {talkNeeds ? (
                <div
                  className="flex w-full flex-col gap-1"
                  data-testid="world-npc-needs"
                >
                  {NEEDS.map((n) => (
                    <div
                      key={n}
                      className="flex items-center gap-1.5 text-[10px] text-white/55"
                      title={`${n} ${Math.round(talkNeeds[n])}`}
                    >
                      <span className="w-9 capitalize">{n}</span>
                      <div className="h-1 flex-1 overflow-hidden rounded-full bg-white/10">
                        <div
                          className={cn(
                            "h-full rounded-full",
                            talkNeeds[n] < 30
                              ? "bg-rose-300"
                              : "bg-emerald-300",
                          )}
                          style={{ width: `${Math.round(talkNeeds[n])}%` }}
                        />
                      </div>
                    </div>
                  ))}
                  {talkDoing ? (
                    <div className="text-center text-[10px] text-white/45">
                      {talkDoing}
                    </div>
                  ) : null}
                </div>
              ) : null}
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
                <ReplyStatus
                  live={live}
                  talking={talkNpc.id}
                  thinking={thinking}
                  warmed={warmed}
                  scroller={scroller}
                />
              </div>
              {talkHeld.length || (flags.trade && ws.auto) ? (
                <div
                  className="flex flex-wrap items-center gap-1.5 text-xs"
                  data-testid="world-trade"
                >
                  {talkHeld.map((id) => (
                    <button
                      key={id}
                      type="button"
                      onClick={() =>
                        deal(
                          (st) => buyBack(spec, st, talkNpc.id, id),
                          "They no longer have it",
                        )
                      }
                      className="rounded-full bg-amber-300 px-3 py-1 font-medium text-black hover:bg-amber-200"
                      data-testid="world-buy-back"
                    >
                      Buy back {itemById.get(id)?.name ?? id} ·{" "}
                      {buyBackPrice(ws)} coins
                    </button>
                  ))}
                  {flags.trade
                    ? talkStock.map(([good, n]) => (
                        <button
                          key={good}
                          type="button"
                          onClick={() =>
                            deal(
                              (st) => playerBuy(spec, st, talkNpc.id, good),
                              "You cannot afford that",
                            )
                          }
                          className={cn(
                            "rounded-full bg-white/10 px-3 py-1 hover:bg-white/20",
                            offer?.npc === talkNpc.id &&
                              offer.good === good &&
                              "ring-1 ring-amber-200",
                          )}
                          data-testid={`world-buy-${good}`}
                        >
                          Buy {GOODS[good].icon} {GOODS[good].name} ·{" "}
                          {priceOf(good, n)}
                        </button>
                      ))
                    : null}
                  {flags.trade
                    ? playerGoods.map(([good]) => (
                        <button
                          key={good}
                          type="button"
                          onClick={() =>
                            deal(
                              (st) => playerSell(spec, st, talkNpc.id, good),
                              `${talkNpc.name} cannot pay for that`,
                            )
                          }
                          className="rounded-full bg-white/5 px-3 py-1 text-white/70 hover:bg-white/15"
                        >
                          Sell {GOODS[good].icon} {GOODS[good].name}
                        </button>
                      ))
                    : null}
                </div>
              ) : null}
              <SayForm
                npcName={talkNpc.name}
                thinking={thinking}
                draft={draft}
                onSend={send}
                onBye={() => {
                  setTalking(null);
                  game.current?.faceNpc(null);
                }}
              />
            </div>
          </div>
        </div>
      ) : null}

      {encounterNpc ? (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/40 p-6">
          <div
            className="w-80 rounded-2xl bg-zinc-950/90 p-5 text-center shadow-2xl ring-1 ring-rose-300/30"
            style={{ animation: "flows-rise 200ms ease-out" }}
            data-testid="world-encounter"
          >
            <p className="mb-1 font-semibold">
              {encounterNpc.name} blocks your way!
            </p>
            <p className="mb-4 text-white/60 text-xs">
              Slip away, stand your ground, or pay them off.
            </p>
            <div className="flex justify-center gap-2">
              {(
                [
                  ["evade", "Evade"],
                  ["defend", "Defend"],
                  ["negotiate", "Negotiate"],
                ] as [Stand, string][]
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  onClick={() => stand(id)}
                  className="rounded-full bg-white/10 px-3 py-1.5 text-sm hover:bg-white/20"
                  data-testid={`world-encounter-${id}`}
                >
                  {label}
                </button>
              ))}
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
