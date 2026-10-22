// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowLeftIcon,
  BackpackIcon,
  MapPinIcon,
  RotateCcwIcon,
  SaveIcon,
  ScrollTextIcon,
  SendIcon,
  SparklesIcon,
  XIcon,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { cn } from "@/lib/utils";

import type { MediaValue, WorldSpec } from "../../types";
import { type PlayerApi, mediaObjectUrl } from "../player-api";
import { PlayerKeyframes } from "../story/backdrop";
import { useMediaUrl } from "../use-media-url";
import type { Interactable, WorldArt, WorldGameHandle } from "./phaser-game";
import { type TileKind, generateMap } from "./procgen";
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
} from "./world-state";

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
): Promise<WorldArt> {
  const jobs: [string, MediaValue][] = [];
  for (const [kind, m] of Object.entries(spec.tileset ?? {}))
    jobs.push([`tile:${kind}`, m]);
  if (spec.player.sprite) jobs.push(["sprite:player", spec.player.sprite]);
  for (const n of spec.npcs)
    if (n.sprite) jobs.push([`sprite:${n.id}`, n.sprite]);
  for (const it of spec.items)
    if (it.sprite) jobs.push([`sprite:${it.id}`, it.sprite]);
  const art: WorldArt = { tiles: {}, sprites: {} };
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
  const lockedRef = useRef(true);
  lockedRef.current = !started || talking !== null || panel === "save";

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

  const onInteract = useCallback(
    (t: Interactable) => {
      if (t.kind === "item") {
        setWs((s) => pickUp(s, t.id));
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
    [npcById, spec, toast],
  );

  // Load art, then boot Phaser (its own chunk).
  useEffect(() => {
    let live = true;
    let handle: WorldGameHandle | null = null;
    setReady(false);
    (async () => {
      try {
        const [art, mod] = await Promise.all([
          loadArt(api, spec, (f) => live && setLoadProgress(f)),
          import("./phaser-game"),
        ]);
        if (!live || !host.current) return;
        handle = mod.createWorldGame(
          host.current,
          spec,
          map,
          art,
          wsRef.current.picked,
          wsRef.current.player,
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
      handle?.destroy();
      game.current = null;
    };
    // `mount` re-creates the game after loading a save.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api, spec, map, mount]);

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
      } else if (e.key === "i" || e.key === "I")
        setPanel((p) => (p === "inventory" ? null : "inventory"));
      else if (e.key === "q" || e.key === "Q")
        setPanel((p) => (p === "quests" ? null : "quests"));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const [draft, setDraft] = useState("");
  const [thinking, setThinking] = useState(false);
  const replyEntry = spec.entry_points?.npc_reply;
  const talkNpc = talking ? npcById.get(talking) : undefined;
  const turns = talking ? (ws.talks[talking] ?? []) : [];
  const scroller = useRef<HTMLDivElement>(null);
  useEffect(() => {
    scroller.current?.scrollTo({
      top: scroller.current.scrollHeight,
      behavior: "smooth",
    });
  }, [turns.length, thinking]);

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
    try {
      const res = await api.invoke(flowId, replyEntry, {
        state: {
          world: spec,
          npc_id: npcId,
          history: (before.talks[npcId] ?? []).slice(0, -1),
          player_message: text,
          vars: before.vars,
          inventory: before.inventory,
          completed_quests: before.completedQuests,
        },
      });
      const reply = res.outputs.reply as NpcReply | undefined;
      if (!reply?.text) throw new Error("No reply");
      const { state, notes } = applyReply(spec, wsRef.current, npcId, reply);
      setWs(state);
      for (const n of notes) toast(n);
    } catch (e) {
      setWs((s) =>
        addTurn(s, npcId, {
          role: "npc",
          text: "(They seem lost in thought. Try again.)",
        }),
      );
      toast(e instanceof Error ? e.message : String(e));
    } finally {
      setThinking(false);
    }
  };

  const save = async (slot: string) => {
    const state = serialize({
      ...wsRef.current,
      player: game.current?.playerTile() ?? null,
    });
    await api.saveSession(
      flowId,
      slot,
      state,
      `${spec.title} · ${new Date().toLocaleString()}`,
    );
    toast("Saved");
    setPanel(null);
  };
  const load = async (slot: string) => {
    const rec = await api.loadSession(flowId, slot);
    const state = rec ? deserialize(spec, rec.state) : null;
    if (!state) {
      toast("That save is empty or from another world");
      return;
    }
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
                {thinking ? (
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
              WASD / arrows or click to walk · E or Space to talk and pick up ·
              I inventory · Q quests
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
