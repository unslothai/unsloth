// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ChevronDownIcon, ChevronUpIcon, SwordsIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cn } from "@/lib/utils";

import type { WorldSpec } from "../../types";
import type { PlayerApi } from "../player-api";
import type { WorldMap } from "./procgen";
import {
  BUILDINGS,
  BUILDING_IDS,
  type Building,
  type DiplomacyOffer,
  POSTURE_KEYS,
  RESOURCES,
  type RivalAction,
  type RivalsState,
  acceptOffer,
  act,
  blocked,
  boardOf,
  declineOffer,
  describe,
  diplomacyRequest,
  initialRivals,
  offerAction,
  playTurn,
  plotKey,
  plotsOf,
  relationOf,
  scoreOf,
} from "./rivals";

const PALETTE = ["#f59e0b", "#38bdf8", "#f472b6", "#a3e635"];
const POSTURE_COLOR = [
  "bg-emerald-400",
  "bg-sky-400",
  "bg-rose-400",
  "bg-violet-400",
];
const RES_ICON = { food: "🍞", wood: "🪵", stone: "🪨", gold: "🪙" } as const;

interface Talk {
  role: "player" | "npc";
  text: string;
}

export interface RivalsPanelProps {
  api: PlayerApi;
  flowId: string;
  entry?: string;
  spec: WorldSpec;
  map: WorldMap;
  state?: RivalsState;
  onChange: (next: RivalsState) => void;
}

export function RivalsPanel({
  api,
  flowId,
  entry,
  spec,
  map,
  state,
  onChange,
}: RivalsPanelProps) {
  const board = useMemo(() => boardOf(spec, map), [spec, map]);
  const s = useMemo(
    () => state ?? initialRivals(spec, map, board),
    [state, spec, map, board],
  );
  useEffect(() => {
    if (!state) onChange(s);
  }, [state, s, onChange]);
  const [open, setOpen] = useState(true);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const [build, setBuild] = useState<Building>("farm");
  const ai = board.factions.filter((f) => !f.player);
  const [raidTarget, setRaidTarget] = useState(ai[0]?.id ?? "");
  const [envoy, setEnvoy] = useState(ai[0]?.id ?? "");
  const [msg, setMsg] = useState("");
  const [talks, setTalks] = useState<Record<string, Talk[]>>({});
  const [deal, setDeal] = useState<{ fid: string; action: RivalAction } | null>(
    null,
  );
  const run = useRef<AbortController | null>(null);
  useEffect(() => () => run.current?.abort(), []);
  const pid = board.player;
  const me = pid ? s.factions[pid] : null;
  const color = (id: string) => {
    const i = board.factions.findIndex((f) => f.id === id);
    return board.factions[i]?.color || PALETTE[i % PALETTE.length];
  };
  const name = (id: string) =>
    board.factions.find((f) => f.id === id)?.name ?? id;

  const doAct = (a: RivalAction) => {
    if (!pid) return;
    const r = act(board, s, pid, a);
    setNote(r.ok ? "" : (r.why ?? ""));
    if (r.ok) onChange(r.state);
  };

  const endTurn = async () => {
    if (busy || s.over) return;
    setBusy(true);
    setNote("");
    const ctl = new AbortController();
    run.current = ctl;
    const ask = entry
      ? async (_fid: string, request: unknown) => {
          const res = await api.invoke(
            flowId,
            entry,
            { state: request },
            { signal: ctl.signal },
          );
          return (res.outputs.reply as { plan?: unknown } | undefined)?.plan;
        }
      : null;
    try {
      const out = await playTurn(spec, board, s, ask);
      if (!ctl.signal.aborted) onChange(out.state);
    } finally {
      if (run.current === ctl) run.current = null;
      setBusy(false);
    }
  };

  const talk = async () => {
    const text = msg.trim();
    if (!text || !entry || !envoy || busy) return;
    setMsg("");
    setDeal(null);
    const history = talks[envoy] ?? [];
    setTalks((t) => ({
      ...t,
      [envoy]: [...history, { role: "player", text }],
    }));
    setBusy(true);
    try {
      const res = await api.invoke(flowId, entry, {
        state: diplomacyRequest(spec, board, s, envoy, text, history),
      });
      const reply = res.outputs.reply as
        | { text?: string; offer?: DiplomacyOffer }
        | undefined;
      const action = offerAction(board, s, envoy, reply?.offer);
      setTalks((t) => ({
        ...t,
        [envoy]: [
          ...(t[envoy] ?? []),
          { role: "npc", text: reply?.text || "..." },
        ],
      }));
      if (action) setDeal({ fid: envoy, action });
    } catch {
      setNote("The envoy did not come back.");
    } finally {
      setBusy(false);
    }
  };

  const takeDeal = () => {
    if (!deal || !pid) return;
    const r = act(board, s, pid, deal.action);
    setNote(r.ok ? "" : (r.why ?? ""));
    if (r.ok) onChange(r.state);
    setDeal(null);
  };

  const button = (a: RivalAction, label = describe(board, a)) => {
    const why = pid ? blocked(board, s, pid, a) : "you have no village";
    return (
      <button
        type="button"
        disabled={!!why || busy}
        title={why ?? label}
        onClick={() => doAct(a)}
        className="rounded-full bg-white/10 px-2.5 py-1 text-xs hover:bg-white/20 disabled:opacity-40"
      >
        {label}
      </button>
    );
  };

  return (
    <div
      className="absolute top-24 left-3 z-20 flex max-h-[calc(100%-8rem)] w-80 flex-col overflow-hidden rounded-2xl bg-zinc-950/85 text-white shadow-2xl ring-1 ring-white/10 backdrop-blur-md"
      data-testid="rivals-panel"
      onKeyDown={(e) => e.stopPropagation()}
    >
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 px-4 py-2.5 text-left"
      >
        <SwordsIcon className="size-4 text-amber-200" />
        <span className="flex-1 font-semibold text-sm">Village Rivals</span>
        <span
          className="text-white/60 text-xs tabular-nums"
          data-testid="rivals-day"
        >
          Day {s.turn} / {spec.rivals?.max_turns ?? 30}
        </span>
        {open ? (
          <ChevronUpIcon className="size-4" />
        ) : (
          <ChevronDownIcon className="size-4" />
        )}
      </button>
      {open ? (
        <div
          className="flex flex-col gap-3 overflow-y-auto px-4 pb-4"
          data-testid="rivals-body"
        >
          {s.over ? (
            <div className="rounded-xl bg-amber-300/15 p-2 text-amber-100 text-xs">
              {s.over.winner ? `${name(s.over.winner)} wins` : "Nobody wins"}:{" "}
              {s.over.reason}.
            </div>
          ) : null}
          <div className="flex flex-col gap-1.5" data-testid="rivals-factions">
            {board.factions.map((f) => {
              const st = s.factions[f.id];
              const rel = pid && f.id !== pid ? relationOf(s, pid, f.id) : null;
              return (
                <div
                  key={f.id}
                  className={cn(
                    "rounded-xl bg-white/5 p-2 text-xs",
                    st.out && "opacity-40",
                  )}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="size-2.5 rounded-full"
                      style={{ background: color(f.id) }}
                    />
                    <span className="flex-1 truncate font-medium">
                      {f.name}
                      {f.player ? " (you)" : ""}
                    </span>
                    {rel ? (
                      <span
                        className={cn(
                          "rounded-full px-1.5 text-[10px]",
                          rel === "war" && "bg-rose-500/30 text-rose-100",
                          rel === "ally" &&
                            "bg-emerald-500/30 text-emerald-100",
                          rel === "peace" && "bg-white/10 text-white/70",
                        )}
                      >
                        {rel}
                      </span>
                    ) : null}
                    <span className="tabular-nums text-white/70">
                      {scoreOf(s, f.id)} pts
                    </span>
                  </div>
                  <div className="mt-1 text-white/60 tabular-nums">
                    {plotsOf(s, f.id).length} land · {st.workers} workers ·{" "}
                    {st.soldiers} soldiers
                  </div>
                  {!f.player ? (
                    <>
                      <div className="mt-1.5 flex h-1.5 overflow-hidden rounded-full">
                        {POSTURE_KEYS.map((k, i) => (
                          <div
                            key={k}
                            title={`${k} ${Math.round(st.posture[k] * 100)}%`}
                            className={POSTURE_COLOR[i]}
                            style={{ width: `${st.posture[k] * 100}%` }}
                          />
                        ))}
                      </div>
                      {st.goals.length ? (
                        <div className="mt-1 text-white/50">
                          {st.goals.join(" · ")}
                          {st.target ? ` · target ${name(st.target)}` : ""}
                        </div>
                      ) : null}
                    </>
                  ) : null}
                </div>
              );
            })}
          </div>
          <div
            className="grid w-40 gap-px self-center rounded-md bg-black/40 p-1"
            style={{ gridTemplateColumns: `repeat(${board.cols}, 1fr)` }}
            data-testid="rivals-map"
          >
            {Array.from({ length: board.cols * board.rows }, (_, i) => {
              const k = plotKey(i % board.cols, Math.floor(i / board.cols));
              const owner = s.plots[k];
              return (
                <div
                  key={k}
                  className="aspect-square rounded-[2px]"
                  style={{
                    background: owner
                      ? color(owner)
                      : board.land[i]
                        ? "rgba(255,255,255,0.12)"
                        : "rgba(56,189,248,0.4)",
                  }}
                />
              );
            })}
          </div>
          {me && pid ? (
            <div className="flex flex-col gap-1.5 text-xs">
              <div
                className="flex flex-wrap gap-1.5"
                data-testid="rivals-stock"
              >
                {RESOURCES.map((r) => (
                  <span
                    key={r}
                    className="rounded-full bg-white/10 px-2 py-0.5"
                  >
                    {RES_ICON[r]} {me.stock[r]}
                  </span>
                ))}
                <span className="rounded-full bg-white/10 px-2 py-0.5">
                  {me.actionsLeft} actions left
                </span>
              </div>
              {me.queue.length ? (
                <div className="text-white/55">
                  Building:{" "}
                  {me.queue.map((q) => BUILDINGS[q.kind].name).join(", ")}
                </div>
              ) : null}
              <div className="flex flex-wrap items-center gap-1.5">
                <select
                  value={build}
                  onChange={(e) => setBuild(e.target.value as Building)}
                  className="rounded-full bg-white/10 px-2 py-1 text-xs"
                  aria-label="Building"
                >
                  {BUILDING_IDS.map((k) => (
                    <option key={k} value={k} className="text-black">
                      {BUILDINGS[k].name} (
                      {Object.entries(BUILDINGS[k].cost)
                        .map(([r, n]) => `${n} ${r}`)
                        .join(", ")}
                      )
                    </option>
                  ))}
                </select>
                {button({ kind: "build", building: build }, "Build")}
                {button({ kind: "claim" })}
                {button({ kind: "train" })}
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <select
                  value={raidTarget}
                  onChange={(e) => setRaidTarget(e.target.value)}
                  className="rounded-full bg-white/10 px-2 py-1 text-xs"
                  aria-label="Raid target"
                >
                  {ai.map((f) => (
                    <option key={f.id} value={f.id} className="text-black">
                      {f.name}
                    </option>
                  ))}
                </select>
                {raidTarget
                  ? button({ kind: "raid", target: raidTarget }, "Raid")
                  : null}
              </div>
            </div>
          ) : null}
          {s.offers.length ? (
            <div className="flex flex-col gap-1.5" data-testid="rivals-offers">
              {s.offers.map((o) => (
                <div
                  key={o.id}
                  className="flex items-center gap-2 rounded-xl bg-sky-400/10 p-2 text-xs"
                >
                  <span className="flex-1">{o.text}</span>
                  <button
                    type="button"
                    className="rounded-full bg-white/15 px-2 py-0.5"
                    onClick={() => {
                      const r = acceptOffer(board, s, o.id);
                      setNote(r.ok ? "" : (r.why ?? ""));
                      onChange(r.state);
                    }}
                  >
                    Accept
                  </button>
                  <button
                    type="button"
                    className="rounded-full px-2 py-0.5 text-white/60"
                    onClick={() => onChange(declineOffer(s, o.id))}
                  >
                    No
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          {entry && ai.length && pid ? (
            <div
              className="flex flex-col gap-1.5 text-xs"
              data-testid="rivals-diplomacy"
            >
              <div className="flex items-center gap-1.5">
                <span className="text-white/60">Talk to</span>
                <select
                  value={envoy}
                  onChange={(e) => {
                    setEnvoy(e.target.value);
                    setDeal(null);
                  }}
                  className="rounded-full bg-white/10 px-2 py-1 text-xs"
                  aria-label="Village to talk to"
                >
                  {ai.map((f) => (
                    <option key={f.id} value={f.id} className="text-black">
                      {f.name}
                    </option>
                  ))}
                </select>
              </div>
              {(talks[envoy] ?? []).slice(-4).map((t, i) => (
                <div
                  // biome-ignore lint/suspicious/noArrayIndexKey: a short append-only transcript
                  key={i}
                  className={cn(
                    "rounded-lg px-2 py-1",
                    t.role === "player"
                      ? "self-end bg-white/10"
                      : "bg-amber-200/10",
                  )}
                >
                  {t.text}
                </div>
              ))}
              {deal && deal.fid === envoy ? (
                <button
                  type="button"
                  onClick={takeDeal}
                  className="self-start rounded-full bg-emerald-400/20 px-2.5 py-1 text-emerald-100 hover:bg-emerald-400/30"
                  data-testid="rivals-deal"
                >
                  Accept: {describe(board, deal.action)}
                </button>
              ) : null}
              <form
                className="flex gap-1.5"
                onSubmit={(e) => {
                  e.preventDefault();
                  void talk();
                }}
              >
                <input
                  value={msg}
                  onChange={(e) => setMsg(e.target.value)}
                  placeholder={`Message ${name(envoy)}...`}
                  className="min-w-0 flex-1 rounded-full bg-white/10 px-3 py-1 outline-none"
                />
                <button
                  type="submit"
                  disabled={busy || !msg.trim()}
                  className="rounded-full bg-white/15 px-2.5 py-1 disabled:opacity-40"
                >
                  Send
                </button>
              </form>
            </div>
          ) : null}
          {note ? <div className="text-amber-200 text-xs">{note}</div> : null}
          <div
            className="flex flex-col gap-0.5 text-[11px] text-white/60"
            data-testid="rivals-log"
          >
            {s.log
              .slice(-6)
              .reverse()
              .map((l, i) => (
                // biome-ignore lint/suspicious/noArrayIndexKey: log lines repeat
                <div key={`${l.turn}-${i}`}>
                  <span className="text-white/35">Day {l.turn}</span> {l.text}
                </div>
              ))}
          </div>
          <button
            type="button"
            onClick={() => void endTurn()}
            disabled={busy || !!s.over}
            data-testid="rivals-end-turn"
            className="rounded-full bg-amber-300 px-4 py-1.5 font-semibold text-black text-sm hover:bg-amber-200 disabled:opacity-50"
          >
            {busy ? "The villages are thinking..." : "End day"}
          </button>
        </div>
      ) : null}
    </div>
  );
}
