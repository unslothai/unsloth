// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type Dispatch,
  type MutableRefObject,
  type SetStateAction,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { cn } from "@/lib/utils";

import type { WorldSpec } from "../../types";
import type { PlayerApi } from "../player-api";
import {
  applyMayorReply,
  cityDigest,
  localAdvice,
  mayorRequest,
  parseMayorReply,
} from "./city-advisor";
import { drawCity } from "./city-draw";
import {
  CI_VALVE_MAX,
  COSTS,
  CYCLES_PER_MONTH,
  CYCLE_MS,
  CitySim,
  type CityState,
  type CityTool,
  OVERLAYS,
  type Overlay,
  R_VALVE_MAX,
  cityGround,
  cityOf,
  newCity,
} from "./city-sim";
import type { WorldGameHandle } from "./phaser-game";
import type { Point, WorldMap } from "./procgen";
import type { BackgroundQueue } from "./world-sim";
import type { WorldState } from "./world-state";

export interface MayorHudProps {
  spec: WorldSpec;
  map: WorldMap;
  city: CityState | undefined;
  setWs: Dispatch<SetStateAction<WorldState>>;
  game: MutableRefObject<WorldGameHandle | null>;
  ready: boolean;
  view: "2d" | "3d";
  locked: MutableRefObject<boolean>;
  queue: BackgroundQueue;
  api: PlayerApi;
  flowId: string;
  replyEntry?: string;
  tileRef: MutableRefObject<(p: Point, drag: boolean) => boolean>;
  toast: (text: string) => void;
}

const TOOLS: { id: CityTool; label: string; tint: string }[] = [
  { id: "R", label: "Residential", tint: "bg-emerald-500" },
  { id: "C", label: "Commercial", tint: "bg-sky-500" },
  { id: "I", label: "Industrial", tint: "bg-amber-400" },
  { id: "road", label: "Road", tint: "bg-zinc-500" },
  { id: "wire", label: "Power line", tint: "bg-yellow-300" },
  { id: "P", label: "Power plant", tint: "bg-zinc-300" },
  { id: "bulldoze", label: "Bulldoze", tint: "bg-rose-400" },
];
const DRAG_TOOLS = new Set<CityTool>(["road", "wire", "bulldoze"]);
const OVERLAY_LABEL: Record<Overlay, string> = {
  power: "Power",
  traffic: "Traffic",
  pollution: "Pollution",
  land_value: "Land value",
  crime: "Crime",
};

function Bar({
  label,
  value,
  max,
}: { label: string; value: number; max: number }) {
  const f = Math.max(-1, Math.min(1, value / max));
  return (
    <div
      className="flex flex-col items-center gap-1"
      data-testid={`mayor-demand-${label}`}
    >
      <div className="relative h-14 w-4 rounded bg-white/10">
        <div className="absolute inset-x-0 top-1/2 h-px bg-white/30" />
        <div
          className={cn(
            "absolute inset-x-0 rounded",
            f >= 0 ? "bottom-1/2 bg-emerald-400" : "top-1/2 bg-rose-400",
          )}
          style={{ height: `${Math.abs(f) * 50}%` }}
        />
      </div>
      <span className="font-semibold text-[10px]">{label}</span>
    </div>
  );
}

function paintOn(
  game: MutableRefObject<WorldGameHandle | null>,
  sim: CitySim,
  overlay: Overlay | null,
): void {
  game.current?.paintCity?.((g, T) => drawCity(g, sim, T, overlay));
}

export function MayorHud({
  spec,
  map,
  city,
  setWs,
  game,
  ready,
  view,
  locked,
  queue,
  api,
  flowId,
  replyEntry,
  tileRef,
  toast,
}: MayorHudProps) {
  const ground = useMemo(() => cityGround(map), [map]);
  const [sim, setSim] = useState(
    () => new CitySim(city ?? newCity(spec), ground),
  );
  const [written, setWritten] = useState<CityState | undefined>(city);
  if (city !== written) {
    // A loaded save or a reset (a save with no city starts a fresh one): rebuild the simulation from it.
    setWritten(city);
    setSim(new CitySim(city ?? newCity(spec), ground));
  }
  const live = useRef(sim);
  useEffect(() => {
    live.current = sim;
  }, [sim]);
  const [tool, setTool] = useState<CityTool | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [open, setOpen] = useState(true);
  const [asking, setAsking] = useState(false);
  const conf = cityOf(spec);

  const paint = useCallback(
    () => paintOn(game, sim, overlay),
    [game, overlay, sim],
  );

  const commit = useCallback(() => {
    const snap = sim.snapshot();
    setWritten(snap);
    setWs((st) => ({ ...st, city: snap }));
    paint();
  }, [paint, setWs, sim]);

  useEffect(() => {
    if (ready) paint();
  }, [ready, view, paint, city]);

  const ask = useCallback(
    async (kind: "month" | "advice", signal: AbortSignal) => {
      const digest = cityDigest(spec, sim, kind);
      if (!replyEntry) {
        if (kind === "month") sim.claimMonth(digest.month);
        sim.setAdvisor(localAdvice(digest), "local");
        commit();
        return;
      }
      setAsking(true);
      try {
        const res = await api.invoke(
          flowId,
          replyEntry,
          { state: mayorRequest(digest) },
          { signal },
        );
        if (signal.aborted || live.current !== sim) return;
        const reply = parseMayorReply(res.outputs.reply);
        if (reply)
          for (const line of applyMayorReply(sim, digest, reply)) toast(line);
        else {
          if (kind === "month") sim.claimMonth(digest.month);
          sim.setAdvisor(localAdvice(digest), "local");
        }
      } catch {
        if (signal.aborted || live.current !== sim) return;
        if (kind === "month") sim.claimMonth(digest.month);
        sim.setAdvisor(localAdvice(digest), "local");
      } finally {
        setAsking(false);
      }
      commit();
    },
    [api, commit, flowId, replyEntry, sim, spec, toast],
  );

  const askLater = useCallback(
    (kind: "month" | "advice") => {
      if (kind === "month" && conf?.advisor === false) return;
      queue.push({
        key: `city:${kind}`,
        kind: "chat",
        run: (signal) => ask(kind, signal),
      });
      queue.pump();
    },
    [ask, conf?.advisor, queue],
  );

  const cycle = useCallback(
    (toMonthEnd: boolean) => {
      let ended = false;
      for (let k = 0; k < CYCLES_PER_MONTH && !ended; k++) {
        ended = sim.step();
        if (!toMonthEnd) break;
      }
      const npcs = spec.npcs;
      for (const id of sim.moveIn(
        npcs.filter((n) => n.role !== "hostile").map((n) => n.id),
      )) {
        const z = sim.zoneById(sim.residents[id]);
        toast(
          `${npcs.find((n) => n.id === id)?.name ?? id} moved into ${z?.name ?? z?.id}`,
        );
      }
      commit();
      if (ended) askLater("month");
      queue.pump();
    },
    [askLater, commit, queue, sim, spec.npcs, toast],
  );

  useEffect(() => {
    if (ready && sim.month > sim.petitioned) askLater("month");
  }, [ready, sim, askLater]);

  useEffect(() => {
    if (!ready) return;
    const timer = window.setInterval(() => {
      if (!locked.current) cycle(false);
    }, CYCLE_MS);
    return () => window.clearInterval(timer);
  }, [ready, locked, cycle]);

  useEffect(() => {
    tileRef.current = (p, drag) => {
      if (!tool || view !== "2d") return false;
      if (drag && !DRAG_TOOLS.has(tool)) return true;
      const r = sim.place(tool, p.x, p.y);
      if (r.ok) commit();
      else if (!drag && r.reason) toast(r.reason);
      return true;
    };
    return () => {
      tileRef.current = () => false;
    };
  }, [commit, sim, tileRef, toast, tool, view]);

  const c = city ?? written ?? sim.snapshot();
  const year = 1900 + Math.floor(c.month / 12);
  const monthName = new Date(2000, c.month % 12, 1).toLocaleString("en", {
    month: "short",
  });
  return (
    <div
      className="absolute top-14 left-3 z-20 w-72 rounded-2xl bg-zinc-950/85 p-3 text-white text-xs shadow-2xl ring-1 ring-white/10 backdrop-blur-md"
      data-testid="mayor-panel"
      data-pop={c.census.total}
      data-res={c.census.res}
      data-zones={c.zones.length}
      data-month={c.month}
      data-funds={c.funds}
    >
      <div className="flex items-center justify-between">
        <button
          type="button"
          className="font-semibold text-sm"
          onClick={() => setOpen((o) => !o)}
        >
          Mayor · {monthName} {year}
        </button>
        <span className="tabular-nums" data-testid="mayor-funds">
          ${c.funds}
        </span>
      </div>
      {view === "2d" ? (
        <div className="mt-2 grid grid-cols-4 gap-1" data-testid="mayor-tools">
          {TOOLS.map((t) => (
            <button
              key={t.id}
              type="button"
              data-testid={`mayor-tool-${t.id}`}
              title={`${t.label} ($${COSTS[t.id]})`}
              onClick={() => setTool((cur) => (cur === t.id ? null : t.id))}
              className={cn(
                "flex items-center gap-1 rounded-lg px-1.5 py-1 text-left ring-1 ring-white/10 hover:bg-white/10",
                tool === t.id && "bg-white/20 ring-white/50",
              )}
            >
              <span className={cn("size-2 shrink-0 rounded-sm", t.tint)} />
              <span className="truncate">{t.label}</span>
            </button>
          ))}
          <button
            type="button"
            data-testid="mayor-tool-walk"
            onClick={() => setTool(null)}
            className={cn(
              "rounded-lg px-1.5 py-1 ring-1 ring-white/10 hover:bg-white/10",
              tool === null && "bg-white/20 ring-white/50",
            )}
          >
            Walk
          </button>
        </div>
      ) : (
        <p className="mt-2 text-white/60">Switch to the map to build.</p>
      )}
      {open ? (
        <>
          <div className="mt-3 flex items-end gap-3">
            <Bar label="R" value={c.valves.r} max={R_VALVE_MAX} />
            <Bar label="C" value={c.valves.c} max={CI_VALVE_MAX} />
            <Bar label="I" value={c.valves.i} max={CI_VALVE_MAX} />
            <div
              className="flex-1 space-y-0.5 tabular-nums"
              data-testid="mayor-budget"
            >
              <div>Population {c.census.total}</div>
              <div>
                Tax income ${c.cash.tax} · roads ${c.cash.roads}
              </div>
              <div>Cash flow ${c.cash.flow}/month</div>
              <div className="flex items-center gap-1">
                Tax {c.tax}%
                <button
                  type="button"
                  aria-label="Lower tax"
                  className="rounded bg-white/10 px-1.5"
                  onClick={() => {
                    sim.setTax(c.tax - 1);
                    commit();
                  }}
                >
                  -
                </button>
                <button
                  type="button"
                  aria-label="Raise tax"
                  className="rounded bg-white/10 px-1.5"
                  onClick={() => {
                    sim.setTax(c.tax + 1);
                    commit();
                  }}
                >
                  +
                </button>
              </div>
            </div>
          </div>
          <div
            className="mt-2 flex flex-wrap gap-1"
            data-testid="mayor-overlays"
          >
            {OVERLAYS.map((o) => (
              <button
                key={o}
                type="button"
                data-testid={`mayor-overlay-${o}`}
                onClick={() => setOverlay((cur) => (cur === o ? null : o))}
                className={cn(
                  "rounded-full px-2 py-0.5 ring-1 ring-white/15",
                  overlay === o && "bg-white/25",
                )}
              >
                {OVERLAY_LABEL[o]}
              </button>
            ))}
          </div>
          <div
            className="mt-2 rounded-xl bg-white/5 p-2"
            data-testid="mayor-advisor"
          >
            <div className="mb-1 flex items-center justify-between font-semibold">
              Advisor
              <button
                type="button"
                data-testid="mayor-ask"
                disabled={asking}
                onClick={() => askLater("advice")}
                className="rounded-full bg-white/10 px-2 py-0.5 font-normal disabled:opacity-50"
              >
                {asking ? "Thinking" : "Ask"}
              </button>
            </div>
            <p className="text-white/80" data-testid="mayor-advice">
              {c.advisor?.text ?? "No report yet. One comes each month."}
            </p>
          </div>
          {c.petitions.length ? (
            <div className="mt-2 space-y-1" data-testid="mayor-petitions">
              {c.petitions
                .slice(-3)
                .reverse()
                .map((p) => (
                  <div
                    key={`${p.month}-${p.citizen}`}
                    className="rounded-lg bg-white/5 px-2 py-1"
                  >
                    <span className="font-semibold">
                      {spec.npcs.find((n) => n.id === p.citizen)?.name ??
                        p.citizen}
                    </span>{" "}
                    ({p.zone}, {p.metric.replace("_", " ")} {p.value}): {p.text}
                  </div>
                ))}
            </div>
          ) : null}
          {c.log.length ? (
            <div className="mt-2 text-white/60">{c.log[c.log.length - 1]}</div>
          ) : null}
          <button
            type="button"
            data-testid="mayor-next-month"
            onClick={() => cycle(true)}
            className="mt-2 w-full rounded-lg bg-emerald-500/80 py-1 font-semibold text-black hover:bg-emerald-400"
          >
            Run to next month
          </button>
        </>
      ) : null}
    </div>
  );
}
