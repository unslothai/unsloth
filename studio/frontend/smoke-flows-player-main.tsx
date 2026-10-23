// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Harness for tests/studio/playwright_flows_player.py: the real Flows players (story and world) against a
// mock backend. `?flow=story` plays the compiled Lantern Picnic example; `?flow=world` walks a fixture
// village whose NPCs answer from a scripted reply. No Studio, no models.

/* eslint-disable no-restricted-imports -- a harness entry point, not app code. */
import { FlowPlayerPage } from "@/features/flows/player/flow-player-page";
import type {
  PlayerApi,
  SessionRecord,
} from "@/features/flows/player/player-api";
import type { WorldGameHandle } from "@/features/flows/player/world/phaser-game";
import { generateMap } from "@/features/flows/player/world/procgen";
import type {
  MediaValue,
  StoryManifest,
  WorldSpec,
} from "@/features/flows/types";
/* eslint-enable no-restricted-imports */
import { createRoot } from "react-dom/client";

import "@/index.css";
import picnicJson from "./tests/fixtures/flows-lantern-picnic.story.json?raw";

const picnic = JSON.parse(picnicJson) as StoryManifest;

const params = new URLSearchParams(location.search);
const kind = params.get("flow") === "world" ? "world" : "story";

const WORLD: WorldSpec = {
  title: "Mossy Hollow",
  seed: 2024,
  width: 40,
  height: 30,
  tile_size: 32,
  biome: "meadow",
  tileset: {},
  player: { name: "Pip" },
  npcs: [
    {
      id: "fern",
      name: "Grandma Fern",
      persona: "village elder",
      greeting: "Ah, a new face! Have you eaten?",
      wander: false,
    },
    {
      id: "tom",
      name: "Old Tom",
      persona: "grumpy fisher",
      greeting: "Hmph. Lost my lucky rod.",
      sprite: { kind: "image", ref: "flowasset:tom", mime: "image/png" },
    },
    { id: "bea", name: "Bea", persona: "baker", greeting: "Fresh bread!" },
  ],
  items: [
    { id: "rod", name: "Lucky Rod", description: "Smells of the pond." },
    { id: "bell", name: "Golden Bell", description: "It rings like laughter." },
    { id: "pie", name: "Berry Pie", description: "Still warm." },
  ],
  quests: [
    {
      id: "rod_q",
      title: "The Lucky Rod",
      description: "Find Old Tom's rod.",
      giver: "tom",
      requires_item: "rod",
      reward_text: "Tom smiles, almost.",
    },
  ],
  regions: [
    {
      id: "pond",
      name: "Lily Pond",
      x: 2,
      y: 2,
      w: 12,
      h: 10,
      on_enter_text: "Frogs go quiet.",
    },
  ],
  vars: { reputation: 0 },
  intro_text:
    "The Lantern Festival starts tonight, but the golden bell is missing.",
  entry_points: { npc_reply: "reply" },
};

const sessions = new Map<string, SessionRecord>();
const calls: { entry: string; inputs: Record<string, unknown> }[] = [];

async function spriteBlob(): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = 96;
  c.height = 128;
  const ctx = c.getContext("2d") as CanvasRenderingContext2D;
  ctx.fillStyle = "#3b82f6";
  ctx.beginPath();
  ctx.roundRect(18, 52, 60, 70, 18);
  ctx.fill();
  ctx.fillStyle = "#fde68a";
  ctx.beginPath();
  ctx.arc(48, 38, 30, 0, Math.PI * 2);
  ctx.fill();
  return new Promise((resolve) =>
    c.toBlob((b) => resolve(b as Blob), "image/png"),
  );
}

const mock: PlayerApi = {
  async flowInfo(id) {
    return {
      id,
      name: kind === "story" ? "Lantern Picnic" : "Mossy Hollow",
      kind,
    };
  },
  async story() {
    return picnic;
  },
  async world() {
    return WORLD;
  },
  async invoke(_flowId, entry, inputs) {
    calls.push({ entry, inputs });
    await new Promise((r) => setTimeout(r, 300));
    const state = inputs.state as {
      npc_id: string;
      inventory: string[];
      player_message: string;
    };
    if (state.npc_id === "tom" && state.inventory.includes("rod")) {
      return {
        outputs: {
          reply: {
            text: "My lucky rod! You found it!",
            complete_quest: "rod_q",
            take_item: "rod",
          },
        },
      };
    }
    return {
      outputs: {
        reply: { text: `(${state.npc_id}) You said: ${state.player_message}` },
      },
    };
  },
  async listSessions() {
    return [...sessions.values()];
  },
  async loadSession(_f, slot) {
    return sessions.get(slot) ?? null;
  },
  async saveSession(_f, slot, state, label) {
    sessions.set(slot, {
      slot,
      state: JSON.parse(JSON.stringify(state)),
      label,
    });
  },
  async reroll() {
    await new Promise((r) => setTimeout(r, 200));
  },
  async mediaBlob(_m: MediaValue) {
    return spriteBlob();
  },
};

declare global {
  interface Window {
    __flowsPlayerSmoke?: {
      calls: typeof calls;
      sessions: typeof sessions;
      exited: number;
      map: ReturnType<typeof generateMap>;
      world: WorldGameHandle | null;
    };
  }
}
window.__flowsPlayerSmoke = {
  calls,
  sessions,
  exited: 0,
  map: generateMap(WORLD),
  world: null,
};

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <div style={{ position: "fixed", inset: 0 }}>
      <FlowPlayerPage
        flowId={`smoke-${kind}`}
        api={mock}
        onWorldReady={(g) => {
          if (window.__flowsPlayerSmoke) window.__flowsPlayerSmoke.world = g;
        }}
        onExit={() => {
          if (window.__flowsPlayerSmoke) window.__flowsPlayerSmoke.exited += 1;
        }}
      />
      <div id="smoke-ready" />
    </div>,
  );
}
