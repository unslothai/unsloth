// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  FLOW_FORMAT,
  FLOW_FORMAT_VERSION,
  type FlowDocument,
  type FlowEdge,
  type FlowKind,
  type FlowNode,
} from "../types";

function newFlowId(): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now()}${Math.random()}`;
  return `flow_${random.replace(/[^a-zA-Z0-9]/g, "").slice(0, 24)}`;
}

function node(
  id: string,
  type: string,
  pos: [number, number],
  params: Record<string, unknown> = {},
  label?: string,
): FlowNode {
  return { id, type, v: 1, params, pos, ...(label ? { label } : {}) };
}

function edge(
  source: string,
  sourcePort: string,
  target: string,
  targetPort: string,
  kind: "data" | "flow" = "data",
): FlowEdge {
  return {
    id: `e_${source}_${sourcePort}_${target}_${targetPort}`.slice(0, 96),
    source,
    source_port: sourcePort,
    target,
    target_port: targetPort,
    kind,
  };
}

function base(kind: FlowKind, name: string, description: string): FlowDocument {
  return {
    format: FLOW_FORMAT,
    version: FLOW_FORMAT_VERSION,
    id: newFlowId(),
    name,
    description,
    kind,
    seed: Math.floor(Math.random() * 2 ** 31),
    params: [],
    requirements: { models: [] },
    nodes: [],
    edges: [],
    groups: [],
    notes: [],
    assets: [],
    baked: {},
    editor: {},
    meta: {},
  };
}

/** Starting points for New. Node params left out fall back to the catalog defaults on the backend. */
export function templateDocument(kind: FlowKind): FlowDocument {
  if (kind === "story") {
    const doc = base(
      "story",
      "My story",
      "A branching story. Scenes connect with story paths; choices branch them.",
    );
    doc.nodes = [
      node("story", "story.start", [0, 120], {
        title: "The Lantern in the Fog",
        vars: { courage: 0 },
        style: "storybook watercolor illustration",
        mode: "fixed",
      }),
      node("hero", "story.character", [0, 360], {
        name: "Mira",
        look: "a young lighthouse keeper with a yellow raincoat",
        color: "#f59e0b",
        side: "left",
      }),
      node("s1", "story.scene", [340, 100], {
        title: "The shore",
        background_prompt: "a foggy shore at dusk with a dark lighthouse",
        lines:
          "Narrator: The lighthouse had been dark for three nights.\nMira: Someone has to climb up there.",
        generate_background: true,
      }),
      node("c1", "story.choice", [700, 100], {
        prompt: "What does Mira do?",
        choices: ["Climb the stairs", "Wait for the boat"],
      }),
      node("s2", "story.scene", [1060, 0], {
        title: "The stairs",
        lines: "Narrator: Each step creaked louder than the last.",
        on_enter: { courage: { op: "+", args: [{ var: "courage" }, 1] } },
        generate_background: true,
        background_prompt: "a spiral staircase inside an old lighthouse",
      }),
      node("s3", "story.scene", [1060, 300], {
        title: "The boat",
        lines: "Narrator: A small boat bumped against the pier, empty.",
        generate_background: true,
        background_prompt: "an empty rowing boat at a misty pier",
      }),
      node("end1", "story.ending", [1420, 0], {
        title: "The light returns",
        kind: "good",
        text: "Mira lit the lamp, and the ships found their way home.",
      }),
      node("end2", "story.ending", [1420, 300], {
        title: "Adrift",
        kind: "neutral",
        text: "The fog swallowed the pier, and the story waits for another night.",
      }),
    ];
    doc.edges = [
      edge("story", "next", "s1", "flow", "flow"),
      edge("hero", "character", "s1", "characters"),
      edge("s1", "next", "c1", "flow", "flow"),
      edge("c1", "choice_1", "s2", "flow", "flow"),
      edge("c1", "choice_2", "s3", "flow", "flow"),
      edge("s2", "next", "end1", "flow", "flow"),
      edge("s3", "next", "end2", "flow", "flow"),
    ];
    return doc;
  }
  if (kind === "world") {
    const doc = base(
      "world",
      "My world",
      "A small top-down world: an LLM designs it, the image model paints it, you explore it.",
    );
    doc.nodes = [
      node(
        "premise",
        "input.text",
        [0, 80],
        {
          value:
            "A cozy seaside village where the lighthouse keeper lost her cat.",
        },
        "Premise",
      ),
      node("gen", "world.generator", [320, 40], {
        npcs: 4,
        items: 4,
        biome: "beach",
      }),
      node("art", "world.art", [680, 40], {}),
      node("play", "world.play", [1040, 40], {}),
      node("reply", "world.npc_reply", [680, 320], {}),
    ];
    doc.edges = [
      edge("premise", "text", "gen", "premise"),
      edge("gen", "world", "art", "world"),
      edge("art", "world", "play", "world"),
    ];
    return doc;
  }
  const doc = base("pipeline", "My flow", "");
  doc.nodes = [
    node(
      "prompt",
      "input.text",
      [0, 80],
      { value: "A tiny sloth astronaut planting a flag on a strawberry moon" },
      "Idea",
    ),
    node("llm", "llm.generate", [320, 40], {
      system: "You write vivid, concise image prompts.",
      prompt: "Turn this idea into one detailed image prompt: {{prompt}}",
      max_tokens: 200,
    }),
    node("image", "image.generate", [680, 40], {}),
    node("preview", "output.preview", [1040, 60], {}),
  ];
  doc.edges = [
    edge("prompt", "text", "llm", "prompt"),
    edge("llm", "text", "image", "prompt"),
    edge("image", "image", "preview", "value"),
  ];
  return doc;
}
