# Canvas Lab Architecture (Current)

Root:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab`

This doc reflects current code shape (React Flow UI node/edge shell + inline/dialog split + derived aux nodes).

## 1) High-level flow

1. Page shell + React Flow canvas:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/canvas-lab-page.tsx`
2. Block picker sheet (plus/import/copy floating controls):
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/block-sheet.tsx`
3. Zustand state + all graph/config mutations:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab.ts`
4. Connection validation + edge side-effects:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/graph.ts`
5. Export/payload map:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/payload/build-payload.ts`
6. Import/rebuild:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/importer.ts`

## 2) Core types

Source of truth:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/types/index.ts`

`NodeConfig` union:

```ts
export type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ExpressionConfig
  | ModelProviderConfig
  | ModelConfig;
```

`CanvasNodeData` is derived from config (`nodeDataFromConfig`), not edited directly.

## 3) Entrypoint wiring

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/canvas-lab-page.tsx`

Current wiring:

```ts
const NODE_TYPES: NodeTypes = { builder: CanvasNode, aux: CanvasAuxNode };
const EDGE_TYPES: EdgeTypes = { canvas: DataEdge, semantic: CanvasSemanticEdge };
```

Default data edge style uses auto path selection:

```ts
defaultEdgeOptions={{
  type: "canvas",
  data: { key: "name", path: "auto" },
  style: { strokeWidth: 1.5, stroke: "var(--border)" },
}}
```

Node click selects config (`selectConfig`), does not auto-open dialog.
Dialog opens via node `Details` button (`openConfig`) or explicit flows.
Aux nodes (prompt/system/scorer) are derived in page and mounted as `type: "aux"`.

## 4) Registry-driven block system

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/blocks/registry.tsx`

Registry owns:
- block metadata for sheet (title/icon/description)
- config factory (`createConfig`)
- dialog renderer (`renderDialog`)

If adding new `NodeConfig.kind`, keep `getBlockDefinitionForConfig` coverage complete.

## 5) Store responsibilities

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab.ts`

Store owns:
- `nodes`, `edges`, `configs`, `processors`
- `auxNodePositions` (derived aux-node position persistence)
- add/update/remove/connect logic
- `layoutDirection` + dagre apply-layout
- config selection/dialog state

Current config-selection API:
- `selectConfig(id)`: select node config, keep dialog closed
- `openConfig(id)`: select + open modal

Aux-node API:
- `setAuxNodePosition(id, position)`: persist independent drag position
- `syncAuxNodePositions(activeIds, defaults)`: cleanup stale aux positions + seed new defaults

Add-node behavior is mode-aware via:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab-helpers.ts`

New nodes:
- become selected
- set `activeConfigId`
- open dialog only for dialog-first config modes

## 6) Inline vs dialog config policy

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-policy.ts`

Inline mode:
- sampler: `uniform`, `gaussian`, `bernoulli`, `uuid`
- `model_provider`
- `model_config`
- llm: `text`, `code`
- `expression`

Dialog mode:
- sampler: `category`, `subcategory`, `datetime`, `timedelta`, `person`, `person_from_faker`
- llm: `structured`, `judge`

Inline editors:
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-sampler.tsx`
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-model.tsx`
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-llm.tsx`
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-expression.tsx`

LLM inline scope:
- text: `model_alias` only
- code: `model_alias`, `code_lang`
- prompt/system prompt edited in dialog or spawned aux prompt nodes

## 7) Node UI architecture (React Flow UI shell)

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/canvas-node.tsx`

Node shell uses feature-local RF UI primitives:
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/rf-ui/base-node.tsx`
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/rf-ui/labeled-handle.tsx`

Current node UX:
- `corner-squircle` + `rounded-lg` container
- inline editor shown by default for inline-capable configs
- summary text for dialog-first configs
- `Details` button opens modal dialog
- node resizer logic enabled (`NodeResizer`), visuals hidden (no corner/box affordance)
- LLM input handles (system/prompt/scorers) render in dedicated content rows (no overlay on inline controls)

## 8) Handles + layout direction

Handle IDs (kept stable):
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/handles.ts`

```ts
dataIn: "data-in"
dataOut: "data-out"
semanticIn: "semantic-in"
semanticOut: "semantic-out"
llmPromptIn: "llm-prompt-in"
llmSystemIn: "llm-system-in"
llmInputOut: "llm-input-out"
getLlmJudgeScoreHandleId(index): `llm-judge-score-in-${index}`
```

`canvas-node.tsx` switches handle positions by layout direction:
- `LR`: data left/right, semantic top/bottom
- `TB`: data top/bottom, semantic left/right

After direction toggle or auto-layout, page refreshes node internals to avoid stale edge anchor offsets.

## 9) Edge architecture

Data edge:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/rf-ui/data-edge.tsx`

Features:
- label from source node data key
- `path: "auto"` chooses straight vs smoothstep/bezier based on geometry/positions

Semantic edge:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/canvas-semantic-edge.tsx`

Features:
- custom dashed smooth-step
- muted stroke styling

Legacy mixed edge component is removed (no `canvas-edge.tsx` path in active flow).

## 9.1) Derived aux nodes (LLM prompt/system/scorer)

Files:
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/canvas-aux-node.tsx`
- `/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/canvas-lab-page.tsx`

Behavior:
- If `llm.prompt` is non-empty, spawn `llm-prompt-input` aux node
- If `llm.system_prompt` is non-empty, spawn `llm-prompt-input` aux node
- For `llm_type === "judge"`, spawn one `llm-judge-score` aux node per `scores[index]`
- Aux edges auto-connect from aux `llmInputOut` to parent LLM target handles
- Aux nodes are draggable independently; positions persist in Zustand `auxNodePositions`
- Aux nodes are derived UI/editor projections only (no payload schema change)

## 10) Connection semantics + side effects

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/graph.ts`

Lane rules:
- semantic lane only: `semantic-out -> semantic-in`
- data lane only: `data-out -> data-in`
- model infra nodes (`model_provider`, `model_config`) blocked from data lane

Semantic relations:
- `model_provider -> model_config`
- `model_config -> llm`

Connect side effects:
- provider edge sets `model_config.provider`
- model config edge sets `llm.model_alias`
- datetime edge sets `timedelta.reference_column_name`
- data edges into llm/expression append `{{ source_name }}` refs
- category -> subcategory syncs mapping scaffold

Single-incoming enforcement (competing refs pruned on connect):
- `provider`
- `model_alias`
- `reference_column_name`
- `subcategory_parent`

Multi data refs remain allowed for llm/expression prompt/expr templates.

## 11) Canvas controls UX

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/block-sheet.tsx`

Top-right floating controls are icon-only:
- `+` opens add-block sheet
- import icon opens import dialog
- copy icon copies recipe (brief check icon state on success)

All use same no-bg bordered button style (`corner-squircle`, hover primary border/icon).

## 12) Dialog routing

Config modal shell:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/dialogs/config-dialog.tsx`

Routes block-specific forms through registry renderer:
`renderBlockDialog(config, categoryOptions, onUpdate)`

Current modal-only edits still live here (structured/judge/category/subcategory/etc).

## 13) Payload + import boundary

Payload map:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/payload/build-payload.ts`

Import map:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/importer.ts`

If `ui.edges` missing on import, inferred edges are built from config refs in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/edges.ts`

## 14) Add new block checklist

1. Add/extend config type in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/types/index.ts`
2. Add factory + `nodeDataFromConfig` mapping in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/index.ts`
3. Add registry entry in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/blocks/registry.tsx`
4. Add dialog and wire in registry `renderDialog`
5. Decide inline vs dialog mode; update:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-policy.ts`
6. If inline, add inline editor component under:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/`
7. Add payload mapping/validation updates in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/payload/`
8. Add import parse/infer updates in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/`
9. Extend connection semantics if needed in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/graph.ts`

## 15) Mental model

- `NodeConfig` = business truth
- `CanvasNodeData` = derived presentational truth
- registry = block metadata + factories + dialog routing
- store = orchestration + consistency
- graph utils = legal edges + side effects
- payload/import = external contract boundary

Keep all 6 synchronized when adding/changing block behavior.
