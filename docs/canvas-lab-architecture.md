# Canvas Lab Architecture (Current)

Root:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab`

Goal of this layout:
- simple ownership
- low coupling
- predictable edit points
- behavior driven by config + store, not view side-effects

## 1) Ownership Map (hard boundaries)

### Page orchestration boundary
File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/canvas-lab-page.tsx`

Owns:
- React Flow mount + wiring
- selector/orchestration glue from Zustand
- derived display graph (`deriveDisplayGraph`)
- modal/sheet open-close UI state

Do not place here:
- config mutation rules
- connection legality rules
- payload/import mapping

### Store mutation boundary
File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab.ts`

Owns:
- source-of-truth state (`configs`, `nodes`, `edges`, `processors`)
- mutation entrypoints (`updateConfig`, `onConnect`, `onNodesChange`, etc)
- selection/dialog state (`selectConfig`, `openConfig`)
- aux node position persistence (`auxNodePositions`)
- aux node size persistence (`auxNodeSizes`)

Helper module:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab-helpers.ts`

Owns:
- pure relation sync helpers (edge/config sync)
- rename/remove propagation helpers
- node data/layout transformation helpers

Do not place in helpers:
- React component logic
- network/API calls

### Graph rules boundary
Files:
- `.../utils/graph/canvas-connection.ts`
- `.../utils/graph/derive-display-graph.ts`
- `.../utils/graph.ts` (re-export shim)

`canvas-connection.ts` owns:
- valid/invalid connection rules
- connect side-effects (config updates from edges)
- single-incoming relation enforcement

`derive-display-graph.ts` owns:
- derived aux nodes/edges (LLM prompt/system/scorer projections)
- default aux positioning

Do not place here:
- dialog form logic
- block creation defaults

### Registry boundary
File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/blocks/registry.tsx`

Owns:
- block metadata for sheet
- config factory per block
- dialog router per block type

Notes:
- registry receives dialog option lists from page and forwards to block dialogs
- avoids dialog -> store dependency cycle

Do not place here:
- cross-node graph mutation logic
- payload/export code

### Import/export boundary
Files:
- `.../utils/payload/build-payload.ts`
- `.../utils/import/importer.ts`
- `.../utils/import/edges.ts`

Owns:
- contract mapping between UI state and backend payload
- edge inference fallback when import payload has no `ui.edges`
- node width persistence via `ui.nodes[].width`

Do not place here:
- ReactFlow render logic
- store actions

## 2) Core Types

Source:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/types/index.ts`

`NodeConfig` is business truth:
```ts
type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ExpressionConfig
  | ModelProviderConfig
  | ModelConfig;
```

`CanvasNodeData` is derived display data (`nodeDataFromConfig`), not primary state.

## 3) React Flow Composition

Page wiring:
```ts
const NODE_TYPES = { builder: CanvasNode, aux: CanvasAuxNode };
const EDGE_TYPES = { canvas: DataEdge, semantic: CanvasSemanticEdge };
```

Data edges use auto path mode:
```ts
defaultEdgeOptions={{
  type: "canvas",
  data: { key: "name", path: "auto" },
}}
```

Dialog flow:
- node click -> `selectConfig` (no forced modal)
- node `Details` button -> `openConfig`

Node sizing:
- default builder/aux node width is `400px`
- users can resize builder + aux nodes
- resized width is kept in canvas state and round-tripped through import/export
- sizing constants live in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/constants.ts`

## 4) UI Mode Policy (Inline vs Dialog)

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/inline/inline-policy.ts`

Inline:
- sampler: `uniform`, `gaussian`, `bernoulli`, `uuid`
- `model_provider`, `model_config`
- llm: `text`, `code`
- `expression`

Dialog:
- sampler: `category`, `subcategory`, `datetime`, `timedelta`, `person`, `person_from_faker`
- llm: `structured`, `judge`

Inline editors:
- `.../components/inline/inline-sampler.tsx`
- `.../components/inline/inline-model.tsx`
- `.../components/inline/inline-llm.tsx`
- `.../components/inline/inline-expression.tsx`

## 5) Handle Contract (stable IDs)

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/handles.ts`

Stable IDs:
- `data-in`, `data-out`
- `semantic-in`, `semantic-out`
- `llm-prompt-in`, `llm-system-in`, `llm-input-out`
- `llm-judge-score-in-${index}`

These are contract-level values for:
- connection validity
- edge inference
- payload consistency

## 6) LLM Derived Aux Nodes

Behavior source:
`.../utils/graph/derive-display-graph.ts`

Rules:
- non-empty `prompt` => spawn editable prompt aux node
- non-empty `system_prompt` => spawn editable system aux node
- `llm_type === "judge"` => spawn scorer aux nodes from `scores[]`

Aux nodes:
- are UI projections, not new payload schema entities
- have independent drag positions persisted in Zustand `auxNodePositions`
- have independent sizes persisted in Zustand `auxNodeSizes`
- are resizable (same hidden-control resize UX as builder nodes)
- are re-anchored near parent nodes after auto-layout/direction change

## 7) Connect Rules (single source of truth)

File:
`.../utils/graph/canvas-connection.ts`

Rules:
- semantic lane only for model infra relations
- data lane for sampler/llm/expression flow
- model infra blocked from data lane

Single incoming enforced for:
- `provider`
- `model_alias`
- `reference_column_name`
- `subcategory_parent`

Connect side-effects:
- provider/model alias/ref-column update target config fields
- category->subcategory scaffolds mapping
- llm/expression data refs append template references

## 8) Circular Dependency Prevention

Current safe flow:
- store state -> page (`configs`)
- page derives dialog option lists (`modelConfigAliases`, `modelProviderOptions`, `datetimeOptions`)
- page passes these options -> `ConfigDialog`
- dialog passes options -> registry -> block dialogs (`LlmDialog`, `ModelConfigDialog`, `TimedeltaDialog`)

No dialog component should import store directly.

## 9) Add New Block (exact flow)

1. Add config type:
`.../types/index.ts`
2. Add defaults + `nodeDataFromConfig` mapping:
`.../utils/index.ts`
3. Add registry definition:
`.../blocks/registry.tsx`
4. Add dialog component + wire via `renderDialog`:
`.../dialogs/...`
5. Choose UI mode policy:
`.../components/inline/inline-policy.ts`
6. If inline, add inline editor:
`.../components/inline/...`
7. Add connect semantics if needed:
`.../utils/graph/canvas-connection.ts`
8. Add payload mapping:
`.../utils/payload/...`
9. Add import parse/inference update:
`.../utils/import/...`

## 10) Keep It Simple Rules

- Keep mutation logic in store/helpers only
- Keep graph legality/side-effects in graph utils only
- Keep view files free of business mutation branching
- Remove dead code in same pass as refactor
- Prefer narrow pure helpers over giant mixed functions

If unsure where code belongs:
- “changes config/edges?” => store/helpers or graph utils
- “changes visuals only?” => components/page
- “changes payload contract?” => payload/import utils
