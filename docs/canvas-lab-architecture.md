# Canvas Lab Architecture (Current)

Root:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab`

This doc explains current architecture, how nodes map to payload/import, and how to add new blocks safely.

## 1) High-level flow

1. UI renders canvas + dialogs in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/canvas-lab-page.tsx`
2. Add-block sheet uses registry metadata to create config objects:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/block-sheet.tsx`
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/blocks/registry.tsx`
3. Zustand store owns nodes/edges/configs and all mutation logic:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab.ts`
4. Graph connection logic updates references + semantic edges:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/graph.ts`
5. Export (preview/copy) converts in-memory graph/config to API payload:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/payload/build-payload.ts`
6. Import reconstructs configs, nodes, edges from JSON:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/importer.ts`

## 2) Core types (single source of truth)

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/types/index.ts`

`NodeConfig` is the main union the whole feature uses:

```ts
export type NodeConfig =
  | SamplerConfig
  | LlmConfig
  | ExpressionConfig
  | ModelProviderConfig
  | ModelConfig;
```

Canvas node UI data (`CanvasNodeData`) is derived from config via `nodeDataFromConfig`.

## 3) Entrypoint wiring

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/canvas-lab-page.tsx`

Key wiring:

```ts
const NODE_TYPES: NodeTypes = { builder: CanvasNode };
const EDGE_TYPES: EdgeTypes = { canvas: CanvasEdge, semantic: CanvasEdge };
```

`CanvasLabPage` pulls actions/state from store and passes add handlers into `BlockSheet`:

```ts
<BlockSheet
  onAddSampler={addSamplerNode}
  onAddLlm={addLlmNode}
  onAddModelProvider={addModelProviderNode}
  onAddModelConfig={addModelConfigNode}
  onAddExpression={addExpressionNode}
/>
```

Preview/copy route through `buildCanvasPayload`, import route through `importCanvasPayload`.

## 4) Registry-driven block system

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/blocks/registry.tsx`

Registry defines each block in one place:
- sheet title/icon/description
- config factory (`createConfig`)
- config dialog (`renderDialog`)

Example (model blocks):

```ts
{
  kind: "llm",
  type: "model_provider",
  createConfig: (id, existing) => makeModelProviderConfig(id, existing),
  renderDialog: ({ config, onUpdate }) =>
    config.kind === "model_provider" ? (
      <ModelProviderDialog config={config} onUpdate={(patch) => onUpdate(config.id, patch)} />
    ) : null,
}
```

Important: `getBlockDefinitionForConfig` must map every new `config.kind`, else dialog won't render.

## 5) Config factories + node label mapping

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/index.ts`

Responsibilities:
- create default config objects (`makeSamplerConfig`, `makeLlmConfig`, `makeModelProviderConfig`, `makeModelConfig`, `makeExpressionConfig`)
- map `NodeConfig -> CanvasNodeData` via `nodeDataFromConfig`
- sampler set includes `category`, `subcategory`, `uniform`, `gaussian`, `bernoulli`, `datetime`, `timedelta`, `uuid`, `person`, `person_from_faker`

Example mapping:

```ts
if (config.kind === "model_provider") {
  return {
    title: "Model Provider",
    kind: "model_provider",
    subtype: config.provider_type || "Provider",
    blockType: "model_provider",
    name: config.name,
    layoutDirection,
  };
}
```

This is what controls visible node title/subtitle in the canvas.

## 6) Store responsibilities

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab.ts`

Store owns:
- graph state (`nodes`, `edges`)
- config map (`configs[id]`)
- add/update/remove/connect operations
- layout direction + apply layout

Add-node pattern (all block types follow same shape):

```ts
const definition = getBlockDefinition("llm", "model_config");
const config = definition.createConfig(id, existing);
return buildNodeUpdate(state, config, state.layoutDirection);
```

When model config `provider` field changes, store auto-syncs semantic edge to matching provider name.

## 7) Edge semantics + connection behavior

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/graph.ts`

Semantic edge classifier:

```ts
function isSemanticEdge(source: NodeConfig, target: NodeConfig): boolean {
  if (source.kind === "model_provider" && target.kind === "model_config") return true;
  return source.kind === "model_config" && target.kind === "llm";
}
```

Handle lanes:
- data edges use `data-out -> data-in` (`right -> left`)
- semantic edges use `semantic-out -> semantic-in` (`bottom -> top`)
- semantic lane only used for `model_provider -> model_config -> llm`

Connection side effects:
- `model_provider -> model_config`: set `model_config.provider = source.name`
- `model_config -> llm`: set `llm.model_alias = source.name`
- `datetime -> timedelta`: set `timedelta.reference_column_name = source.name`
- regular data edges into LLM/expression append `{{ source_name }}` refs

Edge rendering (dotted semantic edges):
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/components/canvas-edge.tsx`

```ts
const nextStyle = type === "semantic"
  ? { ...style, strokeDasharray: "4 4" }
  : style;
```

## 8) Rename/remove propagation

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab-helpers.ts`

Centralized consistency updates:
- rename updates:
  - Jinja refs in `llm.prompt/system_prompt/output_format`
  - expression `expr`
  - subcategory parent
  - `model_config.provider`
  - `llm.model_alias`
- removal clears same references

This keeps graph fields stable when upstream nodes renamed/deleted.

## 9) Payload building (node graph -> API)

File:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/payload/build-payload.ts`

`buildCanvasPayload(configs, nodes, edges)` outputs:

```ts
{
  recipe: {
    model_providers: [...],
    model_configs: [...],
    columns: [...],
    processors: [...],
  },
  run: { rows: 5, preview: true, output_formats: ["jsonl"] },
  ui: { nodes: [...], edges: [...] }
}
```

Current processor UI surface:
- `schema_transform` (sheet -> `Processors` -> `Schema Transform`)
- mapped to recipe processor with `build_stage: "post_batch"` and JSON `template`.

Current drop policy:
- column dialogs (`sampler` / `llm` / `expression`) expose `drop` toggle.
- payload writes column `drop` directly (preferred over drop-columns processor in v1).

How relation is enforced:
- collect `model_alias` values used by LLM columns
- ensure each alias exists in `recipe.model_configs`
- validate `model_config.provider` points to existing provider
- validate `timedelta.reference_column_name` points to a datetime sampler
- require endpoint/provider_type only for providers that are actually referenced
- category sampler supports typed `conditional_params` in payload output

This is why unused provider/config blocks can exist without blocking preview.

## 10) Import pipeline (API -> node graph)

Entry file:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/importer.ts`

Order of reconstruction:
1. parse `recipe.model_providers` -> `ModelProviderConfig`
2. parse `recipe.model_configs` -> `ModelConfig`
3. parse `recipe.columns` -> sampler/llm/expression
4. build nodes with positions
5. build edges

Edge inference file:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/edges.ts`

If UI edges missing, infer edges from fields:
- `subcategory_parent` (canvas edge)
- `model_config.provider`
- `llm.model_alias`
- infer data edge from `timedelta.reference_column_name`

## 11) Dialog routing and edit UIs

Config dialog shell:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/dialogs/config-dialog.tsx`

It calls:
`renderBlockDialog(config, categoryOptions, onUpdate)`

Model dialogs:
- provider:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/dialogs/models/model-provider-dialog.tsx`
- model config:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/dialogs/models/model-config-dialog.tsx`
- processors:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/dialogs/processors-dialog.tsx`

`ModelConfigDialog` and `LlmDialog` use shadcn `Combobox` fed from store configs:
- model config `provider` suggests model-provider node names
- llm `model_alias` suggests model-config aliases
- timedelta dialog suggests datetime columns for `reference_column_name`

## 12) How to add a new block (checklist)

Minimal path for a new block type:

1. Add/extend type in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/types/index.ts`
2. Add default factory + node label mapping in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/index.ts`
3. Add block definition in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/blocks/registry.tsx`
4. Add dialog component and route it via `renderDialog` in registry.
5. Add store add-action if block should be special-cased from sheet:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/stores/canvas-lab.ts`
6. Add payload serialization/validation in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/payload/`
7. Add import parsing + inferred edges in:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/parsers.ts`
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/importer.ts`
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/import/edges.ts`
8. If connection has semantic meaning, extend:
`/Volumes/Expansion/projects/new-ui-prototype/studio/frontend/src/features/canvas-lab/utils/graph.ts`

## 13) Practical mental model

- `NodeConfig` is source-of-truth business state.
- `CanvasNodeData` is derived display state.
- registry = block metadata + factories + dialog routing.
- store = mutation orchestration.
- graph utils = connection semantics.
- payload/import utils = external contract boundary.

If one piece changes, keep all six in sync.

## 14) Processors roadmap (decision)

Current decision: **Option 3 (hybrid)**.

v1 scope:
- add `drop` toggle on column blocks (sampler/llm/expression).
- add processor config surface for `schema_transform`.
- keep payload builder as single mapper to `recipe.processors`.
- keep processor state separate from node graph for now.

Reason:
- fastest ship path.
- matches Data Designer column-level `drop`.
- avoids duplicate/complex processor edge logic in v1.

Future option noted: **Option 2 (processor chain in graph)**.

Option 2 structure:
- add virtual node `Dataset Output`.
- processors become graph nodes: `Schema Transform`, `Drop Columns`, future processors.
- processor order derived from chain edges:
  `Dataset Output -> P1 -> P2 -> ...`
- enforce chain rules:
  - no cycles
  - one incoming max per processor
  - one outgoing max per processor
  - chain must start at `Dataset Output`

Migration from option 3 -> 2:
- keep same processor schema/payload contracts.
- move order source from list/order field to edge traversal.
- UI changes mostly in canvas rendering + validation; payload adapter stays mostly same.
