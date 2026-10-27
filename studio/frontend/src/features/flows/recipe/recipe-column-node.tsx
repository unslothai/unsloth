// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recipe column's content in the shared node card (`node-looks.ts`), and the model / tool cards beside it.

import { Badge } from "@/components/ui/badge";
import { InlineCategoryBadges } from "@/features/recipe-studio/components/inline/inline-category-badges";
import {
  RECIPE_STUDIO_NODE_TONES,
  RECIPE_STUDIO_USER_NODE_TONE,
} from "@/features/recipe-studio/utils/ui-tones";
import {
  BalanceScaleIcon,
  Clock01Icon,
  CodeIcon,
  CodeSimpleIcon,
  DiceFaces03Icon,
  EqualSignIcon,
  FingerPrintIcon,
  FunctionIcon,
  Parabola02Icon,
  PencilEdit02Icon,
  Plant01Icon,
  Plug01Icon,
  Shield02Icon,
  Tag01Icon,
  TagsIcon,
  UserAccountIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import {
  BaseEdge,
  type EdgeProps,
  Handle,
  type NodeProps,
  Position,
  getBezierPath,
} from "@xyflow/react";
import { memo } from "react";
import { inHandle, outHandle } from "../editor/connection-rules";
import type { RfEdge, RfNode } from "../editor/derive-rf";
import {
  CollapsedPorts,
  NodeCard,
  NodeCardHeader,
  NodeDisclosure,
  NodeFieldRows,
  NodePorts,
} from "../editor/node-card";
import type { NodeLook } from "../editor/node-looks";
import { portInk } from "../editor/theme-colors";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { IDLE_RUNTIME } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import { columnName, columnOf } from "./recipe-columns";
import {
  type AuxCard,
  type ColumnLook,
  REF_HANDLE_SEP,
  RECIPE_ACCENT,
  columnInputPorts,
  columnLook,
  columnTakesWires,
} from "./recipe-look";
import { useColumnSamples } from "./recipe-panels";

const BLOCK_ICONS: Record<string, IconSvgElement> = {
  category: Tag01Icon,
  subcategory: TagsIcon,
  uniform: EqualSignIcon,
  gaussian: Parabola02Icon,
  bernoulli: EqualSignIcon,
  datetime: Clock01Icon,
  timedelta: Clock01Icon,
  uuid: FingerPrintIcon,
  person: UserAccountIcon,
  // biome-ignore lint/style/useNamingConvention: api schema
  person_from_faker: UserAccountIcon,
  text: PencilEdit02Icon,
  structured: CodeIcon,
  code: CodeSimpleIcon,
  judge: BalanceScaleIcon,
  expression: FunctionIcon,
};

function columnIcon(look: ColumnLook): IconSvgElement {
  if (look.kind === "validator") return Shield02Icon;
  return BLOCK_ICONS[look.block] ?? DiceFaces03Icon;
}

function columnTone(look: ColumnLook): string {
  if (look.block === "person" || look.block === "person_from_faker")
    return RECIPE_STUDIO_USER_NODE_TONE;
  return RECIPE_STUDIO_NODE_TONES[look.kind];
}

/** Handles the model / tool cards wire through; not for drawing by hand. */
export const AUX_IN = "aux-in";
export const AUX_OUT = "aux-out";
const AUX_HANDLE = "!size-1.5 !min-h-0 !min-w-0 !border-0 !opacity-0";

const portsKey = (id: string) => (s: { graph: Parameters<typeof columnInputPorts>[0] }) =>
  columnInputPorts(s.graph, id)
    .map((p) => `${p.sourceId}\t${p.label}\t${p.type}`)
    .join("\n");

/** One input per column read, named after it; the output carries the column's own name. */
function ColumnPorts({ node, collapsed }: { node: FlowNode; collapsed: boolean }) {
  const key = useDocStore(portsKey(node.id));
  const colors = useCatalogStore((s) => s.catalog?.port_colors);
  const column = columnOf(node);
  const look = columnLook(column);
  const color = (type: string) => colors?.[type] ?? "#cbd5e1";
  const reads = key
    ? key.split("\n").map((line) => {
        const [sourceId, label, type] = line.split("\t");
        return {
          handleId: inHandle(`in${REF_HANDLE_SEP}${sourceId}`),
          label,
          color: color(type),
          labelClassName: "font-mono text-foreground",
        };
      })
    : [];
  const inputs = columnTakesWires(column)
    ? [
        ...reads,
        {
          handleId: inHandle("in"),
          label: "+ column",
          color: color("text"),
        },
      ]
    : reads;
  const output = {
    handleId: outHandle("out"),
    label: columnName(node),
    color: color(look.valueType),
    labelClassName: "font-mono text-foreground",
  };
  if (collapsed)
    return (
      <CollapsedPorts
        inputs={inputs.map((p) => p.handleId)}
        outputs={[output.handleId]}
      />
    );
  // Output on its own row so long column names on either side don't squeeze each other.
  return (
    <>
      <NodePorts inputs={[]} outputs={[output]} />
      <NodePorts inputs={inputs} outputs={[]} />
    </>
  );
}

/** The value this column got in the first row of its recipe's last run. */
function ColumnSample({ node }: { node: FlowNode }) {
  const [sample] = useColumnSamples(node, 1);
  if (!sample) return null;
  return (
    <p className="flow-detail mx-3 line-clamp-3 whitespace-pre-wrap rounded-md border border-emerald-500/25 bg-emerald-500/10 px-2 py-1 text-ui-11 leading-snug text-emerald-800 dark:text-emerald-200">
      {sample}
    </p>
  );
}

function ColumnBody({ node }: { node: FlowNode }) {
  const look = columnLook(columnOf(node));
  return (
    <>
      <NodeFieldRows rows={look.fields} />
      {look.chips.length > 0 && (
        <div className="flow-detail px-3">
          <InlineCategoryBadges values={look.chips} />
        </div>
      )}
      <NodeDisclosure sections={look.details} />
      <ColumnSample node={node} />
    </>
  );
}

function ColumnHeaderExtra({ node }: { node: FlowNode }) {
  const look = columnLook(columnOf(node));
  return (
    <>
      {look.drop && (
        <Badge
          variant="outline"
          className="flow-detail h-5 px-1.5 text-ui-10 text-muted-foreground"
          title="drop=true: used while generating, left out of the dataset"
        >
          Dropped
        </Badge>
      )}
      {look.kind === "llm" && (
        <Handle
          type="target"
          id={AUX_IN}
          position={Position.Bottom}
          isConnectable={false}
          className={AUX_HANDLE}
        />
      )}
    </>
  );
}

/** The `recipe.column` entry of `NODE_LOOKS`. */
export const RECIPE_COLUMN_LOOK: NodeLook = {
  icon: (node) => columnIcon(columnLook(columnOf(node))),
  tone: (node) => columnTone(columnLook(columnOf(node))),
  subtitle: (node) => columnLook(columnOf(node)).subtitle,
  HeaderExtra: ColumnHeaderExtra,
  Ports: ColumnPorts,
  Body: ColumnBody,
};

const AUX_ICON: Record<AuxCard["kind"], IconSvgElement> = {
  // biome-ignore lint/style/useNamingConvention: api schema
  model_provider: Shield02Icon,
  // biome-ignore lint/style/useNamingConvention: api schema
  model_config: Plant01Icon,
  // biome-ignore lint/style/useNamingConvention: api schema
  tool_config: Plug01Icon,
};

export type AuxData = { card: AuxCard };

function RecipeAuxNodeImpl({ data }: NodeProps<RfNode>) {
  const { card } = data as unknown as AuxData;
  return (
    <NodeCard runtime={IDLE_RUNTIME} className="w-56" title="Double-click to edit">
      {card.kind === "model_config" && (
        <Handle
          type="target"
          id={AUX_IN}
          position={Position.Top}
          isConnectable={false}
          className={AUX_HANDLE}
        />
      )}
      <NodeCardHeader
        icon={AUX_ICON[card.kind]}
        tone={RECIPE_STUDIO_NODE_TONES[card.kind]}
        title={card.name}
        subtitle={card.subtitle}
      />
      {card.fields.length > 0 && (
        <div className="py-2">
          <NodeFieldRows rows={card.fields} />
        </div>
      )}
      <Handle
        type="source"
        id={AUX_OUT}
        position={card.kind === "model_provider" ? Position.Bottom : Position.Left}
        isConnectable={false}
        className={AUX_HANDLE}
      />
    </NodeCard>
  );
}

export const RecipeAuxNode = memo(RecipeAuxNodeImpl);

function RecipeAuxEdgeImpl({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: EdgeProps<RfEdge>) {
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  return (
    <BaseEdge
      id={id}
      path={path}
      style={{
        stroke: portInk(RECIPE_ACCENT.model_config),
        strokeWidth: 1.25,
        strokeDasharray: "4 4",
        opacity: 0.6,
      }}
    />
  );
}

export const RecipeAuxEdge = memo(RecipeAuxEdgeImpl);

/** Double-click on a card opens Recipe Studio's own dialog for that provider, preset or tool profile. */
export function editAuxCard(card: AuxCard): void {
  useUiStore
    .getState()
    .setRecipeEditor({ groupId: card.groupId, configName: card.name });
}
