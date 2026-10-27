// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recipe column in Flows' node chrome with Recipe Studio's detail, and the model / tool cards beside it.

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
import { HugeiconsIcon } from "@hugeicons/react";
import {
  BaseEdge,
  type EdgeProps,
  Handle,
  type NodeProps,
  Position,
  getBezierPath,
} from "@xyflow/react";
import { memo, useState } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import type { FlowNode } from "../types";
import { inHandle, outHandle } from "../editor/connection-rules";
import type { RfEdge, RfNode } from "../editor/derive-rf";
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

type Icon = typeof CodeIcon;

const BLOCK_ICONS: Record<string, Icon> = {
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

export function columnIcon(look: ColumnLook): Icon {
  if (look.kind === "validator") return Shield02Icon;
  return BLOCK_ICONS[look.block] ?? DiceFaces03Icon;
}

/** Handle the model / tool cards wire into; not for drawing by hand. */
export const AUX_IN = "aux-in";
export const AUX_OUT = "aux-out";

function usePortColor(type: string): string {
  return (
    useCatalogStore((s) => s.catalog?.port_colors[type]) ?? "#cbd5e1"
  );
}

function PortDot({
  type,
  side,
  id,
}: { type: string; side: "in" | "out"; id: string }) {
  const color = usePortColor(type);
  return (
    <Handle
      type={side === "in" ? "target" : "source"}
      id={id}
      position={side === "in" ? Position.Left : Position.Right}
      className="!h-2.5 !w-2.5 !border-2 !border-zinc-900"
      style={{ background: color, [side === "in" ? "left" : "right"]: -5 }}
    />
  );
}

function inputPortsKey(id: string) {
  return (s: { graph: Parameters<typeof columnInputPorts>[0] }) =>
    columnInputPorts(s.graph, id)
      .map((p) => `${p.sourceId}\t${p.label}\t${p.type}`)
      .join("\n");
}

/** One input per column read, labelled with that column's name; the output carries the column's own name. */
export function RecipeColumnPorts({ node }: { node: FlowNode }) {
  const key = useDocStore(inputPortsKey(node.id));
  const column = columnOf(node);
  const look = columnLook(column);
  const inputs = key
    ? key.split("\n").map((line) => {
        const [sourceId, label, type] = line.split("\t");
        return { sourceId, label, type };
      })
    : [];
  const open = columnTakesWires(column);
  return (
    <div className="flex justify-between gap-2">
      <div className="flex min-w-0 flex-col gap-1">
        {inputs.map((port) => (
          <div
            key={port.sourceId}
            className="relative flex h-5 items-center pl-3 pr-1"
          >
            <PortDot
              type={port.type}
              side="in"
              id={inHandle(`in${REF_HANDLE_SEP}${port.sourceId}`)}
            />
            <span className="flow-detail truncate font-mono text-[11px] text-zinc-300">
              {port.label}
            </span>
          </div>
        ))}
        {open && (
          <div
            className="relative flex h-5 items-center pl-3 pr-1"
            title="Wire a column here to add {{ column }} to this one"
          >
            <PortDot type="text" side="in" id={inHandle("in")} />
            <span className="flow-detail truncate text-[11px] text-zinc-500">
              + column
            </span>
          </div>
        )}
      </div>
      <div className="flex min-w-0 flex-col items-end gap-1">
        <div className="relative flex h-5 items-center pl-1 pr-3">
          <span className="flow-detail truncate font-mono text-[11px] text-zinc-300">
            {columnName(node)}
          </span>
          <PortDot type={look.valueType} side="out" id={outHandle("out")} />
        </div>
      </div>
    </div>
  );
}

function Chips({ values, accent }: { values: string[]; accent: string }) {
  if (!values.length) return null;
  const shown = values.slice(0, 6);
  return (
    <div className="flex flex-wrap gap-1">
      {shown.map((value) => (
        <span
          key={value}
          className="max-w-full truncate rounded px-1.5 py-0.5 text-[10px] leading-tight"
          style={{ background: `${accent}1f`, color: `${accent}` }}
        >
          {value}
        </span>
      ))}
      {values.length > shown.length && (
        <span className="px-1 text-[10px] text-zinc-500">
          +{values.length - shown.length}
        </span>
      )}
    </div>
  );
}

/** Subtitle, key fields, chips and the collapsed prompts of a column. */
export function RecipeColumnDetails({ node }: { node: FlowNode }) {
  const look = columnLook(columnOf(node));
  const [open, setOpen] = useState(false);
  return (
    <div className="flow-detail flex flex-col gap-1.5 px-3">
      {(look.fields.length > 0 || look.drop) && (
        <div className="flex flex-col gap-0.5">
          {look.fields.map((field) => (
            <div
              key={field.label}
              className="flex min-w-0 items-baseline gap-1.5 text-[11px]"
            >
              <span className="shrink-0 text-[9px] uppercase tracking-wide text-zinc-500">
                {field.label}
              </span>
              <span className="truncate font-mono text-zinc-200">
                {field.value}
              </span>
            </div>
          ))}
        </div>
      )}
      <Chips values={look.chips} accent={look.accent} />
      {look.details.length > 0 && (
        <div className="flex flex-col gap-1">
          <button
            type="button"
            className="nodrag self-start text-[10px] text-zinc-500 hover:text-zinc-300"
            onClick={(e) => {
              e.stopPropagation();
              setOpen((v) => !v);
            }}
          >
            {open ? "Hide inputs" : "Show inputs"}
          </button>
          {open &&
            look.details.map((detail) => (
              <div
                key={detail.label}
                className="nowheel max-h-40 overflow-auto rounded bg-white/5 px-2 py-1"
              >
                <div className="text-[9px] uppercase tracking-wide text-zinc-500">
                  {detail.label}
                </div>
                <p className="whitespace-pre-wrap font-mono text-[10px] leading-snug text-zinc-300">
                  {detail.text}
                </p>
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

/** Header identity for a column node: its block icon, accent and subtitle. */
export function RecipeColumnHeader({ node }: { node: FlowNode }) {
  const look = columnLook(columnOf(node));
  return (
    <>
      <span
        className="flex size-6 shrink-0 items-center justify-center rounded-md border"
        style={{
          color: look.accent,
          background: `${look.accent}1a`,
          borderColor: `${look.accent}40`,
        }}
      >
        <HugeiconsIcon icon={columnIcon(look)} size={13} strokeWidth={1.8} />
      </span>
      <span className="flex min-w-0 flex-1 flex-col leading-tight">
        <span className="truncate font-medium text-[13px]">
          {columnName(node) || node.label}
        </span>
        <span className="flow-detail truncate text-[10px] text-zinc-500">
          {look.subtitle}
        </span>
      </span>
      {look.drop && (
        <span
          className="flow-detail shrink-0 rounded bg-white/5 px-1 text-[9px] text-zinc-400"
          title="drop=true: used while generating, left out of the dataset"
        >
          dropped
        </span>
      )}
      {look.kind === "llm" && (
        <Handle
          type="target"
          id={AUX_IN}
          position={Position.Bottom}
          isConnectable={false}
          className="!h-1.5 !w-1.5 !min-h-0 !min-w-0 !border-0 !opacity-0"
        />
      )}
    </>
  );
}

export function columnAccent(node: FlowNode): string {
  return columnLook(columnOf(node)).accent;
}

const AUX_ICON: Record<AuxCard["kind"], Icon> = {
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
  const accent = RECIPE_ACCENT[card.kind];
  return (
    <div
      className="w-56 rounded-lg border border-white/10 bg-zinc-900/95 text-zinc-100 shadow-md shadow-black/30"
      title="Double-click to edit"
    >
      {card.kind === "model_config" && (
        <Handle
          type="target"
          id={AUX_IN}
          position={Position.Top}
          isConnectable={false}
          className="!h-1.5 !w-1.5 !min-h-0 !min-w-0 !border-0 !opacity-0"
        />
      )}
      <div
        className="flex items-center gap-2 rounded-t-lg px-2.5 py-1.5"
        style={{ background: `linear-gradient(90deg, ${accent}26, transparent 80%)` }}
      >
        <span
          className="flex size-5 shrink-0 items-center justify-center rounded"
          style={{ color: accent, background: `${accent}1f` }}
        >
          <HugeiconsIcon icon={AUX_ICON[card.kind]} size={12} strokeWidth={1.8} />
        </span>
        <span className="flex min-w-0 flex-col leading-tight">
          <span className="truncate font-mono text-[12px]">{card.name}</span>
          <span className="truncate text-[10px] text-zinc-500">
            {card.subtitle}
          </span>
        </span>
      </div>
      {card.fields.length > 0 && (
        <div className="flex flex-col gap-0.5 px-2.5 py-1.5">
          {card.fields.map((field) => (
            <div
              key={field.label}
              className="flex min-w-0 items-baseline gap-1.5 text-[11px]"
            >
              <span className="shrink-0 text-[9px] uppercase tracking-wide text-zinc-500">
                {field.label}
              </span>
              <span className="truncate font-mono text-zinc-300">
                {field.value}
              </span>
            </div>
          ))}
        </div>
      )}
      <Handle
        type="source"
        id={AUX_OUT}
        position={card.kind === "model_provider" ? Position.Bottom : Position.Left}
        isConnectable={false}
        className="!h-1.5 !w-1.5 !min-h-0 !min-w-0 !border-0 !opacity-0"
      />
    </div>
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
        stroke: RECIPE_ACCENT.model_config,
        strokeWidth: 1.25,
        strokeDasharray: "4 4",
        opacity: 0.55,
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
