// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Spinner } from "@/components/ui/spinner";
import {
  BaseNode,
  BaseNodeHeader,
  BaseNodeHeaderTitle,
} from "@/features/recipe-studio/components/rf-ui/base-node";
import { LabeledHandle } from "@/features/recipe-studio/components/rf-ui/labeled-handle";
import { RECIPE_STUDIO_WARNING_BADGE_TONE } from "@/features/recipe-studio/utils/ui-tones";
import { cn } from "@/lib/utils";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import { Handle, Position } from "@xyflow/react";
import { type ComponentProps, type ReactNode, useState } from "react";
import type { NodeRuntime } from "../stores/runtime-store";
import { CopyTextButton } from "./copy-text";
import { accentTone } from "./node-tones";
import { portInk } from "./theme-colors";

/** Recipe Studio's node card (BaseNode + header), shared by every Flows node. Per-type
 *  content plugs in through `node-looks.ts`. */

/** Centre of the two-line header: where flow-in and collapsed ports sit. */
export const HEADER_MID = 22;

// Done and cached cards keep the plain border; their badge says so. Green means hover or selection.
const STATUS_BORDER: Record<string, string> = {
  queued: "border-ring",
  waiting: "border-dashed border-amber-400/70 hover:border-amber-400/70",
  loading_model: "border-ring-strong hover:border-ring-strong shadow-md",
  running: "border-ring-strong hover:border-ring-strong shadow-md",
  failed:
    "border-destructive/60 hover:border-destructive/60 ring-1 ring-destructive/20",
  cancelled: "opacity-80",
  skipped: "opacity-60",
};
// Statuses whose own border colour wins over hover and selection.
const LOUD_STATUS = new Set(["waiting", "loading_model", "running", "failed"]);

const STATUS_LABEL: Record<string, string> = {
  queued: "Queued",
  waiting: "Waiting",
  loading_model: "Loading model",
  running: "Running",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
  skipped: "Skipped",
};

const BADGE = "h-5 px-1.5 text-ui-10";
const DONE_BADGE =
  "border-emerald-500/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300";

export function isBusy(runtime: NodeRuntime): boolean {
  return runtime.status === "running" || runtime.status === "loading_model";
}

export function NodeIconTile({
  icon,
  tone,
  accent,
}: { icon: IconSvgElement; tone?: string; accent?: string }) {
  return (
    <div
      className={cn(
        "corner-squircle flex size-7 shrink-0 items-center justify-center rounded-sm border",
        tone,
      )}
      style={tone || !accent ? undefined : accentTone(accent)}
    >
      <HugeiconsIcon icon={icon} className="size-3.5" />
    </div>
  );
}

export function NodeCard({
  runtime,
  disabled,
  lowDetail,
  sized,
  className,
  children,
  ...props
}: {
  runtime: NodeRuntime;
  disabled?: boolean;
  lowDetail?: boolean;
  /** Resized by hand: fill the stored box instead of the default width. */
  sized?: boolean;
  className?: string;
  children: ReactNode;
} & Omit<ComponentProps<"div">, "children">) {
  return (
    <BaseNode
      className={cn(
        "corner-squircle group/node min-w-0 overflow-visible border-border/60 shadow-sm",
        sized ? "flex h-full w-full flex-col" : "w-65",
        !LOUD_STATUS.has(runtime.status) &&
          "hover:border-emerald-500/60 in-[.selected]:border-emerald-500/80",
        // React Flow marks the wrapper .selected; BaseNode's escaped [.react-flow\_\_node.selected_&] form never matched.
        "in-[.selected]:ring-2 in-[.selected]:ring-emerald-500/25",
        STATUS_BORDER[runtime.status],
        disabled && "opacity-45",
        lowDetail && "[&_.flow-detail]:invisible",
        className,
      )}
      data-status={runtime.status}
      data-sized={sized || undefined}
      {...props}
    >
      {children}
      {isBusy(runtime) && (
        <Progress
          value={Math.round((runtime.progress ?? 0) * 100)}
          indeterminate={runtime.progress === null}
          className="absolute inset-x-0 bottom-0 h-1 rounded-none rounded-b-xl"
        />
      )}
    </BaseNode>
  );
}

export function NodeCardHeader({
  icon,
  tone,
  accent,
  title,
  subtitle,
  leading,
  children,
}: {
  icon: IconSvgElement;
  tone?: string;
  accent?: string;
  title: string;
  subtitle?: string;
  /** Handles anchored to the header, such as flow-in. */
  leading?: ReactNode;
  /** Right side: badges and actions. */
  children?: ReactNode;
}) {
  return (
    <BaseNodeHeader className="flow-node-drag relative cursor-grab border-b border-border/50 px-3 py-2 active:cursor-grabbing">
      {leading}
      <div className="flex min-w-0 flex-1 items-center gap-2">
        <NodeIconTile icon={icon} tone={tone} accent={accent} />
        <div className="min-w-0">
          <BaseNodeHeaderTitle className="truncate text-sm">
            {title}
          </BaseNodeHeaderTitle>
          {subtitle && (
            <p className="flow-detail truncate text-ui-11 text-muted-foreground">
              {subtitle}
            </p>
          )}
        </div>
      </div>
      {children && (
        <div className="flex shrink-0 items-center gap-1">{children}</div>
      )}
    </BaseNodeHeader>
  );
}

export function NodeStatusBadges({
  runtime,
  gpu,
}: { runtime: NodeRuntime; gpu?: boolean }) {
  if (runtime.cached && runtime.status !== "running")
    return (
      <Badge variant="outline" className={cn(BADGE, DONE_BADGE)}>
        Cached
      </Badge>
    );
  if (runtime.status === "idle")
    return gpu ? (
      <Badge variant="outline" className={cn(BADGE, "text-muted-foreground")}>
        GPU
      </Badge>
    ) : null;
  const label = STATUS_LABEL[runtime.status] ?? runtime.status;
  if (isBusy(runtime))
    return (
      <Spinner
        className="size-4 text-primary/90"
        label={label}
        data-testid="flow-node-spinner"
      />
    );
  return (
    <Badge
      variant={runtime.status === "failed" ? "destructive" : "outline"}
      className={cn(
        BADGE,
        runtime.status === "done" && DONE_BADGE,
        runtime.status === "waiting" && RECIPE_STUDIO_WARNING_BADGE_TONE,
        (runtime.status === "queued" ||
          runtime.status === "cancelled" ||
          runtime.status === "skipped") &&
          "text-muted-foreground",
      )}
      title={runtime.error ?? undefined}
    >
      {label}
    </Badge>
  );
}

export interface PortRowSpec {
  handleId: string;
  label: string;
  /** Port type colour (catalog `port_colors`); omitted for control-flow ports. */
  color?: string;
  title?: string;
  labelClassName?: string;
}

const DATA_HANDLE = "!size-2.5 !border-2 !border-card";
const FLOW_HANDLE =
  "!size-3 !rounded-[3px] !border-2 !border-foreground !bg-card";

function PortHandle({
  port,
  type,
}: { port: PortRowSpec; type: "target" | "source" }) {
  const position = type === "target" ? Position.Left : Position.Right;
  return (
    <LabeledHandle
      id={port.handleId}
      type={type}
      position={position}
      title={port.title ?? port.label}
      className="h-5 min-w-0 max-w-full"
      handleClassName={port.color ? DATA_HANDLE : FLOW_HANDLE}
      style={port.color ? { background: portInk(port.color) } : undefined}
      labelClassName={cn(
        "flow-detail min-w-0 truncate text-ui-11 text-muted-foreground",
        port.labelClassName,
      )}
    />
  );
}

/** Labelled ports, inputs on the left and outputs on the right. */
export function NodePorts({
  inputs,
  outputs,
}: { inputs: PortRowSpec[]; outputs: PortRowSpec[] }) {
  if (!inputs.length && !outputs.length) return null;
  return (
    <div className="flex justify-between gap-2">
      <div className="flex min-w-0 flex-col gap-1">
        {inputs.map((port) => (
          <PortHandle key={port.handleId} port={port} type="target" />
        ))}
      </div>
      <div className="flex min-w-0 flex-col items-end gap-1">
        {outputs.map((port) => (
          <PortHandle key={port.handleId} port={port} type="source" />
        ))}
      </div>
    </div>
  );
}

/** Ports of a collapsed node: invisible, stacked on the header's middle. */
export function CollapsedPorts({
  inputs,
  outputs,
}: { inputs: string[]; outputs: string[] }) {
  return (
    <div className="relative h-0">
      {inputs.map((id) => (
        <Handle
          key={id}
          type="target"
          id={id}
          position={Position.Left}
          className="!opacity-0"
          style={{ top: -HEADER_MID }}
        />
      ))}
      {outputs.map((id) => (
        <Handle
          key={id}
          type="source"
          id={id}
          position={Position.Right}
          className="!opacity-0"
          style={{ top: -HEADER_MID }}
        />
      ))}
    </div>
  );
}

export function FlowInHandle({ id }: { id: string }) {
  return (
    <Handle
      type="target"
      id={id}
      position={Position.Left}
      className={FLOW_HANDLE}
      style={{ top: HEADER_MID }}
    />
  );
}

export interface FieldRow {
  label: string;
  value: string;
}

/** Read-only key fields, like Recipe Studio's inline fields. */
export function NodeFieldRows({ rows }: { rows: FieldRow[] }) {
  if (!rows.length) return null;
  return (
    <div className="flow-detail flex flex-col gap-0.5 px-3">
      {rows.map((row) => (
        <div
          key={row.label}
          className="flex min-w-0 items-baseline gap-1.5 text-ui-11"
        >
          <span className="shrink-0 text-ui-10 font-semibold uppercase tracking-wide text-muted-foreground">
            {row.label}
          </span>
          <span className="truncate text-foreground">{row.value}</span>
        </div>
      ))}
    </div>
  );
}

/** Long text (a system prompt, a template) clamped on the card, with Recipe Studio's "Show inputs" for the rest. */
export function NodeDisclosure({
  sections,
}: { sections: { label: string; text: string }[] }) {
  const [open, setOpen] = useState(false);
  if (!sections.length) return null;
  return (
    <div className="flow-detail flex flex-col gap-1.5 px-3">
      {!open &&
        sections.map((section) => (
          <p
            key={section.label}
            data-testid="flow-node-long-text"
            className="line-clamp-2 whitespace-pre-wrap text-ui-11 leading-snug text-muted-foreground"
          >
            {section.text}
          </p>
        ))}
      <Button
        type="button"
        size="xs"
        variant="ghost"
        className="nodrag -ml-2 self-start text-muted-foreground"
        onClick={(event) => {
          event.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        {open ? "Hide inputs" : "Show inputs"}
      </Button>
      {open &&
        sections.map((section) => (
          <div
            key={section.label}
            className="nowheel max-h-40 overflow-auto rounded-md border border-border/60 bg-muted/40 px-2 py-1.5"
          >
            <div className="flex items-center justify-between gap-2">
              <p className="text-ui-10 font-semibold uppercase tracking-wide text-muted-foreground">
                {section.label}
              </p>
              <CopyTextButton
                text={section.text}
                label={`Copy ${section.label.toLowerCase()}`}
                className="-my-1 -mr-1 size-5"
              />
            </div>
            <p className="whitespace-pre-wrap text-ui-11 leading-snug text-foreground">
              {section.text}
            </p>
          </div>
        ))}
    </div>
  );
}

export function NodeError({ message }: { message: string }) {
  return (
    <div className="mx-3 flex items-start gap-1 rounded-md border border-destructive/40 bg-destructive/5 py-1 pr-1 pl-2">
      {/* The clamp sits on the text, not the padded box, so a fourth line can't show in the padding. */}
      <p className="line-clamp-3 min-w-0 flex-1 text-ui-11 text-destructive">
        {message}
      </p>
      <CopyTextButton text={message} label="Copy error" className="size-5 text-destructive/70" />
    </div>
  );
}
