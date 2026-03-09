// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  ArrowLeft02Icon,
  ArrowRight01Icon,
  CodeIcon,
  Copy02Icon,
  type Database02Icon,
  DragDropVerticalIcon,
  PlusSignIcon,
  Search01Icon,
  Tick02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type DragEvent as ReactDragEvent,
  type ReactElement,
  useMemo,
  useState,
} from "react";
import { RECIPE_FLOATING_ICON_BUTTON_CLASS } from "./recipe-floating-icon-button-class";
import type { LlmType, SamplerType } from "../types";
import {
  BLOCK_GROUPS,
  getBlocksForKind,
  type BlockType,
  type SeedBlockType,
} from "../blocks/registry";

type SheetView =
  | "root"
  | "sampler"
  | "seed"
  | "llm"
  | "validator"
  | "expression"
  | "note"
  | "processor";
type SheetKind =
  | "sampler"
  | "seed"
  | "llm"
  | "validator"
  | "expression"
  | "note";
type RootSheetView = Exclude<SheetView, "root">;
type RootGroup = {
  kind: RootSheetView;
  title: string;
  description: string;
  icon: typeof Database02Icon;
};

type BlockSheetProps = {
  container: HTMLDivElement | null;
  sheetView: SheetView;
  onViewChange: (sheetView: SheetView) => void;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  onAddSampler: (type: SamplerType) => void;
  onAddSeed: (type: SeedBlockType) => void;
  onAddLlm: (type: LlmType) => void;
  onAddModelProvider: () => void;
  onAddModelConfig: () => void;
  onAddToolProfile: () => void;
  onAddExpression: () => void;
  onAddValidator: (
    type: "validator_python" | "validator_sql" | "validator_oxc",
  ) => void;
  onAddMarkdownNote: () => void;
  onOpenProcessors: () => void;
  copied: boolean;
  onCopy: () => void;
  onImport: () => void;
};

export const RECIPE_BLOCK_DND_MIME = "application/x-recipe-studio-block";
export type RecipeBlockDragPayload = {
  kind: SheetKind;
  type: BlockType;
};

function getSheetTitle(sheetView: SheetView): string {
  if (sheetView === "root") {
    return "Add a block";
  }
  if (sheetView === "sampler") {
    return "Sampler blocks";
  }
  if (sheetView === "seed") {
    return "Seed blocks";
  }
  if (sheetView === "expression") {
    return "Expression blocks";
  }
  if (sheetView === "validator") {
    return "Validator blocks";
  }
  if (sheetView === "note") {
    return "Note blocks";
  }
  if (sheetView === "processor") {
    return "Processor blocks";
  }
  return "LLM blocks";
}

const VIEW_KIND: Record<SheetView, SheetKind | null> = {
  root: null,
  sampler: "sampler",
  seed: "seed",
  llm: "llm",
  validator: "validator",
  expression: "expression",
  note: "note",
  processor: null,
};

const ROOT_GROUPS: RootGroup[] = [
  ...BLOCK_GROUPS,
  {
    kind: "processor",
    title: "Processors",
    description: "Output schema + post batch.",
    icon: CodeIcon,
  },
];
const SEARCHABLE_KINDS: SheetKind[] = [
  "sampler",
  "seed",
  "llm",
  "validator",
  "expression",
  "note",
];
const PROCESSOR_TITLE = "Schema Transform";
const PROCESSOR_DESCRIPTION = "Transform final dataset schema.";

function BlockSheetButton({
  icon,
  title,
  description,
  onClick,
  isActive = false,
  draggable = false,
  onDragStart,
  trailing = "chevron",
  disabled = false,
  badge,
}: {
  icon: typeof Database02Icon;
  title: string;
  description: string;
  onClick: () => void;
  isActive?: boolean;
  draggable?: boolean;
  onDragStart?: (event: ReactDragEvent<HTMLButtonElement>) => void;
  trailing?: "chevron" | "drag" | "none";
  disabled?: boolean;
  badge?: string;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      draggable={disabled ? false : draggable}
      onDragStart={disabled ? undefined : onDragStart}
      className={`flex w-full items-center gap-3 border-l-2 bg-background px-3 py-3 text-left transition ${
        disabled ? "cursor-not-allowed opacity-60" : "hover:bg-muted/35"
      } ${
        isActive
          ? "border-emerald-500"
          : disabled
            ? "border-transparent"
            : "border-transparent hover:border-border/60"
      } ${draggable ? "cursor-grab active:cursor-grabbing" : ""}`}
    >
      <div className="flex size-9 items-center justify-center rounded-xl text-foreground/70">
        <HugeiconsIcon icon={icon} className="size-5" />
      </div>
      <div className="flex-1">
        <div className="flex items-center gap-2">
          <p className="text-sm font-semibold text-foreground">{title}</p>
          {badge ? (
            <Badge variant="outline" className="rounded-full text-[10px]">
              {badge}
            </Badge>
          ) : null}
        </div>
        <p className="text-[11px] text-muted-foreground">{description}</p>
      </div>
      {trailing === "chevron" ? (
        <HugeiconsIcon
          icon={ArrowRight01Icon}
          className="size-3.5 text-muted-foreground"
        />
      ) : trailing === "drag" ? (
        <HugeiconsIcon
          icon={DragDropVerticalIcon}
          strokeWidth={3.5}
          className="size-5 text-foreground"
        />
      ) : null}
    </button>
  );
}

export function BlockSheet({
  container,
  sheetView,
  onViewChange,
  open,
  onOpenChange,
  onAddSampler,
  onAddSeed,
  onAddLlm,
  onAddModelProvider,
  onAddModelConfig,
  onAddToolProfile,
  onAddExpression,
  onAddValidator,
  onAddMarkdownNote,
  onOpenProcessors,
  copied,
  onCopy,
  onImport,
}: BlockSheetProps): ReactElement {
  const sheetTitle = getSheetTitle(sheetView);
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const [search, setSearch] = useState("");
  const expressionBlocks = useMemo(() => getBlocksForKind("expression"), []);
  const noteBlocks = useMemo(() => getBlocksForKind("note"), []);
  const seedBlocks = useMemo(() => getBlocksForKind("seed"), []);
  const isControlled = typeof open === "boolean";
  const sheetOpen = isControlled ? (open as boolean) : uncontrolledOpen;
  const normalizedSearch = search.trim().toLowerCase();
  const hasSearch = normalizedSearch.length > 0;
  const isProcessorView = sheetView === "processor";
  const isRootView = sheetView === "root";
  const isScopedBlockView = !isRootView && !isProcessorView;

  const setSheetOpen = (nextOpen: boolean) => {
    if (!isControlled) {
      setUncontrolledOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);
  };
  const matchesSearch = (title: string, description: string) =>
    title.toLowerCase().includes(normalizedSearch) ||
    description.toLowerCase().includes(normalizedSearch);

  const searchableBlocks = useMemo(
    () => SEARCHABLE_KINDS.flatMap((kind) => getBlocksForKind(kind)),
    [],
  );
  const rootSearchBlocks = useMemo(() => {
    if (!hasSearch) {
      return [];
    }
    return searchableBlocks.filter((item) =>
      matchesSearch(item.title, item.description),
    );
  }, [hasSearch, searchableBlocks, normalizedSearch]);

  const scopedBlocks = useMemo(() => {
    if (!isScopedBlockView) {
      return [];
    }
    const blocks = getBlocksForKind(VIEW_KIND[sheetView] ?? "sampler");
    if (!hasSearch) {
      return blocks;
    }
    return blocks.filter((item) => matchesSearch(item.title, item.description));
  }, [hasSearch, isScopedBlockView, normalizedSearch, sheetView]);

  const rootGroups = useMemo(() => {
    if (!hasSearch) {
      return ROOT_GROUPS;
    }
    return ROOT_GROUPS.filter((group) => {
      if (matchesSearch(group.title, group.description)) {
        return true;
      }
      if (group.kind === "processor") {
        return matchesSearch(PROCESSOR_TITLE, PROCESSOR_DESCRIPTION);
      }
      return getBlocksForKind(group.kind).some((item) =>
        matchesSearch(item.title, item.description),
      );
    });
  }, [hasSearch, normalizedSearch]);
  const showNoMatches =
    (isRootView && hasSearch && rootSearchBlocks.length === 0) ||
    (isScopedBlockView && scopedBlocks.length === 0) ||
    (isProcessorView &&
      hasSearch &&
      !matchesSearch(PROCESSOR_TITLE, PROCESSOR_DESCRIPTION));

  const buildDragStart =
    (kind: SheetKind, type: BlockType) =>
    (event: ReactDragEvent<HTMLButtonElement>) => {
      const payload: RecipeBlockDragPayload = { kind, type };
      const serialized = JSON.stringify(payload);
      event.dataTransfer.setData(RECIPE_BLOCK_DND_MIME, serialized);
      event.dataTransfer.setData("text/plain", serialized);
      event.dataTransfer.effectAllowed = "copy";
    };
  const getTrailing = (_kind: SheetKind): "drag" => "drag";
  const onBlockClick = (kind: SheetKind, type: BlockType) => {
    setSheetOpen(false);
    if (kind === "sampler") {
      onAddSampler(type as SamplerType);
      return;
    }
    if (kind === "seed") {
      onAddSeed(type as SeedBlockType);
      return;
    }
    if (kind === "llm") {
      if (type === "model_provider") {
        onAddModelProvider();
        return;
      }
      if (type === "model_config") {
        onAddModelConfig();
        return;
      }
      if (type === "tool_config") {
        onAddToolProfile();
        return;
      }
      onAddLlm(type as LlmType);
      return;
    }
    if (kind === "validator") {
      onAddValidator(
        type as "validator_python" | "validator_sql" | "validator_oxc",
      );
      return;
    }
    if (kind === "expression") {
      onAddExpression();
      return;
    }
    onAddMarkdownNote();
  };

  return (
    <div className="flex flex-col items-end gap-2">
      <Sheet
        open={sheetOpen}
        onOpenChange={(nextOpen) => {
          setSheetOpen(nextOpen);
          if (nextOpen) {
            onViewChange("root");
            setSearch("");
          }
        }}
      >
        <SheetTrigger asChild={true}>
          <Button
            size="icon"
            className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
            variant="ghost"
          >
            <HugeiconsIcon
              icon={PlusSignIcon}
              className="size-5 text-muted-foreground group-hover:text-primary"
            />
          </Button>
        </SheetTrigger>
        <SheetContent
          side="right"
          container={container}
          position="absolute"
          overlayPosition="absolute"
          className="absolute gap-0 p-0 shadow-none"
          overlayClassName="bg-transparent pointer-events-none backdrop-blur-none supports-backdrop-filter:backdrop-blur-none"
        >
          <SheetHeader className="px-6 py-5">
            <div className="flex items-center gap-2">
              {sheetView !== "root" && (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => onViewChange("root")}
                >
                  <HugeiconsIcon icon={ArrowLeft02Icon} className="size-4" />
                </Button>
              )}
              <SheetTitle>{sheetTitle}</SheetTitle>
            </div>
            <div className="relative mt-3">
              <HugeiconsIcon
                icon={Search01Icon}
                className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
              />
              <Input
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search blocks..."
                className="corner-squircle h-9 pl-8"
              />
            </div>
          </SheetHeader>
          <div className=" py-4">
            <div className="mt-4 flex flex-col gap-2">
              {isRootView &&
                hasSearch &&
                rootSearchBlocks.map((item, index) => (
                  <BlockSheetButton
                    key={`${item.kind}:${item.type}`}
                    icon={item.icon}
                    title={item.title}
                    description={item.description}
                    isActive={index === 0}
                    draggable={true}
                    onDragStart={buildDragStart(item.kind, item.type)}
                    trailing={getTrailing(item.kind)}
                    onClick={() => onBlockClick(item.kind, item.type)}
                  />
                ))}
              {isRootView &&
                !hasSearch &&
                rootGroups.map((item, index) => (
                  <BlockSheetButton
                    key={item.kind}
                    icon={item.icon}
                    title={item.title}
                    description={item.description}
                    isActive={index === 0}
                    draggable={item.kind === "expression" || item.kind === "note"}
                    onDragStart={
                      item.kind === "expression" && expressionBlocks[0]
                        ? buildDragStart("expression", expressionBlocks[0].type)
                        : item.kind === "note" && noteBlocks[0]
                          ? buildDragStart("note", noteBlocks[0].type)
                          : undefined
                    }
                    trailing={
                      item.kind === "expression" || item.kind === "note"
                        ? "drag"
                        : item.kind === "processor"
                          ? "none"
                          : "chevron"
                    }
                    disabled={item.kind === "processor"}
                    badge={item.kind === "processor" ? "Work in progress" : undefined}
                    onClick={() => {
                      if (item.kind === "processor") {
                        setSheetOpen(false);
                        onOpenProcessors();
                        return;
                      }
                      if (item.kind === "seed" && seedBlocks.length === 1) {
                        setSheetOpen(false);
                        onAddSeed(seedBlocks[0].type as SeedBlockType);
                        return;
                      }
                      if (item.kind === "expression" && expressionBlocks.length === 1) {
                        setSheetOpen(false);
                        onAddExpression();
                        return;
                      }
                      if (item.kind === "note" && noteBlocks.length === 1) {
                        setSheetOpen(false);
                        onAddMarkdownNote();
                        return;
                      }
                      onViewChange(item.kind);
                    }}
                  />
                ))}
              {isProcessorView && (
                (!hasSearch ||
                  matchesSearch(PROCESSOR_TITLE, PROCESSOR_DESCRIPTION)) && (
                  <BlockSheetButton
                    icon={CodeIcon}
                    title={PROCESSOR_TITLE}
                    description={PROCESSOR_DESCRIPTION}
                    isActive={true}
                    onClick={onOpenProcessors}
                  />
                )
              )}
              {isScopedBlockView &&
                scopedBlocks.map(
                  (item, index) => (
                    <BlockSheetButton
                      key={item.type}
                      icon={item.icon}
                      title={item.title}
                      description={item.description}
                      isActive={index === 0}
                      draggable={true}
                      onDragStart={buildDragStart(item.kind, item.type)}
                      trailing={getTrailing(item.kind)}
                      onClick={() => onBlockClick(item.kind, item.type)}
                    />
                  ),
                )}
              {showNoMatches && (
                <p className="px-3 py-2 text-xs text-muted-foreground">
                  No blocks match.
                </p>
              )}
            </div>
          </div>
        </SheetContent>
      </Sheet>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={onImport}
      >
        <HugeiconsIcon
          icon={Upload01Icon}
          className="size-5 text-muted-foreground group-hover:text-primary"
        />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={onCopy}
      >
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy02Icon}
          className="size-5 text-muted-foreground group-hover:text-primary"
        />
      </Button>
    </div>
  );
}
