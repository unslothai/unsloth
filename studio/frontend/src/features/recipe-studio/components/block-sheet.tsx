import { Button } from "@/components/ui/button";
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
  PlusSignIcon,
  Tick02Icon,
  Upload01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo, useState } from "react";
import { RECIPE_FLOATING_ICON_BUTTON_CLASS } from "./recipe-floating-icon-button-class";
import type { LlmType, SamplerType } from "../types";
import {
  BLOCK_GROUPS,
  getBlocksForKind,
  type SeedBlockType,
} from "../blocks/registry";

type SheetView =
  | "root"
  | "sampler"
  | "seed"
  | "llm"
  | "expression"
  | "note"
  | "processor";
type SheetKind = "sampler" | "seed" | "llm" | "expression" | "note";
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
  onAddExpression: () => void;
  onAddMarkdownNote: () => void;
  onOpenProcessors: () => void;
  copied: boolean;
  onCopy: () => void;
  onImport: () => void;
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

function BlockSheetButton({
  icon,
  title,
  description,
  onClick,
  isActive = false,
}: {
  icon: typeof Database02Icon;
  title: string;
  description: string;
  onClick: () => void;
  isActive?: boolean;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`flex w-full items-center gap-3 border-l-2 bg-background px-3 py-3 text-left transition hover:bg-muted/35 ${
        isActive
          ? "border-emerald-500"
          : "border-transparent hover:border-border/60"
      }`}
    >
      <div className="flex size-9 items-center justify-center rounded-xl text-foreground/70">
        <HugeiconsIcon icon={icon} className="size-5" />
      </div>
      <div className="flex-1">
        <p className="text-sm font-semibold text-foreground">{title}</p>
        <p className="text-[11px] text-muted-foreground">{description}</p>
      </div>
      <HugeiconsIcon
        icon={ArrowRight01Icon}
        className="size-3.5 text-muted-foreground"
      />
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
  onAddExpression,
  onAddMarkdownNote,
  onOpenProcessors,
  copied,
  onCopy,
  onImport,
}: BlockSheetProps): ReactElement {
  const sheetTitle = getSheetTitle(sheetView);
  const [uncontrolledOpen, setUncontrolledOpen] = useState(false);
  const expressionBlocks = useMemo(() => getBlocksForKind("expression"), []);
  const noteBlocks = useMemo(() => getBlocksForKind("note"), []);
  const seedBlocks = useMemo(() => getBlocksForKind("seed"), []);
  const isControlled = typeof open === "boolean";
  const sheetOpen = isControlled ? (open as boolean) : uncontrolledOpen;

  const setSheetOpen = (nextOpen: boolean) => {
    if (!isControlled) {
      setUncontrolledOpen(nextOpen);
    }
    onOpenChange?.(nextOpen);
  };

  return (
    <div className="flex flex-col items-end gap-2">
      <Sheet
        open={sheetOpen}
        onOpenChange={(nextOpen) => {
          setSheetOpen(nextOpen);
          if (nextOpen) {
            onViewChange("root");
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
          <SheetHeader className="border-b border-border/60 px-6 py-5">
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
          </SheetHeader>
          <div className=" py-4">
            <div className="mt-4 flex flex-col gap-2">
              {sheetView === "root" &&
                ROOT_GROUPS.map((item, index) => (
                  <BlockSheetButton
                    key={item.kind}
                    icon={item.icon}
                    title={item.title}
                    description={item.description}
                    isActive={index === 0}
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
              {sheetView === "processor" && (
                <BlockSheetButton
                  icon={CodeIcon}
                  title="Schema Transform"
                  description="Transform final dataset schema."
                  isActive={true}
                  onClick={onOpenProcessors}
                />
              )}
              {sheetView !== "root" &&
                sheetView !== "processor" &&
                getBlocksForKind(VIEW_KIND[sheetView] ?? "sampler").map(
                  (item, index) => (
                    <BlockSheetButton
                      key={item.type}
                      icon={item.icon}
                      title={item.title}
                      description={item.description}
                      isActive={index === 0}
                      onClick={() => {
                        setSheetOpen(false);
                        if (item.kind === "sampler") {
                          onAddSampler(item.type as SamplerType);
                        } else if (item.kind === "seed") {
                          onAddSeed(item.type as SeedBlockType);
                        } else if (item.kind === "llm") {
                          if (item.type === "model_provider") {
                            onAddModelProvider();
                          } else if (item.type === "model_config") {
                            onAddModelConfig();
                          } else {
                            onAddLlm(item.type as LlmType);
                          }
                        } else if (item.kind === "expression") {
                          onAddExpression();
                        } else {
                          onAddMarkdownNote();
                        }
                      }}
                    />
                  ),
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
