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
  Database02Icon,
  Flowchart01Icon,
  PlusSignIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { ReactElement } from "react";
import type { LlmType, SamplerType } from "../types";

type SheetView = "root" | "sampler" | "llm" | "expression";
type SheetKind = "sampler" | "llm" | "expression";

type BlockSheetProps = {
  container: HTMLDivElement | null;
  view: SheetView;
  onViewChange: (view: SheetView) => void;
  onAddSampler: (type: SamplerType) => void;
  onAddLlm: (type: LlmType) => void;
  onAddExpression: () => void;
};

function getSheetTitle(view: SheetView): string {
  if (view === "root") {
    return "Add a block";
  }
  if (view === "sampler") {
    return "Sampler blocks";
  }
  if (view === "expression") {
    return "Expression blocks";
  }
  return "LLM blocks";
}

const MAIN_SHEET_ITEMS: Array<{
  kind: SheetKind;
  title: string;
  description: string;
  icon: typeof Database02Icon;
}> = [
  {
    kind: "sampler",
    title: "Sampler",
    description: "Numeric + categorical blocks.",
    icon: Database02Icon,
  },
  {
    kind: "llm",
    title: "LLM",
    description: "Text + structured blocks.",
    icon: SparklesIcon,
  },
  {
    kind: "expression",
    title: "Expression",
    description: "Derived columns with Jinja.",
    icon: CodeIcon,
  },
];

const SAMPLER_ITEMS = [
  {
    type: "category" as const,
    title: "Category",
    description: "Pick from a list of values.",
    icon: Database02Icon,
  },
  {
    type: "subcategory" as const,
    title: "Subcategory",
    description: "Map sub-values to a category.",
    icon: Database02Icon,
  },
  {
    type: "uniform" as const,
    title: "Uniform",
    description: "Random number between low/high.",
    icon: Database02Icon,
  },
  {
    type: "gaussian" as const,
    title: "Gaussian",
    description: "Normal distribution sampler.",
    icon: Database02Icon,
  },
  {
    type: "datetime" as const,
    title: "Datetime",
    description: "Date/time range sampler.",
    icon: Database02Icon,
  },
  {
    type: "uuid" as const,
    title: "UUID",
    description: "UUID string sampler.",
    icon: Database02Icon,
  },
  {
    type: "person" as const,
    title: "Person",
    description: "Synthetic person sampler.",
    icon: Database02Icon,
  },
];

const LLM_ITEMS = [
  {
    type: "text" as const,
    title: "LLM Text",
    description: "Free-form prompt generation.",
    icon: SparklesIcon,
  },
  {
    type: "structured" as const,
    title: "LLM Structured",
    description: "JSON output via schema.",
    icon: Flowchart01Icon,
  },
  {
    type: "code" as const,
    title: "LLM Code",
    description: "Generate code or SQL.",
    icon: CodeIcon,
  },
];

const EXPRESSION_ITEMS = [
  {
    title: "Expression",
    description: "Transform columns with Jinja.",
    icon: CodeIcon,
  },
];

function nextViewForKind(kind: SheetKind): SheetView {
  if (kind === "sampler") {
    return "sampler";
  }
  if (kind === "expression") {
    return "expression";
  }
  return "llm";
}

function BlockSheetButton({
  icon,
  title,
  description,
  onClick,
}: {
  icon: typeof Database02Icon;
  title: string;
  description: string;
  onClick: () => void;
}): ReactElement {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center gap-3 rounded-2xl border border-border/60 bg-white px-3 py-3 text-left transition hover:border-border hover:bg-muted/40"
    >
      <div className="flex size-9 items-center justify-center rounded-xl border border-border bg-muted/30 text-muted-foreground">
        <HugeiconsIcon icon={icon} className="size-4" />
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
  view,
  onViewChange,
  onAddSampler,
  onAddLlm,
  onAddExpression,
}: BlockSheetProps): ReactElement {
  const title = getSheetTitle(view);
  return (
    <Sheet>
      <SheetTrigger asChild={true}>
        <Button size="icon-sm" variant="secondary">
          <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        container={container}
        position="absolute"
        overlayPosition="absolute"
        className="absolute gap-0 p-0 shadow-none"
        overlayClassName="bg-transparent pointer-events-none"
      >
        <SheetHeader className="border-b border-border/60 px-6 py-5">
          <div className="flex items-center gap-2">
            {view !== "root" && (
              <Button
                type="button"
                variant="ghost"
                size="icon-sm"
                onClick={() => onViewChange("root")}
              >
                <HugeiconsIcon icon={ArrowLeft02Icon} className="size-4" />
              </Button>
            )}
            <SheetTitle>{title}</SheetTitle>
          </div>
        </SheetHeader>
        <div className="px-6 py-4">
          <div className="mt-4 flex flex-col gap-2">
            {view === "root" &&
              MAIN_SHEET_ITEMS.map((item) => (
                <BlockSheetButton
                  key={item.kind}
                  icon={item.icon}
                  title={item.title}
                  description={item.description}
                  onClick={() => onViewChange(nextViewForKind(item.kind))}
                />
              ))}
            {view === "sampler" &&
              SAMPLER_ITEMS.map((item) => (
                <BlockSheetButton
                  key={item.type}
                  icon={item.icon}
                  title={item.title}
                  description={item.description}
                  onClick={() => onAddSampler(item.type)}
                />
              ))}
            {view === "llm" &&
              LLM_ITEMS.map((item) => (
                <BlockSheetButton
                  key={item.type}
                  icon={item.icon}
                  title={item.title}
                  description={item.description}
                  onClick={() => onAddLlm(item.type)}
                />
              ))}
            {view === "expression" &&
              EXPRESSION_ITEMS.map((item) => (
                <BlockSheetButton
                  key={item.title}
                  icon={item.icon}
                  title={item.title}
                  description={item.description}
                  onClick={onAddExpression}
                />
              ))}
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
}
