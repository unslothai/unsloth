import { SectionCard } from "@/components/section-card";
import { Button } from "@/components/ui/button";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { DATASETS } from "@/config/training";
import { useWizardStore } from "@/stores/training";
import {
  ArrowDown01Icon,
  CloudUploadIcon,
  Database02Icon,
  FileAttachmentIcon,
  InformationCircleIcon,
  Search01Icon,
  ViewIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useState } from "react";
import { useShallow } from "zustand/react/shallow";

export function DatasetSection() {
  const { dataset, setDataset, datasetFormat, setDatasetFormat } =
    useWizardStore(
      useShallow((s) => ({
        dataset: s.dataset,
        setDataset: s.setDataset,
        datasetFormat: s.datasetFormat,
        setDatasetFormat: s.setDatasetFormat,
      })),
    );
  const [recOpen, setRecOpen] = useState(false);

  return (
    <SectionCard
      icon={<HugeiconsIcon icon={Database02Icon} className="size-5" />}
      title="Dataset"
      description="Select or upload training data"
      accent="indigo"
      className="lg:col-span-4 min-h-[450px]"
    >
      <div className="flex flex-col gap-4">
        {/* Load from Hub */}
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            Load from Hub
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-foreground/70 hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                Enter a Hugging Face dataset path like 'username/dataset-name'.{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  Read more
                </a>
              </TooltipContent>
            </Tooltip>
          </span>
          <InputGroup>
            <InputGroupAddon>
              <HugeiconsIcon icon={Search01Icon} className="size-4" />
            </InputGroupAddon>
            <InputGroupInput
              placeholder="yahma/alpaca-cleaned"
              value={dataset ?? ""}
              className=""
              onChange={(e) => setDataset(e.target.value || null)}
            />
          </InputGroup>
        </div>

        {/* Format */}
        <div className="flex flex-col gap-2">
          <span className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            Dataset Format
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-foreground/70 hover:text-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                Format of your training data. Auto-detect works for most
                datasets.{" "}
                <a
                  href="https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline"
                >
                  Read more
                </a>
              </TooltipContent>
            </Tooltip>
          </span>
          <Select
            value={datasetFormat}
            onValueChange={(v) => setDatasetFormat(v as typeof datasetFormat)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="auto">Auto-detect</SelectItem>
              <SelectItem value="alpaca">Alpaca</SelectItem>
              <SelectItem value="chatml">ChatML</SelectItem>
              <SelectItem value="sharegpt">ShareGPT</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Active dataset display */}
        {dataset ? (
          <div className="flex items-center gap-3 rounded-lg border bg-muted/40 px-3.5 py-3">
            <div className="rounded-md bg-indigo-500/10 p-1.5">
              <HugeiconsIcon
                icon={FileAttachmentIcon}
                className="size-4 text-indigo-500"
              />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-mono text-sm font-medium truncate">
                {dataset}
              </p>
              <p className="text-[10px] text-muted-foreground">
                Hugging Face Dataset
              </p>
            </div>
          </div>
        ) : (
          <div className="flex items-center gap-3 rounded-lg border border-dashed bg-muted/20 px-3.5 py-3">
            <HugeiconsIcon
              icon={Database02Icon}
              className="size-4 text-muted-foreground/40"
            />
            <span className="text-xs text-muted-foreground">
              No dataset selected
            </span>
          </div>
        )}

        {/* Recommended */}
        <Collapsible open={recOpen} onOpenChange={setRecOpen}>
          <CollapsibleTrigger className="flex w-full cursor-pointer items-center gap-1.5 text-xs text-muted-foreground">
            <HugeiconsIcon
              icon={ArrowDown01Icon}
              className={`size-3.5 transition-transform ${recOpen ? "rotate-180" : ""}`}
            />
            Common Datasets
          </CollapsibleTrigger>
          <CollapsibleContent className="mt-3 flex flex-col gap-1.5">
            {DATASETS.filter((d) => d.recommended).map((d) => (
              <button
                type="button"
                key={d.id}
                onClick={() => setDataset(d.id)}
                className="flex w-full corner-squircle cursor-pointer items-center gap-2.5 rounded-2xl border bg-muted/30 px-3 py-2.5 text-left text-sm transition-colors hover:bg-muted/60"
              >
                <HugeiconsIcon
                  icon={Database02Icon}
                  className="size-4 shrink-0 text-muted-foreground"
                />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-medium">{d.name}</p>
                  <p className="text-[10px] text-muted-foreground">
                    {d.description}
                  </p>
                </div>
                <span className="text-[10px] font-mono text-muted-foreground mt-0.5">
                  {d.size}
                </span>
              </button>
            ))}
          </CollapsibleContent>
        </Collapsible>

        {/* Action buttons */}
        <div className="grid grid-cols-2 gap-2">
          <Button
            variant="outline"
            size="sm"
            className="cursor-pointer gap-1.5"
          >
            <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
            Upload
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="cursor-pointer gap-1.5 text-muted-foreground"
          >
            <HugeiconsIcon icon={ViewIcon} className="size-3.5" />
            Preview
          </Button>
        </div>
      </div>
    </SectionCard>
  );
}
