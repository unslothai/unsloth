import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import {
  Field,
  FieldDescription,
  FieldGroup,
  FieldLabel,
} from "@/components/ui/field";
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
import { cn } from "@/lib/utils";
import { useWizardStore } from "@/stores/training";
import type { DatasetFormat } from "@/types/training";
import {
  InformationCircleIcon,
  Key01Icon,
  Search01Icon,
  Upload04Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMemo, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

const FORMAT_OPTIONS: { value: DatasetFormat; label: string }[] = [
  { value: "auto", label: "Auto Detect" },
  { value: "alpaca", label: "Alpaca" },
  { value: "chatml", label: "ChatML" },
  { value: "sharegpt", label: "ShareGPT" },
];

export function DatasetStep() {
  const {
    hfToken,
    setHfToken,
    datasetSource,
    setDatasetSource,
    datasetFormat,
    setDatasetFormat,
    dataset,
    setDataset,
    uploadedFile,
    setUploadedFile,
  } = useWizardStore(
    useShallow((s) => ({
      hfToken: s.hfToken,
      setHfToken: s.setHfToken,
      datasetSource: s.datasetSource,
      setDatasetSource: s.setDatasetSource,
      datasetFormat: s.datasetFormat,
      setDatasetFormat: s.setDatasetFormat,
      dataset: s.dataset,
      setDataset: s.setDataset,
      uploadedFile: s.uploadedFile,
      setUploadedFile: s.setUploadedFile,
    })),
  );

  const sortedDatasets = useMemo(
    () =>
      // Sort recommended first
      [...DATASETS].sort(
        (a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0),
      ),
    [],
  );

  const selectedDatasetData = DATASETS.find((d) => d.id === dataset);
  const comboboxAnchorRef = useRef<HTMLDivElement>(null);

  const handleFileUpload = () => {
    // Mock file upload
    setUploadedFile("my_dataset.jsonl");
  };

  return (
    <FieldGroup>
      {/* Source Toggle */}
      <Field>
        <FieldLabel>Source</FieldLabel>
        <div className="flex gap-2">
          <Button
            variant={datasetSource === "huggingface" ? "dark" : "outline"}
            onClick={() => setDatasetSource("huggingface")}
            className="flex-1"
          >
            <img
              src="/huggingface.svg"
              alt=""
              className="size-4 invert"
              data-icon="inline-start"
            />
            Hugging Face
          </Button>
          <Button
            variant={datasetSource === "upload" ? "dark" : "outline"}
            onClick={() => setDatasetSource("upload")}
            className="flex-1"
          >
            <HugeiconsIcon icon={Upload04Icon} data-icon="inline-start" />
            Upload
          </Button>
        </div>
      </Field>

      {datasetSource === "huggingface" ? (
        <>
          <Field>
            <FieldLabel>
              Hugging Face Token{" "}
              <span className="text-muted-foreground font-normal">
                (Optional)
              </span>
            </FieldLabel>
            <FieldDescription>
              Required for gated or private datasets.{" "}
              <a
                href="https://huggingface.co/settings/tokens"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary hover:underline"
              >
                Get token
              </a>
            </FieldDescription>
            <InputGroup>
              <InputGroupAddon>
                <HugeiconsIcon icon={Key01Icon} className="size-4" />
              </InputGroupAddon>
              <InputGroupInput
                type="password"
                placeholder="hf_..."
                value={hfToken}
                onChange={(e) => setHfToken(e.target.value)}
              />
            </InputGroup>
          </Field>

          <Field>
            <FieldLabel>Search datasets</FieldLabel>
            <div ref={comboboxAnchorRef}>
              <Combobox
                items={sortedDatasets.map((d) => d.name)}
                value={selectedDatasetData?.name ?? null}
                onValueChange={(name) => {
                  const ds = sortedDatasets.find((d) => d.name === name);
                  if (ds) {
                    setDataset(ds.id);
                  }
                }}
                autoHighlight={true}
              >
                <ComboboxInput
                  placeholder="Search by name..."
                  className="w-full"
                >
                  <InputGroupAddon>
                    <HugeiconsIcon icon={Search01Icon} className="size-4" />
                  </InputGroupAddon>
                </ComboboxInput>
                <ComboboxContent anchor={comboboxAnchorRef}>
                  <ComboboxEmpty>No datasets found</ComboboxEmpty>
                  <ComboboxList className="p-1">
                    {(name: string) => {
                      const ds = sortedDatasets.find((d) => d.name === name);
                      return (
                        <ComboboxItem key={name} value={name}>
                          <div className="flex flex-col gap-0.5 flex-1">
                            <span>{name}</span>
                            {ds && (
                              <span className="text-xs text-muted-foreground">
                                {ds.description}
                              </span>
                            )}
                          </div>
                          {ds && (
                            <Badge variant="outline" className="ml-auto">
                              {ds.size}
                            </Badge>
                          )}
                        </ComboboxItem>
                      );
                    }}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </div>
          </Field>
        </>
      ) : (
        <>
          <Field>
            <FieldLabel>Upload Dataset</FieldLabel>
            <FieldDescription>
              Supports JSONL, JSON, CSV formats
            </FieldDescription>
            <button
              type="button"
              className={cn(
                "border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer hover:border-primary/50 hover:bg-muted/50",
                uploadedFile && "border-primary/50 bg-primary/5",
              )}
              onClick={handleFileUpload}
            >
              {uploadedFile ? (
                <div className="flex flex-col items-center gap-2">
                  <Badge variant="secondary" className="text-sm">
                    {uploadedFile}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    Click to replace
                  </span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-2">
                  <HugeiconsIcon
                    icon={Upload04Icon}
                    className="size-8 text-muted-foreground"
                  />
                  <span className="text-sm text-muted-foreground">
                    Click to upload or drag and drop
                  </span>
                </div>
              )}
            </button>
          </Field>
        </>
      )}

      {/* Format Selection */}
      <Field>
        <div className="flex items-center justify-between">
          <FieldLabel className="flex items-center gap-1.5">
            Format
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  className="text-muted-foreground/50 hover:text-muted-foreground"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-3.5"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Auto will try to identify and convert your dataset to a
                supported format.{" "}
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
          </FieldLabel>
          <Select
            value={datasetFormat}
            onValueChange={(v) => setDatasetFormat(v as DatasetFormat)}
          >
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {FORMAT_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </Field>
    </FieldGroup>
  );
}
