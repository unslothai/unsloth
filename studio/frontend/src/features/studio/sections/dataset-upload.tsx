


import { PICKER_FOCUS_VISIBLE_CLASS } from "@/components/resource-picker/picker-focus";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { datasetDisplayName } from "@/features/dataset-picker";
import {
  TRAINING_DATASET_UPLOAD_ACCEPT,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  CloudUploadIcon,
  FileAttachmentIcon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useRef } from "react";
import { DocumentUploadRedirectDialog } from "./document-upload-redirect-dialog";
import {
  type DatasetUploads,
  TRAINING_UPLOAD_ACCEPT,
  TRAINING_UPLOAD_LABEL,
} from "./use-dataset-uploads";

export function DatasetUploadField({ uploads }: { uploads: DatasetUploads }) {
  const t = useT();
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="flex flex-col gap-2">
      <span className="flex items-center gap-1.5 text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {t("studio.wizard.uploadLocalLabel")}
        <Tooltip>
          <TooltipTrigger asChild={true}>
            <button
              type="button"
              aria-label={t("studio.dataset.uploadDetails")}
              className="text-foreground/70 hover:text-foreground"
            >
              <HugeiconsIcon icon={InformationCircleIcon} className="size-3" />
            </button>
          </TooltipTrigger>
          <TooltipContent className="max-w-xs">
            {t("studio.dataset.uploadDetailsTooltip", {
              limit: uploads.uploadLimitLabel,
            })}
          </TooltipContent>
        </Tooltip>
      </span>
      <button
        id={uploads.datasetDropTargetId}
        type="button"
        disabled={uploads.isUploading}
        onClick={() => fileInputRef.current?.click()}
        onDrop={uploads.handleDatasetDrop}
        onDragOver={uploads.handleDatasetDragOver}
        onDragLeave={() => uploads.setIsDatasetDragOver(false)}
        className={cn(
          "group relative flex h-9 w-full select-none items-center justify-center gap-2 rounded-[12px] border border-dashed px-3 text-center transition-colors",
          "border-foreground/15 dark:border-white/15",
          "hover:border-foreground/30 hover:bg-foreground/[0.02] dark:hover:border-white/30 dark:hover:bg-white/[0.025]",
          PICKER_FOCUS_VISIBLE_CLASS,
          uploads.isDatasetDragOver &&
            "border-foreground/45 bg-foreground/[0.04] dark:border-white/40 dark:bg-white/[0.05]",
          uploads.isUploading && "cursor-progress opacity-80",
        )}
      >
        {uploads.isUploading ? (
          <Spinner className="size-3.5 text-muted-foreground" />
        ) : (
          <HugeiconsIcon
            icon={CloudUploadIcon}
            strokeWidth={1.5}
            className="size-3.5 text-muted-foreground"
          />
        )}
        <span className="truncate text-ui-12p5 text-foreground/85">
          {uploads.isUploading
            ? t("studio.dataset.uploading")
            : uploads.isDatasetDragOver
              ? t("studio.wizard.releaseToUpload")
              : t("studio.dataset.dropFileOrClick")}
        </span>
      </button>
      <p className="truncate text-ui-10 text-muted-foreground">
        {TRAINING_UPLOAD_LABEL}
      </p>
      <input
        ref={fileInputRef}
        type="file"
        accept={TRAINING_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(event) => {
          uploads.handleDatasetFileChange(event).catch(() => undefined);
        }}
      />
      <DocumentUploadRedirectDialog
        open={uploads.documentRedirectOpen}
        onOpenChange={uploads.setDocumentRedirectOpen}
        fileName={uploads.redirectFileName}
        onOpenLearningRecipes={uploads.handleOpenLearningRecipes}
      />
    </div>
  );
}

export function EvaluationDatasetUpload({
  uploads,
}: {
  uploads: DatasetUploads;
}) {
  const t = useT();
  const evalFileInputRef = useRef<HTMLInputElement>(null);
  const datasetSource = useTrainingConfigStore((state) => state.datasetSource);
  const uploadedFile = useTrainingConfigStore((state) => state.uploadedFile);
  const uploadedEvalFile = useTrainingConfigStore(
    (state) => state.uploadedEvalFile,
  );
  const setUploadedEvalFile = useTrainingConfigStore(
    (state) => state.setUploadedEvalFile,
  );

  if (datasetSource !== "upload" || !uploadedFile) {
    return null;
  }

  return (
    <div className="rounded-lg border bg-muted/20 px-3.5 py-3">
      <p className="mb-2 text-xs font-medium text-muted-foreground">
        {t("studio.dataset.evalDataset")}
      </p>
      {uploadedEvalFile ? (
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 overflow-hidden">
            <HugeiconsIcon
              icon={FileAttachmentIcon}
              className="size-3.5 shrink-0 text-muted-foreground"
            />
            <span className="truncate text-xs">
              {datasetDisplayName(uploadedEvalFile)}
            </span>
          </div>
          <Button
            variant="ghost"
            size="sm"
            aria-label={`${t("studio.dataset.clear")} ${t(
              "studio.dataset.evalDataset",
            )}`}
            className="h-6 w-6 shrink-0 cursor-pointer p-0"
            onClick={() => setUploadedEvalFile(null)}
          >
            <HugeiconsIcon icon={Cancel01Icon} className="size-3.5" />
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          <Button
            variant="outline"
            size="sm"
            className="w-full cursor-pointer gap-1.5"
            disabled={uploads.isUploading}
            onClick={() => evalFileInputRef.current?.click()}
          >
            {uploads.isUploading ? (
              <Spinner className="size-3.5" />
            ) : (
              <HugeiconsIcon icon={CloudUploadIcon} className="size-3.5" />
            )}
            {uploads.isUploading
              ? t("studio.dataset.uploading")
              : t("studio.dataset.uploadEvalFile")}
          </Button>
          <p className="text-ui-10 text-muted-foreground/80">
            {t("studio.dataset.evalDatasetDescription")}
          </p>
        </div>
      )}
      <input
        ref={evalFileInputRef}
        type="file"
        accept={TRAINING_DATASET_UPLOAD_ACCEPT}
        className="hidden"
        onChange={(event) => {
          uploads.handleEvalFileChange(event).catch(() => undefined);
        }}
      />
    </div>
  );
}
