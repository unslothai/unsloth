// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { bumpInventoryVersion } from "@/features/hub";
import {
  formatUploadSize,
  getCachedUploadLimitBytes,
  getCachedUploadLimitLabel,
  loadUploadLimitSettings,
  subscribeUploadLimitSettings,
} from "@/features/settings";
import {
  uploadTrainingDataset,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useState,
} from "react";
import { getFileExtension } from "./dataset-panel-helpers";

const TRAINING_UPLOAD_EXTENSIONS = [
  ".csv",
  ".jsonl",
  ".json",
  ".parquet",
  ".pdf",
  ".docx",
  ".txt",
] as const;
const TRAINING_UPLOAD_EXTENSION_SET = new Set<string>(
  TRAINING_UPLOAD_EXTENSIONS,
);
export const TRAINING_UPLOAD_ACCEPT = TRAINING_UPLOAD_EXTENSIONS.join(",");
const TRAINING_UPLOAD_LABEL = "CSV, JSONL, JSON, Parquet, PDF, DOCX, TXT";
const DOCUMENT_REDIRECT_EXTENSIONS = new Set([".pdf", ".docx", ".txt"]);
const OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY =
  "data-recipes:open-learning-recipes";

export function useDatasetUploads() {
  const t = useT();
  const navigate = useNavigate();
  const selectLocalDataset = useTrainingConfigStore(
    (state) => state.selectLocalDataset,
  );
  const setUploadedEvalFile = useTrainingConfigStore(
    (state) => state.setUploadedEvalFile,
  );
  const [isUploading, setIsUploading] = useState(false);
  const [isDatasetDragOver, setIsDatasetDragOver] = useState(false);
  const [uploadLimitBytes, setUploadLimitBytes] = useState(
    getCachedUploadLimitBytes,
  );
  const [uploadLimitLabel, setUploadLimitLabel] = useState(
    getCachedUploadLimitLabel,
  );
  const [documentRedirectOpen, setDocumentRedirectOpen] = useState(false);
  const [redirectFileName, setRedirectFileName] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const applyLimit = (settings: {
      maxUploadSizeBytes: number;
      maxUploadSizeLabel: string;
    }) => {
      setUploadLimitBytes(settings.maxUploadSizeBytes);
      setUploadLimitLabel(settings.maxUploadSizeLabel);
    };
    const unsubscribe = subscribeUploadLimitSettings(applyLimit);
    loadUploadLimitSettings()
      .then((settings) => {
        if (!cancelled) {
          applyLimit(settings);
        }
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  const getLatestUploadLimit = async () => {
    try {
      const settings = await loadUploadLimitSettings();
      setUploadLimitBytes(settings.maxUploadSizeBytes);
      setUploadLimitLabel(settings.maxUploadSizeLabel);
      return settings;
    } catch {
      return {
        maxUploadSizeBytes: uploadLimitBytes,
        maxUploadSizeLabel: uploadLimitLabel,
      };
    }
  };

  const uploadFile = async (
    file: File,
    onSuccess: (storedPath: string) => void,
    successMessage: string,
  ) => {
    const latestLimit = await getLatestUploadLimit();
    if (file.size > latestLimit.maxUploadSizeBytes) {
      toast.error(t("studio.dataset.fileTooLarge"), {
        description: t("studio.dataset.fileTooLargeDescription", {
          file: file.name,
          size: formatUploadSize(file.size),
          limit: latestLimit.maxUploadSizeLabel,
        }),
      });
      return;
    }
    setIsUploading(true);
    try {
      const uploaded = await uploadTrainingDataset(file);
      bumpInventoryVersion();
      onSuccess(uploaded.stored_path);
      toast.success(successMessage, { description: uploaded.filename });
    } catch (error) {
      toast.error(t("studio.dataset.uploadFailed"), {
        description:
          error instanceof Error
            ? error.message
            : t("studio.dataset.unknownError"),
      });
    } finally {
      setIsUploading(false);
    }
  };

  const handleDatasetFile = async (file: File) => {
    const extension = getFileExtension(file.name);
    if (!TRAINING_UPLOAD_EXTENSION_SET.has(extension)) {
      toast.error(t("studio.dataset.unsupportedFileType"), {
        description: t("studio.dataset.uploadOneFileType", {
          types: TRAINING_UPLOAD_LABEL,
        }),
      });
      return;
    }
    if (DOCUMENT_REDIRECT_EXTENSIONS.has(extension)) {
      setRedirectFileName(file.name);
      setDocumentRedirectOpen(true);
      return;
    }
    await uploadFile(
      file,
      selectLocalDataset,
      t("studio.dataset.datasetUploaded"),
    );
  };

  const handleDatasetFileChange = async (
    event: ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) {
      await handleDatasetFile(file);
    }
  };

  const handleDatasetDrop = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    setIsDatasetDragOver(false);
    if (isUploading) {
      return;
    }
    const files = Array.from(event.dataTransfer.files);
    if (files.length === 0) {
      return;
    }
    if (files.length > 1) {
      toast.error(t("studio.dataset.uploadOneFileAtATime"), {
        description: t("studio.dataset.uploadSingleFileDescription"),
      });
      return;
    }
    handleDatasetFile(files[0]).catch(() => undefined);
  };

  const handleDatasetDragOver = (event: DragEvent<HTMLButtonElement>) => {
    event.preventDefault();
    if (isUploading) {
      return;
    }
    event.dataTransfer.dropEffect = "copy";
    setIsDatasetDragOver(true);
  };

  const handleEvalFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) {
      await uploadFile(
        file,
        setUploadedEvalFile,
        t("studio.dataset.evalDatasetUploaded"),
      );
    }
  };

  const handleOpenLearningRecipes = useCallback(() => {
    sessionStorage.setItem(OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY, "1");
    setDocumentRedirectOpen(false);
    navigate({ to: "/data-recipes" }).catch(() => undefined);
  }, [navigate]);

  return {
    documentRedirectOpen,
    handleDatasetDragOver,
    handleDatasetDrop,
    handleDatasetFileChange,
    handleEvalFileChange,
    handleOpenLearningRecipes,
    isDatasetDragOver,
    isUploading,
    redirectFileName,
    setDocumentRedirectOpen,
    setIsDatasetDragOver,
    uploadLimitLabel,
  };
}

export type DatasetUploads = ReturnType<typeof useDatasetUploads>;
