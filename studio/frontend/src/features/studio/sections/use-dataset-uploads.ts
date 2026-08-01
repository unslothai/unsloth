// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { bumpInventoryVersion } from "@/features/hub";
import {
  consumeNativePathToken,
  registerNativeDatasetPath,
} from "@/features/native-intents";
import {
  formatUploadSize,
  getCachedUploadLimitBytes,
  getCachedUploadLimitLabel,
  loadUploadLimitSettings,
  subscribeUploadLimitSettings,
} from "@/features/settings";
import {
  uploadNativeTrainingDataset,
  uploadTrainingDataset,
  useTrainingConfigStore,
} from "@/features/training";
import {
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
  classifyNativeTrainingDatasetDrop,
  nativeDropPositionHitsBounds,
} from "@/features/training/lib/native-dataset-drop";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { getFileExtension } from "./dataset-panel-helpers";

const TRAINING_UPLOAD_EXTENSIONS = [
  ...TRAINING_DATASET_UPLOAD_EXTENSIONS,
  ...TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
] as const;
const TRAINING_UPLOAD_EXTENSION_SET = new Set<string>(
  TRAINING_UPLOAD_EXTENSIONS,
);
export const TRAINING_UPLOAD_ACCEPT = TRAINING_UPLOAD_EXTENSIONS.join(",");
const TRAINING_UPLOAD_LABEL = "CSV, JSONL, JSON, Parquet, PDF, DOCX, TXT";
const DOCUMENT_REDIRECT_EXTENSIONS = new Set<string>(
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
);
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
  const datasetDropTargetRef = useRef<HTMLButtonElement>(null);

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
            : typeof error === "string"
              ? error
              : t("studio.dataset.unknownError"),
      });
    } finally {
      setIsUploading(false);
    }
  };

  const uploadNativeFile = async (path: string, filename: string) => {
    const latestLimit = await getLatestUploadLimit();
    const intent = await registerNativeDatasetPath(path);
    if (
      intent.path.sizeBytes != null &&
      intent.path.sizeBytes > latestLimit.maxUploadSizeBytes
    ) {
      toast.error(t("studio.dataset.fileTooLarge"), {
        description: t("studio.dataset.fileTooLargeDescription", {
          file: filename,
          size: formatUploadSize(intent.path.sizeBytes),
          limit: latestLimit.maxUploadSizeLabel,
        }),
      });
      return;
    }
    setIsUploading(true);
    try {
      const grant = await consumeNativePathToken(
        intent.path.token,
        "dataset-import",
      );
      const uploaded = await uploadNativeTrainingDataset(grant.nativePathLease);
      bumpInventoryVersion();
      selectLocalDataset(uploaded.stored_path);
      toast.success(t("studio.dataset.datasetUploaded"), {
        description: uploaded.filename,
      });
    } catch (error) {
      toast.error(t("studio.dataset.uploadFailed"), {
        description:
          error instanceof Error
            ? error.message
            : typeof error === "string"
              ? error
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

  const nativeDropHandlerRef = useRef<(paths: string[]) => void>(
    () => undefined,
  );
  nativeDropHandlerRef.current = (paths) => {
    const dropped = classifyNativeTrainingDatasetDrop(paths);
    if (dropped.kind === "multiple") {
      toast.error(t("studio.dataset.uploadOneFileAtATime"), {
        description: t("studio.dataset.uploadSingleFileDescription"),
      });
      return;
    }
    if (dropped.kind === "unsupported") {
      toast.error(t("studio.dataset.unsupportedFileType"), {
        description: t("studio.dataset.uploadOneFileType", {
          types: TRAINING_UPLOAD_LABEL,
        }),
      });
      return;
    }
    if (dropped.kind === "document") {
      setRedirectFileName(dropped.filename);
      setDocumentRedirectOpen(true);
      return;
    }
    uploadNativeFile(dropped.path, dropped.filename).catch((error) => {
      toast.error(t("studio.dataset.uploadFailed"), {
        description:
          error instanceof Error
            ? error.message
            : typeof error === "string"
              ? error
              : t("studio.dataset.unknownError"),
      });
    });
  };

  useEffect(() => {
    if (!isTauri) {
      return;
    }
    let disposed = false;
    let unlisten: (() => void) | undefined;
    let scaleFactor = window.devicePixelRatio || 1;
    let eligible = false;
    const hitsTarget = (position: { x: number; y: number }) => {
      const target = datasetDropTargetRef.current;
      return (
        target != null &&
        nativeDropPositionHitsBounds(
          position,
          scaleFactor,
          target.getBoundingClientRect(),
        )
      );
    };

    void import("@tauri-apps/api/window")
      .then(async ({ getCurrentWindow }) => {
        const currentWindow = getCurrentWindow();
        scaleFactor = await currentWindow.scaleFactor();
        return currentWindow.onDragDropEvent((event) => {
          if (disposed) {
            return;
          }
          if (event.payload.type === "enter") {
            const dropped = classifyNativeTrainingDatasetDrop(
              event.payload.paths,
            );
            eligible =
              !isUploading &&
              (dropped.kind === "dataset" || dropped.kind === "document");
            setIsDatasetDragOver(
              eligible && hitsTarget(event.payload.position),
            );
            return;
          }
          if (event.payload.type === "over") {
            setIsDatasetDragOver(
              eligible && hitsTarget(event.payload.position),
            );
            return;
          }
          if (event.payload.type === "leave") {
            eligible = false;
            setIsDatasetDragOver(false);
            return;
          }
          const shouldHandle =
            !isUploading && hitsTarget(event.payload.position);
          eligible = false;
          setIsDatasetDragOver(false);
          if (shouldHandle) {
            nativeDropHandlerRef.current(event.payload.paths);
          }
        });
      })
      .then((cleanup) => {
        if (disposed) {
          cleanup();
        } else {
          unlisten = cleanup;
        }
      })
      .catch(() => undefined);

    return () => {
      disposed = true;
      unlisten?.();
    };
  }, [isUploading]);

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
    try {
      sessionStorage.setItem(OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY, "1");
    } catch {}
    setDocumentRedirectOpen(false);
    navigate({ to: "/data-recipes" }).catch(() => undefined);
  }, [navigate]);

  return {
    documentRedirectOpen,
    datasetDropTargetRef,
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
