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
  TRAINING_DATASET_UPLOAD_EXTENSIONS,
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
  classifyNativeTrainingDatasetDrop,
  isTrainingDatasetUploadPath,
  nativeDropPositionHitsBounds,
  uploadNativeTrainingDataset,
  uploadTrainingDataset,
  useTrainingConfigStore,
} from "@/features/training";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { OPEN_LEARNING_RECIPES_ON_ARRIVAL_KEY } from "@/lib/navigation-intents";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import {
  type ChangeEvent,
  type DragEvent,
  useCallback,
  useEffect,
  useEffectEvent,
  useId,
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
export const TRAINING_UPLOAD_LABEL = TRAINING_UPLOAD_EXTENSIONS.map(
  (extension) => extension.slice(1).toUpperCase(),
).join(", ");
const TRAINING_DATASET_UPLOAD_LABEL = TRAINING_DATASET_UPLOAD_EXTENSIONS.map(
  (extension) => extension.slice(1).toUpperCase(),
).join(", ");
const DOCUMENT_REDIRECT_EXTENSIONS = new Set<string>(
  TRAINING_DOCUMENT_REDIRECT_EXTENSIONS,
);
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
  const uploadLockRef = useRef(false);
  const [isDatasetDragOver, setIsDatasetDragOver] = useState(false);
  const [uploadLimitBytes, setUploadLimitBytes] = useState(
    getCachedUploadLimitBytes,
  );
  const [uploadLimitLabel, setUploadLimitLabel] = useState(
    getCachedUploadLimitLabel,
  );
  const [documentRedirectOpen, setDocumentRedirectOpen] = useState(false);
  const [redirectFileName, setRedirectFileName] = useState<string | null>(null);
  const datasetDropTargetId = useId();

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

  const acquireUploadLock = () => {
    if (uploadLockRef.current) {
      return false;
    }
    uploadLockRef.current = true;
    setIsUploading(true);
    return true;
  };

  const releaseUploadLock = () => {
    uploadLockRef.current = false;
    setIsUploading(false);
  };

  const uploadFile = async (
    file: File,
    onSuccess: (storedPath: string) => void,
    successMessage: string,
  ) => {
    if (!acquireUploadLock()) {
      return;
    }
    try {
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
      releaseUploadLock();
    }
  };

  const uploadNativeFile = async (path: string, filename: string) => {
    if (!acquireUploadLock()) {
      return;
    }
    try {
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
      releaseUploadLock();
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
    if (uploadLockRef.current) {
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
    if (uploadLockRef.current) {
      return;
    }
    event.dataTransfer.dropEffect = "copy";
    setIsDatasetDragOver(true);
  };

  const handleNativeDrop = useEffectEvent((paths: string[]) => {
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
    void uploadNativeFile(dropped.path, dropped.filename);
  });
  const canHandleNativeDrop = useEffectEvent(() => !uploadLockRef.current);

  useEffect(() => {
    if (!isTauri) {
      return;
    }
    let disposed = false;
    let unlistenDragDrop: (() => void) | undefined;
    let unlistenScaleChanged: (() => void) | undefined;
    let scaleFactor = window.devicePixelRatio || 1;
    let scaleFactorRevision = 0;
    let eligible = false;
    const stopListening = () => {
      const stopDragDrop = unlistenDragDrop;
      const stopScaleChanged = unlistenScaleChanged;
      unlistenDragDrop = undefined;
      unlistenScaleChanged = undefined;
      stopDragDrop?.();
      stopScaleChanged?.();
    };
    const hitsTarget = (position: { x: number; y: number }) => {
      const target = document.getElementById(datasetDropTargetId);
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
        const stopScaleChanged = await currentWindow.onScaleChanged(
          ({ payload }) => {
            if (disposed) {
              return;
            }
            scaleFactor = payload.scaleFactor;
            scaleFactorRevision += 1;
          },
        );
        if (disposed) {
          stopScaleChanged();
          return;
        }
        unlistenScaleChanged = stopScaleChanged;

        const revisionBeforeRead = scaleFactorRevision;
        const currentScaleFactor = await currentWindow.scaleFactor();
        if (disposed) {
          return;
        }
        if (scaleFactorRevision === revisionBeforeRead) {
          scaleFactor = currentScaleFactor;
        }

        const stopDragDrop = await currentWindow.onDragDropEvent((event) => {
          if (disposed) {
            return;
          }
          if (event.payload.type === "enter") {
            const dropped = classifyNativeTrainingDatasetDrop(
              event.payload.paths,
            );
            eligible =
              canHandleNativeDrop() &&
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
            canHandleNativeDrop() && hitsTarget(event.payload.position);
          eligible = false;
          setIsDatasetDragOver(false);
          if (shouldHandle) {
            handleNativeDrop(event.payload.paths);
          }
        });
        if (disposed) {
          stopDragDrop();
        } else {
          unlistenDragDrop = stopDragDrop;
        }
      })
      .catch(stopListening);

    return () => {
      disposed = true;
      stopListening();
    };
  }, [datasetDropTargetId]);

  const handleEvalFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) {
      if (!isTrainingDatasetUploadPath(file.name)) {
        toast.error(t("studio.dataset.unsupportedFileType"), {
          description: t("studio.dataset.uploadOneFileType", {
            types: TRAINING_DATASET_UPLOAD_LABEL,
          }),
        });
        return;
      }
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
    } catch {
      // Continue when session storage is unavailable.
    }
    setDocumentRedirectOpen(false);
    navigate({ to: "/data-recipes" }).catch(() => undefined);
  }, [navigate]);

  return {
    datasetDropTargetId,
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
