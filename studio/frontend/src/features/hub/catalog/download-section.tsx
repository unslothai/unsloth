


import type { ModelInventoryFormat } from "../inventory";
import { GgufDownloadCard } from "./gguf-download-card";
import { SafetensorsDownloadCard } from "./safetensors-download-card";

export function DownloadSection({
  repoId,
  isGguf,
  isDownloaded,
  isPartial = false,
  partialTransport = null,
  partialResumable = false,
  modelFormat,
  canRun = true,
  isActive,
  activeQuant,
  preferredGgufFile = null,

  preferredGgufFileIntent = 0,
  isLoadingThisModel,
  gpuGb,
  systemRamGb,
  cachePath,
  knownBytes,
  onLoad,
  onUseInChat,
  onEject,
  onTrain,
  onChange,
}: {
  repoId: string;
  isGguf: boolean;
  isDownloaded: boolean;
  isPartial?: boolean;
  partialTransport?: string | null;
  partialResumable?: boolean;
  modelFormat?: ModelInventoryFormat | null;
  canRun?: boolean;
  isActive: boolean;
  activeQuant: string | null;
  preferredGgufFile?: string | null;

  preferredGgufFileIntent?: number;
  isLoadingThisModel: boolean;
  gpuGb?: number;
  systemRamGb?: number;
  cachePath?: string | null;
  knownBytes?: number | null;
  onLoad: (opts: { ggufVariant?: string; expectedBytes?: number }) => void;
  onUseInChat?: () => void;
  onEject?: () => void;
  onTrain?: () => void;
  onChange?: () => void;
}) {
  if (isGguf || preferredGgufFile) {
    return (
      <GgufDownloadCard
        repoId={repoId}
        isActive={isActive}
        activeQuant={activeQuant}
        preferredFile={preferredGgufFile}

        preferredFileIntent={preferredGgufFileIntent}
        isLoadingThisModel={isLoadingThisModel}
        gpuGb={gpuGb}
        systemRamGb={systemRamGb}
        cachePath={cachePath}
        isPartial={isPartial}
        onLoad={onLoad}
        onUseInChat={onUseInChat}
        onEject={onEject}
        onChange={onChange}
      />
    );
  }
  return (
    <SafetensorsDownloadCard
      repoId={repoId}
      isDownloaded={isDownloaded}
      isPartial={isPartial}
      partialTransport={partialTransport}
      partialResumable={partialResumable}
      modelFormat={modelFormat}
      canRun={canRun}
      isActive={isActive}
      isLoadingThisModel={isLoadingThisModel}
      cachePath={cachePath}
      knownBytes={knownBytes}
      onLoad={onLoad}
      onUseInChat={onUseInChat}
      onEject={onEject}
      onTrain={onTrain}
      onChange={onChange}
    />
  );
}
