const GPU_FALLBACK_CACHE_TYPES = new Set(["q4_1", "q5_0", "q5_1", "iq4_nl"]);

export function shouldWarnKvCacheGpuFallback({
  backend,
  cacheType,
  gpuMemoryMode,
  gpuLayers,
  isDiffusion,
}: {
  backend: string | null;
  cacheType: string;
  gpuMemoryMode: "auto" | "manual";
  gpuLayers: number | null | undefined;
  isDiffusion: boolean;
}): boolean {
  const placementWarns =
    gpuMemoryMode === "auto" ||
    (gpuMemoryMode === "manual" &&
      (gpuLayers == null || gpuLayers < 0 || gpuLayers > 0));
  return (
    !isDiffusion &&
    (backend === "cuda" || backend === "rocm") &&
    placementWarns &&
    GPU_FALLBACK_CACHE_TYPES.has(cacheType.trim().toLowerCase())
  );
}
