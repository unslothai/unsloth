


interface InferenceGpuDiscovery {
  available: boolean;
  backend?: string;
}

export function shouldRetrySystemDiscovery(
  cacheIsCold: boolean,
  inferenceGpu: InferenceGpuDiscovery | undefined,
  retrySubscribers: number,
): boolean {
  if (retrySubscribers <= 0) {
    return false;
  }
  if (cacheIsCold) {
    return true;
  }
  return inferenceGpu?.backend === "vulkan" && !inferenceGpu.available;
}
