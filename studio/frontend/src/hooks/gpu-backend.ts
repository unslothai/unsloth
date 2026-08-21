type InferenceSystemSnapshot = {
  device_backend?: string;
  inference_gpu?: { backend?: string };
};

export function inferenceBackendFromSystem(
  data: InferenceSystemSnapshot | null,
): string {
  return data?.inference_gpu?.backend ?? data?.device_backend ?? "";
}
