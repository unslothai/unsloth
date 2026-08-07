


export function shouldUseVisionDatasetCheck(
  state: {
    isVisionModel: boolean;
    isDatasetImage: boolean | null;
  },
  detectedIsImage = false,
): boolean {
  return (
    state.isVisionModel && (state.isDatasetImage === true || detectedIsImage)
  );
}
