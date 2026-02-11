const DEFAULT_BASE = "";

export const DATA_DESIGNER_API_BASE =
  import.meta.env.VITE_DATA_DESIGNER_API ?? DEFAULT_BASE;

type PreviewResponse = {
  dataset?: unknown[];
  processorArtifacts?: Record<string, unknown>;
};

export async function previewRecipe(
  payload: unknown,
): Promise<PreviewResponse> {
  const response = await fetch(`${DATA_DESIGNER_API_BASE}/preview`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || "Preview request failed.");
  }

  return response.json();
}
