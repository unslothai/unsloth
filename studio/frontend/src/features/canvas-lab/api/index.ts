const DEFAULT_BASE = "http://127.0.0.1:8000";

export const CANVAS_LAB_API_BASE =
  import.meta.env.VITE_DATA_DESIGNER_API ?? DEFAULT_BASE;

type PreviewResponse = {
  dataset?: unknown[];
  processorArtifacts?: Record<string, unknown>;
};

export async function previewCanvas(
  payload: unknown,
): Promise<PreviewResponse> {
  const response = await fetch(`${CANVAS_LAB_API_BASE}/preview`, {
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
