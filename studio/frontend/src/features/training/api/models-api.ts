import { authFetch } from "@/features/auth";

interface VisionCheckResponse {
    model_name: string;
    is_vision: boolean;
}

/**
 * Check whether a model is a vision model by asking the backend.
 * Calls GET /api/models/check-vision/{model_name}.
 */
export async function checkVisionModel(modelName: string): Promise<boolean> {
    const encoded = encodeURIComponent(modelName);
    const response = await authFetch(`/api/models/check-vision/${encoded}`);
    if (!response.ok) {
        // If the check fails (e.g. network error), default to non-vision
        return false;
    }
    const data = (await response.json()) as VisionCheckResponse;
    return data.is_vision;
}
