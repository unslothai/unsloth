// src/mindmodel/classifier.ts
import type { MindmodelManifest } from "./types";

export function buildClassifierPrompt(task: string, manifest: MindmodelManifest): string {
  const categoriesText = manifest.categories.map((c) => `- ${c.path}: ${c.description}`).join("\n");

  return `You are a task classifier. Given a coding task and a list of available example categories, determine which categories are relevant.

Task: "${task}"

Available categories:
${categoriesText}

Return ONLY a JSON array of relevant category paths. Example: ["components/form.md", "patterns/data-fetching.md"]

If no categories are relevant, return an empty array: []

Respond with ONLY the JSON array, no explanation.`;
}

export function parseClassifierResponse(response: string, manifest: MindmodelManifest): string[] {
  try {
    // Extract JSON array from response (handle markdown code blocks)
    const jsonMatch = response.match(/\[[\s\S]*?\]/);
    if (!jsonMatch) return [];

    const parsed = JSON.parse(jsonMatch[0]);
    if (!Array.isArray(parsed)) return [];

    // Validate paths exist in manifest
    const validPaths = new Set(manifest.categories.map((c) => c.path));
    return parsed.filter((p): p is string => typeof p === "string" && validPaths.has(p));
  } catch {
    return [];
  }
}
