// src/mindmodel/loader.ts
import { access, readFile } from "node:fs/promises";
import { join } from "node:path";

import { config } from "../utils/config";
import { type MindmodelManifest, parseManifest } from "./types";

export interface LoadedMindmodel {
  directory: string;
  manifest: MindmodelManifest;
}

export interface LoadedExample {
  path: string;
  description: string;
  content: string;
}

export async function loadMindmodel(projectDir: string): Promise<LoadedMindmodel | null> {
  const mindmodelDir = join(projectDir, config.paths.mindmodelDir);

  try {
    await access(mindmodelDir);
  } catch {
    return null;
  }

  const manifestPath = join(mindmodelDir, config.paths.mindmodelManifest);

  try {
    const manifestContent = await readFile(manifestPath, "utf-8");
    const manifest = parseManifest(manifestContent);

    return {
      directory: mindmodelDir,
      manifest,
    };
  } catch (error) {
    console.warn(`[micode] Failed to load mindmodel manifest: ${error}`);
    return null;
  }
}

export async function loadExamples(mindmodel: LoadedMindmodel, categoryPaths: string[]): Promise<LoadedExample[]> {
  const examples: LoadedExample[] = [];

  for (const categoryPath of categoryPaths) {
    const category = mindmodel.manifest.categories.find((c) => c.path === categoryPath);
    if (!category) continue;

    const fullPath = join(mindmodel.directory, categoryPath);

    try {
      const content = await readFile(fullPath, "utf-8");
      examples.push({
        path: categoryPath,
        description: category.description,
        content,
      });
    } catch {
      console.warn(`[micode] Failed to load mindmodel example: ${categoryPath}`);
    }
  }

  return examples;
}
