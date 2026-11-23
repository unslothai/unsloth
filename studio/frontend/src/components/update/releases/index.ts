import release912 from "./v0.1.912-beta/notes.json";
import release913 from "./v0.1.913-beta/notes.json";
import release914 from "./v0.1.914-beta/notes.json";

interface Release {
  version: string;
  title: string;
  date: string;
  features: { image: string; alt: string; title: string; summary: string }[];
  changes: Partial<Record<"new" | "improved" | "fixed", string[][]>>;
}

// Artwork lives outside the frontend bundle and is requested only for the selected release.
export function releaseImageUrl(version: string, filename: string): string {
  return `https://raw.githubusercontent.com/unslothai/unsloth/main/images/releases/v${encodeURIComponent(version)}/${encodeURIComponent(filename)}`;
}

export const releases: Release[] = [release914, release913, release912];
