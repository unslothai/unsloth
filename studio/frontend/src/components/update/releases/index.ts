import release905 from "./v0.1.905-beta/notes.json";
import release906 from "./v0.1.906-beta/notes.json";
import release907 from "./v0.1.907-beta/notes.json";
import release908 from "./v0.1.908-beta/notes.json";
import release909 from "./v0.1.909-beta/notes.json";
import release910 from "./v0.1.910-beta/notes.json";
import release911 from "./v0.1.911-beta/notes.json";
import release912 from "./v0.1.912-beta/notes.json";
import release913 from "./v0.1.913-beta/notes.json";
import release914 from "./v0.1.914-beta/notes.json";
import release915 from "./v0.1.915-beta/notes.json";
import release916 from "./v0.1.916-beta/notes.json";
import release917 from "./v0.1.917-beta/notes.json";
import release918 from "./v0.1.918-beta/notes.json";
import release919 from "./v0.1.919-beta/notes.json";
import release920 from "./v0.1.920-beta/notes.json";
import release921 from "./v0.1.921-beta/notes.json";
import release922 from "./v0.1.922-beta/notes.json";

type Row = [string, string];

interface Release {
  version: string;
  title: string;
  date: string;
  features: {
    image: string;
    // Artwork for the dark theme; falls back to `image`.
    imageDark?: string;
    alt: string;
    title: string;
    summary: string;
    // Docs page, and where the card's button goes (a route, or `settings:<tab>`) with its label.
    link?: string;
    open?: string;
    action?: string;
  }[];
  changes: Partial<Record<"new" | "improved" | "fixed", Row[]>>;
  // Changes the release page lists beyond the ones shown here.
  more?: number;
  // Docs pages for change rows, keyed by the row's label.
  guides?: Record<string, string>;
}

// Artwork lives outside the frontend bundle and is requested only for the selected release.
export function releaseImageUrl(version: string, filename: string): string {
  return `https://raw.githubusercontent.com/unslothai/unsloth/main/images/releases/v${encodeURIComponent(version)}/${encodeURIComponent(filename)}`;
}

// Newest first.
export const releases = [
  release922,
  release921,
  release920,
  release919,
  release918,
  release917,
  release916,
  release915,
  release914,
  release913,
  release912,
  release911,
  release910,
  release909,
  release908,
  release907,
  release906,
  release905,
] as Release[];
