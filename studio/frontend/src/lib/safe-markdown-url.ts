import { type UrlTransform, defaultUrlTransform } from "streamdown";

const PROTOCOL_RELATIVE_RE = /^[/\\]{2}/;
const SCHEME_RE = /^[a-zA-Z][a-zA-Z0-9+\-.]*:/;

function stripAsciiControls(value: string): string {
  return Array.from(value, (character) => {
    const code = character.charCodeAt(0);
    return code <= 0x1f || code === 0x7f ? "" : character;
  }).join("");
}

export const safeMarkdownUrl: UrlTransform = (url, key, node) => {
  if (node.tagName !== "img") {
    return defaultUrlTransform(url, key, node);
  }

  // Browsers discard ASCII controls while parsing URLs, so strip them before
  // rejecting remote schemes and protocol-relative image locations.
  const normalized = stripAsciiControls(url).trim();
  const lower = normalized.toLowerCase();

  if (lower.startsWith("data:") || lower.startsWith("blob:")) {
    return normalized;
  }
  if (PROTOCOL_RELATIVE_RE.test(normalized)) {
    return null;
  }
  if (SCHEME_RE.test(normalized)) {
    return null;
  }
  return normalized;
};
