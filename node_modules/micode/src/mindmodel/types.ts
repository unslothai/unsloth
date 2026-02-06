// src/mindmodel/types.ts
import * as v from "valibot";
import { parse as parseYaml } from "yaml";

export const CategorySchema = v.object({
  path: v.string(),
  description: v.string(),
  group: v.optional(v.string()),
});

export const ManifestSchema = v.object({
  name: v.string(),
  version: v.pipe(v.number(), v.minValue(1)),
  categories: v.pipe(v.array(CategorySchema), v.minLength(1)),
});

export type Category = v.InferOutput<typeof CategorySchema>;
export type MindmodelManifest = v.InferOutput<typeof ManifestSchema>;

export function parseManifest(yamlContent: string): MindmodelManifest {
  const parsed = parseYaml(yamlContent);
  return v.parse(ManifestSchema, parsed);
}

export interface ConstraintExample {
  title: string;
  code: string;
  language: string;
}

export interface ConstraintFile {
  title: string;
  rules: string[];
  examples: ConstraintExample[];
  antiPatterns: ConstraintExample[];
}

export function parseConstraintFile(content: string): ConstraintFile {
  // Extract title from first H1
  const titleMatch = content.match(/^#\s+(.+)$/m);
  const title = titleMatch ? titleMatch[1] : "Untitled";

  // Extract rules
  const rulesSection = content.match(/## Rules\n([\s\S]*?)(?=\n##|$)/);
  const rules: string[] = [];
  if (rulesSection) {
    const ruleLines = rulesSection[1].match(/^-\s+(.+)$/gm);
    if (ruleLines) {
      rules.push(...ruleLines.map((r) => r.replace(/^-\s+/, "")));
    }
  }

  // Extract examples
  const examples = extractCodeBlocks(content, "## Examples");

  // Extract anti-patterns
  const antiPatterns = extractCodeBlocks(content, "## Anti-patterns");

  return { title, rules, examples, antiPatterns };
}

function extractCodeBlocks(content: string, sectionHeader: string): ConstraintExample[] {
  const results: ConstraintExample[] = [];

  // Find section
  const sectionIndex = content.indexOf(sectionHeader);
  if (sectionIndex === -1) return results;

  // Find next section or end
  const nextSectionMatch = content.slice(sectionIndex + sectionHeader.length).match(/\n## /);
  const sectionEnd = nextSectionMatch ? sectionIndex + sectionHeader.length + nextSectionMatch.index! : content.length;

  const section = content.slice(sectionIndex, sectionEnd);

  // Extract H3 titles and code blocks
  const blockRegex = /### (.+)\n```(\w+)?\n([\s\S]*?)```/g;
  const matches = section.matchAll(blockRegex);
  for (const match of matches) {
    results.push({
      title: match[1],
      language: match[2] || "",
      code: match[3].trim(),
    });
  }

  return results;
}
