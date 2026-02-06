/**
 * Core skill discovery and management logic.
 *
 * Handles skill discovery from multiple locations (project > user > marketplace),
 * validation against the Anthropic Agent Skills Spec, and skill resolution.
 */

import * as fs from "node:fs/promises";
import * as path from "node:path";
import { homedir } from "node:os";
import { z } from "zod";
import {
  findFile,
  parseYamlFrontmatter,
  injectSyntheticContent,
  type FileDiscoveryResult,
  type OpencodeClient,
  type SessionContext,
} from "./utils";
import { discoverMarketplaceSkills, discoverPluginCacheSkills } from "./claude";

/**
 * Skill label indicating the source/location of a skill.
 * - project: .opencode/skills/ in project directory
 * - user: ~/.config/opencode/skills/
 * - claude-project: .claude/skills/ in project directory
 * - claude-user: ~/.claude/skills/
 * - claude-plugins: ~/.claude/plugins/ (cache or marketplace)
 */
export type SkillLabel = "project" | "user" | "claude-project" | "claude-user" | "claude-plugins";

/**
 * Script metadata with both relative and absolute paths.
 */
interface Script {
  relativePath: string;
  absolutePath: string;
}

/**
 * Complete metadata for a discovered skill.
 */
interface Skill {
  name: string;
  description: string;
  path: string;
  relativePath: string;
  namespace?: string;
  label: SkillLabel;
  scripts: Script[];
  template: string;
}

/**
 * Recursively find executable scripts in a skill's directory.
 * Skips hidden directories (starting with .) and common dependency dirs.
 * Only files with executable bit set are returned.
 */
async function findScripts(skillPath: string, maxDepth: number = 10): Promise<Script[]> {
  const scripts: Script[] = [];
  const skipDirs = new Set(['node_modules', '__pycache__', '.git', '.venv', 'venv', '.tox', '.nox']);

  async function recurse(dir: string, depth: number, relPath: string) {
    if (depth > maxDepth) return;

    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        if (entry.name.startsWith('.')) continue;
        if (skipDirs.has(entry.name)) continue;

        const fullPath = path.join(dir, entry.name);
        const newRelPath = relPath ? `${relPath}/${entry.name}` : entry.name;

        let stats;
        try {
          stats = await fs.stat(fullPath);
        } catch {
          continue;
        }

        if (stats.isDirectory()) {
          await recurse(fullPath, depth + 1, newRelPath);
        } else if (stats.isFile()) {
          if (stats.mode & 0o111) {
            scripts.push({
              relativePath: newRelPath,
              absolutePath: fullPath
            });
          }
        }
      }
    } catch { }
  }

  await recurse(skillPath, 0, '');
  return scripts.sort((a, b) => a.relativePath.localeCompare(b.relativePath));
}

/**
 * Anthropic Agent Skills Spec v1.0 compliant schema.
 * @see https://github.com/anthropics/skills/blob/main/agent_skills_spec.md
 */
const SkillFrontmatterSchema = z.object({
  name: z.string()
    .regex(/^[\p{Ll}\p{N}-]+$/u, { message: "Name must be lowercase alphanumeric with hyphens" })
    .min(1, { message: "Name cannot be empty" }),
  description: z.string()
    .min(1, { message: "Description cannot be empty" }),
  license: z.string().optional(),
  "allowed-tools": z.array(z.string()).optional(),
  metadata: z.record(z.string(), z.string()).optional()
});

type SkillFrontmatter = z.infer<typeof SkillFrontmatterSchema>;

/**
 * Parse a SKILL.md file and validate its frontmatter.
 * Returns null if parsing fails (with error logging).
 */
async function parseSkillFile(
  skillPath: string,
  relativePath: string,
  label: SkillLabel
): Promise<Skill | null> {
  const content = await fs.readFile(skillPath, 'utf-8').catch(() => null);
  if (!content) {
    return null;
  }

  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
  if (!frontmatterMatch?.[1] || !frontmatterMatch[2]) {
    return null;
  }

  const frontmatterText = frontmatterMatch[1];
  const skillContent = frontmatterMatch[2].trim();

  let frontmatterObj: unknown;
  try {
    frontmatterObj = parseYamlFrontmatter(frontmatterText);
  } catch {
    return null;
  }

  let frontmatter: SkillFrontmatter;
  try {
    frontmatter = SkillFrontmatterSchema.parse(frontmatterObj);
  } catch (error) {
    return null;
  }

  const skillDirPath = path.dirname(skillPath);
  const scripts = await findScripts(skillDirPath);

  return {
    name: frontmatter.name,
    description: frontmatter.description,
    path: skillDirPath,
    relativePath,
    namespace: frontmatter.metadata?.namespace,
    label,
    scripts,
    template: skillContent
  };
}

/** Discovery result with label attached */
export type LabeledDiscoveryResult = FileDiscoveryResult & { label: SkillLabel };

/**
 * Recursively find SKILL.md files in a directory.
 */
export async function findSkillsRecursive(
  baseDir: string,
  label: SkillLabel,
  maxDepth: number = 3
): Promise<LabeledDiscoveryResult[]> {
  const results: LabeledDiscoveryResult[] = [];

  async function recurse(dir: string, depth: number, relPath: string) {
    if (depth > maxDepth) return;

    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);

        let stats;
        try {
          stats = await fs.stat(fullPath);
        } catch {
          continue;
        }

        if (!stats.isDirectory()) continue;

        const newRelPath = relPath ? `${relPath}/${entry.name}` : entry.name;
        const found = await findFile(fullPath, newRelPath, 'SKILL.md');

        if (found) {
          results.push({ ...found, label });
        } else {
          await recurse(fullPath, depth + 1, newRelPath);
        }
      }
    } catch { }
  }

  try {
    await fs.access(baseDir);
    await recurse(baseDir, 0, '');
  } catch { }

  return results;
}

/** Configuration for a skill discovery path */
interface DiscoveryPath {
  path: string;
  label: SkillLabel;
  maxDepth: number;
}

/**
 * Discover all skills from all locations.
 *
 * Discovery order (first found wins, OpenCode trumps Claude at each level):
 * 1. .opencode/skills/                 (project - OpenCode)
 * 2. .claude/skills/                   (project - Claude)
 * 3. ~/.config/opencode/skills/        (user - OpenCode)
 * 4. ~/.claude/skills/                 (user - Claude)
 * 5. ~/.claude/plugins/cache/          (cached plugin skills)
 * 6. ~/.claude/plugins/marketplaces/   (installed plugins)
 *
 * No shadowing - unique names only. First match wins, duplicates are warned.
 */
export async function discoverAllSkills(directory: string): Promise<Map<string, Skill>> {
  const discoveryPaths: DiscoveryPath[] = [
    { path: path.join(directory, '.opencode', 'skills'), label: 'project', maxDepth: 3 },
    { path: path.join(directory, '.claude', 'skills'), label: 'claude-project', maxDepth: 1 },
    { path: path.join(homedir(), '.config', 'opencode', 'skills'), label: 'user', maxDepth: 3 },
    { path: path.join(homedir(), '.claude', 'skills'), label: 'claude-user', maxDepth: 1 }
  ];

  const allResults: LabeledDiscoveryResult[] = [];
  for (const { path: baseDir, label, maxDepth } of discoveryPaths) {
    allResults.push(...await findSkillsRecursive(baseDir, label, maxDepth));
  }
  allResults.push(...await discoverPluginCacheSkills());
  allResults.push(...await discoverMarketplaceSkills());

  const skillsByName = new Map<string, Skill>();
  for (const { filePath, relativePath, label } of allResults) {
    const skill = await parseSkillFile(filePath, relativePath, label);
    if (!skill || skillsByName.has(skill.name)) continue;
    skillsByName.set(skill.name, skill);
  }

  return skillsByName;
}

/**
 * Resolve a skill by name, handling namespace prefixes.
 * Supports: "skill-name", "project:skill-name", "user:skill-name", etc.
 */
export function resolveSkill(
  skillName: string,
  skillsByName: Map<string, Skill>
): Skill | null {
  if (skillName.includes(':')) {
    const [namespace, name] = skillName.split(':');
    for (const skill of skillsByName.values()) {
      if (skill.name === name && (skill.label === namespace || skill.namespace === namespace)) {
        return skill;
      }
    }
    return null;
  }
  return skillsByName.get(skillName) || null;
}

/**
 * Recursively list all files in a directory, returning relative paths.
 * Excludes SKILL.md since it's already loaded as the main content.
 */
export async function listSkillFiles(skillPath: string, maxDepth: number = 3): Promise<string[]> {
  const files: string[] = [];

  async function recurse(dir: string, depth: number, relPath: string) {
    if (depth > maxDepth) return;

    try {
      const entries = await fs.readdir(dir, { withFileTypes: true });

      for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        const newRelPath = relPath ? `${relPath}/${entry.name}` : entry.name;

        try {
          const stats = await fs.stat(fullPath);
          if (stats.isDirectory()) {
            await recurse(fullPath, depth + 1, newRelPath);
          } else if (stats.isFile() && entry.name !== 'SKILL.md') {
            files.push(newRelPath);
          }
        } catch { }
      }
    } catch { }
  }

  await recurse(skillPath, 0, '');
  return files.sort();
}

/**
 * Skill summary for preflight evaluation.
 */
export interface SkillSummary {
  name: string;
  description: string;
}

/**
 * Get summaries of all available skills (name + description only).
 * Used by preflight LLM call to evaluate which skills are relevant.
 *
 * @param directory - Project directory to discover skills from
 * @returns Array of skill summaries
 */
export async function getSkillSummaries(directory: string): Promise<SkillSummary[]> {
  const skillsByName = await discoverAllSkills(directory);
  return Array.from(skillsByName.values()).map(skill => ({
    name: skill.name,
    description: skill.description,
  }));
}

/**
 * Inject the available skills list into a session.
 * Used on session start and after compaction.
 */
export async function injectSkillsList(
  directory: string,
  client: OpencodeClient,
  sessionID: string,
  context?: SessionContext
): Promise<void> {
  const skillsByName = await discoverAllSkills(directory);
  const skills = Array.from(skillsByName.values());

  if (skills.length === 0) return;

  const skillsList = skills
    .map(s => `- ${s.name}: ${s.description}`)
    .join('\n');

  await injectSyntheticContent(
    client,
    sessionID,
    `<available-skills>
Use the use_skill, read_skill_file, run_skill_script, and get_available_skills tools to work with skills.

${skillsList}
</available-skills>`,
    context
  );
}
