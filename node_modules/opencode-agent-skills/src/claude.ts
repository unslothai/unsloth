/**
 * Claude Code compatibility utilities
 *
 * Functions and types for discovering skills from Claude's plugin system
 * (marketplaces and plugin cache directories).
 */

import * as fs from "node:fs/promises";
import { type Dirent } from "node:fs";
import * as path from "node:path";
import { homedir } from "node:os";
import { findFile } from "./utils";
import type { LabeledDiscoveryResult } from "./skills";

/**
 * Tool translation guide for skills written for Claude Code.
 * Injected into skill content to help the AI use OpenCode equivalents.
 */
export const toolTranslation = `<tool-translation>
This skill may reference Claude Code tools. Use OpenCode equivalents:
- TodoWrite/TodoRead -> todowrite/todoread
- Task (subagents) -> task tool with subagent_type parameter
- Skill tool -> use_skill tool
- Read/Write/Edit/Bash/Glob/Grep/WebFetch -> lowercase (read/write/edit/bash/glob/grep/webfetch)
</tool-translation>`;

/** Structure of Claude's marketplace.json file */
interface MarketplaceManifest {
  plugins: Array<{
    name: string;
    skills?: string[];
  }>;
}

/** v1 format: single installation object per plugin */
interface PluginInstallationV1 {
  installPath: string;
  version?: string;
  installedAt?: string;
  lastUpdated?: string;
  gitCommitSha?: string;
}

/** v2 format: array of installations per plugin (supports multiple scopes) */
interface PluginInstallationV2 {
  scope: "managed" | "user" | "project" | "local";
  installPath: string;
  version?: string;
  projectPath?: string;
  installedAt?: string;
  lastUpdated?: string;
  gitCommitSha?: string;
}

/** Structure of Claude's installed_plugins.json file (supports v1 and v2) */
interface InstalledPlugins {
  version?: 1 | 2;
  plugins: {
    [key: string]: PluginInstallationV1 | PluginInstallationV2[];
  };
}

/**
 * Get install paths from a plugin entry, handling both v1 and v2 formats.
 * v1: single object with installPath
 * v2: array of installation objects with installPath
 */
function getPluginInstallPaths(
  pluginData: PluginInstallationV1 | PluginInstallationV2[]
): string[] {
  if (Array.isArray(pluginData)) {
    // v2 format: array of installations
    return pluginData.map((p) => p.installPath).filter(Boolean);
  }
  // v1 format: single object
  return pluginData.installPath ? [pluginData.installPath] : [];
}

/**
 * Discover skills from a plugin directory by scanning its skills/ subdirectory.
 */
async function discoverSkillsFromPluginDir(
  pluginDir: string
): Promise<LabeledDiscoveryResult[]> {
  const results: LabeledDiscoveryResult[] = [];
  const skillsDir = path.join(pluginDir, "skills");

  try {
    const skillDirs = await fs.readdir(skillsDir, { withFileTypes: true });

    for (const skillDir of skillDirs) {
      if (!skillDir.isDirectory()) continue;

      const directory = path.join(skillsDir, skillDir.name);
      const found = await findFile(directory, skillDir.name, "SKILL.md");
      if (found) {
        results.push({ ...found, label: "claude-plugins" });
      }
    }
  } catch {
    // Skills directory doesn't exist or isn't readable
  }

  return results;
}

/**
 * Discover skills from Claude plugin marketplaces.
 * Only loads skills from INSTALLED plugins (checked via installed_plugins.json).
 *
 * Supports both v1 and v2 formats of installed_plugins.json:
 * - v1: plugins[key] = { installPath: string }
 * - v2: plugins[key] = [{ scope, installPath, version, ... }]
 */
export async function discoverMarketplaceSkills(): Promise<
  LabeledDiscoveryResult[]
> {
  const results: LabeledDiscoveryResult[] = [];
  const claudeDir = path.join(homedir(), ".claude", "plugins");
  const installedPath = path.join(claudeDir, "installed_plugins.json");
  const marketplacesDir = path.join(claudeDir, "marketplaces");

  let installed: InstalledPlugins;
  try {
    const content = await fs.readFile(installedPath, "utf-8");
    installed = JSON.parse(content);
  } catch {
    return results;
  }

  const isV2 = installed.version === 2;

  for (const pluginKey of Object.keys(installed.plugins || {})) {
    const pluginData = installed.plugins[pluginKey];
    if (!pluginData) continue;

    if (isV2 || Array.isArray(pluginData)) {
      // v2 format: use installPath directly from each installation entry
      const installPaths = getPluginInstallPaths(pluginData);
      for (const installPath of installPaths) {
        const skills = await discoverSkillsFromPluginDir(installPath);
        results.push(...skills);
      }
    } else {
      // v1 format: use marketplace manifest to find skills
      const [pluginName, marketplaceName] = pluginKey.split("@");
      if (!pluginName || !marketplaceName) continue;

      const manifestPath = path.join(
        marketplacesDir,
        marketplaceName,
        ".claude-plugin",
        "marketplace.json"
      );
      let manifest: MarketplaceManifest;
      try {
        const manifestContent = await fs.readFile(manifestPath, "utf-8");
        manifest = JSON.parse(manifestContent);
      } catch {
        continue;
      }

      const plugin = manifest.plugins?.find((p) => p.name === pluginName);
      if (!plugin?.skills) continue;

      for (const skillRelPath of plugin.skills) {
        const cleanPath = skillRelPath.replace(/^\.\//, "");
        const directory = path.join(marketplacesDir, marketplaceName, cleanPath);
        const skillName = path.basename(cleanPath);

        const found = await findFile(directory, skillName, "SKILL.md");
        if (found) {
          results.push({ ...found, label: "claude-plugins" });
        }
      }
    }
  }

  return results;
}

/**
 * Discover skills from Claude Code's plugin cache directory.
 *
 * Supports both old and new cache directory structures:
 * - Old (v1): ~/.claude/plugins/cache/<plugin-name>/skills/<skill-name>/SKILL.md
 * - New (v2): ~/.claude/plugins/cache/<marketplace>/<plugin-name>/<version>/skills/<skill-name>/SKILL.md
 *
 * Note: For v2, discoverMarketplaceSkills() using installed_plugins.json is preferred
 * as it provides direct paths. This function serves as a fallback for discovery
 * without relying on installed_plugins.json.
 */
export async function discoverPluginCacheSkills(): Promise<
  LabeledDiscoveryResult[]
> {
  const results: LabeledDiscoveryResult[] = [];
  const cacheDir = path.join(homedir(), ".claude", "plugins", "cache");

  try {
    await fs.access(cacheDir);
  } catch {
    return [];
  }

  // First level: could be plugin name (v1) or marketplace name (v2)
  const level1Entries = await fs.readdir(cacheDir, { withFileTypes: true });

  for (const level1 of level1Entries) {
    if (!level1.isDirectory()) continue;

    const level1Path = path.join(cacheDir, level1.name);

    // Check if this directory has a skills/ subdirectory (v1 structure)
    const v1Skills = await discoverSkillsFromPluginDir(level1Path);
    if (v1Skills.length > 0) {
      results.push(...v1Skills);
      continue;
    }

    // Otherwise, assume v2 structure: marketplace/<plugin>/<version>/skills/
    // Second level: plugin names
    let level2Entries: Dirent[];
    try {
      level2Entries = await fs.readdir(level1Path, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const level2 of level2Entries) {
      if (!level2.isDirectory()) continue;

      const level2Path = path.join(level1Path, level2.name);

      // Third level: version directories
      let level3Entries: Dirent[];
      try {
        level3Entries = await fs.readdir(level2Path, { withFileTypes: true });
      } catch {
        continue;
      }

      for (const level3 of level3Entries) {
        if (!level3.isDirectory()) continue;

        const versionDir = path.join(level2Path, level3.name);
        const skills = await discoverSkillsFromPluginDir(versionDir);
        results.push(...skills);
      }
    }
  }

  return results;
}
