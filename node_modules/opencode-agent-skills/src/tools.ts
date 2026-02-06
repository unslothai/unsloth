/**
 * OpenCode Agent Skills - Tool Definitions
 *
 * Factory functions that create the 4 skill tools with injected dependencies:
 * - GetAvailableSkills: Get available skills with optional filtering
 * - ReadSkillFile: Read supporting files from skill directories
 * - RunSkillScript: Execute scripts from skill directories
 * - UseSkill: Load a skill's SKILL.md into context
 */

import type { PluginInput } from "@opencode-ai/plugin";
import { tool } from "@opencode-ai/plugin";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { toolTranslation } from "./claude";
import {
  getSessionContext,
  injectSyntheticContent,
  findClosestMatch,
  isPathSafe,
  type OpencodeClient,
} from "./utils";
import {
  discoverAllSkills,
  resolveSkill,
  listSkillFiles,
} from "./skills";

export const GetAvailableSkills = (directory: string) => {
  return tool({
    description: "Get available skills with their descriptions. Optionally filter by query.",
    args: {
      query: tool.schema.string().optional()
        .describe("Search query to filter skills (matches name and description)")
    },
    async execute(args) {
      const skillsByName = await discoverAllSkills(directory);
      const allSkills = Array.from(skillsByName.values());

      let filtered = allSkills;

      if (args.query) {
        const pattern = new RegExp(args.query.replace(/\*/g, '.*'), 'i');
        filtered = filtered.filter(s =>
          pattern.test(s.name) || pattern.test(s.description)
        );
      }

      if (filtered.length === 0) {
        if (args.query) {
          const allSkillNames = allSkills.map(s => s.name);
          const suggestion = findClosestMatch(args.query, allSkillNames);

          if (suggestion) {
            return `No skills found matching "${args.query}". Did you mean "${suggestion}"?`;
          }
        }

        return "No skills found matching your query.";
      }

      return filtered
        .map(s => {
          const scripts = s.scripts.length > 0
            ? ` [scripts: ${s.scripts.map(sc => sc.relativePath).join(', ')}]`
            : '';
          return `${s.name} (${s.label})\n  ${s.description}${scripts}`;
        })
        .join('\n\n');
    }
  });
};

export const ReadSkillFile = (directory: string, client: OpencodeClient) => {
  return tool({
    description: "Read a supporting file from a skill's directory (docs, examples, configs).",
    args: {
      skill: tool.schema.string()
        .describe("Name of the skill"),
      filename: tool.schema.string()
        .describe("File to read, relative to skill directory (e.g., 'anthropic-best-practices.md', 'scripts/helper.sh')")
    },
    async execute(args, ctx) {
      const skillsByName = await discoverAllSkills(directory);
      const allSkills = Array.from(skillsByName.values());

      const skill = resolveSkill(args.skill, skillsByName);

      if (!skill) {
        const allSkillNames = allSkills.map(s => s.name);
        const suggestion = findClosestMatch(args.skill, allSkillNames);

        if (suggestion) {
          return `Skill "${args.skill}" not found. Did you mean "${suggestion}"?`;
        }

        return `Skill "${args.skill}" not found. Use get_available_skills to list available skills.`;
      }

      // Security: ensure path doesn't escape skill directory
      if (!isPathSafe(skill.path, args.filename)) {
        return `Invalid path: cannot access files outside skill directory.`;
      }

      const filePath = path.join(skill.path, args.filename);

      try {
        const content = await fs.readFile(filePath, 'utf-8');

        // Inject via noReply for context persistence
        const wrappedContent = `<skill-file skill="${skill.name}" file="${args.filename}">
  <metadata>
    <directory>${skill.path}</directory>
  </metadata>

  <content>
${content}
  </content>
</skill-file>`;

        const context = await getSessionContext(client, ctx.sessionID);
        await injectSyntheticContent(client, ctx.sessionID, wrappedContent, context);

        return `File "${args.filename}" from skill "${skill.name}" loaded.`;
      } catch {
        try {
          const files = await fs.readdir(skill.path);
          return `File "${args.filename}" not found. Available files: ${files.join(', ')}`;
        } catch {
          return `File "${args.filename}" not found in skill "${skill.name}".`;
        }
      }
    }
  });
};

export const RunSkillScript = (directory: string, $: PluginInput["$"]) => {
  return tool({
    description: "Execute a script from a skill's directory. Scripts are run with the skill directory as CWD.",
    args: {
      skill: tool.schema.string()
        .describe("Name of the skill"),
      script: tool.schema.string()
        .describe("Relative path to the script (e.g., 'build.sh', 'tools/deploy.sh')"),
      arguments: tool.schema.array(tool.schema.string()).optional()
        .describe("Arguments to pass to the script")
    },
    async execute(args) {
      const skillsByName = await discoverAllSkills(directory);
      const allSkills = Array.from(skillsByName.values());

      const skill = resolveSkill(args.skill, skillsByName);

      if (!skill) {
        const allSkillNames = allSkills.map(s => s.name);
        const suggestion = findClosestMatch(args.skill, allSkillNames);

        if (suggestion) {
          return `Skill "${args.skill}" not found. Did you mean "${suggestion}"?`;
        }

        return `Skill "${args.skill}" not found. Use get_available_skills to list available skills.`;
      }

      const script = skill.scripts.find(s => s.relativePath === args.script);

      if (!script) {
        const scriptPaths = skill.scripts.map(s => s.relativePath);
        const suggestion = findClosestMatch(args.script, scriptPaths);

        if (suggestion) {
          return `Script "${args.script}" not found in skill "${skill.name}". Did you mean "${suggestion}"?`;
        }

        const available = scriptPaths.join(', ') || 'none';
        return `Script "${args.script}" not found in skill "${skill.name}". Available scripts: ${available}`;
      }

      try {
        $.cwd(skill.path);
        const scriptArgs = args.arguments || [];
        const result = await $`${script.absolutePath} ${scriptArgs}`.text();
        return result;
      } catch (error: unknown) {
        if (error instanceof Error && 'exitCode' in error) {
          const shellError = error as Error & { exitCode: number; stderr?: Buffer; stdout?: Buffer };
          const stderr = shellError.stderr?.toString() || '';
          const stdout = shellError.stdout?.toString() || '';
          return `Script failed (exit ${shellError.exitCode}): ${stderr || stdout || shellError.message}`;
        }
        if (error instanceof Error) {
          return `Script failed: ${error.message}`;
        }
        return `Script failed: ${String(error)}`;
      }
    }
  });
};

export const UseSkill = (
  directory: string,
  client: OpencodeClient,
  onSkillLoaded?: (sessionID: string, skillName: string) => void
) => {
  return tool({
    description: "Load a skill's SKILL.md content into context. Skills contain proven workflows, techniques, and patterns.",
    args: {
      skill: tool.schema.string()
        .describe("Name of the skill (e.g., 'brainstorming', 'project:my-skill', 'user:my-skill')")
    },
    async execute(args, ctx) {
      const skillsByName = await discoverAllSkills(directory);
      const allSkills = Array.from(skillsByName.values());

      const skill = resolveSkill(args.skill, skillsByName);

      if (!skill) {
        const allSkillNames = allSkills.map(s => s.name);
        const suggestion = findClosestMatch(args.skill, allSkillNames);

        if (suggestion) {
          return `Skill "${args.skill}" not found. Did you mean "${suggestion}"?`;
        }

        return `Skill "${args.skill}" not found. Use get_available_skills to list available skills.`;
      }

      const skillFiles = await listSkillFiles(skill.path);

      const scriptsXml = skill.scripts.length > 0
        ? `\n    <scripts>\n${skill.scripts.map(s => `      <script>${s.relativePath}</script>`).join('\n')}\n    </scripts>`
        : '';

      const filesXml = skillFiles.length > 0
        ? `\n    <files>\n${skillFiles.map(f => `      <file>${f}</file>`).join('\n')}\n    </files>`
        : '';

      const skillContent = `<skill name="${skill.name}">
  <metadata>
    <source>${skill.label}</source>
    <directory>${skill.path}</directory>${scriptsXml}${filesXml}
  </metadata>

  ${toolTranslation}

  <content>
${skill.template}
  </content>
</skill>`;

      const context = await getSessionContext(client, ctx.sessionID);
      await injectSyntheticContent(client, ctx.sessionID, skillContent, context);

      onSkillLoaded?.(ctx.sessionID, skill.name);

      const scriptInfo = skill.scripts.length > 0
        ? `\nAvailable scripts: ${skill.scripts.map(s => s.relativePath).join(', ')}`
        : '';

      const filesInfo = skillFiles.length > 0
        ? `\nAvailable files: ${skillFiles.join(', ')}`
        : '';

      return `Skill "${skill.name}" loaded.${scriptInfo}${filesInfo}`;
    }
  });
};
