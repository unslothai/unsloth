/**
 * Superpowers bootstrap logic for OpenCode Agent Skills
 *
 * Provides automatic injection of the "using-superpowers" skill content
 * when OPENCODE_AGENT_SKILLS_SUPERPOWERS_MODE=true environment variable is set.
 */

import type { OpencodeClient, SessionContext } from "./utils";
import { injectSyntheticContent, getSessionContext } from "./utils";
import { discoverAllSkills } from "./skills";

const toolMapping = `**Tool Mapping for OpenCode:**
- \`TodoWrite\` → \`todowrite\`
- \`Task\` tool with subagents → Use the \`task\` tool with \`subagent_type\`
- \`Skill\` tool → \`use_skill\`
- \`Read\`, \`Write\`, \`Edit\`, \`Bash\`, \`Glob\`, \`Grep\`, \`WebFetch\` → Use the native lowercase OpenCode tools`;

const skillsNamespace = `**Skill namespace priority:**
1. Project: \`project:skill-name\`
2. Claude project: \`claude-project:skill-name\`
3. User: \`skill-name\`
4. Claude user: \`claude-user:skill-name\`
5. Marketplace: \`claude-plugins:skill-name\`

The first discovered match wins.`;

/**
 * Maybe inject superpowers bootstrap content into a session.
 * Only injects if superpowers mode is enabled and using-superpowers skill exists.
 */
export const maybeInjectSuperpowersBootstrap = async (
  directory: string,
  client: OpencodeClient,
  sessionID: string,
  context?: SessionContext
) => {
  const superpowersModeEnabled = process.env.OPENCODE_AGENT_SKILLS_SUPERPOWERS_MODE === 'true';
  if (!superpowersModeEnabled) return;

  const skillsByName = await discoverAllSkills(directory);
  const usingSuperpowersSkill = skillsByName.get('using-superpowers');
  if (!usingSuperpowersSkill) return;

  const content = `<EXTREMELY_IMPORTANT>
You have superpowers.

**IMPORTANT: The using-superpowers skill content is included below. It is ALREADY LOADED - do not call use_skill for it again. Use use_skill only for OTHER skills.**

${usingSuperpowersSkill.template}

${toolMapping}

${skillsNamespace}
</EXTREMELY_IMPORTANT>`;

  const ctx = context ?? await getSessionContext(client, sessionID);
  await injectSyntheticContent(client, sessionID, content, ctx);
};
