// src/tools/artifact-search.ts
import { tool } from "@opencode-ai/plugin/tool";
import { getArtifactIndex } from "./artifact-index";

export const artifact_search = tool({
  description: `Search past plans and ledgers for relevant precedent.
Use this to find:
- Similar problems you've solved before
- Patterns and approaches that worked
- Lessons learned from past sessions
Returns ranked results with file paths for further reading.`,
  args: {
    query: tool.schema.string().describe("Search query - describe what you're looking for"),
    limit: tool.schema.number().optional().describe("Max results to return (default: 10)"),
    type: tool.schema.enum(["all", "plan", "ledger"]).optional().describe("Filter by artifact type (default: all)"),
  },
  execute: async (args) => {
    try {
      const index = await getArtifactIndex();
      const results = await index.search(args.query, args.limit || 10);

      // Filter by type if specified
      const filtered = args.type && args.type !== "all" ? results.filter((r) => r.type === args.type) : results;

      if (filtered.length === 0) {
        return `No results found for "${args.query}". Try broader search terms.`;
      }

      let output = `## Search Results for "${args.query}"\n\n`;
      output += `Found ${filtered.length} result(s):\n\n`;

      for (const result of filtered) {
        const typeLabel = result.type.charAt(0).toUpperCase() + result.type.slice(1);
        output += `### ${typeLabel}: ${result.title || result.id}\n`;
        output += `**File:** \`${result.filePath}\`\n`;
        if (result.summary) {
          output += `**Summary:** ${result.summary}\n`;
        }
        output += `**Relevance Score:** ${result.score.toFixed(2)}\n\n`;
      }

      output += `---\n*Use the Read tool to view full content of relevant files.*`;

      return output;
    } catch (e) {
      return `Error searching artifacts: ${e instanceof Error ? e.message : String(e)}`;
    }
  },
});
