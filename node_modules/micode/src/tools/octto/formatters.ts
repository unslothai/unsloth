// src/tools/octto/formatters.ts

import type { Answer } from "../../octto/session";
import type { BrainstormState, Branch, BranchQuestion } from "../../octto/state";
import { extractAnswerSummary } from "./extractor";

function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

export function formatBranchFinding(branch: Branch): string {
  return `<branch id="${branch.id}">
    <scope>${escapeXml(branch.scope)}</scope>
    <finding>${escapeXml(branch.finding || "no finding")}</finding>
  </branch>`;
}

export function formatBranchStatus(branch: Branch): string {
  return `<branch id="${branch.id}" status="${branch.status}">
    <scope>${escapeXml(branch.scope)}</scope>
    <finding>${escapeXml(branch.finding || "pending")}</finding>
  </branch>`;
}

export function formatFindings(state: BrainstormState): string {
  const branches = state.branch_order.map((id) => formatBranchFinding(state.branches[id])).join("\n");
  return `<findings>\n${branches}\n</findings>`;
}

export function formatFindingsList(state: BrainstormState): string {
  const items = state.branch_order
    .map((id) => {
      const b = state.branches[id];
      return `  <finding scope="${escapeXml(b.scope)}">${escapeXml(b.finding || "no finding")}</finding>`;
    })
    .join("\n");
  return `<findings>\n${items}\n</findings>`;
}

export function formatQASummary(branch: Branch): string {
  const answered = branch.questions.filter((q): q is BranchQuestion & { answer: Answer } => q.answer !== undefined);

  if (answered.length === 0) {
    return "<qa_summary>no questions answered</qa_summary>";
  }

  const qas = answered
    .map((q) => {
      const answerText = extractAnswerSummary(q.type, q.answer);
      return `    <qa>
      <question>${escapeXml(q.text)}</question>
      <answer>${escapeXml(answerText)}</answer>
    </qa>`;
    })
    .join("\n");

  return qas;
}
