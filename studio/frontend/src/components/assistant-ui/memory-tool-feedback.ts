// SPDX-License-Identifier: AGPL-3.0-only
export function memoryToolFeedback({
  phase,
  forgetting,
  project,
  resultText,
}: {
  phase: "approval" | "running" | "cancelled" | "settled";
  forgetting: boolean;
  project: boolean;
  resultText: string;
}) {
  const idle = { detail: "", canManage: false };
  if (phase === "approval")
    return { ...idle, label: "Waiting for your approval" };
  if (phase === "running") {
    return { ...idle, label: forgetting ? "Forgetting…" : "Remembering…" };
  }
  if (phase === "cancelled") {
    return {
      ...idle,
      label: "Cancelled",
      detail: "The update wasn't completed.",
    };
  }
  if (resultText.startsWith("Memory updated:")) {
    return {
      detail: "",
      canManage: true,
      label: forgetting
        ? project
          ? "Forgotten for this project"
          : "Forgotten"
        : project
          ? "Saved for this project"
          : "Saved for future chats",
    };
  }
  if (resultText.startsWith("Memory unchanged:")) {
    return {
      detail: "",
      canManage: true,
      label: forgetting ? "Already forgotten" : "Already remembered",
    };
  }
  const label = forgetting
    ? "Couldn't forget this memory"
    : "Couldn't save this memory";
  if (resultText.startsWith("Error: Memory is full.")) {
    return {
      label,
      detail: "Your memory is full. Remove a saved memory to make room.",
      canManage: true,
    };
  }
  if (resultText.startsWith("Error: Memory changed.")) {
    return {
      label,
      detail: "Your saved memories changed. Try again.",
      canManage: true,
    };
  }
  if (resultText === "Error: Memory is unavailable for this chat.") {
    return { ...idle, label, detail: "Memory isn't available in this chat." };
  }
  if (
    resultText === "Error: Choose personal or the current project memory." ||
    resultText === "Error: Project is unavailable."
  ) {
    return {
      ...idle,
      label,
      detail: "Open a chat in the relevant project to remember this there.",
    };
  }
  return { ...idle, label, detail: "Nothing was changed. Try again." };
}
