===== PR 5704 / chatgpt-codex-connector[bot] / 2026-06-26T10:24:56Z / COMMENTED =====

### 💡 Codex Review

Here are some automated review suggestions for this pull request.

**Reviewed commit:** `1d0a2a1b13`
    

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

[Your team has set up Codex to review pull requests in this repo](https://chatgpt.com/codex/cloud/settings/general). Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




Codex can also answer questions or update the PR. Try commenting "@codex address that feedback".
            
</details>

===== PR 5704 / gemini-code-assist[bot] / 2026-06-26T10:33:04Z / COMMENTED =====
## Code Review

This pull request adds support for parsing and stripping Mistral bracket-tag (`[TOOL_CALLS]`) and rehearsal (`[ARGS]`) tool-call formats, as well as stripping thinking blocks (`<think>`/`[THINK]`) before parsing. The reviewer feedback highlights potential streaming leaks where partial or unclosed tool-call prefixes can leak to the UI before the opening brace is generated, suggesting regex updates to match these prefixes immediately. Additionally, the reviewer recommends consolidating duplicate parsing loops for the new tool-call formats to improve maintainability.

> [!IMPORTANT]
> The [consumer version of Gemini Code Assist on GitHub](https://developers.google.com/gemini-code-assist/docs/review-repo-code) is being sunset. Starting **June 18, 2026**, new organization installations will be blocked, and all code review activity will officially cease on **July 17, 2026**.
> For more details on the timeline and next steps, please review the [Help Documentation](https://developers.google.com/gemini-code-assist/docs/deprecations/consumer-code-review).

===== PR 5704 / chatgpt-codex-connector[bot] / 2026-06-26T10:39:16Z / COMMENTED =====

### 💡 Codex Review

Here are some automated review suggestions for this pull request.

**Reviewed commit:** `5fc0ecc531`
    

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

[Your team has set up Codex to review pull requests in this repo](https://chatgpt.com/codex/cloud/settings/general). Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




Codex can also answer questions or update the PR. Try commenting "@codex address that feedback".
            
</details>

