===== PR 5624 / gemini-code-assist[bot] / 2026-06-26T10:32:33Z / COMMENTED =====
## Code Review

This pull request adds tool-calling support and parsers for DeepSeek (R1, V3, V3.1), GLM (4.5, 4.6, 4.7), and Kimi K2 models, along with handling wrapper-less Gemma 4 streams and stripping leaked markup for these new formats. The review feedback focuses on improving robustness and performance: it suggests defensively checking for `None` tokenizers to prevent `AttributeError`, caching tool-ignoring template properties to avoid redundant rendering passes, simplifying position tracking in DeepSeek and Kimi parsers to prevent skipping subsequent calls when closing tags are missing, and adding end-of-string anchors (`\Z`) to the regex stripping patterns to handle truncated tool-call blocks robustly.

> [!IMPORTANT]
> The [consumer version of Gemini Code Assist on GitHub](https://developers.google.com/gemini-code-assist/docs/review-repo-code) is being sunset. Starting **June 18, 2026**, new organization installations will be blocked, and all code review activity will officially cease on **July 17, 2026**.
> For more details on the timeline and next steps, please review the [Help Documentation](https://developers.google.com/gemini-code-assist/docs/deprecations/consumer-code-review).

===== PR 5624 / chatgpt-codex-connector[bot] / 2026-06-26T10:45:21Z / COMMENTED =====

### 💡 Codex Review

Here are some automated review suggestions for this pull request.

**Reviewed commit:** `b360e6ad4d`
    

<details> <summary>ℹ️ About Codex in GitHub</summary>
<br/>

[Your team has set up Codex to review pull requests in this repo](https://chatgpt.com/codex/cloud/settings/general). Reviews are triggered when you
- Open a pull request for review
- Mark a draft as ready
- Comment "@codex review".

If Codex has suggestions, it will comment; otherwise it will react with 👍.




Codex can also answer questions or update the PR. Try commenting "@codex address that feedback".
            
</details>

