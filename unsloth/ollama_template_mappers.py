# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "OLLAMA_TEMPLATES",
    "OLLAMA_TEMPLATE_TO_MODEL_MAPPER",
    "MODEL_TO_OLLAMA_TEMPLATE_MAPPER",
]

OLLAMA_TEMPLATES = {}

# =========================================== Unsloth

unsloth_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }}
{{ end }}{{ if .Prompt }}>>> User: {{ .Prompt }}
{{ end }}>>> Assistant: {{ .Response }}{__EOS_TOKEN__}
"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """You are a helpful assistant to the user"""
'''

OLLAMA_TEMPLATES["unsloth"] = unsloth_ollama

# =========================================== Zephyr

zephyr_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}{__EOS_TOKEN__}
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}{__EOS_TOKEN__}
{{ end }}<|assistant|>
{{ .Response }}{__EOS_TOKEN__}
"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["zephyr"] = zephyr_ollama

# =========================================== ChatML
chatml_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["chatml"] = chatml_ollama

# =========================================== Mistral-1
# Ollama from https://www.ollama.com/library/mistral
# Mistral v0.1 https://ollama.com/library/mistral:v0.1/blobs/22e1b2e8dc2f
# Mistral v0.2 https://ollama.com/library/mistral:v0.2/blobs/e6836092461f
mistral_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
'''

# mistral:v0.3 https://ollama.com/library/mistral:v0.3/blobs/1ff5b64b61b9
# mistral-large https://ollama.com/library/mistral-large:latest/blobs/96adabcf2c08
mistral_v03_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .Messages }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and (eq (len (slice $.Messages $index)) 1) $.Tools }}[AVAILABLE_TOOLS] {{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST] {{ if and $.System (eq (len (slice $.Messages $index)) 1) }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }}{{ .Content }}
{{- else if .ToolCalls }}[TOOL_CALLS] [
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]
{{- end }}</s>
{{- else if eq .Role "tool" }}[TOOL_RESULTS] {"content": {{ .Content }}} [/TOOL_RESULTS]
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }}[/INST]
{{- end }}{{ .Response }}
{{- if .Response }}</s>
{{- end }}"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
PARAMETER stop "</s>"
'''

# Mistral-small https://ollama.com/library/mistral-small:latest/blobs/6db27cd4e277
mistral_small_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $index, $_ := .Messages }}
{{- if eq .Role "system" }}[SYSTEM_PROMPT]{{ .Content }}[/SYSTEM_PROMPT]
{{- else if eq .Role "user" }}
{{- if and (le (len (slice $.Messages $index)) 2) $.Tools }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST]{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }}{{ .Content }}
{{- if not (eq (len (slice $.Messages $index)) 1) }}</s>
{{- end }}
{{- else if .ToolCalls }}[TOOL_CALLS][
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
PARAMETER temperature 0.15
SYSTEM """You are Mistral Small 3, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris. Your knowledge base was last updated on 2023-10-01. When you're not sure about some information, you say that you don't have the information and don't make up anything. If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?")"""
'''

# mistral-small-3.1 https://ollama.com/library/mistral-small3.1:latest/blobs/6db27cd4e277
mistral_small_31_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $index, $_ := .Messages }}
{{- if eq .Role "system" }}[SYSTEM_PROMPT]{{ .Content }}[/SYSTEM_PROMPT]
{{- else if eq .Role "user" }}
{{- if and (le (len (slice $.Messages $index)) 2) $.Tools }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST]{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }}{{ .Content }}
{{- if not (eq (len (slice $.Messages $index)) 1) }}</s>
{{- end }}
{{- else if .ToolCalls }}[TOOL_CALLS][
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
PARAMETER num_ctx 4096
SYSTEM """You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos."""
'''

# mistral-small-3.2 https://ollama.com/library/mistral-small3.2:latest/blobs/706c4d1164f7
mistral_small_32_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $index, $_ := .Messages }}
{{- if eq .Role "system" }}[SYSTEM_PROMPT]{{ .Content }}[/SYSTEM_PROMPT]
{{- else if eq .Role "user" }}
{{- if and (le (len (slice $.Messages $index)) 2) $.Tools }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST]{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }}{{ .Content }}
{{- if not (eq (len (slice $.Messages $index)) 1) }}</s>
{{- end }}
{{- else if .ToolCalls }}
{{- range $i, $_ := .ToolCalls }}[TOOL_CALLS]{{ .Function.Name }}[CALL_ID]{{ $i }}[ARGS]{{ .Function.Arguments }}
{{- end }}</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
PARAMETER temperature 0.15
SYSTEM """You are Mistral Small 3.2, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.

When you're not sure about some information or when the user's request requires up-to-date or specific data, you must use the available tools to fetch the information. Do not hesitate to use tools whenever they can provide a more accurate or complete response. If no relevant tools are available, then clearly state that you don't have the information and avoid making up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos.

TOOL CALLING INSTRUCTIONS

You may have access to tools that you can use to fetch information or perform actions. You must use these tools in the following situations:

1. When the request requires up-to-date information.
2. When the request requires specific data that you do not have in your knowledge base.
3. When the request involves actions that you cannot perform without tools.

Always prioritize using tools to provide the most accurate and helpful response. If tools are not available, inform the user that you cannot perform the requested action at the moment."""
'''


# https://ollama.com/library/mixtral:latest/blobs/53d74de0d84c
mixtral_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST] {{ .Response }}"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
'''

# https://registry.ollama.ai/library/mistral-nemo:latest/blobs/438402ddac75
mistral_nemo_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{- range $i, $_ := .Messages }}
{{- if eq .Role "user" }}
{{- if and $.Tools (le (len (slice $.Messages $i)) 2) }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST]{{ if and $.System (eq (len (slice $.Messages $i)) 1) }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }} {{ .Content }}{{ if not (eq (len (slice $.Messages $i)) 1) }}</s>{{ end }}
{{- else if .ToolCalls }}[TOOL_CALLS][
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
'''

# https://ollama.com/library/codestral:latest/blobs/51707752a87c
codestral_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{- if .Suffix }}[SUFFIX]{{ .Suffix }}[PREFIX] {{ .Prompt }}
{{- else if .Messages }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}[INST] {{ if and $.System (eq (len (slice $.Messages $index)) 1) }}{{ $.System }}

{{ end }}{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }} {{ .Content }}</s>
{{- end }}
{{- end }}
{{- else }}[INST] {{ if .System }}{{ .System }}

{{ end }}{{ .Prompt }} [/INST]
{{- end }} {{ .Response }}
{{- if .Response }}</s>
{{- end }}
"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
PARAMETER stop "[PREFIX]"
PARAMETER stop "[MIDDLE]"
PARAMETER stop "[SUFFIX]"
'''

# https://ollama.com/library/devstral:latest/blobs/ea9ec42474e0
devstral_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- $lastUserIndex := -1 }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "user" }}{{ $lastUserIndex = $index }}{{ end }}
{{- end }}
{{- range $index, $_ := .Messages }}
{{- if eq .Role "system" }}[SYSTEM_PROMPT]{{ .Content }}[/SYSTEM_PROMPT]
{{- else if eq .Role "user" }}
{{- if and (eq $lastUserIndex $index) $.Tools }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST]{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if .Content }}{{ .Content }}
{{- if not (eq (len (slice $.Messages $index)) 1) }}</s>
{{- end }}
{{- else if .ToolCalls }}[TOOL_CALLS][
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}]</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS]{"content": {{ .Content }}}[/TOOL_RESULTS]
{{- end }}
{{- end }}"""
SYSTEM """You are Devstral, a helpful agentic model trained by Mistral AI and using the OpenHands scaffold. You can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
</CODE_QUALITY>

<VERSION_CONTROL>
* When configuring git credentials, use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION: Make focused, minimal changes to address the problem
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.
* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.
</SECURITY>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Document your reasoning process
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>"""
'''

# https://ollama.com/library/magistral:latest/blobs/35f7a1efc383
magistral_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1}}
{{- if eq .Role "system" }}[SYSTEM_PROMPT]{{ .Content }}[/SYSTEM_PROMPT]
{{- else if eq .Role "user" }}
{{- if and (le (len (slice $.Messages $i)) 2) $.Tools }}[AVAILABLE_TOOLS]{{ $.Tools }}[/AVAILABLE_TOOLS]
{{- end }}[INST]{{ .Content }}[/INST]
{{- else if eq .Role "assistant" }}
{{- if and $.IsThinkSet (and $last .Thinking) -}}
<think>
{{ .Thinking }}
</think>
{{ end }}
{{- if .Content }}{{ .Content }}
{{- end }}
{{- if .ToolCalls }}{{ range $i, $_ := .ToolCalls }}[TOOL_CALLS]{{ .Function.Name }}[CALL_ID]{{ $i }}[ARGS]{{ .Function.Arguments }}{{ end }}
{{- end }}
{{- if not (eq (len (slice $.Messages $i)) 1) }}</s>
{{- end }}
{{- else if eq .Role "tool" }}[TOOL_RESULTS]0[TOOL_CONTENT]{{ .Content }}[/TOOL_RESULTS]
{{- end }}
{{- if and $last (ne .Role "assistant") }}{{ if and $.IsThinkSet (not $.Think) -}}<think>
</think>
{{ end }}
{{- end }}
{{- end }}"""
PARAMETER temperature 0.7
PARAMETER top_p 0.95
SYSTEM """A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown and Latex to format your response. Write both your thoughts and summary in the same language as the task posed by the user.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user.

Problem:"""
'''

OLLAMA_TEMPLATES["mistral"] = mistral_ollama
OLLAMA_TEMPLATES["mistral-v03"] = mistral_v03_ollama
OLLAMA_TEMPLATES["mistral-small"] = mistral_small_ollama
OLLAMA_TEMPLATES["mistral-small-31"] = mistral_small_31_ollama
OLLAMA_TEMPLATES["mistral-small-32"] = mistral_small_32_ollama
OLLAMA_TEMPLATES["mixtral"] = mixtral_ollama
OLLAMA_TEMPLATES["mistral-nemo"] = mistral_nemo_ollama
OLLAMA_TEMPLATES["devstral"] = devstral_ollama
OLLAMA_TEMPLATES["magistral"] = magistral_ollama
OLLAMA_TEMPLATES["codestral"] = codestral_ollama


# =========================================== Llama-2
# Ollama from https://www.ollama.com/library/llama3
llama_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """[INST] <<SYS>>{{ .System }}<</SYS>>

{{ .Prompt }} [/INST]"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["llama"] = llama_ollama

# ===========================================  Vicuna
# Ollama from https://www.ollama.com/library/vicuna
vicuna_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }} {{ end }}{{ if .Prompt }}USER: {{ .Prompt }} {{ end }}ASSISTANT: {{ .Response }} {__EOS_TOKEN__}"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["vicuna"] = vicuna_ollama

# =========================================== Vicuna Old
vicuna_old_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }}
{{ end }}{{ if .Prompt }}### Human: {{ .Prompt }}
{{ end }}### Assistant: {{ .Response }}{__EOS_TOKEN__}
"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."""
'''

OLLAMA_TEMPLATES["vicuna_old"] = vicuna_old_ollama
OLLAMA_TEMPLATES["vicuna old"] = OLLAMA_TEMPLATES["vicuna_old"]

# =========================================== Alpaca multi turn
alpaca_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}{{ .System }}

{{ end }}{{ if .Prompt }}### Instruction:
{{ .Prompt }}{{ end }}

### Response:
{{ .Response }}{__EOS_TOKEN__}

"""
PARAMETER stop "{__EOS_TOKEN__}"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """Below are some instructions that describe some tasks. Write responses that appropriately complete each request."""
'''

OLLAMA_TEMPLATES["alpaca"] = alpaca_ollama

# =========================================== Gemma
# Ollama from https://www.ollama.com/library/gemma
gemma_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """<start_of_turn>user
{{ if .System }}{{ .System }} {{ end }}{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ .Response }}<end_of_turn>
"""
PARAMETER repeat_penalty 1
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER penalize_newline false
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["gemma"] = gemma_ollama

# =========================================== Gemma with ChatML instead
gemma_chatml_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""
PARAMETER repeat_penalty 1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER penalize_newline false
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["gemma_chatml"] = gemma_chatml_ollama

# =========================================== Gemma 2
# Same as Gemma 1, but with sliding window attention!
# https://ollama.com/library/gemma2/blobs/6522ca797f47
gemma2_ollama = gemma_ollama + "PARAMETER num_ctx 4096\n"
OLLAMA_TEMPLATES["gemma2"] = gemma2_ollama

# =========================================== Gemma 2 with ChatML instead
gemma2_chatml_ollama = gemma_chatml_ollama + "PARAMETER num_ctx 4096\n"
OLLAMA_TEMPLATES["gemma2_chatml"] = gemma2_chatml_ollama

# =========================================== Llama-3
# Ollama from https://www.ollama.com/library/llama3
llama3_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""
PARAMETER num_keep 24
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["llama-3"] = llama3_ollama
OLLAMA_TEMPLATES["llama3"] = llama3_ollama


# =========================================== Phi-3
# Ollama from https://www.ollama.com/library/phi3
phi3_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|system|>
{{ .System }}<|end|>
{{ end }}{{ if .Prompt }}<|user|>
{{ .Prompt }}<|end|>
{{ end }}<|assistant|>
{{ .Response }}<|end|>
"""
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["phi-3"] = phi3_ollama
OLLAMA_TEMPLATES["phi-35"] = OLLAMA_TEMPLATES["phi-3"]
OLLAMA_TEMPLATES["phi-3.5"] = OLLAMA_TEMPLATES["phi-3"]

# =========================================== Llama-3.1
"""
No trimming in Llama 3.1 Instruct!
Also an extra newline for Cutting Knowledge Date
See https://colab.research.google.com/drive/1Xpqq5xpIgO-B00MQ-UccYMwN2J8QFgBM?usp=sharing

Also should be

import datetime
tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True,
    tokenize = False,
    date_string = datetime.today().strftime("%d %B %Y")),
)
"""

# Ollama from https://ollama.com/library/llama3.1 (needs updating!)
llama31_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .Messages }}
{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

You are a helpful assistant with tool calling capabilities. When you receive a tool call response, use the output to format an answer to the original use question.
{{- end }}
{{- end }}<|eot_id|>
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>
{{- if and $.Tools $last }}

Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ $.Tools }}
{{- end }}

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}

{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
{{- else }}

{{ .Content }}{{ if not $last }}<|eot_id|>{{ end }}
{{- end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}
{{- end }}
{{- else }}
{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}{{ .Response }}{{ if .Response }}<|eot_id|>{{ end }}"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|eom_id|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

# https://ollama.com/ajindal/llama3.1-storm:8b/blobs/1970553b62f4
llama_31_storm_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{ if .Messages }}
{{- if or .System .Tools }}<|start_header_id|>system<|end_header_id|>
{{- if .System }}

{{ .System }}
{{- end }}
{{- if .Tools }}

You are a function calling AI model. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into function. The user may use the terms function calling or tool use interchangeably.

Here are the available functions:
<tools>{{ json .Tools }}</tools>

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags in the format:
<tool_call>{"tool_name": <function-name>, "tool_arguments": <args-dict>}</tool_call>
{{- end }}
{{- end }}<|eot_id|>
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
{{ end }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>
{{- if .ToolCalls }}

{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }}}{{ end }}
{{- else }}

{{ .Content }}{{ if not $last }}<|eot_id|>{{ end }}
{{- end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>{{ if $last }}<|start_header_id|>assistant<|end_header_id|>
{{ end }}
{{- end }}
{{- end }}
{{- else }}
{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}{{ .Response }}{{ if .Response }}<|eot_id|>{{ end }}
"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
'''

# https://ollama.com/library/nemotron:latest/blobs/4863fe3335f3
llama_31_nemotron_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """<|start_header_id|>system<|end_header_id|>

{{ if .Tools }}You have access to the following functions. To call a function, please respond with JSON for a function call. Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{{ range .Tools }}{{ . }}

{{ end }}
{{- end }}{{ .System }}<|eot_id|>
{{- range $i, $_ := .Messages }}
{{- $isLastMessage := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "system" }}
{{- else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "parameters": {{ .Function.Arguments }} }
{{- end }}
{{- end }}
{{- if not $isLastMessage }}<|eot_id|>
{{- end }}
{{- else if eq .Role "tool" }}<|start_header_id|>ipython<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- if $isLastMessage }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- else }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}<|eot_id|>
{{- if $isLastMessage }}<|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}
{{- end }}
"""
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
'''

# https://ollama.com/library/llama3.2-vision:latest/blobs/715415638c895a1f8e8c6
llama_32_vision_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $index, $_ := .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>

{{ .Content }}
{{- if gt (len (slice $.Messages $index)) 1 }}<|eot_id|>
{{- else if ne .Role "assistant" }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ end }}
{{- end }}"""
PARAMETER temperature 0.6
PARAMETER top_p 0.9
'''

OLLAMA_TEMPLATES["llama-3.1"] = llama31_ollama
OLLAMA_TEMPLATES["llama-31"] = llama31_ollama
OLLAMA_TEMPLATES["llama-31-nemotron"] = llama_31_nemotron_ollama
OLLAMA_TEMPLATES["llama-31-storm"] = llama_31_storm_ollama
OLLAMA_TEMPLATES["llama-32-vision"] = llama_32_vision_ollama

for version in ("llama-3.2", "llama-3.3", "llama-32", "llama-33"):
    OLLAMA_TEMPLATES[version] = OLLAMA_TEMPLATES["llama-3.1"]

# =========================================== tinyllama
# tinyllama-chat https://ollama.com/library/tinyllama:latest/blobs/af0ddbdaaa26
tinyllama_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """<|system|>
{{ .System }}</s>
<|user|>
{{ .Prompt }}</s>
<|assistant|>"""
PARAMETER stop "<|system|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"
PARAMETER "</s>"
SYSTEM """You are a helpful AI assistant."""
'''

OLLAMA_TEMPLATES["tinyllama"] = tinyllama_ollama


# =========================================== Qwen 2/2.5
# Qwen2 https://ollama.com/library/qwen2:latest/blobs/77c91b422cc9
# Qwen2.5 from https://ollama.com/library/qwen2.5/blobs/eb4402837c78
qwen25_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""
'''

# https://ollama.com/library/qwen2.5-coder:latest/blobs/1e65450c3067
qwen_25_coder_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .Suffix }}<|fim_prefix|>{{ .Prompt }}<|fim_suffix|>{{ .Suffix }}<|fim_middle|>
{{- else if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools>:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> with NO other text. Do not include any backticks or ```json.
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
SYSTEM """You are Qwen, created by Alibaba Cloud. You are a helpful assistant."""
'''

# https://ollama.com/library/qwen2.5vl:latest/blobs/a242d8dfdc8f
qwen_25_vl_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .System -}}
<|im_start|>system
{{ .System }}<|im_end|>
{{- end -}}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}
<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}
<|im_start|>assistant
{{ if .Content }}{{ .Content }}{{ if not $last }}<|im_end|>
{{- else -}}<|im_end|>{{- end -}}
{{- end -}}
{{- end -}}
{{- if and (ne .Role "assistant") $last }}
<|im_start|>assistant
{{ end -}}
{{- end }}"""
PARAMETER temperature 0.0001
SYSTEM """You are a helpful assistant."""
'''

# https://ollama.com/library/openthinker:latest/blobs/32695b892af8
openthinker_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
<|im_start|>{{ .Role }}<|im_sep|>
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_end|>
<|im_start|>assistant<|im_sep|>
{{ end }}
{{- end }}"""
'''


OLLAMA_TEMPLATES["qwen-25"] = qwen25_ollama
OLLAMA_TEMPLATES["qwen-25-coder"] = qwen_25_coder_ollama
OLLAMA_TEMPLATES["qwen-25-vl"] = qwen_25_vl_ollama
OLLAMA_TEMPLATES["openthinker"] = openthinker_ollama
OLLAMA_TEMPLATES["qwen-2"] = qwen25_ollama

# =========================================== Phi-4
_phi4_ollama_template = (
    "{{ if .System }}<|im_start|><|system|><|im_sep|>{{ .System }}<|im_end|>{{ end }}"
    "{{ if .Prompt }}<|im_start|><|user|><|im_sep|>{{ .Prompt }}<|im_end|>{{ end }}"
    "<|im_start|><|assistant|><|im_sep|>{{ .Response }}<|im_end|>"
)

# Ollama from https://www.ollama.com/library/phi4 is different
phi_4_ollama = f'''
FROM {{__FILE_LOCATION__}}
TEMPLATE """{_phi4_ollama_template}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_sep|>"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

# https://ollama.com/library/phi4-reasoning:latest/blobs/32695b892af8
phi_4_reasoning_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
<|im_start|>{{ .Role }}<|im_sep|>
{{ .Content }}{{ if not $last }}<|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_end|>
<|im_start|>assistant<|im_sep|>
{{ end }}
{{- end }}"""
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_sep|>"
'''

# https://ollama.com/library/phi4-mini:latest/blobs/813f53fdc6e5
phi_4_mini_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if or .System .Tools }}<|system|>{{ if .System }}{{ .System }}{{ end }}
{{- if .Tools }}{{ if not .System }}You are a helpful assistant with some tools.{{ end }}<|tool|>{{ .Tools }}<|/tool|><|end|>
{{- end }}
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if ne .Role "system" }}<|{{ .Role }}|>{{ .Content }}
{{- if .ToolCalls }}<|tool_call|>[{{ range .ToolCalls }}{"name":"{{ .Function.Name }}","arguments":{{ .Function.Arguments }}{{ end }}]<|/tool_call|>
{{- end }}
{{- if not $last }}<|end|>
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|end|><|assistant|>{{ end }}
{{- end }}
{{- end }}"""
'''

# https://ollama.com/library/phi4-mini-reasoning:latest/blobs/c895a1f8e8c6
phi_4_mini_reasoning_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{- if .System }}<|system|>{{ .System }}
{{- end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if ne .Role "system" }}<|{{ .Role }}|>{{ .Content }}
{{- if not $last }}<|end|>
{{- end }}
{{- if and (ne .Role "assistant") $last }}<|end|><|assistant|>{{ end }}
{{- end }}
{{- end }}"""
SYSTEM """Your name is Phi, an AI math expert developed by Microsoft."""
'''
OLLAMA_TEMPLATES["phi-4"] = phi_4_ollama
OLLAMA_TEMPLATES["phi-4-reasoning"] = phi_4_reasoning_ollama
OLLAMA_TEMPLATES["phi-4-mini"] = phi_4_mini_ollama
OLLAMA_TEMPLATES["phi-4-mini-reasoning"] = phi_4_mini_reasoning_ollama


# =========================================== Gemma-3
# Ollama from https://ollama.com/library/gemma3/blobs/e0a42594d802
gemma3_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""
PARAMETER stop "<end_of_turn>"
PARAMETER stop "<eos>"
PARAMETER temperature 1.0
PARAMETER min_p 0.0
PARAMETER top_k 64
PARAMETER top_p 0.95
PARAMETER num_predict 32768
'''

# https://ollama.com/library/gemma3:270m/blobs/4b19ac7dd2fb
gemma3_270m_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- $systemPromptAdded := false }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<start_of_turn>user
{{- if (and (not $systemPromptAdded) $.System) }}
{{- $systemPromptAdded = true }}
{{ $.System }}
{{ end }}
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}
"""
PARAMETER stop "<end_of_turn>"
PARAMETER top_k 64
PARAMETER top_p 0.95
'''

OLLAMA_TEMPLATES["gemma-3"] = gemma3_ollama
OLLAMA_TEMPLATES["gemma3"] = gemma3_ollama
OLLAMA_TEMPLATES["gemma3-270m"] = gemma3_270m_ollama


# =========================================== Qwen-3
# Ollama template for Qwen-3 (see https://ollama.com/library/qwen3/blobs/eb4402837c78)
qwen3_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- if .Messages }}
{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
{{- else }}
{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ end }}{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.6
PARAMETER min_p 0.0
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1
'''

qwen3_template_eos_token = "<|im_end|>"
OLLAMA_TEMPLATES["qwen-3"] = qwen3_ollama
OLLAMA_TEMPLATES["qwen3"] = qwen3_ollama


# =========================================== Gemma-3n
# Ollama from https://ollama.com/library/gemma3n/blobs/e0a42594d802
gemma3n_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if or (eq .Role "user") (eq .Role "system") }}<start_of_turn>user
{{ .Content }}<end_of_turn>
{{ if $last }}<start_of_turn>model
{{ end }}
{{- else if eq .Role "assistant" }}<start_of_turn>model
{{ .Content }}{{ if not $last }}<end_of_turn>
{{ end }}
{{- end }}
{{- end }}"""
'''

OLLAMA_TEMPLATES["gemma-3n"] = gemma3n_ollama
OLLAMA_TEMPLATES["gemma3n"] = gemma3n_ollama

# =========================================== GPT-OSS

# Ollama from https://ollama.com/library/gpt-oss:latest/blobs/fa6710a93d78
gptoss_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: {{ currentDate }}
{{- if and .IsThinkSet .Think (ne .ThinkLevel "") }}

Reasoning: {{ .ThinkLevel }}
{{- else if or (not .IsThinkSet) (and .IsThinkSet .Think) }}

Reasoning: medium
{{- end }}

{{- $hasNonBuiltinTools := false }}
{{- if .Tools -}}
{{- $hasBrowserSearch := false }}
{{- $hasBrowserOpen := false }}
{{- $hasBrowserFind := false }}
{{- $hasPython := false }}
  {{- range .Tools }}
    {{- if eq .Function.Name "browser.search" -}}{{- $hasBrowserSearch = true -}}
    {{- else if eq .Function.Name "browser.open" -}}{{- $hasBrowserOpen = true -}}
    {{- else if eq .Function.Name "browser.find" -}}{{- $hasBrowserFind = true -}}
    {{- else if eq .Function.Name "python" -}}{{- $hasPython = true -}}
    {{- else }}{{ $hasNonBuiltinTools = true -}}
    {{- end }}
  {{- end }}
{{- if or $hasBrowserSearch $hasBrowserOpen $hasBrowserFind $hasPython }}

# Tools
{{- if or $hasBrowserSearch $hasBrowserOpen $hasBrowserFind }}

## browser

// Tool for browsing.
// The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
// Cite information from the tool using the following format:
// `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
// Do not quote more than 10 words directly from the tool output.
// sources=web (default: web)
namespace browser {
{{- if $hasBrowserSearch }}

// Searches for information related to `query` and displays `topn` results.
type search = (_: {
query: string,
topn?: number, // default: 10
source?: string,
}) => any;
{{- end }}
{{- if $hasBrowserOpen }}

// Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
// Valid link ids are displayed with the formatting: `【{id}†.*】`.
// If `cursor` is not provided, the most recent page is implied.
// If `id` is a string, it is treated as a fully qualified URL associated with `source`.
// If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
// Use this function without `id` to scroll to a new location of an opened page.
type open = (_: {
id?: number | string, // default: -1
cursor?: number, // default: -1
loc?: number, // default: -1
num_lines?: number, // default: -1
view_source?: boolean, // default: false
source?: string,
}) => any;
{{- end }}
{{- if $hasBrowserFind }}

// Finds exact matches of `pattern` in the current page, or the page given by `cursor`.
type find = (_: {
pattern: string,
cursor?: number, // default: -1
}) => any;
{{- end }}

} // namespace browser
{{- end }}{{/* end if has browser tools */}}
{{- if $hasPython }}

## python

Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.
{{- end }}{{/* end if hasPython */}}
{{- end }}{{/* end if has any built-in tools */}}
{{- end }}{{/* end if .Tools */}}

# Valid channels: analysis, commentary, final. Channel must be included for every message.{{ if $hasNonBuiltinTools }}
Calls to these tools must go to the commentary channel: 'functions'.
{{- end -}}<|end|>{{/* end of system */ -}}
{{- if or $hasNonBuiltinTools .System -}}
<|start|>developer<|message|>{{- if $hasNonBuiltinTools }}# Tools

## functions

namespace functions {
{{- range .Tools }}
{{- if not (or (eq .Function.Name "browser.search") (eq .Function.Name "browser.open") (eq .Function.Name "browser.find") (eq .Function.Name "python")) }}
{{if .Function.Description }}
// {{ .Function.Description }}
{{- end }}
{{- if and .Function.Parameters.Properties (gt (len .Function.Parameters.Properties) 0) }}
type {{ .Function.Name }} = (_: {
{{- range $name, $prop := .Function.Parameters.Properties }}
{{- if $prop.Description }}
  // {{ $prop.Description }}
{{- end }}
  {{ $name }}: {{ if gt (len $prop.Type) 1 }}{{ range $i, $t := $prop.Type }}{{ if $i }} | {{ end }}{{ $t }}{{ end }}{{ else }}{{ index $prop.Type 0 }}{{ end }},
{{- end }}
}) => any;
{{- else }}
type {{ .Function.Name }} = () => any;
{{- end }}
{{- end }}{{/* end if not browser tool */}}
{{- end }}{{/* end of range .Tools */}}

} // namespace functions
{{- end }}{{/* end if hasNonBuiltinTools */}}
{{- if .System}}

# Instructions

{{ .System }}
{{- end -}}
<|end|>
{{- end -}}
{{- /* Find the index of the last user message */ -}}
{{- $lastUserIdx := -1 }}
{{- $prefillingContent := false }}
{{- $prefillingThinkingOnly := false }}
{{- range $i, $msg := .Messages }}
  {{- $last := eq (len (slice $.Messages $i)) 1 -}}
  {{- if eq $msg.Role "user" }}
    {{- $lastUserIdx = $i }}
  {{- end -}}
  {{- if and $last (eq $msg.Role "assistant") (gt (len $msg.Content) 0) }}
    {{- $prefillingContent = true }}
  {{- else if and $last (eq $msg.Role "assistant") (gt (len $msg.Thinking) 0) }}
    {{- $prefillingThinkingOnly = true }}
  {{- end }}
{{- end -}}
{{- /* Now render messages */ -}}
{{- range $i, $msg := .Messages }}
  {{- $last := eq (len (slice $.Messages $i)) 1 -}}
  {{- if (ne $msg.Role "system") -}}
    {{- if eq $msg.Role "tool" -}}
      {{- if or (eq $msg.ToolName "python") (eq $msg.ToolName "browser.search") (eq $msg.ToolName "browser.open") (eq $msg.ToolName "browser.find") -}}
        <|start|>{{ $msg.ToolName }} to=assistant<|message|>{{ $msg.Content }}<|end|>
      {{- else -}}
        <|start|>functions.{{ $msg.ToolName }} to=assistant<|message|>{{ $msg.Content }}<|end|>
      {{- end -}}
    {{- else if eq $msg.Role "assistant" -}}
      {{- if and $msg.Thinking (gt $i $lastUserIdx) -}}{{- /* Show thinking only after last user message */ -}}
      <|start|>assistant<|channel|>analysis<|message|>{{ $msg.Thinking }}{{- if not $prefillingThinkingOnly -}}<|end|>{{- end -}}
      {{- end -}}
      {{- if gt (len $msg.Content) 0 -}}
        <|start|>assistant<|channel|>final<|message|>{{ $msg.Content }}{{- if not $prefillingContent -}}<|end|>{{- end -}}
      {{- end -}}
      {{- if gt (len $msg.ToolCalls) 0 -}}
        {{- range $j, $toolCall := $msg.ToolCalls -}}
          {{- $isBuiltin := or (eq $toolCall.Function.Name "python") (eq $toolCall.Function.Name "browser.search") (eq $toolCall.Function.Name "browser.open") (eq $toolCall.Function.Name "browser.find") -}}
          <|start|>assistant<|channel|>{{ if $isBuiltin }}analysis{{ else }}commentary{{ end }} to={{ if not $isBuiltin}}functions.{{end}}{{ $toolCall.Function.Name }} <|constrain|>json<|message|>{{ $toolCall.Function.Arguments }}<|call|>
        {{- end -}}
      {{- end -}}
    {{- else if eq $msg.Role "user" -}}
      <|start|>{{ $msg.Role }}<|message|>{{ $msg.Content }}<|end|>
    {{- end }}
  {{- else }}
  {{- end }}
{{- end -}}
{{- if not (or $prefillingContent $prefillingThinkingOnly) -}}
<|start|>assistant
{{- end -}}"""
PARAMETER temperature 1.0
PARAMETER top_k 0
PARAMETER top_p 1.0
'''

OLLAMA_TEMPLATES["gpt-oss"] = gptoss_ollama
OLLAMA_TEMPLATES["gptoss"] = gptoss_ollama


# =========================================== Qwen3

# Ollama from https://ollama.com/library/qwen3/blobs/53e4ea15e8f5
qwen3_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """
{{- $lastUserIdx := -1 -}}
{{- range $idx, $msg := .Messages -}}
{{- if eq $msg.Role "user" }}{{ $lastUserIdx = $idx }}{{ end -}}
{{- end }}
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if (and $.IsThinkSet (and .Thinking (or $last (gt $i $lastUserIdx)))) -}}
<think>{{ .Thinking }}</think>
{{ end -}}
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}
"""
'''

OLLAMA_TEMPLATES["qwen3-instruct"] = qwen3_ollama
OLLAMA_TEMPLATES["qwen3-thinking"] = qwen3_ollama


# =========================================== Starling-LM


# Ollama from https://ollama.com/library/starling-lm:7b/blobs/4b21bfc435b4
starling_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}GPT4 Correct System: {{ .System }}<|end_of_turn|>
{{ end }}{{ if .Prompt }}GPT4 Correct User: {{ .Prompt }}<|end_of_turn|>
{{ end }}GPT4 Correct Assistant: {{ .Response }}<|end_of_turn|>"""
PARAMETER stop "<|end_of_turn|>"
PARAMETER stop "GPT4 Correct User:"
PARAMETER stop "GPT4 Correct Assistant:"
PARAMETER stop "GPT4 Correct System:"
PARAMETER temperature 1.5
PARAMETER min_p 0.1
'''

OLLAMA_TEMPLATES["starling"] = starling_ollama


# =========================================== Yi-chat


# Ollama from https://ollama.com/library/yi:34b-chat/blobs/62fbfd9ed093
yi_chat_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
'''

OLLAMA_TEMPLATES["yi-chat"] = yi_chat_ollama

# =========================================== Granite

# Ollama from https://ollama.com/library/granite3.2:latest/blobs/3e7ca51acd6e
granite_32_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- /*

------ MESSAGE PARSING ------

*/}}
{{- /*
Declare the prompt structure variables to be filled in from messages
*/}}
{{- $system := "" }}
{{- $documents := "" }}
{{- $documentCounter := 0 }}
{{- $thinking := false }}
{{- $citations := false }}
{{- $hallucinations := false }}
{{- $length := "" }}

{{- /*
Loop over messages and look for a user-provided system message and documents
*/ -}}
{{- range .Messages }}

    {{- /* User defined system prompt(s) */}}
    {{- if (eq .Role "system")}}
        {{- if (ne $system "") }}
            {{- $system = print $system " " }}
        {{- end}}
        {{- $system = print $system .Content }}
    {{- end}}

    {{- /*
    NOTE: Since Ollama collates consecutive roles, for control and documents, we
        work around this by allowing the role to contain a qualifier after the
        role string.
    */ -}}

    {{- /* Role specified thinking */ -}}
    {{- if (and (ge (len .Role) 7) (eq (slice .Role 0 7) "control")) }}
        {{- if (eq .Content "thinking")}}{{- $thinking = true }}{{- end}}
        {{- if (eq .Content "citations")}}{{- $citations = true }}{{- end}}
        {{- if (eq .Content "hallucinations")}}{{- $hallucinations = true }}{{- end}}
        {{- if (and (ge (len .Content) 7) (eq (slice .Content 0 7) "length "))}}
            {{- $length = print ` {"length": "` (slice .Content 7) `"}` }}
        {{- end}}
    {{- end}}

    {{- /* Role specified document */ -}}
    {{- if (and (ge (len .Role) 8) (eq (slice .Role 0 8) "document")) }}
        {{- if (ne $documentCounter 0)}}
            {{- $documents = print $documents " "}}
        {{- end}}
        {{- $identifier := $documentCounter}}
        {{- if (ge (len .Role) 9) }}
            {{- $identifier = (slice .Role 8)}}
        {{- end}}
        {{- $documents = print $documents "Document " $identifier "" .Content}}
        {{- $documentCounter = len (printf "a%*s" $documentCounter "")}}
    {{- end}}
{{- end}}

{{- /*
If no user message provided, build the default system message
*/ -}}
{{- if eq $system "" }}
    {{- $system = "Knowledge Cutoff Date: April 2024.You are Granite, developed by IBM."}}

    {{- /* Add Tools prompt */}}
    {{- if .Tools }}
        {{- $system = print $system " You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request." }}
    {{- end}}

    {{- /* Add documents prompt */}}
    {{- if $documents }}
        {{- if .Tools }}
            {{- $system = print $system " "}}
        {{- else }}
            {{- $system = print $system " "}}
        {{- end}}
        {{- $system = print $system "Write the response to the user's input by strictly aligning with the facts in the provided documents. If the information needed to answer the question is not available in the documents, inform the user that the question cannot be answered based on the available data." }}
        {{- if $citations}}
            {{- $system = print $system " In your response, use the symbols <co> and </co> to indicate when a fact comes from a document in the search result, e.g <co>0</co> for a fact from document 0. Afterwards, list all the citations with their corresponding documents in an ordered list."}}
        {{- end}}
        {{- if $hallucinations}}
            {{- $system = print $system "Finally, after the response is written, include a numbered list of sentences from the response that are potentially hallucinated and not based in the documents."}}
        {{- end}}
    {{- end}}

    {{- /* Prompt without tools or documents */}}
    {{- if (and (not .Tools) (not $documents)) }}
        {{- $system = print $system " You are a helpful AI assistant."}}
        {{- if $thinking}}
            {{- $system = print $system "Respond to every user query in a comprehensive and detailed way. You can write down your thought process before responding. Write your thoughts after 'Here is my thought process:' and write your response after 'Here is my response:' for each user query."}}
        {{- end}}
    {{- end}}

    {{- /* Add thinking prompt if no tools or documents */}}
    {{- if (and $thinking (not .Tools) (not $documents)) }}
        {{- $system = print $system " You are a helpful AI assistant.Respond to every user query in a comprehensive and detailed way. You can write down your thought process before responding. Write your thoughts after 'Here is my thought process:' and write your response after 'Here is my response:' for each user query."}}
    {{- end}}

{{- end}}
{{- /*

------ TEMPLATE EXPANSION ------

*/}}
{{- /* System Prompt */ -}}
<|start_of_role|>system<|end_of_role|>{{- $system }}<|end_of_text|>

{{- /* Tools */ -}}
{{- if .Tools }}
<|start_of_role|>tools<|end_of_role|>[
{{- range $index, $_ := .Tools }}
{{ . }}
{{- if and (ne (len (slice $.Tools $index)) 1) (gt (len $.Tools) 1) }},
{{- end}}
{{- end }}
]
{{- end}}

{{- /* Documents */ -}}
{{- if $documents }}
<|start_of_role|>documents<|end_of_role|>
{{ $documents }}<|end_of_text|>
{{- end}}

{{- /* Standard Messages */}}
{{- range $index, $_ := .Messages }}
{{- if (and
    (ne .Role "system")
    (or (lt (len .Role) 7) (ne (slice .Role 0 7) "control"))
    (or (lt (len .Role) 8) (ne (slice .Role 0 8) "document"))
)}}
<|start_of_role|>
{{- if eq .Role "tool" }}tool_response
{{- else }}{{ .Role }}
{{- end }}<|end_of_role|>
{{- if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<|tool_call|>
{{- range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{- end }}
{{- end }}
{{- if eq (len (slice $.Messages $index)) 1 }}
{{- if eq .Role "assistant" }}
{{- else }}<|end_of_text|>
<|start_of_role|>assistant<|end_of_role|>
{{- end -}}
{{- else }}<|end_of_text|>
{{- end }}
{{- end }}
{{- end }}
"""
'''

# granite-3.2-vision https://ollama.com/library/granite3.2-vision:latest/blobs/579046ba1157
granite_32_vision_ollama = '''
FROM {__FILE_LOCATION__}
TEMPLATE """{{- /* Tools */ -}}
{{- if .Tools -}}
<|start_of_role|>available_tools<|end_of_role|>
{{- range $index, $_ := .Tools }}
{{- $last := eq (len (slice $.Tools $index)) 1 }}
{{ . }}
{{- if not $last }}
{{ end}}
{{- end -}}
<|end_of_text|>
{{ end }}

{{- /* System Prompt */ -}}
{{- if and (gt (len .Messages) 0) (eq (index .Messages 0).Role "system") -}}
<|system|>
{{(index .Messages 0).Content}}
{{- else -}}
<|system|>
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
{{- end }}

{{- /*Main message loop*/ -}}
{{- range $index, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $index)) 1 }}
{{- if eq .Role "system" }}

{{- else if eq .Role "user" }}
<|user|>
{{.Content}}

{{- else if eq .Role "assistant" }}
<|assistant|>
{{- if .Content }}
{{.Content}}
<|end_of_text|>
{{ end }}

{{- else if eq .Role "assistant_tool_call" }}
<|start_of_role|>assistant<|end_of_role|><|tool_call|>{{.Content}}<|end_of_text|>

{{- else if eq .Role "tool_response" }}
<|start_of_role|>tool_response<|end_of_role|>{{.Content}}<|end_of_text|>
{{- end }}

{{- /* Add generation prompt */ -}}
{{ if $last }}
{{- if eq .Role "assistant" }}
{{- else }}
<|assistant|>
{{- end }}
{{- end }}
{{- end }}"""
PARAMETER num_ctx 16384
PARAMETER temperature 0
SYSTEM """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
'''

OLLAMA_TEMPLATES["granite-32"] = granite_32_ollama
OLLAMA_TEMPLATES["granite-32-vision"] = granite_32_vision_ollama


OLLAMA_TEMPLATE_TO_MODEL_MAPPER = {
    "phi-3.5": (
        "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        "unsloth/Phi-3.5-mini-instruct",
        "microsoft/Phi-3.5-mini-instruct",
    ),
    "phi-3": (
        "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-mini-4k-instruct",
        "unsloth/Phi-3-medium-4k-instruct-bnb-4bit",
        "unsloth/Phi-3-medium-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
        "unsloth/Phi-3-mini-4k-instruct-v0-bnb-4bit",
        "unsloth/Phi-3-mini-4k-instruct-v0",
    ),
    "phi-4": (
        "unsloth/phi-4-unsloth-bnb-4bit",
        "unsloth/phi-4",
        "microsoft/phi-4",
        "unsloth/phi-4-bnb-4bit",
    ),
    "phi-4-reasoning": (
        "unsloth/phi-4-reasoning-unsloth-bnb-4bit",
        "unsloth/phi-4-reasoning",
        "microsoft/Phi-4-reasoning",
        "unsloth/phi-4-reasoning-bnb-4bit",
        "unsloth/phi-4-reasoning-plus-unsloth-bnb-4bit",
        "unsloth/phi-4-reasoning-plus",
        "microsoft/Phi-4-reasoning-plus",
        "unsloth/phi-4-reasoning-plus-bnb-4bit",
    ),
    "phi-4-mini": (
        "unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit",
        "unsloth/Phi-4-mini-instruct",
        "microsoft/Phi-4-mini-instruct",
        "unsloth/Phi-4-mini-instruct-bnb-4bit",
    ),
    "phi-4-mini-reasoning": (
        "unsloth/phi-4-mini-reasoning-unsloth-bnb-4bit",
        "unsloth/phi-4-mini-reasoning",
        "microsoft/Phi-4-mini-reasoning",
        "unsloth/phi-4-mini-reasoning-bnb-4bit",
    ),
    "mistral": (
        "unsloth/mistral-7b-instruct-v0.1-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ),
    "mistral-v03": (
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        "unsloth/mistral-7b-instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "unsloth/Mistral-Large-Instruct-2407-bnb-4bit",
        "mistralai/Mistral-Large-Instruct-2407",
    ),
    "mistral-small": (
        "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
        "unsloth/Mistral-Small-Instruct-2409",
        "mistralai/Mistral-Small-Instruct-2409",
        "unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-24B-Instruct-2501",
        "mistralai/Mistral-Small-24B-Instruct-2501",
        "unsloth/Mistral-Small-24B-Instruct-2501-bnb-4bit",
    ),
    "mistral-small-31": (
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503",
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "unsloth/Mistral-Small-3.1-24B-Instruct-2503-bnb-4bit",
    ),
    "mistral-small-32": (
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-unsloth-bnb-4bit",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506",
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "unsloth/Mistral-Small-3.2-24B-Instruct-2506-bnb-4bit",
    ),
    "mixtral": (
        "unsloth/Mixtral-8x7B-Instruct-v0.1-unsloth-bnb-4bit",
        "unsloth/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "unsloth/Mixtral-8x7B-Instruct-v0.1-bnb-4bit",
    ),
    "mistral-nemo": (
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ),
    "codestral": (
        "mistralai/Codestral-22B-v0.1",
        "mistral-community/Codestral-22B-v0.1",
    ),
    "devstral": (
        "unsloth/Devstral-Small-2505-unsloth-bnb-4bit",
        "unsloth/Devstral-Small-2505",
        "mistralai/Devstral-Small-2505",
        "unsloth/Devstral-Small-2505-bnb-4bit",
        "unsloth/Devstral-Small-2507-unsloth-bnb-4bit",
        "unsloth/Devstral-Small-2507",
        "mistralai/Devstral-Small-2507",
        "unsloth/Devstral-Small-2507-bnb-4bit",
    ),
    "magistral": (
        "unsloth/Magistral-Small-2506-unsloth-bnb-4bit",
        "unsloth/Magistral-Small-2506",
        "mistralai/Magistral-Small-2506",
        "unsloth/Magistral-Small-2506-bnb-4bit",
        "unsloth/Magistral-Small-2507-unsloth-bnb-4bit",
        "unsloth/Magistral-Small-2507",
        "mistralai/Magistral-Small-2507",
        "unsloth/Magistral-Small-2507-bnb-4bit",
        "unsloth/Magistral-Small-2509-unsloth-bnb-4bit",
        "unsloth/Magistral-Small-2509",
        "mistralai/Magistral-Small-2509",
        "unsloth/Magistral-Small-2509-bnb-4bit",
    ),
    "tinyllama": (
        "unsloth/tinyllama-chat-bnb-4bit",
        "unsloth/tinyllama-chat",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ),
    "llama": (
        "unsloth/llama-2-7b-bnb-4bit",
        "unsloth/llama-2-7b",
        "meta-llama/Llama-2-7b-hf",
        "unsloth/llama-2-13b-bnb-4bit",
        "unsloth/llama-2-13b",
        "meta-llama/Llama-2-13b-hf",
        "unsloth/llama-2-7b-chat-bnb-4bit",
        "unsloth/llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ),
    "llama3": (
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-8b-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "unsloth/llama-3-70b-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3-70B-Instruct",
    ),
    "llama-3.1": (
        "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit",
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "unsloth/Hermes-3-Llama-3.1-8B-bnb-4bit",
        "unsloth/Hermes-3-Llama-3.1-8B",
        "NousResearch/Hermes-3-Llama-3.1-8B",
        "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit",
        "unsloth/Hermes-3-Llama-3.1-70B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
        "unsloth/Hermes-3-Llama-3.1-405B-bnb-4bit",
        "NousResearch/Hermes-3-Llama-3.1-405B",
        "unsloth/Llama-3.1-Tulu-3-8B-bnb-4bit",
        "unsloth/Llama-3.1-Tulu-3-8B",
        "allenai/Llama-3.1-Tulu-3-8B",
        "unsloth/Llama-3.1-Tulu-3-70B-bnb-4bit",
        "unsloth/Llama-3.1-Tulu-3-70B",
        "allenai/Llama-3.1-Tulu-3-70B",
    ),
    "llama-31-storm": (
        "unsloth/Llama-3.1-Storm-8B-bnb-4bit",
        "unsloth/Llama-3.1-Storm-8B",
        "akjindal53244/Llama-3.1-Storm-8B",
    ),
    "llama-31-nemotron": (
        "unsloth/Llama-3.1-Nemotron-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.1-Nemotron-70B-Instruct",
        "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
    ),
    "llama-3.2": (
        "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    ),
    "llama-32-vision": (
        "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct",
        "meta-llama/Llama-3.2-90B-Vision-Instruct",
    ),
    "llama-3.3": (
        "unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        "unsloth/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct",
    ),
    "gemma": (
        "unsloth/gemma-7b-it-bnb-4bit",
        "unsloth/gemma-7b-it",
        "google/gemma-7b-it",
        "google/gemma-2b-it",
        "unsloth/gemma-1.1-2b-it-bnb-4bit",
        "unsloth/gemma-1.1-2b-it",
        "google/gemma-1.1-2b-it",
        "unsloth/gemma-1.1-7b-it-bnb-4bit",
        "unsloth/gemma-1.1-7b-it",
        "google/gemma-1.1-7b-it",
    ),
    "gemma2": (
        "unsloth/gemma-2-9b-it-bnb-4bit",
        "unsloth/gemma-2-9b-it",
        "google/gemma-2-9b-it",
        "unsloth/gemma-2-27b-it-bnb-4bit",
        "unsloth/gemma-2-27b-it",
        "google/gemma-2-27b-it",
        "unsloth/gemma-2-2b-it-bnb-4bit",
        "unsloth/gemma-2-2b-it",
        "google/gemma-2-2b-it",
    ),
    "gemma-3": (
        "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-1b-it",
        "google/gemma-3-1b-it",
        "unsloth/gemma-3-1b-it-bnb-4bit",
        "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-4b-it",
        "google/gemma-3-4b-it",
        "unsloth/gemma-3-4b-it-bnb-4bit",
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-12b-it",
        "google/gemma-3-12b-it",
        "unsloth/gemma-3-12b-it-bnb-4bit",
        "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-27b-it",
        "google/gemma-3-27b-it",
        "unsloth/gemma-3-27b-it-bnb-4bit",
        "unsloth/medgemma-4b-it-unsloth-bnb-4bit",
        "unsloth/medgemma-4b-it",
        "google/medgemma-4b-it",
        "unsloth/medgemma-4b-it-bnb-4bit",
        "unsloth/medgemma-27b-text-it-unsloth-bnb-4bit",
        "unsloth/medgemma-27b-text-it",
        "google/medgemma-27b-text-it",
        "unsloth/medgemma-27b-text-it-bnb-4bit",
    ),
    "gemma3n": (
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E4B-it",
        "google/gemma-3n-E4B-it",
        "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
        "unsloth/gemma-3n-E2B-it",
        "google/gemma-3n-E2B-it",
        "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    ),
    "gemma3-270m": (
        "unsloth/gemma-3-270m-it-unsloth-bnb-4bit",
        "unsloth/gemma-3-270m-it",
        "google/gemma-3-270m-it",
        "unsloth/gemma-3-270m-it-bnb-4bit",
    ),
    "qwen-25": (
        "unsloth/Qwen2.5-0.5B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "unsloth/Qwen2.5-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-1.5B-Instruct",
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-7B-Instruct",
        "Qwen/Qwen2.5-Math-7B-Instruct",
        "unsloth/Qwen2.5-Math-72B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Math-72B-Instruct",
        "Qwen/Qwen2.5-Math-72B-Instruct",
    ),
    "qwen-25-coder": (
        "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-0.5B-Instruct",
        "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-14B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "unsloth/Qwen2.5-Coder-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    "qwen-25-vl": (
        "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "unsloth/Qwen2.5-VL-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-32B-Instruct",
        "Qwen/Qwen2.5-VL-32B-Instruct",
        "unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct-unsloth-bnb-4bit",
        "unsloth/Qwen2.5-VL-72B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "unsloth/Qwen2.5-VL-72B-Instruct-bnb-4bit",
    ),
    "openthinker": (
        "unsloth/OpenThinker-7B-unsloth-bnb-4bit",
        "unsloth/OpenThinker-7B",
        "open-thoughts/OpenThinker-7B",
        "unsloth/OpenThinker-7B-bnb-4bit",
    ),
    "qwen-2": (
        "unsloth/Qwen2-0.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
        "unsloth/Qwen2-1.5B-Instruct-bnb-4bit",
        "unsloth/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "unsloth/Qwen2-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "unsloth/Qwen2-70B-Instruct-bnb-4bit",
        "Qwen/Qwen2-70B-Instruct",
    ),
    "qwen3": (
        "unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
        "unsloth/Qwen3-0.6B",
        "Qwen/Qwen3-0.6B",
        "unsloth/Qwen3-0.6B-bnb-4bit",
        "unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        "unsloth/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B",
        "unsloth/Qwen3-1.7B-bnb-4bit",
        "unsloth/Qwen3-4B-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B",
        "Qwen/Qwen3-4B",
        "unsloth/Qwen3-4B-bnb-4bit",
        "unsloth/Qwen3-8B-unsloth-bnb-4bit",
        "unsloth/Qwen3-8B",
        "Qwen/Qwen3-8B",
        "unsloth/Qwen3-8B-bnb-4bit",
        "unsloth/Qwen3-14B-unsloth-bnb-4bit",
        "unsloth/Qwen3-14B",
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-14B-bnb-4bit",
        "unsloth/Qwen3-32B-unsloth-bnb-4bit",
        "unsloth/Qwen3-32B",
        "Qwen/Qwen3-32B",
        "unsloth/Qwen3-32B-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-unsloth-bnb-4bit",
        "unsloth/Qwen3-30B-A3B",
        "Qwen/Qwen3-30B-A3B",
        "unsloth/Qwen3-30B-A3B-bnb-4bit",
    ),
    "qwen3-instruct": (
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "unsloth/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Instruct-2507",
        "Qwen/Qwen3-4B-Instruct-2507",
        "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit",
    ),
    "qwen3-thinking": (
        "unsloth/QwQ-32B-Preview-bnb-4bit",
        "unsloth/QwQ-32B-Preview",
        "Qwen/QwQ-32B-Preview",
        "unsloth/QwQ-32B-unsloth-bnb-4bit",
        "unsloth/QwQ-32B",
        "Qwen/QwQ-32B",
        "unsloth/QwQ-32B-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
        "unsloth/Qwen3-4B-Thinking-2507",
        "Qwen/Qwen3-4B-Thinking-2507",
        "unsloth/Qwen3-4B-Thinking-2507-bnb-4bit",
        "unsloth/Qwen3-30B-A3B-Thinking-2507",
        "Qwen/Qwen3-30B-A3B-Thinking-2507",
    ),
    "zephyr": (
        "unsloth/zephyr-sft-bnb-4bit",
        "unsloth/zephyr-sft",
        "HuggingFaceH4/mistral-7b-sft-beta",
    ),
    "chatml": (
        "unsloth/Hermes-2-Pro-Mistral-7B-bnb-4bit",
        "unsloth/Hermes-2-Pro-Mistral-7B",
        "NousResearch/Hermes-2-Pro-Mistral-7B",
        "unsloth/OpenHermes-2.5-Mistral-7B-bnb-4bit",
        "unsloth/OpenHermes-2.5-Mistral-7B",
        "teknium/OpenHermes-2.5-Mistral-7B",
    ),
    "gpt-oss": (
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-20b",
        "openai/gpt-oss-20b",
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
        "unsloth/gpt-oss-120b",
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-120b-unsloth-bnb-4bit",
    ),
    "starling": (
        "unsloth/Starling-LM-7B-beta-bnb-4bit",
        "unsloth/Starling-LM-7B-beta",
        "Nexusflow/Starling-LM-7B-beta",
    ),
    "yi-chat": (
        "unsloth/yi-34b-chat-bnb-4bit",
        "01-ai/Yi-6B-Chat",
        "01-ai/Yi-34B-Chat",
    ),
    "granite-32": (
        "unsloth/granite-3.2-2b-instruct-unsloth-bnb-4bit",
        "unsloth/granite-3.2-2b-instruct",
        "ibm-granite/granite-3.2-2b-instruct",
        "unsloth/granite-3.2-2b-instruct-bnb-4bit",
        "unsloth/granite-3.2-8b-instruct-unsloth-bnb-4bit",
        "unsloth/granite-3.2-8b-instruct",
        "ibm-granite/granite-3.2-8b-instruct",
        "unsloth/granite-3.2-8b-instruct-bnb-4bit",
    ),
    "granite-32-vision": (
        "unsloth/granite-vision-3.2-2b-unsloth-bnb-4bit",
        "unsloth/granite-vision-3.2-2b",
        "ibm-granite/granite-vision-3.2-2b",
        "unsloth/granite-vision-3.2-2b-bnb-4bit",
    ),
}

MODEL_TO_OLLAMA_TEMPLATE_MAPPER = {}

for key, values in OLLAMA_TEMPLATE_TO_MODEL_MAPPER.items():
    for value in values:
        MODEL_TO_OLLAMA_TEMPLATE_MAPPER[value] = key

    # Get lowercased
    lowered_key = key.lower()
    for value in values:
        MODEL_TO_OLLAMA_TEMPLATE_MAPPER[value.lower()] = lowered_key
