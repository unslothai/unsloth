# Copyright 2026-present the Unforgettable contributors.
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

"""OpenAI-shaped function specs for durable memory and contact tools."""

from __future__ import annotations

MEMORY_WRITE = {
    "type": "function",
    "function": {
        "name": "memory_write",
        "description": (
            "Write a durable memory record (claim, procedure, error_fix, entity, "
            "directive, or twin_note). Use this when the user asks to remember "
            "something or a lesson should persist past this episode. Do not write "
            "episode summaries; the runner owns those."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "claim",
                        "procedure",
                        "error_fix",
                        "entity",
                        "directive",
                        "twin_note",
                    ],
                },
                "title": {"type": "string"},
                "body": {"type": "string"},
                "provenance": {
                    "type": "string",
                    "enum": ["world", "sim", "mixed", "infer"],
                    "description": (
                        "Where this was observed. Tools cannot claim human; "
                        "the runner treats that as infer. Sim contact cannot claim world."
                    ),
                },
                "speaker": {
                    "type": "string",
                    "enum": ["world", "sim", "user", "model", "other"],
                    "description": (
                        "Who asserted this. Tools cannot claim user; that "
                        "becomes model. Sim contact cannot claim world. "
                        "Directives are stored as speaker=user."
                    ),
                },
                "speaker_label": {
                    "type": "string",
                    "description": "Optional which-user or which-document label.",
                },
                "warrant": {
                    "type": "string",
                    "description": (
                        "Internal proof or explanation. Empty means unbacked. "
                        "Unbacked user/other claims stay proposed."
                    ),
                },
                "namespace": {"type": "string"},
            },
            "required": ["kind", "title", "body", "provenance"],
        },
    },
}

MEMORY_SEARCH = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": "Search active durable memories by text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "kinds": {"type": "string", "description": "Comma-separated kinds."},
                "provenance": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}

MEMORY_GET = {
    "type": "function",
    "function": {
        "name": "memory_get",
        "description": "Read one memory record by id, including supersession pointer.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
}

MEMORY_SUPERSEDE = {
    "type": "function",
    "function": {
        "name": "memory_supersede",
        "description": (
            "Replace a memory with a corrected version. The old record is kept "
            "as superseded history."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "body": {"type": "string"},
                "title": {"type": "string"},
                "provenance": {"type": "string"},
                "speaker": {"type": "string"},
                "speaker_label": {"type": "string"},
                "warrant": {"type": "string"},
            },
            "required": ["id", "body"],
        },
    },
}

MEMORY_DEPRECATE = {
    "type": "function",
    "function": {
        "name": "memory_deprecate",
        "description": (
            "Archive a memory so it is excluded from default retrieval. Does not hard-delete."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["id"],
        },
    },
}

MEMORY_COMPACT = {
    "type": "function",
    "function": {
        "name": "memory_compact",
        "description": (
            "Hygiene pass on durable memory: drop old empty proposed rows, "
            "deprecate duplicate claim/procedure/entity titles, fold long "
            "superseded chains. Does not invent or merge bodies; title-dedupe "
            "losers only get the existing [deprecated] suffix. Never "
            "title-dedupes twin_note, episode, error_fix, or directive. "
            "dry_run defaults true (preview)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "Default true (preview). Pass false to mutate.",
                    "default": True,
                }
            },
        },
    },
}

MEMORY_COMPILE = {
    "type": "function",
    "function": {
        "name": "memory_compile",
        "description": (
            "Pin an admitted procedure into the standing prompt cache, or "
            "preview/run auto-compile of procedures that have enough world-pass hits. "
            "Source of truth stays the B record. dry_run defaults true."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Procedure id to pin. Omit to run maybe_compile.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Default true (preview). Pass false to mutate.",
                    "default": True,
                },
            },
        },
    },
}

MEMORY_TOOLS = [
    MEMORY_WRITE,
    MEMORY_SEARCH,
    MEMORY_GET,
    MEMORY_SUPERSEDE,
    MEMORY_DEPRECATE,
    MEMORY_COMPACT,
    MEMORY_COMPILE,
]

MEMORY_TOOL_NAMES = frozenset(spec["function"]["name"] for spec in MEMORY_TOOLS)

RIMS_ENTER_SIM = {
    "type": "function",
    "function": {
        "name": "rims_enter_sim",
        "description": (
            "Request a sim clone of the world tree after a recognized failure. "
            "Calling this tool is itself a recognized failure and enters sim."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
            },
        },
    },
}

CONTACT_TOOLS = [RIMS_ENTER_SIM]
CONTACT_TOOL_NAMES = frozenset(spec["function"]["name"] for spec in CONTACT_TOOLS)
