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

"""Record kinds, statuses, and provenance. Phase 1 implements the full set."""

from __future__ import annotations

KINDS = frozenset(
    {"claim", "procedure", "error_fix", "entity", "episode", "directive", "twin_note"}
)
# Tools may not mint episode rows; the runner owns those.
TOOL_WRITE_KINDS = frozenset(
    {"claim", "procedure", "error_fix", "entity", "directive", "twin_note"}
)
STATUSES = frozenset({"active", "superseded", "deprecated", "proposed", "rejected"})
PROVENANCES = frozenset({"world", "sim", "mixed", "human", "infer"})
ADMISSION_MODES = frozenset({"auto", "propose", "deny"})
# Operator CLI admit without --force.
ADMIT_FROM_STATUSES = frozenset({"proposed", "deprecated"})

# Hard caps so a tool dump cannot blow up FTS or become uncapped gold.
RECORD_TITLE_CHARS = 200
RECORD_BODY_CHARS = 4000
EVENT_SUMMARY_CHARS = 240
ROLLOUT_SUMMARY_CHARS = 400
SEARCH_TOP_K_MAX = 32
SEARCH_FTS_SCAN_CAP = 256

DEFAULT_NAMESPACE_ID = "default"
DEFAULT_NAMESPACE_NAME = "default"

# Lower is preferred at retrieve time.
PROVENANCE_WEIGHT = {
    "world": 0,
    "mixed": 1,
    "human": 2,
    "sim": 3,
    "infer": 4,
}

ACTIVE_RETRIEVE_STATUSES = frozenset({"active"})
