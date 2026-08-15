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

from .db import default_db_path, get_connection
from .records import (
    create_namespace,
    deprecate_record,
    ensure_default_namespace,
    get_namespace,
    get_record,
    insert_record,
    insert_rollout,
    list_admissions,
    list_records,
    list_rollouts,
    set_record_status,
    summarize_records,
    supersede_record,
    update_proposed_record,
)
from .search import search_records

__all__ = [
    "create_namespace",
    "default_db_path",
    "deprecate_record",
    "ensure_default_namespace",
    "get_connection",
    "get_namespace",
    "get_record",
    "insert_record",
    "insert_rollout",
    "list_admissions",
    "list_records",
    "list_rollouts",
    "search_records",
    "set_record_status",
    "summarize_records",
    "supersede_record",
    "update_proposed_record",
]
