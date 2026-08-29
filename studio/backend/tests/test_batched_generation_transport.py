# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One command, several replies, all the way from the route to the backend."""

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import pytest

from core.inference.orchestrator import GenStreamError, InferenceOrchestrator
from routes.inference import _choice_seed
from core.inference.worker import _dispatch_generate
from core.inference.api_monitor import ApiMonitor
from fastapi import HTTPException

from models.inference import ChatCompletionRequest, ChatMessage
from routes.inference import openai_chat_completions




_MISSING = object()

