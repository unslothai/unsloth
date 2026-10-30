# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""lm_eval's ``gguf`` backend, pointed at Studio instead of a bare llama-server.

The upstream backend tokenizes through ``<server>/tokenize``. Studio requires a
bearer key on every route and serves tokenize under ``/v1``. This subclass
sends the run's key, uses ``/v1/tokenize``, and supplies neutral
sampling defaults so evaluation does not inherit Studio's chat recommendations.
"""

import logging
import time

import requests
from requests.exceptions import RequestException

from lm_eval.api.registry import register_model
from lm_eval.models.gguf import GGUFLM

logger = logging.getLogger(__name__)

STUDIO_GGUF_MODEL = "unsloth-studio-gguf"


@register_model(STUDIO_GGUF_MODEL)
class StudioGGUFLM(GGUFLM):
    def __init__(
        self,
        api_key = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tokenize_url = self.server_url + "/v1/tokenize"
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _post_with_retries(
        self,
        url,
        payload,
        retries = 3,
        delay = 5,
    ):
        if url == self.completions_url:
            # Omitted values otherwise inherit per-model chat recommendations
            # in Studio's completions route. Penalties affect greedy decoding
            # too, and must also stay neutral for teacher-forced scoring.
            payload = {
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repeat_penalty": 1.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                **payload,
            }
        from core.benchmark.orchestrator import worker_cancelled

        for _ in range(retries):
            if worker_cancelled():
                raise RuntimeError("The benchmark was stopped.")
            try:
                response = self._session.post(url, json = payload, timeout = self.timeout)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                logger.error("RequestException: %s", e)
                time.sleep(delay)
        raise RuntimeError(f"Failed to get a valid response after {retries} retries.")

    def _detect_total_slots(self):
        try:
            params = {"model": self.model} if self.model is not None else None
            response = self._session.get(f"{self.server_url}/props", params = params, timeout = 10)
            response.raise_for_status()
            total_slots = response.json().get("total_slots")
            if isinstance(total_slots, int) and total_slots > 0:
                return total_slots
        except (RequestException, ValueError) as e:
            logger.debug("Could not query /props for slot count: %s", e)
        return None
