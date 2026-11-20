# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One Python or Terminal call's network verdicts: the owner's lists first, then a question in the chat."""

from __future__ import annotations

import threading

from state import network_requests
from utils import network_firewall

ASK_TIMEOUT_SECONDS = 300.0


class NetworkGate:
    """Built on the tool's own thread, asked from the proxy's: the account is read once, here."""

    def __init__(self, tool: str, session_id, thread_id, output_callback, cancel_event) -> None:
        from utils.account_context import current_account_id, is_owner_context, via_api_key_context

        self.tool = tool
        self.session_id = session_id or ""
        self.chat = thread_id or session_id or ""
        self.account_id = current_account_id()
        self.can_always = is_owner_context() and not via_api_key_context()
        # No chat to show a question in: an API caller or a scheduled run.
        self._output = None if via_api_key_context() else output_callback
        self._cancel = cancel_event
        self._closed = threading.Event()
        self._one_question = threading.Lock()
        self.refusals: list[str] = []

    def _stopped(self) -> bool:
        return self._closed.is_set() or (self._cancel is not None and self._cancel.is_set())

    def _question(self, kind: str, host: str, port: int, **extra) -> bool:
        request_id, slot = network_requests.begin_request(self.session_id, self.account_id, host)
        question = {
            "id": request_id,
            "session_id": self.session_id,
            "kind": kind,
            "host": host,
            "port": port,
            "tool": self.tool,
            "can_always": kind == "website" and self.can_always,
            **extra,
        }
        try:
            self._output({"network_request": question})
            decision = network_requests.wait_request(
                request_id,
                slot,
                self._stopped,
                ASK_TIMEOUT_SECONDS,
                # A reloaded tab hears only what is sent after it rejoins the run.
                remind = lambda: self._output({"network_request": question}),
            )
        finally:
            self._output({"network_request": {"id": request_id, "done": True}})
        return decision != "deny"

    def _refuse(self, host: str, port: int, refusal: str) -> None:
        network_requests.record(self.tool, host, port, "refused")
        if refusal not in self.refusals:
            self.refusals.append(refusal)

    def _ask(self, host: str, port: int) -> str | None:
        if self._output is None:
            return f"{host} is not an allowed website and nobody is in the chat to approve it"
        with self._one_question:
            # Answered, or the lists changed, while this connection waited its turn.
            if network_requests.granted(self.account_id, self.chat, host):
                return None
            verdict = network_firewall.decide(host)
            if verdict != "ask":
                return None if verdict == "allow" else _refusal(host)
            declined = f"the connection to {host} was not approved in the chat"
            # A client that retries must not put the same question back up.
            if self._stopped() or declined in self.refusals:
                return declined
            if not self._question("website", host, port):
                return declined
            network_requests.grant(self.account_id, self.chat, host)
            return None

    def check(self, host: str, port: int) -> str | None:
        host = host.lower().rstrip(".")
        verdict = network_firewall.decide(host)
        if verdict == "allow":
            return None
        if verdict == "deny":
            refusal = _refusal(host)
        elif network_requests.granted(self.account_id, self.chat, host):
            return None
        else:
            refusal = self._ask(host, port)
        if refusal:
            self._refuse(host, port, refusal)
        return refusal

    def sending(self, host: str, port: int, sent: int) -> bool:
        """False stops the connection: the chat has sent this website its budget and may send no more."""
        host = host.lower().rstrip(".")
        total = network_requests.add_sent(self.account_id, self.chat, host, sent)
        if total <= network_firewall.UPLOAD_BUDGET_BYTES or network_requests.granted(
            self.account_id, self.chat, host, "uploads"
        ):
            return True
        megabytes = network_firewall.UPLOAD_BUDGET_BYTES // 1_000_000
        refusal = f"sending more than {megabytes} MB to {host} was not approved in the chat"
        with self._one_question:
            allowed = network_requests.granted(self.account_id, self.chat, host, "uploads") or (
                self._output is not None
                and not self._stopped()
                and refusal not in self.refusals
                and self._question("upload", host, port, sent = total)
            )
        if not allowed:
            self._refuse(host, port, refusal)
            return False
        network_requests.grant(self.account_id, self.chat, host, "uploads")
        return True

    def note(self) -> str:
        """For the end of the tool's output: most clients show a refused tunnel as a bare 403."""
        if not self.refusals:
            return ""
        return "\n[Unsloth network] Refused: " + "; ".join(self.refusals[:10]) + "\n"

    def record(self, host: str, port: int, sent: int, received: int) -> None:
        network_requests.record(
            self.tool, host.lower().rstrip("."), port, "allowed", sent, received
        )

    def close(self) -> None:
        self._closed.set()


def _refusal(host: str) -> str:
    if network_firewall.blocks(host):
        return f"{host} is a blocked website (Settings > Security)"
    return f"{host} is not an allowed website (Settings > Security > Allowed websites)"
