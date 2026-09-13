# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The request and approval path shared by all Studio MCP tool loops."""

from __future__ import annotations

import copy
import base64
import functools
import inspect
import json
import secrets
import time
import threading
from urllib.parse import quote

from core.inference.mcp_image_disclosure import (
    McpImageDisclosureError,
    canonical_arguments_digest,
    issue_mcp_image_reference,
    model_schema_for_mapping,
    resolve_mcp_image_reference,
    revoke_mcp_image_references,
    validate_image_input_mappings,
    resolve_tool_only_image,
)
from state.tool_approvals import (
    McpImageDisclosureBinding,
    abort_mcp_image_disclosure,
    abort_tool_decision,
    begin_mcp_image_disclosure,
    consume_mcp_image_disclosure,
    revoke_mcp_image_disclosures,
    wait_mcp_image_disclosure,
    wait_tool_decision,
)


def prepare_image_tool_request(payload, *, subject, tools, cancel_event, ui_events):
    """Validate the private selection before any model receives this request."""
    selection = getattr(payload, "mcp_image_attachment", None)
    if selection is None:
        if tools is not None and getattr(payload, "mcp_enabled", False):
            from storage.studio_db import get_chat_setting_with_revision
            enabled, _ = get_chat_setting_with_revision("mcpImageAttachmentsEnabled")
            if enabled is True:
                public = copy.deepcopy(tools)
                for tool in public:
                    function = tool.get("function", {})
                    match = _mapping_for_name(function.get("name", ""))
                    if match:
                        _, mapping, schema, _ = match
                        function["parameters"] = model_schema_for_mapping(schema, mapping["field"])
                return None, public
        return None, tools
    _feature_revision()
    if not payload.stream or not ui_events or not payload.mcp_enabled:
        raise McpImageDisclosureError("Private MCP images require a Studio interactive tool stream")
    from storage.studio_db import get_latest_chat_user_message_id

    if (
        not payload.thread_id
        or get_latest_chat_user_message_id(payload.thread_id) != selection.message_id
    ):
        raise McpImageDisclosureError("Select an image from the current conversation message")
    durable_id = getattr(payload, "generation_run_id", None)
    if durable_id:
        from storage.chat_generation_runs_db import get_run
        durable = get_run(durable_id, owner_subject = subject)
        if (
            not durable
            or durable.get("thread_id") != payload.thread_id
            or durable.get("user_message_id") != selection.message_id
        ):
            raise McpImageDisclosureError("The image does not belong to this generation")
    image = resolve_tool_only_image(
        thread_id = payload.thread_id,
        message_id = selection.message_id,
        attachment_id = selection.attachment_id,
    )
    fingerprints = (
        base64.b64encode(image.data).decode("ascii").rstrip("="),
        base64.urlsafe_b64encode(image.data).decode("ascii").rstrip("="),
    )

    def contains_private(value):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, str):
            compact = "".join(value.split())
            return any(fingerprint in compact for fingerprint in fingerprints)
        if isinstance(value, dict):
            return any(contains_private(child) for child in value.values())
        if isinstance(value, (list, tuple)):
            return any(contains_private(child) for child in value)
        return False

    if contains_private(payload.messages):
        raise McpImageDisclosureError("The private image must be removed from model messages")
    if tools is None:
        return None, None
    run = McpImageToolRun(
        subject = subject,
        thread_id = payload.thread_id,
        session_id = payload.session_id,
        generation_id = durable_id or payload.cancel_id,
        selection = selection,
        cancel_event = cancel_event,
    )
    try:
        return run, run.rewrite_tools(tools)
    except BaseException:
        run.close()
        raise


def _feature_revision():
    from storage.studio_db import get_chat_setting_with_revision

    enabled, revision = get_chat_setting_with_revision("mcpImageAttachmentsEnabled")
    if enabled is not True or not revision:
        raise McpImageDisclosureError("Private MCP image sharing is disabled")
    return revision


def _mapping_for_name(name):
    from core.inference.mcp_client import get_cached_tools
    from storage import mcp_servers_db

    if not name.startswith("mcp__"):
        return None
    parts = name.split("__", 2)
    if len(parts) != 3:
        return None
    server = mcp_servers_db.get_server_for_tool(parts[1])
    if not server or not server.get("is_enabled"):
        return None
    try:
        mappings = json.loads(server.get("image_input_mappings_json") or "[]")
    except (TypeError, ValueError):
        raise McpImageDisclosureError("The image input configuration is invalid") from None
    if not isinstance(mappings, list) or any(not isinstance(mapping, dict) for mapping in mappings):
        raise McpImageDisclosureError("The image input configuration is invalid")
    mapping = next((m for m in mappings if m.get("tool") == parts[2]), None)
    if mapping is None:
        return None
    tools = get_cached_tools(server["id"])
    if not tools:
        raise McpImageDisclosureError("Refresh the MCP tools before sharing an image")
    _, digest = validate_image_input_mappings(mappings, tools)
    if digest != server.get("image_input_schema_digest"):
        raise McpImageDisclosureError("The MCP image input schema changed; configure it again")
    tool = next(t for t in tools if t.get("name") == parts[2])
    schema = tool.get("inputSchema")
    if not isinstance(schema, dict):
        schema = tool.get("input_schema")
    return server, mapping, schema, digest


class McpImageToolRun:
    """Only opaque selectors and scope live on a run; image bytes stay per-call."""

    def __init__(self, *, subject, thread_id, session_id, generation_id, selection, cancel_event):
        self.subject = subject
        self.thread_id = thread_id
        self.session_id = session_id or ""
        self.generation_id = generation_id or secrets.token_urlsafe(24)
        self.cancel_event = cancel_event
        self.feature_revision = _feature_revision()
        self.reference = issue_mcp_image_reference(
            subject = subject,
            thread_id = thread_id,
            generation_id = self.generation_id,
            message_id = selection.message_id,
            attachment_id = selection.attachment_id,
        )
        self.approvals = []
        self._lock = threading.Lock()
        self.closed = False

    def rewrite_tools(self, tools):
        public = copy.deepcopy(tools)
        mapped = 0
        for tool in public:
            function = tool.get("function", {})
            match = _mapping_for_name(function.get("name", ""))
            if match:
                _, mapping, schema, _ = match
                function["parameters"] = model_schema_for_mapping(
                    schema, mapping["field"], self.reference.reference
                )
                mapped += 1
        if not mapped:
            raise McpImageDisclosureError("Enable a configured MCP image tool to share this image")
        return public

    def prepare_call(self, name, arguments, call_id):
        match = _mapping_for_name(name)
        if match is None:
            return None
        if self.closed or (self.cancel_event is not None and self.cancel_event.is_set()):
            raise McpImageDisclosureError("Image sharing was cancelled")
        if _feature_revision() != self.feature_revision:
            raise McpImageDisclosureError("Image sharing settings changed")
        server, mapping, schema, digest = match
        if arguments.get(mapping["field"]) != self.reference.reference:
            raise McpImageDisclosureError("Select the supplied image attachment reference")
        arguments_digest = canonical_arguments_digest(arguments)
        ref, image = resolve_mcp_image_reference(
            self.reference.reference,
            subject = self.subject,
            thread_id = self.thread_id,
            generation_id = self.generation_id,
        )
        from core.inference.mcp_client import (
            prepare_mcp_image_recipient,
            mcp_image_recipient_location,
            parse_server_headers,
            close_mcp_image_recipient,
        )

        recipient = None
        try:
            recipient = prepare_mcp_image_recipient(
                server["url"],
                parse_server_headers(server),
                scope = "s={}:t={}".format(
                    quote(self.session_id, safe = ""), quote(self.thread_id, safe = "")
                ),
                use_oauth = bool(server.get("use_oauth")),
            )
            destination = mcp_image_recipient_location(recipient)
            current = _mapping_for_name(name)
            if (
                self.closed
                or current is None
                or current[0].get("config_revision") != server["config_revision"]
                or current[3] != digest
                or _feature_revision() != self.feature_revision
                or (self.cancel_event is not None and self.cancel_event.is_set())
            ):
                raise McpImageDisclosureError("The approved MCP recipient changed")
        except Exception:
            if recipient is not None:
                close_mcp_image_recipient(recipient)
            raise McpImageDisclosureError(
                "This MCP transport cannot safely share an image"
            ) from None
        binding = McpImageDisclosureBinding(
            subject = self.subject,
            session_id = self.session_id,
            thread_id = self.thread_id,
            generation_id = self.generation_id,
            call_id = call_id or secrets.token_urlsafe(16),
            attachment_ref = ref.reference,
            message_id = ref.message_id,
            attachment_id = ref.attachment_id,
            attachment_sha256 = image.sha256,
            mime_type = image.mime_type,
            size_bytes = image.size_bytes,
            server_id = server["id"],
            config_revision = server["config_revision"],
            tool_name = mapping["tool"],
            field = mapping["field"],
            encoding = mapping["encoding"],
            schema_digest = digest,
            public_arguments_digest = arguments_digest,
            feature_revision = self.feature_revision,
            recipient = recipient,
        )
        try:
            approval = McpImageApproval(self, binding, arguments, image, schema)
        except BaseException:
            close_mcp_image_recipient(recipient)
            raise
        approval.metadata = {
            "purpose": "mcp_image_disclosure",
            "previewUrl": "/api/chat/attachments/{}/{}/file".format(
                quote(ref.message_id, safe = ""), quote(ref.attachment_id, safe = "")
            ),
            "sizeBytes": image.size_bytes,
            "serverName": server.get("display_name") or server["id"],
            "toolName": mapping["tool"],
            "destination": destination,
            "field": mapping["field"],
            "encoding": mapping["encoding"],
            "status": "pending",
            "expiresAt": int((time.time() + 300) * 1000),
        }
        with self._lock:
            if self.closed:
                approval.close()
                raise McpImageDisclosureError("Image sharing was cancelled")
            self.approvals.append(approval)
        return approval

    def close(self):
        with self._lock:
            self.closed = True
            approvals, self.approvals = self.approvals, []
        for approval in approvals:
            approval.close()
        revoke_mcp_image_references(subject = self.subject, generation_id = self.generation_id)
        revoke_mcp_image_disclosures(subject = self.subject, generation_id = self.generation_id)


class McpImageApproval:
    def __init__(self, run, binding, arguments, image, schema):
        from core.inference.mcp_image_redaction import McpImageCallContext

        self.run, self.binding = run, binding
        self.approval_id, self.slot = begin_mcp_image_disclosure(binding)
        self.metadata = {}

        def commit(recipient):
            try:
                if run.closed or (run.cancel_event is not None and run.cancel_event.is_set()):
                    return False
                if _feature_revision() != binding.feature_revision:
                    return False
                ref, live_image = resolve_mcp_image_reference(
                    binding.attachment_ref,
                    subject = run.subject,
                    thread_id = run.thread_id,
                    generation_id = run.generation_id,
                )
                match = _mapping_for_name(
                    "mcp__{}__{}".format(binding.server_id, binding.tool_name)
                )
                if not match:
                    return False
                row, mapping, _, digest = match
                if (
                    row["config_revision"] != binding.config_revision
                    or digest != binding.schema_digest
                    or mapping["field"] != binding.field
                    or mapping["encoding"] != binding.encoding
                    or live_image.sha256 != binding.attachment_sha256
                    or canonical_arguments_digest(arguments) != binding.public_arguments_digest
                ):
                    return False
                return consume_mcp_image_disclosure(self.approval_id, binding, recipient)
            except Exception:
                return False

        try:
            self.context = McpImageCallContext(
                public_arguments = arguments,
                image = image,
                field = binding.field,
                encoding = binding.encoding,
                original_schema = schema,
                recipient = binding.recipient,
                commit = commit,
                tool_name = binding.tool_name,
            )
        except BaseException:
            abort_mcp_image_disclosure(self.slot, self.approval_id)
            raise

    def close(self):
        from core.inference.mcp_client import close_mcp_image_recipient

        abort_mcp_image_disclosure(self.slot, self.approval_id)
        self.context.close()
        close_mcp_image_recipient(self.binding.recipient)


def wait_call_decision(
    image_approval,
    slot,
    approval_id,
    cancel_event = None,
    ordinary_wait = wait_tool_decision,
):
    if image_approval is not None:
        decision = wait_mcp_image_disclosure(slot, approval_id, cancel_event, timeout = 300)
        if decision != "allow":
            image_approval.close()
        return decision
    return ordinary_wait(slot, approval_id, cancel_event)


def abort_call_decision(
    image_approval,
    slot,
    approval_id,
    ordinary_abort = abort_tool_decision,
):
    if image_approval is not None:
        image_approval.close()
    else:
        ordinary_abort(slot, approval_id)


def mcp_image_run_lifetime(function):
    """Revoke even when a consumer closes immediately after the approval card."""
    if inspect.isasyncgenfunction(function):

        @functools.wraps(function)
        async def async_wrapper(*args, **kwargs):
            run = kwargs.get("mcp_image_run")
            generator = function(*args, **kwargs)
            try:
                async for event in generator:
                    yield event
            finally:
                try:
                    await generator.aclose()
                finally:
                    if run is not None:
                        run.close()

        return async_wrapper

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        run = kwargs.get("mcp_image_run")
        try:
            yield from function(*args, **kwargs)
        finally:
            if run is not None:
                run.close()

    return wrapper
