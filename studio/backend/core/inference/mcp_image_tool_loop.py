# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The request and approval path shared by all Studio MCP tool loops."""

from __future__ import annotations

import copy
import functools
import inspect
import json
import secrets
import threading
from urllib.parse import quote

from core.inference.mcp_image_disclosure import (
    McpImageDisclosureError,
    canonical_arguments_digest,
    issue_mcp_image_reference,
    model_schema_for_mapping,
    resolve_mcp_image_reference,
    revoke_mcp_image_references,
    stored_image_input_mappings,
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


_COUNT_ATTACHMENT_REFERENCE = "mcp-image-ref-xQ7m9K2vP4sN8dF1hJ6cL0wR3tY5uB7eG9aZ2iC4oEU"


def _image_policy_revisions():
    from storage import mcp_servers_db

    current = []
    for server in mcp_servers_db.list_servers():
        if (
            server.get("is_enabled")
            and server.get("allow_image_attachments")
            and stored_image_input_mappings(server)
        ):
            current.append((server["id"], int(server.get("config_revision") or 0)))
    return sorted(current)


def _request_contains_model_image(payload):
    if getattr(payload, "image_base64", None):
        return True
    for message in getattr(payload, "messages", None) or []:
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if not isinstance(content, list):
            continue
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type in {"image_url", "input_image", "image"}:
                return True
    return False


def _request_contains_new_model_image(payload):
    """Detect an image on the newest user turn, including a distinct legacy field."""
    latest_user_seen = False
    latest_image = None
    newest_has_image = False
    for message in reversed(list(getattr(payload, "messages", None) or [])):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        values = []
        has_image_part = False
        for part in content if isinstance(content, list) else []:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type not in {"image_url", "input_image", "image"}:
                continue
            has_image_part = True
            value = (
                part.get("image_url")
                if isinstance(part, dict)
                else getattr(part, "image_url", None)
            )
            if isinstance(value, dict):
                value = value.get("url")
            elif value is not None and not isinstance(value, str):
                value = getattr(value, "url", None)
            if value is None and part_type == "image":
                value = (
                    part.get("image") if isinstance(part, dict) else getattr(part, "image", None)
                )
            if isinstance(value, str) and value:
                values.append(value.partition(",")[2] if value.startswith("data:") else value)
        if not latest_user_seen:
            newest_has_image = has_image_part
            latest_user_seen = True
        if values and latest_image is None:
            latest_image = values[0]
    legacy = getattr(payload, "image_base64", None)
    return newest_has_image or bool(legacy and legacy != latest_image)


def _validate_image_policy_snapshot(payload):
    snapshot = getattr(payload, "mcp_image_policy", None)
    selection = getattr(payload, "mcp_image_attachment", None)
    if snapshot is None:
        if selection is not None:
            raise McpImageDisclosureError("MCP image sharing settings must be checked again")
        if (
            getattr(payload, "mcp_enabled", False)
            and _request_contains_model_image(payload)
            and _image_policy_revisions()
        ):
            raise McpImageDisclosureError("MCP image sharing settings must be checked again")
        return
    current = _image_policy_revisions() if getattr(payload, "mcp_enabled", False) else []
    supplied = [(item.server_id, item.config_revision) for item in snapshot.servers]
    if (
        supplied != sorted(set(supplied))
        or current != supplied
        or snapshot.tool_only != bool(current)
        or (selection is not None and not snapshot.tool_only)
        or (selection is None and snapshot.tool_only and _request_contains_new_model_image(payload))
    ):
        raise McpImageDisclosureError(
            "MCP image sharing settings changed. Remove and attach the image again."
        )


def prepare_image_tool_request(payload, *, subject, tools, cancel_event, ui_events):
    """Validate the private selection before any model receives this request."""
    _validate_image_policy_snapshot(payload)
    selection = getattr(payload, "mcp_image_attachment", None)
    if selection is None:
        if tools is not None and getattr(payload, "mcp_enabled", False):
            return None, rewrite_image_tool_schemas(tools)
        return None, tools
    if not payload.stream or not ui_events or not payload.mcp_enabled:
        raise McpImageDisclosureError("Private MCP images require a Studio interactive tool stream")
    if not payload.thread_id:
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
    from core.inference.mcp_image_redaction import contains_mcp_image_echo

    def dump_models(value):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, dict):
            return {key: dump_models(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [dump_models(child) for child in value]
        return value

    model_inputs = {
        "messages": dump_models(payload.messages),
        "image_base64": getattr(payload, "image_base64", None),
    }
    try:
        contains_private = contains_mcp_image_echo(model_inputs, image.data)
    except Exception:
        raise McpImageDisclosureError("Private image message validation failed") from None
    if contains_private:
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


def _mapping_for_name(name):
    from core.inference.mcp_client import get_cached_tools
    from storage import mcp_servers_db

    if not name.startswith("mcp__"):
        return None
    parts = name.split("__", 2)
    if len(parts) != 3:
        return None
    server = mcp_servers_db.get_server_for_tool(parts[1])
    if not server or not server.get("is_enabled") or not server.get("allow_image_attachments"):
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
    _, digest = validate_image_input_mappings(mappings, tools, server_key = parts[1])
    if digest != server.get("image_input_schema_digest"):
        raise McpImageDisclosureError("The MCP image input schema changed; configure it again")
    tool = next(t for t in tools if t.get("name") == parts[2])
    schema = tool.get("inputSchema")
    if not isinstance(schema, dict):
        schema = tool.get("input_schema")
    return server, mapping, schema, digest


def rewrite_image_tool_schemas(
    tools,
    attachment_ref = None,
    *,
    require_mapped = False,
):
    """Copy tool schemas and replace configured image payload fields with public selectors."""
    public = copy.deepcopy(tools)
    mapped = 0
    for tool in public:
        function = tool.get("function", {})
        match = _mapping_for_name(function.get("name", ""))
        if match:
            _, mapping, schema, _ = match
            function["parameters"] = model_schema_for_mapping(
                schema, mapping["field"], attachment_ref
            )
            mapped += 1
    if require_mapped and not mapped:
        raise McpImageDisclosureError("Enable a configured MCP image tool to share this image")
    return public


def rewrite_image_tool_schemas_for_count(payload, tools):
    """Render the same public MCP schemas a generation will put in the model prompt."""
    selection = getattr(payload, "mcp_image_attachment", None)
    if selection is not None:
        # Generation references contain 32 urlsafe bytes (43 encoded characters). The count
        # must not mint a live disclosure reference, but it still prices the enum and description
        # that the model will receive.
        attachment_ref = _COUNT_ATTACHMENT_REFERENCE
    else:
        attachment_ref = None
    return rewrite_image_tool_schemas(tools, attachment_ref, require_mapped = selection is not None)


class McpImageToolRun:
    """Only opaque selectors and scope live on a run; image bytes stay per-call."""

    def __init__(self, *, subject, thread_id, session_id, generation_id, selection, cancel_event):
        self.subject = subject
        self.thread_id = thread_id
        self.session_id = session_id or ""
        self.generation_id = generation_id or secrets.token_urlsafe(24)
        self.cancel_event = cancel_event
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
        return rewrite_image_tool_schemas(tools, self.reference.reference, require_mapped = True)

    def prepare_call(self, name, arguments, call_id):
        match = _mapping_for_name(name)
        if match is None:
            return None
        if self.closed or (self.cancel_event is not None and self.cancel_event.is_set()):
            raise McpImageDisclosureError("Image sharing was cancelled")
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
            mcp_image_recipient_remaining_ms,
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
                cancel_event = self.cancel_event,
            )
            destination = mcp_image_recipient_location(recipient)
            expires_in_ms = mcp_image_recipient_remaining_ms(recipient)
            if expires_in_ms <= 0:
                raise McpImageDisclosureError("Private MCP image recipient expired")
            current = _mapping_for_name(name)
            if (
                self.closed
                or current is None
                or current[0].get("config_revision") != server["config_revision"]
                or current[3] != digest
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
            "expiresInMs": expires_in_ms,
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


def begin_call_decision(
    decision, image_approval, needs_confirm, session_id, ordinary_new, ordinary_begin
):
    """Register consent before publishing the same tool-start event in every loop."""
    if image_approval:
        approval_id, slot = image_approval.approval_id, image_approval.slot
    else:
        approval_id = ordinary_new() if needs_confirm else ""
        slot = ordinary_begin(session_id, approval_id) if needs_confirm else None
    event = decision.tool_start_event()
    event["approval_id"] = approval_id
    event["awaiting_confirmation"] = needs_confirm
    if image_approval is not None:
        event["image_disclosure"] = image_approval.metadata
    return approval_id, slot, event


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
