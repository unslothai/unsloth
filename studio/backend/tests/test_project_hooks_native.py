import pytest
from core.agent_workspace import hook_runtime
from core.agent_workspace.hook_context import AgentWorkspaceError
from routes import project_hooks
from .test_project_verification_native import native_project


def _review_hooks(
    workspace,
    command,
    *,
    event = "PreToolUse",
):
    import json

    path = workspace.root / ".codex" / "hooks.json"
    path.parent.mkdir(exist_ok = True)
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    event: [{"hooks": [{"type": "command", "command": command, "timeout": 10}]}]
                }
            }
        )
    )
    snapshot = project_hooks.project_hooks(workspace.project_id, _current_subject = "ui")
    return project_hooks.project_hooks_trust(
        workspace.project_id,
        project_hooks.TrustProjectHooksRequest(
            contentHash = snapshot["contentHash"], workspaceRevision = 0, revision = 0
        ),
        via_api_key = False,
        _current_subject = "ui",
    )


def test_native_hook_revalidates_trust_after_bind_before_user_code(native_project, monkeypatch):
    native, workspace = native_project
    _review_hooks(workspace, "echo must-not-run > forbidden")
    original_bind = native._BubblewrapLifecycle.bind

    def revoke_after_bind(self, *args, **kwargs):
        bound = original_bind(self, *args, **kwargs)
        project_hooks.project_hooks_revoke(
            workspace.project_id,
            project_hooks.RevokeProjectHooksRequest(revision = 1),
            via_api_key = False,
            _current_subject = "ui",
        )
        return bound

    monkeypatch.setattr(native._BubblewrapLifecycle, "bind", revoke_after_bind)
    with pytest.raises(AgentWorkspaceError, match = "changed"):
        hook_runtime.run_tool_hooks(workspace.project_id, "PreToolUse", "terminal", {})
    assert not (workspace.root / "forbidden").exists()


def test_native_reviewed_pre_hook_blocks_public_edit_tool(native_project):
    _native, workspace = native_project
    from core.inference.tools import execute_tool, project_session_id

    _review_hooks(workspace, "exit 2")
    target = workspace.root / "source.txt"
    target.write_text("before")
    result = execute_tool(
        "edit_file",
        {"path": "source.txt", "old_string": "before", "new_string": "after"},
        session_id = project_session_id(workspace.project_id),
    )
    assert "hook" in result.lower() and "failed" in result.lower(), result
    assert target.read_text() == "before"
