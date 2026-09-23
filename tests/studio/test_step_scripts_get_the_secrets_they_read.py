# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A step that reads a secret from its environment has to be given it.

Secrets here are step-scoped on purpose (test_cached_paths_hold_no_credentials.py says why),
and scripts read them through the environment rather than a `${{ secrets.* }}` expression
spliced into the script, so the value never lands in the step's temporary shell file or in
argv. That leaves one way to get it wrong that nothing else sees: a script that reads
`os.environ["DOCKER_API_KEY"]` or `$DOCKER_API_KEY` in a step whose `env:` never maps it.
Python raises KeyError, the shell expands to an empty string, and the step fails or quietly
does nothing, only on the privileged run that holds the secret, which a pull request never
exercises.

That is what happened when the Docker Hub token exchanges moved to the environment: the Hub
README step was given `DOCKER_API_KEY`, the cleanup step beside it and the ROCm workflow's
README step were not, and every publish run from then on left its handle tags on Docker Hub.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[2] / ".github" / "workflows").glob("*.yml"))

_SECRET = re.compile(r"secrets\.([A-Za-z_][A-Za-z0-9_]*)")
_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.S)


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _strings(key)
            yield from _strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _strings(item)


def _reads(script: str, name: str) -> bool:
    """Whether a run script reads NAME from its environment, in shell or in inline Python.

    Includes bash indirect expansion over a list of names, the shape release-desktop.yml's
    notarization check uses: `for required in APPLE_ID ...; do [ -z "${!required:-}" ]`.
    """
    n = re.escape(name)
    if (
        re.search(rf"os\.environ\[\s*['\"]{n}['\"]\s*\]", script)
        or re.search(rf"os\.(?:environ\.get|getenv)\(\s*['\"]{n}['\"]", script)
        or re.search(rf"\$\{{?{n}(?![A-Za-z0-9_])", script)
        # PowerShell and cmd, for the Windows steps.
        or re.search(rf"\$env:{n}(?![A-Za-z0-9_])", script, re.I)
        or re.search(rf"%{n}%", script)
        or re.search(rf"GetEnvironmentVariable\(\s*['\"]{n}['\"]", script)
    ):
        return True
    for var, words in re.findall(r"\bfor\s+(\w+)\s+in\s+([^;\n]+)", script):
        if name in words.split() and re.search(rf"\$\{{!{var}(?![A-Za-z0-9_])", script):
            return True
    return False


def _env_blocks(doc: dict):
    yield doc.get("env") or {}
    for job in (doc.get("jobs") or {}).values():
        yield (job or {}).get("env") or {}
        for step in (job or {}).get("steps") or []:
            yield step.get("env") or {}


# The secret each env key is supplied from, as every workflow maps it today. GitHub expands an
# unknown `secrets.*` name to an empty string without complaint, and CI cannot list the
# repository's secret names to check against, so a typo (`secrets.DOCKER_API_KE`) or a mapping to
# the wrong existing secret reads as a valid expression everywhere except the privileged run that
# needs it. Pinning the pairs turns both into a failure here. A genuinely new secret goes in this
# table in the same change that adds it, once it is confirmed to exist.
_SECRET_FOR = {
    "APPLE_CERTIFICATE": "APPLE_CERTIFICATE",
    "APPLE_CERTIFICATE_PASSWORD": "APPLE_CERTIFICATE_PASSWORD",
    "APPLE_ID": "APPLE_ID",
    "APPLE_PASSWORD": "APPLE_PASSWORD",
    "APPLE_SIGNING_IDENTITY": "APPLE_SIGNING_IDENTITY",
    "APPLE_TEAM_ID": "APPLE_TEAM_ID",
    "AZURE_CERTIFICATE_PROFILE_NAME": "AZURE_CERTIFICATE_PROFILE_NAME",
    "AZURE_CLIENT_ID": "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET": "AZURE_CLIENT_SECRET",
    "AZURE_TENANT_ID": "AZURE_TENANT_ID",
    "AZURE_TRUSTED_SIGNING_ACCOUNT_NAME": "AZURE_TRUSTED_SIGNING_ACCOUNT_NAME",
    "DOCKER_API_KEY": "DOCKER_API_KEY",
    "GH_TOKEN": "GITHUB_TOKEN",
    "GITHUB_TOKEN": "GITHUB_TOKEN",
    "HF_TOKEN": "HF_TOKEN",
    "KAGGLE_API_TOKEN": "KAGGLE_API_TOKEN",
    "KAGGLE_API_TOKEN_2": "KAGGLE_API_TOKEN_2",
    "KEYCHAIN_PASSWORD": "KEYCHAIN_PASSWORD",
    "TAURI_SIGNING_PRIVATE_KEY": "TAURI_SIGNING_PRIVATE_KEY",
    "VT_API_KEY": "VIRUS_TOTAL_API_TOKEN",
}

# Indexed lookups, `${{ secrets[matrix.secret_name] }}`: the Kaggle jobs pick an account at run
# time, so the name is a matrix value. Each key pins the one index expression it may use and the
# secrets that index may resolve to; a static matrix is checked value by value.
_INDEXED_FOR = {
    "KAGGLE_API_TOKEN": ("matrix.secret_name", {"KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_2"}),
}
_INDEXED = re.compile(r"secrets\[\s*([^\]]+?)\s*\]")


def _secret_backed_names() -> frozenset[str]:
    """Every name a step could be expected to receive a secret under, across all workflows.

    A secret's own name, and every env key any workflow maps from a `secrets.*` expression:
    `VT_API_KEY: ${{ secrets.VIRUS_TOTAL_API_TOKEN }}` makes VT_API_KEY secret-backed even in a
    workflow that has lost its only mapping of it, which is exactly the file a per-file scan
    would call clean.
    """
    # Seeded from the reviewed table, so a name whose only reference was the mapping that got
    # deleted is still tracked.
    names = set(_SECRET_FOR) | set(_SECRET_FOR.values())
    for allowed in _INDEXED_FOR.values():
        names |= allowed[1]
    for path in WORKFLOWS:
        doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
        # From parsed values, not the raw text: a comment that explains `secrets.A || secrets.B`
        # would otherwise make B a secret.
        for value in _strings(doc):
            for expression in _EXPRESSION.findall(value):
                names.update(_SECRET.findall(expression))
        for env in _env_blocks(doc):
            for key, value in env.items():
                if isinstance(value, str) and any(
                    _SECRET.search(e) or _INDEXED.search(e) for e in _EXPRESSION.findall(value)
                ):
                    names.add(key)
    return frozenset(names)


SECRET_BACKED = _secret_backed_names()


def _supplies_a_secret(value) -> bool:
    """An env entry counts only when its value draws on a `secrets.*` expression: an empty
    string, a `vars.*` lookup or a misspelled expression is present and still hands the
    script nothing. `github.token` is the run's own token and counts, as GH_TOKEN mappings use it."""
    return isinstance(value, str) and any(
        _SECRET.search(expression)
        or _INDEXED.search(expression)
        or re.search(r"\bgithub\.token\b", expression)
        for expression in _EXPRESSION.findall(value)
    )


def _unmapped(path: Path) -> list[str]:
    doc = yaml.safe_load(path.read_text(encoding = "utf-8")) or {}
    workflow_env = doc.get("env") or {}
    found = []
    for job_name, job in (doc.get("jobs") or {}).items():
        job_env = (job or {}).get("env") or {}
        for index, step in enumerate((job or {}).get("steps") or []):
            script = step.get("run")
            if not isinstance(script, str):
                continue
            env = {**workflow_env, **job_env, **(step.get("env") or {})}
            for name in sorted(SECRET_BACKED):
                if _reads(script, name) and not _supplies_a_secret(env.get(name)):
                    label = step.get("name") or f"step {index}"
                    why = (
                        "without mapping it" if name not in env else f"but maps it to {env[name]!r}"
                    )
                    found.append(f"{path.name} :: {job_name} :: {label} reads {name} {why}")
    return found


def test_the_workflows_are_found():
    assert len(WORKFLOWS) > 20, "the glob stopped finding the workflows"


@pytest.mark.parametrize("path", WORKFLOWS, ids = [p.name for p in WORKFLOWS])
def test_every_secret_a_step_reads_is_in_its_env(path):
    unmapped = _unmapped(path)
    assert not unmapped, (
        "these steps read a secret from the environment without an env: entry that supplies it, "
        "so it is empty or a KeyError on the run that holds it:\n  " + "\n  ".join(unmapped)
    )


def test_the_reader_sees_every_spelling_the_workflows_use():
    assert _reads("python3 -c 'import os; os.environ[\"K\"]'", "K")
    assert _reads("os.environ.get('K', '')", "K")
    assert _reads('os.getenv("K")', "K")
    assert _reads('curl -H "Bearer $K"', "K")
    assert _reads('echo "${K}"', "K")
    assert _reads("Write-Host $env:K", "K")
    assert _reads("echo %K%", "K")
    assert _reads("[Environment]::GetEnvironmentVariable('K')", "K")
    assert not _reads("Write-Host $env:K_OTHER", "K")
    assert _reads('for v in J K L; do [ -z "${!v:-}" ] && exit 1; done', "K")
    # A plain loop over the names, with no indirect read, reads none of them.
    assert not _reads("for v in J K L; do echo $v; done", "K")
    # A longer name that starts with K is not K, and an expression is not an env read.
    assert not _reads('echo "$K_OTHER"', "K")
    assert not _reads("echo ${{ secrets.K }}", "K")


def test_an_aliased_secret_is_tracked_under_the_name_the_script_reads():
    # The repository secret is VIRUS_TOTAL_API_TOKEN and the scripts read VT_API_KEY.
    assert "VT_API_KEY" in SECRET_BACKED
    assert "VIRUS_TOTAL_API_TOKEN" in SECRET_BACKED
    # kaggle-t4-notebook-ci.yml explains `secrets.A || secrets.B` in a comment; neither is real.
    assert "A" not in SECRET_BACKED and "B" not in SECRET_BACKED
    # GitHub does not export the run token to the environment by itself; a script reading
    # $GITHUB_TOKEN needs the mapping like any other secret.
    assert "GITHUB_TOKEN" in SECRET_BACKED


def test_an_inline_read_of_an_alias_is_caught_in_a_workflow_that_never_maps_it(tmp_path):
    """The alias is known from the workflow that maps it, so a file that lost its only mapping
    is still checked. Scripts the step merely invokes are out of scope: many read a secret
    optionally by design (the pinned-symbol suites read GH_TOKEN only when present), and from
    the file alone an intended absence and a lost mapping look the same."""
    workflow = tmp_path / "w.yml"
    workflow.write_text(
        "on: push\n"
        "jobs:\n"
        "  scan:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Scan\n"
        "        run: |\n"
        '          curl -H "x-apikey: $VT_API_KEY" https://example.invalid\n',
        encoding = "utf-8",
    )
    assert _unmapped(workflow) == ["w.yml :: scan :: Scan reads VT_API_KEY without mapping it"]
    mapped = workflow.read_text(encoding = "utf-8").replace(
        "      - name: Scan\n",
        "      - name: Scan\n        env:\n          VT_API_KEY: ${{ secrets.VIRUS_TOTAL_API_TOKEN }}\n",
    )
    workflow.write_text(mapped, encoding = "utf-8")
    assert _unmapped(workflow) == []


def test_an_entry_that_supplies_no_secret_does_not_count():
    assert _supplies_a_secret("${{ secrets.DOCKER_API_KEY }}")
    assert _supplies_a_secret("${{ github.event_name == 'push' && secrets.HF_TOKEN || '' }}")
    assert not _supplies_a_secret("")
    assert not _supplies_a_secret(None)
    assert not _supplies_a_secret("${{ vars.DOCKER_API_KEY }}")
    assert not _supplies_a_secret("${{ secret.DOCKER_API_KEY }}")
    assert not _supplies_a_secret("secrets.DOCKER_API_KEY")
    assert _supplies_a_secret("${{ github.token }}")


def _static_matrix_values(job: dict, field: str):
    """The values a static matrix gives `field`, or None when the matrix is built at run time."""
    matrix = ((job or {}).get("strategy") or {}).get("matrix")
    if not isinstance(matrix, dict):
        return None
    values = []
    if isinstance(matrix.get(field), list):
        values += matrix[field]
    for entry in matrix.get("include") or []:
        if isinstance(entry, dict) and field in entry:
            values.append(entry[field])
    return values


def _check_mapping(key, value, job, name, wrong):
    if not isinstance(value, str):
        return
    expressions = _EXPRESSION.findall(value)
    drawn = {n for e in expressions for n in _SECRET.findall(e)}
    indexed = [x for e in expressions for x in _INDEXED.findall(e)]
    if not drawn and not indexed:
        return
    if drawn:
        if key not in _SECRET_FOR:
            wrong.append(f"{name}: {key} is a new secret mapping; add it to _SECRET_FOR")
        elif drawn != {_SECRET_FOR[key]}:
            wrong.append(f"{name}: {key} draws on {sorted(drawn)}, not {_SECRET_FOR[key]}")
    for index in indexed:
        if key not in _INDEXED_FOR:
            wrong.append(f"{name}: {key} is a new indexed secret mapping; add it to _INDEXED_FOR")
            continue
        expected, allowed = _INDEXED_FOR[key]
        if index != expected:
            wrong.append(f"{name}: {key} indexes secrets with {index}, not {expected}")
            continue
        field = expected.split(".", 1)[1]
        values = _static_matrix_values(job, field)
        if values is not None:
            if not values:
                wrong.append(f"{name}: {key} indexes {expected}, which the matrix never sets")
            for v in values:
                if v not in allowed:
                    wrong.append(f"{name}: {key} resolves to {v!r}, not one of {sorted(allowed)}")


def _misdrawn(doc: dict, name: str) -> list[str]:
    wrong = []
    for key, value in (doc.get("env") or {}).items():
        _check_mapping(key, value, None, name, wrong)
    for job in (doc.get("jobs") or {}).values():
        for key, value in ((job or {}).get("env") or {}).items():
            _check_mapping(key, value, job, name, wrong)
        for step in (job or {}).get("steps") or []:
            for key, value in (step.get("env") or {}).items():
                _check_mapping(key, value, job, name, wrong)
    return wrong


@pytest.mark.parametrize("path", WORKFLOWS, ids = [p.name for p in WORKFLOWS])
def test_every_secret_mapping_draws_on_the_secret_its_key_is_known_by(path):
    wrong = _misdrawn(yaml.safe_load(path.read_text(encoding = "utf-8")) or {}, path.name)
    assert not wrong, "\n".join(wrong)


def test_a_misspelled_or_swapped_secret_is_caught():
    def doc(value):
        return {"jobs": {"j": {"steps": [{"env": {"DOCKER_API_KEY": value}, "run": "true"}]}}}

    assert _misdrawn(doc("${{ secrets.DOCKER_API_KEY }}"), "w") == []
    assert _misdrawn(doc("${{ secrets.DOCKER_API_KE }}"), "w") == [
        "w: DOCKER_API_KEY draws on ['DOCKER_API_KE'], not DOCKER_API_KEY"
    ]
    assert _misdrawn(doc("${{ secrets.HF_TOKEN }}"), "w") == [
        "w: DOCKER_API_KEY draws on ['HF_TOKEN'], not DOCKER_API_KEY"
    ]


def test_a_step_reading_the_run_token_must_map_it(tmp_path):
    workflow = tmp_path / "t.yml"
    workflow.write_text(
        "on: push\n"
        "jobs:\n"
        "  api:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - name: Call\n"
        "        run: |\n"
        '          curl -H "Authorization: Bearer $GITHUB_TOKEN" https://api.github.com\n',
        encoding = "utf-8",
    )
    assert _unmapped(workflow) == ["t.yml :: api :: Call reads GITHUB_TOKEN without mapping it"]


def test_a_deleted_only_mapping_is_still_tracked():
    # APPLE_CERTIFICATE has one reference in the whole repository, the mapping itself.
    assert {"APPLE_CERTIFICATE", "KAGGLE_API_TOKEN_2"} <= SECRET_BACKED


def test_an_indexed_lookup_is_checked():
    def doc(index, values):
        return {
            "jobs": {
                "j": {
                    "strategy": {"matrix": {"include": [{"secret_name": v} for v in values]}},
                    "steps": [{"env": {"KAGGLE_API_TOKEN": "${{ secrets[" + index + "] }}"}}],
                }
            }
        }

    good = ["KAGGLE_API_TOKEN", "KAGGLE_API_TOKEN_2"]
    assert _misdrawn(doc("matrix.secret_name", good), "w") == []
    assert _misdrawn(doc("matrix.secert_name", good), "w") == [
        "w: KAGGLE_API_TOKEN indexes secrets with matrix.secert_name, not matrix.secret_name"
    ]
    assert _misdrawn(doc("matrix.secret_name", ["KAGGLE_API_TOKN"]), "w") == [
        "w: KAGGLE_API_TOKEN resolves to 'KAGGLE_API_TOKN', not one of "
        "['KAGGLE_API_TOKEN', 'KAGGLE_API_TOKEN_2']"
    ]
    assert _supplies_a_secret("${{ secrets[matrix.secret_name] }}")
