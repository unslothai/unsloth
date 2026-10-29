# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validate a Flow document and compile it into an execution plan.

Checks: schema and limits, known node types, ports and port types, one edge per non-multi input, flow edges only
between story control handles, acyclic data edges, For Each bodies, required inputs. A For Each node's body is the
nodes whose ``group`` is the For Each node's id, plus the other nodes of the For Each node's own group. Edges into a
For Each node's ``collect`` port come from inside its body and are not dependencies. A Loop node's body
is built the same way, plus the recipe group its ``feedback`` edge feeds; that edge carries one round's
failures into the next round and is not a dependency either, so the data edges stay acyclic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from pydantic import ValidationError

from core.flows import schema
from core.flows.recipe_compile import RecipeCompileError, compile_recipe, recipe_graph
from core.flows.recipe_import import COLUMN_TYPE, GENERATE_TYPE
from core.flows.specs import SPECS_BY_TYPE

FOR_EACH = "logic.for_each"
LOOP = "logic.loop"
REPEATS = (FOR_EACH, LOOP)
FEEDBACK = "feedback"
COLLECT_PORT = "collect"
# The compiled `{recipe, run}` a Recipe Dataset node runs, filled in from its group at compile time.
RECIPE_PARAM = "_recipe"
FOR_EACH_INNER_PORTS = ("item", "index")
FLOW_TARGET_PORT = "in"
JSON_LIKE = (
    "text",
    "number",
    "bool",
    "json",
    "list",
    "story",
    "world",
    "table",
    "chart",
    "heatmap",
    "tensor",
)


class CompileError(Exception):
    def __init__(self, issues: list[dict]):
        super().__init__("; ".join(i["message"] for i in issues[:5]))
        self.issues = issues


def issue(
    message: str,
    *,
    node_id: Optional[str] = None,
    edge_id: Optional[str] = None,
) -> dict:
    found: dict[str, Any] = {"message": message}
    if node_id:
        found["node_id"] = node_id
    if edge_id:
        found["edge_id"] = edge_id
    return found


def json_depth(value: Any, limit: int = schema.MAX_JSON_DEPTH) -> int:
    """Iterative depth, so a hostile document cannot blow the stack; stops counting past ``limit``."""
    deepest = 0
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > deepest:
            deepest = depth
            if deepest > limit:
                return deepest
        if isinstance(item, dict):
            stack.extend((v, depth + 1) for v in item.values())
        elif isinstance(item, list):
            stack.extend((v, depth + 1) for v in item)
    return deepest


def parse_document(raw: Any) -> tuple[Optional[dict], list[dict]]:
    """Schema-validate ``raw`` (dict or JSON text). Returns (normalized doc dict, errors)."""
    if isinstance(raw, (bytes, str)):
        if len(raw) > schema.MAX_DOC_BYTES:
            return None, [issue("The flow is larger than 8 MB")]
        try:
            raw = json.loads(raw)
        except ValueError as exc:
            return None, [issue(f"Not valid JSON: {exc}")]
    if not isinstance(raw, dict):
        return None, [issue("A flow is a JSON object")]
    if json_depth(raw) > schema.MAX_JSON_DEPTH:
        return None, [issue("The flow is nested too deeply")]
    if len(json.dumps(raw)) > schema.MAX_DOC_BYTES:
        return None, [issue("The flow is larger than 8 MB")]
    if raw.get("format", schema.FORMAT) != schema.FORMAT:
        return None, [issue("Not an Unsloth flow (format must be 'unsloth-flow')")]
    if isinstance(raw.get("version"), int) and raw["version"] > schema.FORMAT_VERSION:
        return None, [
            issue(f"This flow needs a newer Unsloth Studio (format version {raw['version']})")
        ]
    try:
        doc = schema.FlowDocument.model_validate(raw)
    except ValidationError as exc:
        errors = []
        for err in exc.errors()[:20]:
            loc = ".".join(str(p) for p in err.get("loc", ()))
            errors.append(issue(f"{loc}: {err.get('msg')}"))
        return None, errors
    if len(doc.nodes) > schema.MAX_NODES:
        return None, [issue(f"Too many nodes (at most {schema.MAX_NODES})")]
    if len(doc.edges) > schema.MAX_EDGES:
        return None, [issue(f"Too many edges (at most {schema.MAX_EDGES})")]
    if len(doc.assets) > schema.MAX_ASSETS:
        return None, [issue(f"Too many assets (at most {schema.MAX_ASSETS})")]
    return doc.model_dump(mode = "json", by_alias = True), []


def port_compatible(src: dict, dst: dict) -> bool:
    s, d = src["type"], dst["type"]
    if "any" in (s, d):
        return True
    if s == d:
        if s == "list":
            si, di = src.get("item"), dst.get("item")
            return si in (None, "any") or di in (None, "any") or si == di
        return True
    if d == "json" and s in JSON_LIKE:
        return True
    if {s, d} == {"image", "mask"}:
        return True
    if d == "list" and dst.get("item") in (None, "any", s):
        return True
    return False


@dataclass
class Plan:
    doc: dict
    nodes: dict[str, dict]
    specs: dict[str, dict]
    executable: set[str]
    inputs: dict[str, dict[str, list[tuple[str, str]]]]
    collect: dict[str, tuple[str, str]]
    consumers: dict[str, set[str]]
    bodies: dict[str, list[str]]
    owner: dict[str, Optional[str]]
    order: list[str]
    waits: dict[str, set[str]] = field(default_factory = dict)
    flow_edges: list[dict] = field(default_factory = list)
    warnings: list[dict] = field(default_factory = list)
    # Loop node id -> its feeds-the-next-round edge.
    feedback: dict[str, dict] = field(default_factory = dict)

    def top_level(self) -> list[str]:
        return [n for n in self.order if self.owner.get(n) is None]

    def body_order(self, for_each_id: str) -> list[str]:
        members = set(self.bodies.get(for_each_id, ()))
        return [n for n in self.order if n in members]

    def ancestors(self, targets: Iterable[str]) -> set[str]:
        """Every node ``targets`` need, including whole For Each bodies (and their owners) they touch."""
        need: set[str] = set()
        stack = list(targets)
        while stack:
            node = stack.pop()
            if node in need or node not in self.nodes:
                continue
            need.add(node)
            for sources in self.inputs.get(node, {}).values():
                stack.extend(src for src, _ in sources)
            owner = self.owner.get(node)
            if owner:
                stack.append(owner)
            if node in self.bodies:
                stack.extend(self.bodies[node])
                if node in self.collect:
                    stack.append(self.collect[node][0])
        return need

    def gpu_nodes(self, among: Optional[Iterable[str]] = None) -> list[str]:
        pool = self.order if among is None else [n for n in self.order if n in set(among)]
        return [n for n in pool if self.specs[n]["lane"] == "gpu"]


def _members(doc_nodes: dict[str, dict], edges: list[dict]) -> dict[str, set[str]]:
    members: dict[str, set[str]] = {}
    for fid, node in doc_nodes.items():
        if node["type"] not in REPEATS:
            continue
        found = {nid for nid, n in doc_nodes.items() if n.get("group") == fid}
        if node.get("group"):
            found |= {
                nid
                for nid, n in doc_nodes.items()
                if nid != fid and n.get("group") == node["group"]
            }
        members[fid] = found
    # A Loop also repeats the recipe group its feedback edge feeds, wherever that frame sits.
    for edge in edges:
        if edge.get("kind") != FEEDBACK or edge["target"] not in doc_nodes:
            continue
        group = doc_nodes[edge["target"]].get("group")
        for fid, found in members.items():
            if doc_nodes[fid]["type"] == LOOP and edge["source"] in found:
                found.add(edge["target"])
                if group:
                    found |= {nid for nid, n in doc_nodes.items() if n.get("group") == group}
                found.discard(fid)
    return members


def compile_flow(doc: dict) -> Plan:
    """Compile a schema-valid document; raises CompileError listing every problem found."""
    errors: list[dict] = []
    warnings: list[dict] = []
    nodes: dict[str, dict] = {}
    for node in doc.get("nodes", []):
        if node["id"] in nodes:
            errors.append(issue(f"Duplicate node id {node['id']}", node_id = node["id"]))
        nodes[node["id"]] = node
    specs: dict[str, dict] = {}
    for nid, node in nodes.items():
        spec = SPECS_BY_TYPE.get(node["type"])
        if spec is None:
            errors.append(issue(f"Unknown node type {node['type']}", node_id = nid))
            continue
        if node.get("v", 1) > spec["v"]:
            errors.append(
                issue(f"{spec['title']} v{node['v']} needs a newer Unsloth Studio", node_id = nid)
            )
        specs[nid] = spec
        for key, value in (node.get("params") or {}).items():
            if isinstance(value, str) and len(value) > schema.MAX_STRING_PARAM:
                errors.append(issue(f"Parameter {key} is too long", node_id = nid))

    members = _members(nodes, list(doc.get("edges", [])))
    owner: dict[str, Optional[str]] = {nid: None for nid in nodes}
    for fid, found in members.items():
        for nid in found:
            current = owner.get(nid)
            if current is None or fid in members.get(current, set()):
                owner[nid] = fid
    for fid in members:
        cursor, seen = owner.get(fid), {fid}
        while cursor is not None:
            if cursor in seen:
                errors.append(issue("For Each groups are nested in a loop", node_id = fid))
                break
            seen.add(cursor)
            cursor = owner.get(cursor)
    if any(e["message"] == "For Each groups are nested in a loop" for e in errors):
        raise CompileError(errors)
    bodies = {fid: [n for n in nodes if owner.get(n) == fid] for fid in members}

    def inside(nid: str, fid: str) -> bool:
        cursor = owner.get(nid)
        while cursor is not None:
            if cursor == fid:
                return True
            cursor = owner.get(cursor)
        return False

    inputs: dict[str, dict[str, list[tuple[str, str]]]] = {nid: {} for nid in nodes}
    collect: dict[str, tuple[str, str]] = {}
    consumers: dict[str, set[str]] = {nid: set() for nid in nodes}
    flow_edges: list[dict] = []
    feedback_edges: list[dict] = []
    edge_ids: set[str] = set()
    for edge in doc.get("edges", []):
        eid = edge["id"]
        if eid in edge_ids:
            errors.append(issue(f"Duplicate edge id {eid}", edge_id = eid))
            continue
        edge_ids.add(eid)
        src, dst = edge["source"], edge["target"]
        if src not in nodes or dst not in nodes:
            errors.append(issue("Edge connects a missing node", edge_id = eid))
            continue
        if src == dst:
            errors.append(issue("A node cannot connect to itself", edge_id = eid, node_id = src))
            continue
        sspec, dspec = specs.get(src), specs.get(dst)
        if sspec is None or dspec is None:
            continue
        if COLUMN_TYPE in (sspec["type"], dspec["type"]) and (
            sspec["type"] != dspec["type"] or nodes[src].get("group") != nodes[dst].get("group")
        ):
            errors.append(
                issue("A recipe column only wires to columns of its own recipe", edge_id = eid)
            )
            continue
        if edge.get("kind") == FEEDBACK:
            feedback_edges.append(edge)
            continue
        if edge.get("kind", "data") == "flow":
            if edge["source_port"] not in sspec["flow_out"]:
                errors.append(
                    issue(f"{sspec['title']} has no story exit {edge['source_port']}", edge_id = eid)
                )
            elif not dspec["flow_in"]:
                errors.append(issue(f"{dspec['title']} cannot be a story step", edge_id = eid))
            else:
                flow_edges.append(edge)
            continue
        sport = next((p for p in sspec["outputs"] if p["id"] == edge["source_port"]), None)
        dport = next((p for p in dspec["inputs"] if p["id"] == edge["target_port"]), None)
        if sport is None:
            errors.append(
                issue(f"{sspec['title']} has no output {edge['source_port']}", edge_id = eid)
            )
            continue
        if dport is None:
            errors.append(
                issue(f"{dspec['title']} has no input {edge['target_port']}", edge_id = eid)
            )
            continue
        if not port_compatible(sport, dport):
            errors.append(
                issue(
                    f"{sspec['title']}.{sport['label']} ({sport['type']}) cannot connect to "
                    f"{dspec['title']}.{dport['label']} ({dport['type']})",
                    edge_id = eid,
                )
            )
            continue
        if dspec["type"] == FOR_EACH and dport["id"] == COLLECT_PORT:
            if not inside(src, dst):
                errors.append(
                    issue("For Each collects only from a node inside its group", edge_id = eid)
                )
            elif dst in collect:
                errors.append(issue("For Each collects from one node", edge_id = eid))
            else:
                collect[dst] = (src, sport["id"])
            continue
        src_owner = owner.get(src)
        if src_owner is not None and not inside(dst, src_owner):
            errors.append(
                issue(
                    "A step inside a Loop only feeds its Loop; use the Loop's own outputs"
                    if nodes[src_owner]["type"] == LOOP
                    else "A node inside a For Each group can only feed its group; connect it to collect",
                    edge_id = eid,
                )
            )
            continue
        if (
            sspec["type"] == FOR_EACH
            and sport["id"] in FOR_EACH_INNER_PORTS
            and not inside(dst, src)
        ):
            errors.append(
                issue("For Each item and index are only for nodes inside its group", edge_id = eid)
            )
            continue
        existing = inputs[dst].setdefault(dport["id"], [])
        if existing and not dport.get("multi"):
            errors.append(
                issue(
                    f"{dspec['title']}.{dport['label']} takes one connection",
                    edge_id = eid,
                    node_id = dst,
                )
            )
            continue
        existing.append((src, sport["id"]))
        consumers[src].add(dst)

    feedback = _check_feedback(feedback_edges, nodes, specs, owner, inputs, errors)

    executable = {
        nid
        for nid, node in nodes.items()
        if nid in specs
        and not specs[nid]["player"]
        and not specs[nid].get("design")
        and not node.get("disabled")
    }
    for nid in executable:
        if nodes[nid]["type"] == LOOP and nid not in feedback:
            errors.append(
                issue(
                    "Draw the dashed feeds-the-next-round wire from the score step back into the recipe's seed",
                    node_id = nid,
                )
            )
    recipe_groups = {g["id"] for g in doc.get("groups") or [] if g.get("kind") == "recipe"}
    steps: dict[str, int] = {}
    for node in nodes.values():
        if node["type"] == GENERATE_TYPE and not node.get("disabled") and node.get("group"):
            steps[node["group"]] = steps.get(node["group"], 0) + 1
    for nid, node in nodes.items():
        if node["type"] == COLUMN_TYPE and node.get("group") not in recipe_groups:
            errors.append(issue("A recipe column belongs inside a Data Recipe group", node_id = nid))
        if nid not in executable or node["type"] != GENERATE_TYPE:
            continue
        if steps.get(node.get("group") or "", 0) > 1:
            errors.append(
                issue(
                    "A Data Recipe runs from one Recipe Dataset step; delete or disable the extra one",
                    node_id = nid,
                )
            )
            continue
        try:
            compiled = compile_recipe(recipe_graph(doc, node.get("group") or ""))
        except RecipeCompileError as exc:
            errors.append(issue(str(exc), node_id = nid))
            continue
        nodes[nid] = {**node, "params": {**(node.get("params") or {}), RECIPE_PARAM: compiled}}
    for nid in executable:
        spec = specs[nid]
        params = nodes[nid].get("params") or {}
        param_ids = {p["id"] for p in spec["params"]}
        for port in spec["inputs"]:
            if port.get("optional") or port["id"] == COLLECT_PORT or port["id"] in inputs[nid]:
                continue
            if port["id"] in param_ids and params.get(port["id"]) not in (None, ""):
                continue
            errors.append(
                issue(f"{spec['title']} needs its {port['label']} input connected", node_id = nid)
            )

    for nid in executable:
        if specs[nid]["type"] == BENCHMARK:
            errors.extend(_check_benchmark(nid, nodes, specs, inputs))

    output_names: dict[str, str] = {}
    for nid in nodes:
        if nid in executable and specs[nid]["type"] == "output.result":
            name = str((nodes[nid].get("params") or {}).get("name") or "result")
            if name in output_names:
                errors.append(
                    issue(f"Another Flow Output is already named {name!r}; rename one", node_id = nid)
                )
            output_names.setdefault(name, nid)

    # Column wires are design-only; compile_recipe orders a recipe's enabled columns itself.
    design = {nid for nid, node in nodes.items() if node["type"] == COLUMN_TYPE}
    flow_inputs = {
        nid: {
            port: [s for s in sources if not (nid in design and s[0] in design)]
            for port, sources in ports.items()
        }
        for nid, ports in inputs.items()
    }
    order = _topo(nodes, flow_inputs, owner, collect)
    if order is None:
        errors.append(
            issue("The data connections form a loop; use a For Each group to repeat steps")
        )
        order = []
    if errors:
        raise CompileError(errors)
    order = [n for n in order if n in executable]
    for nid in nodes:
        if nid not in specs:
            continue
        if (
            nid not in executable
            and not specs[nid]["player"]
            and not specs[nid].get("design")
            and nodes[nid].get("disabled")
        ):
            warnings.append(issue("Disabled: skipped", node_id = nid))
    waits: dict[str, set[str]] = {}
    for nid, fid in owner.items():
        sources = {src for ports in inputs.get(nid, {}).values() for src, _ in ports}
        cursor = fid
        while cursor is not None:
            for src in sources:
                if src != cursor and src != nid and not _within(src, cursor, owner):
                    waits.setdefault(cursor, set()).add(src)
            cursor = owner.get(cursor)
    return Plan(
        doc = doc,
        nodes = nodes,
        specs = specs,
        executable = executable,
        inputs = inputs,
        collect = collect,
        consumers = consumers,
        bodies = bodies,
        owner = owner,
        order = order,
        waits = waits,
        flow_edges = flow_edges,
        warnings = warnings,
        feedback = feedback,
    )


def _check_feedback(
    edges: list[dict],
    nodes: dict[str, dict],
    specs: dict[str, dict],
    owner: dict[str, Optional[str]],
    inputs: dict[str, dict[str, list[tuple[str, str]]]],
    errors: list[dict],
) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for edge in edges:
        eid, src, dst = edge["id"], edge["source"], edge["target"]
        sspec, dspec = specs.get(src), specs.get(dst)
        if sspec is None or dspec is None:
            continue
        loop = owner.get(src)
        while loop is not None and nodes[loop]["type"] != LOOP:
            loop = owner.get(loop)
        if loop is None or not _within(dst, loop, owner):
            errors.append(
                issue("A feeds-the-next-round wire runs inside one Loop group", edge_id = eid)
            )
            continue
        if edge["source_port"] != "failures" or not {"failures", "score"} <= {
            p["id"] for p in sspec["outputs"]
        }:
            errors.append(
                issue(
                    "Feed the next round from a step that scores the model (its failures and score)",
                    edge_id = eid,
                )
            )
            continue
        task = next((p for p in sspec["params"] if p["id"] == "task"), None)
        if (
            task
            and (nodes[src].get("params") or {}).get("task", task["default"]) != task["default"]
        ):
            errors.append(
                issue(
                    f"A Loop scores its own held-out rows: set {sspec['title']}'s Task to {task['default']}",
                    node_id = src,
                )
            )
            continue
        dport = next((p for p in dspec["inputs"] if p["id"] == edge["target_port"]), None)
        if dspec["type"] != GENERATE_TYPE or dport is None or dport["id"] != "seed":
            errors.append(
                issue("The next round's failures go into a recipe's seed input", edge_id = eid)
            )
            continue
        if inputs[dst].get(dport["id"]):
            errors.append(
                issue(f"{dspec['title']}.{dport['label']} takes one connection", edge_id = eid)
            )
            continue
        held = [
            src_node
            for port in sspec["inputs"]
            if port["type"] == "dataset"
            for src_node, src_port in inputs[src].get(port["id"], ())
            if src_port == "holdout" and _within(src_node, loop, owner)
        ]
        if not held:
            errors.append(
                issue(
                    f"{sspec['title']} needs held-out rows from a step inside the Loop "
                    "(Fine-tune's Held-out rows)",
                    node_id = src,
                )
            )
            continue
        scored = {
            src_node
            for port in sspec["inputs"]
            if port["type"] == "checkpoint"
            for src_node, _ in inputs[src].get(port["id"], ())
        }
        if scored != {held[0]}:
            errors.append(
                issue(
                    f"Wire the Checkpoint of the step that held the rows out into {sspec['title']}, "
                    "so each round scores the model it just trained",
                    node_id = src,
                )
            )
            continue
        if loop in found:
            errors.append(issue("A Loop takes one feeds-the-next-round wire", edge_id = eid))
            continue
        found[loop] = edge
    return found


BENCHMARK = "train.benchmark"
EXPORT = "train.export"


def _check_benchmark(nid: str, nodes: dict, specs: dict, inputs: dict) -> list[dict]:
    wired = inputs.get(nid, {})
    if "checkpoint" in wired and "model" in wired:
        return [
            issue(
                "Benchmark scores one model: wire its Checkpoint or its Model, not both",
                node_id = nid,
            )
        ]
    task = str((nodes[nid].get("params") or {}).get("task") or "Recipe held-out")
    if task == "Recipe held-out":
        return []
    if "checkpoint" in wired:
        return [
            issue(
                f"{task} is an lm-eval task, and lm-eval scores a GGUF, not a Fine-tune checkpoint. Put Export "
                "Model (Format GGUF) between them and wire it into Benchmark's Model, or set Task to Recipe held-out",
                node_id = nid,
            )
        ]
    for src, _port in wired.get("model", []):
        if (
            specs[src]["type"] == EXPORT
            and str((nodes[src].get("params") or {}).get("format") or "gguf") != "gguf"
        ):
            return [issue(f"{task} scores a GGUF: set Export Model's Format to GGUF", node_id = nid)]
    return []


def _topo(
    nodes: dict[str, dict],
    inputs: dict[str, dict[str, list[tuple[str, str]]]],
    owner: dict[str, Optional[str]],
    collect: dict[str, tuple[str, str]],
) -> Optional[list[str]]:
    deps: dict[str, set[str]] = {nid: set() for nid in nodes}
    for nid, ports in inputs.items():
        for sources in ports.values():
            for src, _ in sources:
                deps[nid].add(src)
    for nid, fid in owner.items():
        if fid is not None:
            deps[nid].add(fid)
    for nid, fid in owner.items():
        cursor = fid
        while cursor is not None:
            for src in deps[nid]:
                if src != cursor and not _within(src, cursor, owner) and src != nid:
                    deps[cursor].add(src)
            cursor = owner.get(cursor)
    position = {nid: i for i, nid in enumerate(nodes)}
    remaining = {nid: set(d) for nid, d in deps.items()}
    ready = sorted((n for n, d in remaining.items() if not d), key = position.__getitem__)
    order: list[str] = []
    dependents: dict[str, set[str]] = {nid: set() for nid in nodes}
    for nid, d in deps.items():
        for src in d:
            dependents.setdefault(src, set()).add(nid)
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for nxt in sorted(dependents.get(nid, ()), key = position.__getitem__):
            remaining[nxt].discard(nid)
            if not remaining[nxt] and nxt not in order and nxt not in ready:
                ready.append(nxt)
        ready.sort(key = position.__getitem__)
    if len(order) != len(nodes):
        return None
    return order


def _within(nid: str, fid: str, owner: dict[str, Optional[str]]) -> bool:
    cursor = owner.get(nid)
    while cursor is not None:
        if cursor == fid:
            return True
        cursor = owner.get(cursor)
    return False


def validate(raw: Any) -> dict:
    """The /validate answer: {ok, errors, warnings}."""
    doc, errors = parse_document(raw)
    if doc is None:
        return {"ok": False, "errors": errors, "warnings": []}
    try:
        plan = compile_flow(doc)
    except CompileError as exc:
        return {"ok": False, "errors": exc.issues, "warnings": []}
    return {"ok": True, "errors": [], "warnings": plan.warnings}


def model_needs(plan: Plan, among: Optional[Iterable[str]] = None) -> list[dict]:
    """The models GPU nodes name, deduplicated, plus the document's declared requirements."""
    seen: dict[tuple, dict] = {}
    for req in (plan.doc.get("requirements") or {}).get("models") or []:
        key = (req["kind"], req["repo"].lower(), (req.get("variant") or "").lower())
        seen[key] = {**req}
    pool = plan.order if among is None else [n for n in plan.order if n in set(among)]
    for nid in pool:
        spec = plan.specs[nid]
        if not spec.get("model_kind"):
            continue
        choice = (plan.nodes[nid].get("params") or {}).get("model")
        if not isinstance(choice, dict) or not choice.get("repo"):
            continue
        key = (spec["model_kind"], choice["repo"].lower(), (choice.get("variant") or "").lower())
        seen.setdefault(
            key,
            {"kind": spec["model_kind"], "repo": choice["repo"], "variant": choice.get("variant")},
        )
    return list(seen.values())
