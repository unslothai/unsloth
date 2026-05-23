# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .comparators import get_comparator
from .schema import ArrayNode, LeafNode, Node, ObjectNode


@dataclass
class ScoreNode:
    """A node in the per-field score breakdown.

    score:          mean score over this subtree's leaves, in [0, 1]
    n_leaves:       number of leaf comparisons this subtree contributes
    children:       dict (objects) | list (arrays) | None (leaves)
    note:           explanation set on penalties / mismatches
    matched_option: index into a ground-truth alternatives tuple, when applicable
    """

    score: float
    n_leaves: int
    children: Any = None  # dict | list | None
    note: str | None = None
    matched_option: int | None = None


def _leaf_count(value: Any, node: Node | None) -> int:
    """Number of leaf comparisons in `value` under schema `node`."""
    if isinstance(value, tuple):
        return _leaf_count(value[0], node) if value else 1
    if isinstance(node, ObjectNode) or (node is None and isinstance(value, dict)):
        if not isinstance(value, dict):
            return 1
        total = 0
        for k, v in value.items():
            child = node.fields.get(k) if isinstance(node, ObjectNode) else None
            total += _leaf_count(v, child)
        return total
    if isinstance(node, ArrayNode) or (node is None and isinstance(value, list)):
        if not isinstance(value, list):
            return 1
        # each array item is one slot, matching _score_array's n_leaves = max(len)
        return len(value)
    return 1


def _score_leaf(gt: Any, pred: Any, comparator: str, params: dict) -> ScoreNode:
    if gt is None and pred is None:
        return ScoreNode(1.0, 1)
    if (gt is None) != (pred is None):
        return ScoreNode(0.0, 1, note="missing or hallucinated")
    cmp = get_comparator(comparator, **params)
    return ScoreNode(float(cmp(gt, pred)), 1)


def _score(gt: Any, pred: Any, node: Node | None, default: str) -> ScoreNode:
    # Options: a ground-truth tuple lists acceptable alternatives -> take the best
    if isinstance(gt, tuple):
        if not gt:
            return ScoreNode(0.0, 1, note="empty alternatives tuple")
        best: ScoreNode | None = None
        best_i = -1
        for i, opt in enumerate(gt):
            cand = _score(opt, pred, node, default)
            if best is None or cand.score > best.score:
                best, best_i = cand, i
        assert best is not None
        best.matched_option = best_i
        return best

    # Object: schema says object, or no schema and the data is a dict
    if isinstance(node, ObjectNode) or (node is None and isinstance(gt, dict)):
        if not isinstance(gt, dict) or not isinstance(pred, dict):
            return _mismatch(gt, pred, node)
        return _score_object(gt, pred, node, default)

    # Array: schema says array, or no schema and the data is a list
    if isinstance(node, ArrayNode) or (node is None and isinstance(gt, list)):
        if not isinstance(gt, list) or not isinstance(pred, list):
            return _mismatch(gt, pred, node)
        return _score_array(gt, pred, node, default)

    # Leaf — but a container on either side can't be scored by a leaf
    # comparator (it would stringify and Levenshtein the repr), so treat a
    # leaf-vs-container shape clash as a type mismatch.
    if isinstance(gt, (dict, list)) or isinstance(pred, (dict, list)):
        return _mismatch(gt, pred, node)
    if isinstance(node, LeafNode):
        return _score_leaf(gt, pred, node.comparator, node.params)
    return _score_leaf(gt, pred, default, {})


def _mismatch(gt: Any, pred: Any, node: Node | None) -> ScoreNode:
    # Weight the mismatch by what ground truth claims: when GT is the degenerate
    # (scalar) side, _leaf_count is 1; when GT carries the structure, it's that
    # subtree's size. Either way the local score is 0.
    n = _leaf_count(gt, node) if gt is not None else 1
    return ScoreNode(0.0, max(n, 1), note="type mismatch")


def _score_object(gt: dict, pred: dict, node: Node | None, default: str) -> ScoreNode:
    children: dict[str, ScoreNode] = {}
    total_sum, total_n = 0.0, 0
    keys = list(dict.fromkeys([*gt.keys(), *pred.keys()]))  # ordered union
    for k in keys:
        child_node = node.fields.get(k) if isinstance(node, ObjectNode) else None
        if k in gt and k in pred:
            cn = _score(gt[k], pred[k], child_node, default)
        elif k in gt:  # present in gt, missing from prediction
            cn = ScoreNode(
                0.0, _leaf_count(gt[k], child_node), note="missing in prediction"
            )
        else:  # present in prediction only -> hallucinated, one zero-leaf
            cn = ScoreNode(0.0, 1, note="hallucinated")
        children[k] = cn
        total_sum += cn.score * cn.n_leaves
        total_n += cn.n_leaves
    if total_n == 0:
        return ScoreNode(1.0, 0, children=children)
    return ScoreNode(total_sum / total_n, total_n, children=children)


def _score_array(gt: list, pred: list, node: Node | None, default: str) -> ScoreNode:
    item_node = node.item if isinstance(node, ArrayNode) else None
    n_g, n_p = len(gt), len(pred)
    slots = max(n_g, n_p)
    if slots == 0:
        return ScoreNode(1.0, 0, children=[])
    if n_g == 0 or n_p == 0:
        # all hallucinated or all missing
        return ScoreNode(0.0, slots, children=[])

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # full ScoreNode for every (gt_i, pred_j) pair; matching maximises mean score
    node_matrix = [
        [_score(g, p, item_node, default) for p in pred] for g in gt
    ]
    score_matrix = np.array([[n.score for n in row] for row in node_matrix])
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    matched = list(zip(row_ind.tolist(), col_ind.tolist()))
    total = sum(score_matrix[i][j] for i, j in matched)
    children = [node_matrix[i][j] for i, j in matched]
    # Surface the unmatched items in the breakdown — they drove the penalty.
    # (Score and n_leaves already account for them via the `slots` denominator;
    # these nodes are for diagnosis only.)
    matched_g = {i for i, _ in matched}
    matched_p = {j for _, j in matched}
    children += [
        ScoreNode(0.0, 1, note="missing in prediction")
        for i in range(n_g)
        if i not in matched_g
    ]
    children += [
        ScoreNode(0.0, 1, note="hallucinated")
        for j in range(n_p)
        if j not in matched_p
    ]
    return ScoreNode(float(total) / slots, slots, children=children)
