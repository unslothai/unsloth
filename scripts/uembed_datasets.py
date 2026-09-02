#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Streamed SCIENCE-TECH training pairs for the UEmbed (Qwen3.5) fine-tune example.

Opt-in loader for ``scripts/uembed_finetune.py``. Nothing here touches the network at
import time: every Hub call lives inside ``load_uembed_science_tech_dataset()``, which the
fine-tune script only calls when ``UNSLOTH_UEMBED_TRAIN_DATASET`` is set. Importing this
module - or importing the fine-tune script - therefore downloads nothing.

What it builds
--------------
Two ``{anchor, positive}`` subsets, a few hundred pairs each, streamed (``streaming =
True``), so no full dataset is ever materialised on disk:

- ``"multimodal"``: ``vidore/arxivqa_test_subsampled`` - a natural-language question about
  an arXiv page as the anchor, the rendered page image as the positive. This is the
  screenshot-retrieval half UEmbed is built for.
- ``"text"``: SciFact (``allenai/scifact``) - a scientific claim as the anchor, the cited
  paper's title + abstract as the positive. The claims config carries ``cited_doc_ids``
  and the corpus config carries the abstracts, so the pair is a two-stream join.

Why a dict of two datasets and not one concatenated ``Dataset``
---------------------------------------------------------------
A single ``Dataset`` has a single Arrow schema, so merging the two subsets would force the
``positive`` column to be a struct with a null ``image`` on every text row, and
sentence-transformers reads such a dict as an image input. ``SentenceTransformerTrainer``
accepts ``dict[str, Dataset]`` directly and never mixes two datasets inside one batch, so
each batch stays single-modality and the in-batch negatives of
``MultipleNegativesRankingLoss`` / ``UEmbedUnifiedLoss`` stay meaningful. Same
``{anchor, positive}`` column shape either way, so the loss, trainer and collator are
unchanged.

Scope: this is a short domain-adaptation slice, deliberately a few hundred pairs. The real
convergence run - loss slope, sparsity, recall@k over a few hundred steps - is Todo 11 on
the Brev GPU box, not this file.

Needs ``datasets`` (imported lazily, inside the loaders).
"""

from __future__ import annotations

from typing import Any


# Environment variable that opts the fine-tune script into this loader.
TRAIN_DATASET_ENV_VAR = "UNSLOTH_UEMBED_TRAIN_DATASET"

# -- multimodal: text question -> arXiv page image ---------------------------------------
ARXIVQA_REPO = "vidore/arxivqa_test_subsampled"
ARXIVQA_SPLIT = "test"
ARXIVQA_QUERY_COLUMN = "query"
ARXIVQA_IMAGE_COLUMN = "image"

# -- text: scientific claim -> cited abstract ---------------------------------------------
SCIFACT_REPO = "allenai/scifact"
SCIFACT_CLAIMS_CONFIG = "claims"
SCIFACT_CORPUS_CONFIG = "corpus"
SCIFACT_CLAIMS_SPLIT = "train"
SCIFACT_CORPUS_SPLIT = "train"

# Subset names of the returned mapping; they become the trainer's dataset names.
MULTIMODAL_SUBSET = "multimodal"
TEXT_SUBSET = "text"

DEFAULT_NUM_MULTIMODAL_PAIRS = 200
DEFAULT_NUM_TEXT_PAIRS = 200


def _column(row: dict[str, Any], column: str, repo: str) -> Any:
    """Read `column` off a streamed row, saying which repo drifted when it is missing."""
    if column not in row:
        raise KeyError(
            f"Unsloth: `{repo}` has no `{column}` column; it exposes {sorted(row)}. The "
            f"dataset's schema changed - update scripts/uembed_datasets.py."
        )
    return row[column]


def _joined_text(value: Any) -> str:
    """SciFact abstracts arrive as a list of sentences; titles arrive as a plain string."""
    if isinstance(value, str):
        return value.strip()
    return " ".join(str(sentence).strip() for sentence in value).strip()


def load_arxivqa_pairs(num_pairs: int = DEFAULT_NUM_MULTIMODAL_PAIRS, streaming: bool = True):
    """`{anchor: question (str), positive: arXiv page (PIL image)}` from ViDoRe ArxivQA.

    The positive column is declared as a ``datasets.Image`` feature, which is what lets
    ``Dataset.from_list`` hold the streamed PIL pages, and what sentence-transformers reads
    as the image modality.
    """
    from datasets import Dataset, Features, Image, Value, load_dataset

    stream = load_dataset(ARXIVQA_REPO, split = ARXIVQA_SPLIT, streaming = streaming)
    rows: list[dict[str, Any]] = []
    for row in stream:
        query = _column(row, ARXIVQA_QUERY_COLUMN, ARXIVQA_REPO)
        image = _column(row, ARXIVQA_IMAGE_COLUMN, ARXIVQA_REPO)
        # A handful of ViDoRe rows carry no question; they cannot form a pair.
        if not query or image is None:
            continue
        rows.append({"anchor": str(query), "positive": image})
        if len(rows) >= num_pairs:
            break

    if not rows:
        raise RuntimeError(
            f"Unsloth: `{ARXIVQA_REPO}` yielded no usable (question, page) pair."
        )
    features = Features({"anchor": Value("string"), "positive": Image()})
    return Dataset.from_list(rows, features = features)


def load_scifact_pairs(num_pairs: int = DEFAULT_NUM_TEXT_PAIRS, streaming: bool = True):
    """`{anchor: claim, positive: title + abstract}` joined across SciFact's two configs.

    Pass 1 streams the claims and collects the ``cited_doc_ids`` needed; pass 2 streams the
    corpus and stops as soon as every wanted document has been seen. Both passes are
    bounded by `num_pairs`, so neither config is ever fully downloaded.
    """
    from datasets import Dataset, load_dataset

    claims = load_dataset(
        SCIFACT_REPO, SCIFACT_CLAIMS_CONFIG, split = SCIFACT_CLAIMS_SPLIT, streaming = streaming
    )
    wanted: dict[str, list[str]] = {}
    collected = 0
    for row in claims:
        claim = _column(row, "claim", SCIFACT_REPO)
        cited = _column(row, "cited_doc_ids", SCIFACT_REPO)
        # Claims without a cited document have no positive to pair with.
        if not claim or not cited:
            continue
        wanted.setdefault(str(cited[0]), []).append(str(claim))
        collected += 1
        if collected >= num_pairs:
            break

    if not wanted:
        raise RuntimeError(
            f"Unsloth: `{SCIFACT_REPO}` ({SCIFACT_CLAIMS_CONFIG}) yielded no cited claim."
        )

    corpus = load_dataset(
        SCIFACT_REPO, SCIFACT_CORPUS_CONFIG, split = SCIFACT_CORPUS_SPLIT, streaming = streaming
    )
    rows: list[dict[str, str]] = []
    for document in corpus:
        doc_id = str(_column(document, "doc_id", SCIFACT_REPO))
        claims_for_document = wanted.pop(doc_id, None)
        if not claims_for_document:
            continue
        title = _joined_text(_column(document, "title", SCIFACT_REPO))
        abstract = _joined_text(_column(document, "abstract", SCIFACT_REPO))
        positive = f"{title} {abstract}".strip()
        rows.extend({"anchor": claim, "positive": positive} for claim in claims_for_document)
        if not wanted:
            break

    if not rows:
        raise RuntimeError(
            f"Unsloth: no `{SCIFACT_REPO}` claim could be joined to its cited abstract."
        )
    return Dataset.from_list(rows[:num_pairs])


def load_uembed_science_tech_dataset(
    num_multimodal_pairs: int = DEFAULT_NUM_MULTIMODAL_PAIRS,
    num_text_pairs: int = DEFAULT_NUM_TEXT_PAIRS,
    streaming: bool = True,
) -> dict[str, Any]:
    """The two science-tech subsets, ready for ``SentenceTransformerTrainer``.

    Args:
        num_multimodal_pairs: ArxivQA (question -> page image) pairs; 0 skips the subset.
        num_text_pairs: SciFact (claim -> abstract) pairs; 0 skips the subset.
        streaming: keep the Hub streams lazy (default). ``False`` downloads the full
            datasets and is only worth it when the same slice is reused many times.

    Returns:
        ``{"multimodal": Dataset, "text": Dataset}``, each with the ``anchor`` / ``positive``
        columns the UEmbed losses expect. Pass it straight to the trainer's
        ``train_dataset``. Calling this function is what triggers the download.
    """
    subsets: dict[str, Any] = {}
    if num_multimodal_pairs > 0:
        subsets[MULTIMODAL_SUBSET] = load_arxivqa_pairs(num_multimodal_pairs, streaming)
    if num_text_pairs > 0:
        subsets[TEXT_SUBSET] = load_scifact_pairs(num_text_pairs, streaming)
    if not subsets:
        raise ValueError(
            "Unsloth: `num_multimodal_pairs` and `num_text_pairs` are both 0, so there is "
            "nothing to train on."
        )
    return subsets
