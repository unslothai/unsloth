# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

# Public datasets as FastDecisionModel.build_dataset rows {state, questions, gold}, schema augmentation
# and a mixture builder. Converters take plain dict rows, so they run offline.

__all__ = [
    "DecisionSource",
    "SOURCES",
    "EVAL_ONLY_SOURCES",
    "load_source",
    "augment_row",
    "build_decision_mixture",
    "Decontaminator",
]

import json
import random
import re
import string
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

NLI_LABELS = ("entailment", "neutral", "contradiction")
NLI_CRITERIA = {
    "entailment": "The premise shows the hypothesis is true.",
    "neutral": "The premise neither confirms nor rules out the hypothesis.",
    "contradiction": "The premise shows the hypothesis is false.",
}
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
AG_NEWS_CRITERIA = {
    "world": "World news and international affairs",
    "sports": "Sports",
    "business": "Business and economy",
    "sci_tech": "Science and technology",
}


def _humanize(label: str) -> str:
    return re.sub(r"[_\-/]+", " ", str(label)).strip()


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_") or "option"


def _row(
    source,
    state,
    qid,
    question,
    gold,
    instructions = (),
    ids = (),
) -> dict:
    # `instructions` / `ids`: paraphrases and alternative question ids augment_row may pick.
    return {
        "source": source,
        "state": state,
        "questions": {qid: question},
        "gold": {qid: gold},
        "_variants": {qid: {"instructions": list(instructions), "ids": list(ids)}},
    }


def _names(row, label_names, column):
    value = row[column]
    if isinstance(value, str) and not value.lstrip("-").isdigit():
        return value
    return label_names[int(value)]


def convert_intent(
    source,
    label_column,
    out_of_scope = None,
) -> Callable:
    def convert(rows, label_names) -> Iterator[dict]:
        criteria = {
            _slug(name): (
                "None of the other intents applies." if name == out_of_scope else _humanize(name)
            )
            for name in label_names
        }
        for row in rows:
            label = _slug(_names(row, label_names, label_column))
            yield _row(
                source,
                {"message": row["text"]},
                "intent",
                {
                    "type": "choice",
                    "instructions": "Which intent does the customer's message express?",
                    "criteria": criteria,
                },
                label,
                instructions = (
                    "What does the customer want?",
                    "Classify the intent of this message.",
                    "Which request type is this?",
                ),
                ids = ("customer_intent", "request_type", "category"),
            )

    return convert


def convert_nli(source, label_column = "label") -> Callable:
    def convert(rows, label_names) -> Iterator[dict]:
        for row in rows:
            value = row[label_column]
            if value is None or (not isinstance(value, str) and int(value) < 0):
                continue
            label = value if isinstance(value, str) else NLI_LABELS[int(value)]
            if label not in NLI_LABELS:
                continue
            yield _row(
                source,
                {"premise": row["premise"], "hypothesis": row["hypothesis"]},
                "relation",
                {
                    "type": "choice",
                    "instructions": "How does the premise relate to the hypothesis?",
                    "criteria": dict(NLI_CRITERIA),
                },
                label,
                instructions = (
                    "Does the premise support, contradict, or say nothing about the hypothesis?",
                    "Classify the logical relation between premise and hypothesis.",
                ),
                ids = ("entailment", "nli_label", "inference"),
            )

    return convert


def convert_boolq(rows, label_names = None) -> Iterator[dict]:
    for row in rows:
        yield _row(
            "boolq",
            {"passage": row["passage"], "question": row["question"]},
            "answer",
            {
                "type": "noul",
                "instructions": "Based on the passage, is the answer to the question yes?",
            },
            bool(row["answer"]),
            instructions = (
                "Does the passage answer the question with yes?",
                "Is the statement in the question true according to the passage?",
            ),
            ids = ("is_true", "yes_no", "passage_says_yes"),
        )


def convert_ag_news(rows, label_names) -> Iterator[dict]:
    keys = list(AG_NEWS_CRITERIA)
    for row in rows:
        yield _row(
            "ag_news",
            {"article": row["text"]},
            "topic",
            {
                "type": "choice",
                "instructions": "What is the topic of this news article?",
                "criteria": dict(AG_NEWS_CRITERIA),
            },
            keys[int(row["label"])],
            instructions = (
                "Which section does this story belong in?",
                "Classify the article's topic.",
            ),
            ids = ("section", "news_topic", "category"),
        )


def convert_sst5(rows, label_names = None) -> Iterator[dict]:
    for row in rows:
        yield _row(
            "sst5",
            {"review": row["text"]},
            "sentiment",
            {
                "type": "score",
                "instructions": "How positive is this review?",
                "criteria": list(SST5_LEVELS),
            },
            int(row["label"]),
            instructions = (
                "Rate the sentiment of the review.",
                "How much did the reviewer like it?",
            ),
            ids = ("rating", "sentiment_level", "positivity"),
        )


def convert_mcq(
    source,
    question_key = "question",
    subject_key = None,
) -> Callable:
    def convert(rows, label_names = None) -> Iterator[dict]:
        for row in rows:
            choices = row["choices"]
            if isinstance(choices, dict):
                labels, texts = list(choices["label"]), list(choices["text"])
                answer = str(row.get("answerKey"))
            else:
                texts = list(choices)
                labels = list(string.ascii_uppercase[: len(texts)])
                answer = labels[int(row["answer"])]
            if answer not in labels or len(set(labels)) != len(labels):
                continue
            state = {"question": row[question_key]}
            if subject_key and row.get(subject_key):
                state["subject"] = _humanize(row[subject_key])
            yield _row(
                source,
                state,
                "answer",
                {
                    "type": "choice",
                    "instructions": "Which option answers the question correctly?",
                    "criteria": dict(zip(labels, texts)),
                },
                answer,
                instructions = ("Pick the correct answer.", "Which choice is right?"),
                ids = ("correct_option", "best_answer", "choice"),
            )

    return convert


def convert_xlam(rows, label_names = None) -> Iterator[dict]:
    for row in rows:
        tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
        names = [tool.get("name") for tool in tools]
        if not answers or len(set(names)) != len(names) or answers[0].get("name") not in names:
            continue
        yield _row(
            "xlam",
            {"request": row["query"]},
            "tool",
            {
                "type": "choice",
                "instructions": "Which tool should be called first to handle the request?",
                "criteria": {
                    tool["name"]: tool.get("description") or tool["name"] for tool in tools
                },
            },
            answers[0]["name"],
            instructions = ("Pick the tool to call first.", "Which function serves this request?"),
            ids = ("first_tool", "function", "tool_choice"),
        )


def convert_prompt_injection(rows, label_names = None) -> Iterator[dict]:
    for row in rows:
        yield _row(
            "prompt_injections",
            {"user_input": row["text"]},
            "injection",
            {
                "type": "noul",
                "instructions": "Is this input trying to override or hijack the assistant's instructions?",
            },
            bool(int(row["label"])),
            instructions = (
                "Is this a prompt injection attempt?",
                "Does the text try to change the assistant's rules?",
            ),
            ids = ("prompt_injection", "is_attack", "jailbreak"),
        )


def convert_typed_decisions(rows, label_names = None) -> Iterator[dict]:
    # Already {state, questions, gold} as JSON strings, with soft labels.
    for row in rows:
        questions = (
            json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"]
        )
        gold = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]
        state = row["state"]
        if isinstance(state, str) and state.strip()[:1] in "{[":
            try:
                state = json.loads(state)
            except ValueError:
                pass
        yield {
            "source": "typed_decisions",
            "state": state,
            "questions": questions,
            "gold": gold,
            "_variants": {},
        }


@dataclass(frozen = True)
class DecisionSource:
    name: str
    hf_id: str
    config: Optional[str]
    train_split: str
    test_split: Optional[str]
    license: str
    convert: Callable
    label_column: Optional[str] = None
    eval_only: bool = False
    note: str = ""


SOURCES = {
    s.name: s
    for s in (
        DecisionSource(
            "banking77",
            "legacy-datasets/banking77",
            None,
            "train",
            "test",
            "cc-by-4.0",
            convert_intent("banking77", "label"),
            "label",
        ),
        DecisionSource(
            "clinc150",
            "clinc/clinc_oos",
            "plus",
            "train",
            "test",
            "cc-by-3.0",
            convert_intent("clinc150", "intent", out_of_scope = "oos"),
            "intent",
        ),
        DecisionSource(
            "mnli",
            "nyu-mll/multi_nli",
            None,
            "train",
            "validation_matched",
            "mixed cc-by-3.0 / cc-by-sa-3.0 / mit / other",
            convert_nli("mnli"),
        ),
        DecisionSource(
            "snli", "stanfordnlp/snli", None, "train", "test", "cc-by-sa-4.0", convert_nli("snli")
        ),
        DecisionSource(
            "wanli",
            "alisawuffles/WANLI",
            None,
            "train",
            "test",
            "cc-by-4.0",
            convert_nli("wanli", "gold"),
        ),
        DecisionSource(
            "boolq", "google/boolq", None, "train", "validation", "cc-by-sa-3.0", convert_boolq
        ),
        DecisionSource(
            "ag_news",
            "fancyzhx/ag_news",
            None,
            "train",
            "test",
            "unknown (check before redistributing)",
            convert_ag_news,
            "label",
        ),
        DecisionSource(
            "sst5",
            "SetFit/sst5",
            None,
            "train",
            "test",
            "none listed (check before redistributing)",
            convert_sst5,
        ),
        DecisionSource(
            "mmlu",
            "cais/mmlu",
            "all",
            "auxiliary_train",
            "test",
            "mit",
            convert_mcq("mmlu", subject_key = "subject"),
            note = "MMLU is a Decision Index benchmark: train on auxiliary_train only",
        ),
        DecisionSource(
            "commonsense_qa",
            "tau/commonsense_qa",
            None,
            "train",
            "validation",
            "mit",
            convert_mcq("commonsense_qa"),
            note = "CSQA is a Decision Index benchmark: report it as in-distribution",
        ),
        DecisionSource(
            "arc",
            "allenai/ai2_arc",
            "ARC-Challenge",
            "train",
            "test",
            "cc-by-sa-4.0",
            convert_mcq("arc"),
            note = "ARC is a Decision Index benchmark: report it as in-distribution",
        ),
        DecisionSource(
            "xlam",
            "Salesforce/xlam-function-calling-60k",
            None,
            "train",
            None,
            "cc-by-4.0 (gated: accept the terms on the Hub)",
            convert_xlam,
        ),
        DecisionSource(
            "prompt_injections",
            "deepset/prompt-injections",
            None,
            "train",
            "test",
            "apache-2.0",
            convert_prompt_injection,
        ),
        DecisionSource(
            "typed_decisions",
            "LocalLLaMA/typed-decisions",
            "all",
            "train",
            "test",
            "apache-2.0",
            convert_typed_decisions,
        ),
    )
}
# Decision Index benchmarks: evaluation only, never trained on.
EVAL_ONLY_SOURCES = frozenset({"bfcl", "when2call"})


def _label_names(dataset, column) -> list:
    feature = (getattr(dataset, "features", None) or {}).get(column) if column else None
    return list(getattr(feature, "names", None) or [])


def load_source(
    name: str,
    split: str = "train",
    limit: Optional[int] = None,
    seed: int = 3407,
    token = None,
) -> list:
    """Rows of one public source in the decision format (no augmentation)."""
    from datasets import load_dataset

    source = SOURCES[name]
    hf_split = {"train": source.train_split, "test": source.test_split}.get(split, split)
    if hf_split is None:
        raise ValueError(f"Unsloth: {name} has no {split} split.")
    dataset = load_dataset(source.hf_id, source.config, split = hf_split, token = token)
    if limit is not None and len(dataset) > limit:
        dataset = dataset.shuffle(seed = seed).select(range(limit))
    return list(source.convert(dataset, _label_names(dataset, source.label_column)))


GENERIC_IDS = ("q1", "field_a", "decision", "answer_1", "label")


@dataclass
class AugmentConfig:
    rename_question_ids: float = 0.3
    rename_options: float = 0.3
    paraphrase_instructions: float = 0.4
    drop_instructions: float = 0.1
    derived_questions: float = 0.5
    state_as_text: float = 0.3
    max_options: int = 24
    shuffle_fields: bool = True


def _random_code(rng, used) -> str:
    while True:
        code = "".join(rng.choice(string.ascii_lowercase) for _ in range(3)) + str(
            rng.randint(0, 9)
        )
        if code not in used:
            used.add(code)
            return code


def _gold_label(gold):
    if isinstance(gold, dict):
        if gold.get("probabilities") is not None or gold.get("noul") is not None:
            return None
        gold = gold.get("label")
    return gold


def _subsample_options(question, gold, rng, max_options):
    criteria = question["criteria"]
    label = _gold_label(gold)
    if len(criteria) <= max_options or label is None or str(label) not in criteria:
        return question, gold
    others = [key for key in criteria if key != str(label)]
    kept = set(rng.sample(others, max_options - 1)) | {str(label)}
    return {**question, "criteria": {k: v for k, v in criteria.items() if k in kept}}, gold


def _rename_options(question, gold, rng):
    label = _gold_label(gold)
    criteria = question["criteria"]
    if label is None or str(label) not in criteria:
        return question, gold
    style = rng.choice(("letters", "codes"))
    keys = list(criteria)
    if style == "letters" and len(keys) <= 26:
        new = list(string.ascii_uppercase[: len(keys)])
    else:
        used = set()
        new = [_random_code(rng, used) for _ in keys]
    mapping = dict(zip(keys, new))
    # The meaning moves into the description when the id stops carrying it.
    renamed = {mapping[k]: (v if v else _humanize(k)) for k, v in criteria.items()}
    return {**question, "criteria": renamed}, mapping[str(label)]


def _derived(qid, question, gold, rng):
    label = _gold_label(gold)
    kind = question["type"]
    if label is None:
        return None
    if kind == "choice" and str(label) in question["criteria"]:
        options = list(question["criteria"])
        option = str(label) if rng.random() < 0.5 else rng.choice(options)
        description = question["criteria"][option] or _humanize(option)
        return (
            f"is_{_slug(option)[:24]}",
            {
                "type": "noul",
                "instructions": f"Is the answer to '{question.get('instructions') or qid}' this: {description}?",
            },
            option == str(label),
        )
    if kind == "score" and isinstance(label, int) and len(question["criteria"]) > 1:
        levels = question["criteria"]
        threshold = rng.randint(1, len(levels) - 1)
        return (
            f"at_least_{_slug(levels[threshold])[:24]}",
            {
                "type": "noul",
                "instructions": f"For '{question.get('instructions') or qid}', is it at least '{levels[threshold]}'?",
            },
            label >= threshold,
        )
    return None


def _state_text(state) -> str:
    if not isinstance(state, dict):
        return state if isinstance(state, str) else json.dumps(state, ensure_ascii = False)
    if len(state) == 1:
        return str(next(iter(state.values())))
    return "\n".join(f"{_humanize(key).capitalize()}: {value}" for key, value in state.items())


def augment_row(
    row: dict,
    rng: random.Random,
    config: Optional[AugmentConfig] = None,
) -> dict:
    """One schema variation of a converted row: ids, option ids, instructions, extra fields, order."""
    config = config or AugmentConfig()
    variants = row.get("_variants") or {}
    questions, gold = {}, {}
    for qid, question in row["questions"].items():
        question, answer = dict(question), row["gold"][qid]
        variant = variants.get(qid) or {}
        if question["type"] == "choice" and isinstance(question.get("criteria"), dict):
            question, answer = _subsample_options(question, answer, rng, config.max_options)
            if rng.random() < config.rename_options:
                question, answer = _rename_options(question, answer, rng)
        dropped = False
        if rng.random() < config.drop_instructions:
            question.pop("instructions", None)
            dropped = True
        elif variant.get("instructions") and rng.random() < config.paraphrase_instructions:
            question["instructions"] = rng.choice(variant["instructions"])
        new_id = qid
        # A dropped instruction leaves the id as the only description, so it stays meaningful.
        if rng.random() < config.rename_question_ids:
            pool = list(variant.get("ids") or ())
            if not dropped:
                pool += list(GENERIC_IDS)
            if pool:
                new_id = rng.choice(pool)
        if new_id in questions or (new_id != qid and new_id in row["questions"]):
            new_id = qid
        while new_id in questions:
            new_id = f"{new_id}_"
        questions[new_id], gold[new_id] = question, answer
        if rng.random() < config.derived_questions:
            extra = _derived(new_id, question, answer, rng)
            if extra is not None and extra[0] not in questions:
                questions[extra[0]], gold[extra[0]] = extra[1], extra[2]
    order = list(questions)
    if config.shuffle_fields:
        rng.shuffle(order)
    state = row["state"]
    if isinstance(state, dict) and rng.random() < config.state_as_text:
        state = _state_text(state)
    return {
        "source": row.get("source"),
        "state": state,
        "questions": {qid: questions[qid] for qid in order},
        "gold": {qid: gold[qid] for qid in order},
    }


def canonical_row(
    row: dict,
    max_options: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """The row as converted, for evaluation: no augmentation, option subsampling only if asked."""
    out = {k: v for k, v in row.items() if k != "_variants"}
    if max_options is not None:
        out["questions"], out["gold"] = dict(out["questions"]), dict(out["gold"])
        rng = random.Random(seed)
        for qid, question in list(out["questions"].items()):
            if question["type"] == "choice" and isinstance(question.get("criteria"), dict):
                out["questions"][qid], out["gold"][qid] = _subsample_options(
                    question, out["gold"][qid], rng, max_options
                )
    return out


def _words(text) -> list:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii = False, sort_keys = True)
    return re.findall(r"\w+", text.lower())


class Decontaminator:
    """Drops rows whose state shares any n-word shingle (default 13) with an evaluation text."""

    def __init__(
        self,
        eval_texts: Iterable,
        n: int = 13,
    ):
        self.n = n
        self.shingles = set()
        for text in eval_texts:
            self.shingles.update(self._shingles(text))

    def _shingles(self, text) -> set:
        words = _words(text)
        if len(words) < self.n:
            return {" ".join(words)} if words else set()
        return {" ".join(words[i : i + self.n]) for i in range(len(words) - self.n + 1)}

    def contaminated(self, row) -> bool:
        return not self.shingles.isdisjoint(self._shingles(row["state"]))


def build_decision_mixture(
    sources,
    n_rows: int,
    seed: int = 3407,
    split: str = "train",
    augment: bool = True,
    augment_config: Optional[AugmentConfig] = None,
    decontaminate_against: Optional[Iterable] = None,
    token = None,
) -> list:
    """A shuffled mix of decision rows from several sources.

    `sources`: names from SOURCES, {name: weight}, or {name: [converted rows]} (weights equal).
    Each source contributes n_rows * weight / total rows, sampled with replacement only when it
    has fewer rows than its share (each repeat gets a fresh augmentation).
    """
    rng = random.Random(seed)
    if isinstance(sources, (list, tuple)):
        sources = {name: 1.0 for name in sources}
    pools, weights = {}, {}
    for name, value in sources.items():
        if name in EVAL_ONLY_SOURCES:
            raise ValueError(
                f"Unsloth: {name} is a Decision Index benchmark and stays out of training."
            )
        if isinstance(value, (int, float)):
            weights[name] = float(value)
            share = max(
                1,
                round(
                    n_rows
                    * float(value)
                    / sum(
                        float(v) if isinstance(v, (int, float)) else 1.0 for v in sources.values()
                    )
                ),
            )
            pools[name] = load_source(name, split, limit = share, seed = seed, token = token)
        else:
            weights[name] = 1.0
            pools[name] = list(value)
    if decontaminate_against is not None:
        cleaner = Decontaminator(decontaminate_against)
        pools = {
            name: [r for r in rows if not cleaner.contaminated(r)] for name, rows in pools.items()
        }
    live = [name for name in pools if pools[name]]
    total = sum(weights[name] for name in live)
    exact = {name: n_rows * weights[name] / total for name in live}
    shares = {name: max(1, int(exact[name])) for name in live}
    for name in sorted(live, key = lambda n: int(exact[n]) - exact[n])[
        : max(0, n_rows - sum(shares.values()))
    ]:
        shares[name] += 1
    rows = []
    for name in live:
        pool, share = pools[name], shares[name]
        picks = (
            rng.sample(pool, share)
            if share <= len(pool)
            else [rng.choice(pool) for _ in range(share)]
        )
        for row in picks:
            rows.append(augment_row(row, rng, augment_config) if augment else canonical_row(row))
    rng.shuffle(rows)
    return rows[:n_rows]
