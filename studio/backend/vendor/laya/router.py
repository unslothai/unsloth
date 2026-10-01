"""Route a request to the Laya checkpoint best suited to it.

Three checkpoints, measured on a shared benchmark (17,416 questions, one T4, identical questions
per model -- see the repository's benchmark notebook):

  english          convaiinnovations/laya                421M  ModernBERT-large, 512 tokens
  multilingual     convaiinnovations/laya-multilingual   322M  mmBERT-base, 1024 tokens, 100+ langs
  typed-decisions  convaiinnovations/laya-typed-decisions 421M  ModernBERT-large, 1024 tokens,
                                                                fine-tuned on the typed-decisions
                                                                workflows

Why routing is worth it -- accuracy by language family:

                      english   multilingual
  MASSIVE intent  en    0.783       0.657        <- English checkpoint wins
  MASSIVE intent  non-en 0.306      0.451
  XNLI            en    0.860       0.843
  XNLI            non-en 0.521      0.731        <- +21 points for multilingual
  English suites        0.684       0.619

The English checkpoint does not gently degrade off English, it collapses: on 20-option MASSIVE
intent it scores 0.100 on Hindi and 0.103 on Korean, against 0.050 for random guessing -- and it
reports high confidence while doing so (ECE 0.855 on Hindi). Script detection is therefore the
primary routing signal.

`typed-decisions` is never selected automatically unless you opt in with
`auto_task_detection=True` or pass `task="typed_decisions"`: it is fine-tuned on four specific
synthetic workflows and should not be a silent default.
"""
import os
import threading
from typing import Any, Dict, List, Optional, Union

from .lang import analyse

# The hub repo bundles all three checkpoints; only the requested subfolder is downloaded.
BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}

# The same checkpoints also live in their own repos, for anyone who prefers them.
STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}


def _repo_str(spec):
    """Human-readable id for a model spec: 'repo' or 'repo/subfolder'."""
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


def _split(spec):
    """Normalise a model spec to (repo_or_path, subfolder)."""
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None

# Aliases people are likely to type.
_ALIASES = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

# Question-id signatures of the four typed-decisions workflows, used only when
# auto_task_detection is enabled.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}


class RouteDecision(dict):
    """The routing outcome: which model, why, and what was detected.

    Behaves as a dict so it serialises straight into an API response.
    """

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self):
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError("unknown model %r; choose one of %s (or an alias: %s)"
                         % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES)))
    return key


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """Name of the typed-decisions workflow whose question ids these are, else None.

    Requires an exact id-set match, so an unrelated schema that happens to contain 'urgency'
    is never captured.
    """
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


class Router:
    """Lazily loads Laya checkpoints and sends each request to the right one.

        from laya import Router

        r = Router()
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)   # -> multilingual
        r.predict({"message": "I was charged twice"}, questions)                 # -> english
        r.predict(state, questions, model="typed-decisions")                     # explicit

    Models are downloaded and built on first use. `max_loaded` caps how many stay resident
    (least-recently-used is evicted), because all three together are ~1.16B parameters.

    For a server or a demo, preload instead: a cold load costs seconds, while detection costs
    microseconds, so anything that alternates languages at `max_loaded=1` reloads on every
    request.

        r = Router(preload=True)                    # all three resident, routing is free
        r = Router(preload=True, device="cuda")
        r.preload(["english", "multilingual"])      # or just the two you serve
    """

    def __init__(
        self,
        models: Optional[Dict[str, str]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_loaded: int = 1,
        default: str = "english",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
    ):
        self.models = dict(STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): v for k, v in models.items()})
        self.device = device
        self.token = token or os.environ.get("HF_TOKEN")
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []          # least-recently-used first
        # Re-entrant lock guarding model lifecycle (load/unload/attach/preload) and the
        # LRU bookkeeping. RLock so the public methods can call the private `_touch`/`_evict`
        # helpers without deadlocking. Inference (`Agent.system_one`) is deliberately left
        # outside the lock so concurrent predictions share a checkpoint without serialising.
        self._lock = threading.RLock()
        if preload:
            self.preload()

    # ------------------------------------------------------------------ loading
    def load(self, name: str):
        """Return the Agent for `name`, downloading and building it on first use.

        Concurrent callers share a single Agent instead of building duplicates.
        """
        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            from .agent import Agent
            repo, sub = _split(self.models[key])
            agent = Agent(repo, device=self.device, token=self.token, subfolder=sub)
            self._agents[key] = agent
            self._order.append(key)
            self._evict()
            return agent

    def _touch(self, key: str):
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict(self):
        with self._lock:
            while len(self._order) > self.max_loaded:
                victim = self._order.pop(0)
                self._agents.pop(victim, None)
            if len(self._order) < len(self._agents):     # keep the two views consistent
                for k in list(self._agents):
                    if k not in self._order:
                        self._agents.pop(k, None)

    def attach(self, name: str, agent: Any):
        """Register an already-built Agent under `name` instead of loading a second copy.

        Useful when the process has a checkpoint loaded for other reasons: a demo that already
        built `convaiinnovations/laya` can hand it to the router rather than pay for -- and hold
        in memory -- a duplicate 421M parameters.
        """
        key = normalise_name(name)
        with self._lock:
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: Optional[List[str]] = None):
        """Download and build checkpoints up front so no request ever pays a model load.

        A cold load costs seconds; language detection costs microseconds. With every
        checkpoint resident, routing is effectively free -- which is what you want in a
        server or a demo. `max_loaded` is raised to fit whatever is preloaded, otherwise
        the LRU would immediately evict what this just built.
        """
        names = [normalise_name(n) for n in (names or list(self.models))]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(names), len(self._agents))
            for n in names:
                if n not in self._agents:      # an attached agent is already built
                    self.load(n)
        return self

    def unload(self, name: Optional[str] = None):
        """Free one model, or all of them."""
        with self._lock:
            if name is None:
                self._agents.clear()
                self._order.clear()
            else:
                key = normalise_name(name)
                self._agents.pop(key, None)
                if key in self._order:
                    self._order.remove(key)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    # ------------------------------------------------------------------ routing
    def route(
        self,
        state: Union[str, dict, list, None],
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> RouteDecision:
        """Decide which checkpoint to use, without loading or running anything.

        Precedence: explicit `model` > explicit `task` > detected workflow (opt-in) >
        explicit `lang` > detected script/language > default.
        """
        if model is not None:
            key = normalise_name(model)
            return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit model=%r" % model,
                                 detection=None, workflow=None)

        if task is not None:
            key = normalise_name("typed-decisions" if str(task).lower().replace("-", "_") == "typed_decisions" else task)
            return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit task=%r" % task,
                                 detection=None, workflow=None)

        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            return RouteDecision(model="typed-decisions", repo=self.models["typed-decisions"],
                                 reason="question ids match the %r typed-decisions workflow" % workflow,
                                 detection=None, workflow=workflow)

        if lang is not None:
            key = "english" if str(lang).lower().split("-")[0] in ("en", "eng", "english") else "multilingual"
            return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason="explicit lang=%r" % lang,
                                 detection=None, workflow=workflow)

        det = analyse(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it" % (
                det["script"], 100 * float(det["non_latin_fraction"]))
        elif not det["is_english"]:
            key = "multilingual"
            if det["language"]:
                reason = "Latin script but language looks like %r, not English" % det["language"]
            else:
                # Unidentified Latin-script language: routed on the non-English letters alone,
                # because no stopword list here covers it.
                reason = ("Latin script, language not identified but %.0f%% non-English letters; "
                          "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"])))
        else:
            key = "english"
            reason = "English Latin text"
        return RouteDecision(model=key, repo=_repo_str(self.models[key]), reason=reason,
                             detection=det, workflow=workflow)

    # ------------------------------------------------------------------ running
    def predict(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Route, then answer every question in one forward pass on the chosen checkpoint.

        The result is the usual `system_one` payload plus a `routing` key recording the decision.
        """
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        agent = self.load(decision["model"])
        result = agent.system_one(state, questions)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    def __repr__(self):
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (self.loaded, self.max_loaded, self.default)
