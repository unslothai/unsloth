"""The shipped controller must order victims the way the benchmark ranked best.

`scripts/preempt_bench.py` chose newest-first (with parked holders taken first) over six
alternatives across nine load regimes. That choice is only worth anything if the
controller actually implements it, and nothing else in the suite compares the two: the
policy tests assert individual cases, which a subtly different ordering can still pass.

So this replays the simulator's own key function against the controller's real victim
order on randomised populations. A divergence means either the implementation drifted or
the benchmark measured a policy nobody ships.
"""

import random
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from core.inference.llama_admission import reset_llama_admission_queues  # noqa: E402
from core.inference.llama_preemption import (  # noqa: E402
    ParticipantState,
    PreemptionController,
    reset_preemption_controllers,
)

preempt_sim = pytest.importorskip(
    "preempt_sim", reason = "the simulator lives beside the repo, not inside it"
)


@pytest.fixture(autouse = True)
def _clean():
    reset_llama_admission_queues()
    reset_preemption_controllers()
    yield
    reset_llama_admission_queues()
    reset_preemption_controllers()


class _SimChat:
    """The fields preempt_sim's key functions read, taken from a real Participant."""

    def __init__(self, participant, arrival):
        self.cid = arrival
        self.arrival = arrival
        self.resident = participant.tokens
        self.remaining = 0
        self.state = (
            "parked" if participant.state == ParticipantState.PARKED_ON_TOOL else "decoding"
        )


def _controller(budget = 16384, slots = 4):
    controller = PreemptionController("conformance")
    controller.configure(budget = budget, kv_unified = True, draft_tokens = 2, slots = slots)
    return controller


class TestTheShippedOrderIsTheBenchmarkedOne:
    @pytest.mark.parametrize("seed", range(40))
    def test_victim_order_matches_newest_first(self, seed):
        rng = random.Random(seed)
        controller = _controller()
        snapshot = controller.snapshot()
        ceiling = snapshot.budget - snapshot.buffer

        population = []
        for i in range(rng.randint(3, 7)):
            tokens = rng.randint(200, ceiling // 2)
            parked = rng.random() < 0.3
            controller.register(
                f"gen{i}",
                tokens = tokens,
                state = (ParticipantState.PARKED_ON_TOOL if parked else ParticipantState.DECODING),
            )
            population.append(i)

        # BEFORE the sweep: plan_preemptions marks each victim PREEMPTING, so reading
        # state afterwards loses the parked flag and makes a correct implementation look
        # like a divergence. Cost 31 failing tests before it was spotted.
        sim_chats = [_SimChat(controller.participant(f"gen{i}"), i) for i in population]

        victims = [p.gen_id for p in controller.plan_preemptions(needed = ceiling)]
        if not victims:
            pytest.skip("this population never crossed the watermark")
        expected = [
            f"gen{c.cid}" for c in sorted(sim_chats, key = preempt_sim.POLICIES["newest_first"])
        ]
        # The sweep stops once it fits and always leaves one holder, so compare the
        # prefix it actually took rather than the whole ordering.
        assert victims == expected[: len(victims)], (
            f"shipped order {victims} is not the benchmarked newest-first "
            f"{expected[: len(victims)]}"
        )

    def test_parked_holders_come_first_in_both(self):
        controller = _controller()
        controller.register("decoding_new", tokens = 3000)
        controller.register("parked_old", tokens = 3000, state = ParticipantState.PARKED_ON_TOOL)
        controller.register("decoding_newest", tokens = 3000)
        snapshot = controller.snapshot()
        victims = [
            p.gen_id for p in controller.plan_preemptions(needed = snapshot.budget - snapshot.buffer)
        ]
        assert victims[0] == "parked_old", (
            "a chat holding cells while consuming no compute is the cheapest room to "
            "take, in the simulator and here"
        )

    def test_the_benchmark_still_prefers_what_is_shipped(self):
        """Guards the other direction: if the ranking flips, the code should follow.

        Cheap enough to run in CI at low seed counts, and it fails loudly if a future
        change to the model makes a different policy win.
        """
        rows = preempt_sim.sweep(
            ["newest_first", "largest_first", "oldest_first"],
            seeds = 12,
            epoch_winner = False,
            chats = 8,
            target_hi = 14000,
            evict_latency = 2,
        )
        assert (
            rows["newest_first"]["makespan"] <= rows["largest_first"]["makespan"]
        ), "largest-first now wins on makespan; the shipped policy should be revisited"
