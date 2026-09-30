"""The reorder buffer that hides provider response order from everything downstream.

A transcription provider answers when it feels like it. Media order is a fact
about the recording, so nothing downstream may inherit the network's order.
"""

from __future__ import annotations

import math

import pytest

from livestream_transcriber.audio.ordering import MediaOrderGate


def gate_with_three() -> tuple[MediaOrderGate[str], list[int]]:
    gate: MediaOrderGate[str] = MediaOrderGate()
    tickets = [gate.submit(100.0, 105.0), gate.submit(105.0, 110.0), gate.submit(110.0, 115.0)]
    return gate, tickets


def test_in_order_completion_releases_immediately():
    gate, (a, b, c) = gate_with_three()
    assert [s.start for s in gate.complete(a, "A")] == [100.0]
    assert [s.start for s in gate.complete(b, "B")] == [105.0]
    assert [s.start for s in gate.complete(c, "C")] == [110.0]
    assert gate.reordered == 0


def test_a_later_chunk_waits_for_the_earlier_one():
    gate, (a, b, c) = gate_with_three()
    assert gate.complete(b, "B") == []
    assert gate.complete(c, "C") == []
    released = gate.complete(a, "A")
    assert [s.payload for s in released] == ["A", "B", "C"]
    assert [s.start for s in released] == [100.0, 105.0, 110.0]


def test_the_frontier_never_runs_past_an_outstanding_chunk():
    gate, (a, b, c) = gate_with_three()
    assert gate.frontier == -math.inf
    gate.complete(b, "B")
    gate.complete(c, "C")
    # 105-110 and 110-115 are back, but 100-105 might still say something,
    # so the lane has covered nothing at all.
    assert gate.frontier == -math.inf
    gate.complete(a, "A")
    assert gate.frontier == 115.0


def test_out_of_order_arrivals_are_counted_but_not_hidden():
    gate, (a, b, c) = gate_with_three()
    gate.complete(c, "C")
    gate.complete(b, "B")
    gate.complete(a, "A")
    assert gate.reordered == 2
    assert gate.outstanding == 0
    assert gate.describe() == {"outstanding": 0, "reordered": 2, "frontier": 115.0}


def test_an_empty_result_still_advances_the_lane():
    """Silence is not an excuse to stall: the chunk was handled."""
    gate: MediaOrderGate[str] = MediaOrderGate()
    ticket = gate.submit(0.0, 5.0)
    released = gate.complete(ticket, None)
    assert len(released) == 1 and released[0].payload is None
    assert gate.frontier == 5.0


def test_extra_fields_travel_with_the_slot():
    gate: MediaOrderGate[str] = MediaOrderGate()
    ticket = gate.submit(0.0, 5.0)
    (slot,) = gate.complete(ticket, "text", latency=0.4)
    assert slot.extra == {"latency": 0.4}


def test_an_unknown_ticket_is_an_error_not_a_silent_drop():
    gate: MediaOrderGate[str] = MediaOrderGate()
    with pytest.raises(KeyError):
        gate.complete(7, "x")


def test_a_released_ticket_cannot_be_completed_again():
    gate: MediaOrderGate[str] = MediaOrderGate()
    ticket = gate.submit(0.0, 5.0)
    gate.complete(ticket, "x")
    with pytest.raises(KeyError):
        gate.complete(ticket, "again")


def test_a_long_shuffled_run_releases_everything_in_order():
    gate: MediaOrderGate[int] = MediaOrderGate()
    tickets = [gate.submit(i * 5.0, i * 5.0 + 5.0) for i in range(50)]
    order = tickets[::2] + tickets[1::2]  # evens first, then odds
    released: list[int] = []
    for ticket in order:
        released.extend(s.ticket for s in gate.complete(ticket, ticket))
    assert released == tickets
    assert gate.outstanding == 0 and gate.frontier == 250.0
