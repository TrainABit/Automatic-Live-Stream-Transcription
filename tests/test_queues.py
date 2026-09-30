from __future__ import annotations

import asyncio

import pytest

from livestream_transcriber.stream.queues import QUEUE_CLOSED, DropOldestQueue


def test_fifo_order_when_not_full() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(4)
    for i in range(3):
        assert q.put(i) is True
    assert q.qsize() == 3
    assert q.dropped == 0


def test_overflow_drops_oldest_and_counts() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(3)
    for i in range(3):
        q.put(i)
    assert q.put(99) is False  # signals that a drop happened
    assert q.dropped == 1
    assert q.qsize() == 3


async def test_consumer_sees_newest_after_overflow() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    for i in range(5):
        q.put(i)
    q.close()
    got = [item async for item in q]
    assert got == [3, 4], "a live pipeline must keep the newest, not the oldest"
    assert q.dropped == 3


async def test_close_terminates_iteration() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(4)
    q.put(1)
    q.close()
    assert [x async for x in q] == [1]


async def test_get_returns_the_sentinel_after_close() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    q.close()
    assert await q.get() is QUEUE_CLOSED


async def test_every_read_after_close_sees_the_end() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    q.put(1)
    q.close()
    assert [x async for x in q] == [1]
    # A second reader must not wait forever for a marker the first one consumed.
    assert await asyncio.wait_for(q.get(), 1) is QUEUE_CLOSED
    assert [x async for x in q] == []


def test_close_is_idempotent_and_blocks_further_puts() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    q.close()
    q.close()
    assert q.closed is True
    assert q.put(1) is False


async def test_get_raises_the_close_error() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    q.close(RuntimeError("producer died"))
    with pytest.raises(RuntimeError, match="producer died"):
        await q.get()


async def test_second_close_keeps_the_first_non_none_error() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    q.close(RuntimeError("first"))
    q.close(RuntimeError("second"))
    q.close()
    with pytest.raises(RuntimeError, match="first"):
        await q.get()


async def test_second_close_can_set_error_if_first_was_clean() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    q.close()
    q.close(RuntimeError("late"))
    with pytest.raises(RuntimeError, match="late"):
        await q.get()


def test_maxsize_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DropOldestQueue(0)


async def test_put_wait_applies_backpressure_instead_of_dropping() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(2)
    assert await q.put_wait(0)
    assert await q.put_wait(1)

    blocked = asyncio.create_task(q.put_wait(2))
    await asyncio.sleep(0.05)
    assert not blocked.done(), "a full queue must make the producer wait"

    assert await q.get() == 0
    assert await asyncio.wait_for(blocked, 5) is True
    assert q.dropped == 0
    q.close()
    assert [x async for x in q] == [1, 2]


async def test_put_wait_returns_false_when_closed_while_waiting() -> None:
    q: DropOldestQueue[int] = DropOldestQueue(1)
    await q.put_wait(0)
    blocked = asyncio.create_task(q.put_wait(1))
    await asyncio.sleep(0.05)
    q.close()
    assert await asyncio.wait_for(blocked, 5) is False
