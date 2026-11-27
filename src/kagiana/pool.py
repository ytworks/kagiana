"""A worker pool whose tasks can be aborted by force.

Why not ``concurrent.futures``: a task already running there cannot be
cancelled, and the expensive part of 3D screening happens inside RDKit's C++
layer, where a Python-level interrupt never arrives.  A molecule whose
distance-geometry embedding fails to converge would therefore stall the whole
run.

This pool keeps one long-lived process per worker slot, dispatches a single
task at a time to each, and records when it was dispatched.  When a task
overruns its deadline the parent kills that process outright and starts a
replacement, so the run continues while the offending molecule is recorded as
``timeout``.  Results are emitted in input order; because a slot accepts a new
task only after its previous result has been read, at most ``n_workers``
results are ever buffered.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from time import monotonic
from typing import Any, Callable, Iterable, Iterator, Sequence

from .schema import Status

#: How often the parent wakes up to re-check deadlines.
_POLL_INTERVAL = 0.05

_STARTING = "starting"
_IDLE = "idle"
_BUSY = "busy"

# Set inside a worker process while a task is running, so that a task can
# report partial progress the parent can still use if the worker is killed.
_CURRENT: dict[str, Any] = {"conn": None, "index": -1}


def report_progress(value: Any) -> None:
    """Report partial progress from inside a worker task.

    A task that is killed on its deadline returns no value, but work it had
    already completed need not be thrown away: whatever it last reported here
    arrives as :attr:`TaskResult.partial` alongside the timeout status.  Calling
    this outside a worker is a no-op, so tasks stay usable in-process.
    """
    conn = _CURRENT.get("conn")
    if conn is None:
        return
    try:
        conn.send(("progress", (_CURRENT["index"], value)))
    except (OSError, BrokenPipeError, ValueError):
        pass


class WorkerInitError(RuntimeError):
    """A worker process failed to build its per-process state."""


def preferred_start_method() -> str:
    """``fork`` on Linux, ``spawn`` elsewhere.

    ``fork`` avoids re-importing RDKit in every worker, but it is unsafe in a
    process that has already started threads, which is the normal situation on
    macOS, so that platform uses ``spawn``.
    """
    return "fork" if platform.system() == "Linux" else "spawn"


def default_jobs() -> int:
    """Number of worker processes to use when ``--jobs`` is not given."""
    try:
        # Respects CPU affinity masks and cgroup-style restrictions.
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _worker_loop(
    conn: Connection,
    initializer: Callable[..., Callable[[Any], Any]],
    initargs: Sequence[Any],
) -> None:
    """Child-process entry point: build state once, then serve tasks."""
    try:
        task = initializer(*initargs)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent verbatim
        try:
            conn.send(("init_error", f"{type(exc).__name__}: {exc}"))
        finally:
            conn.close()
        return

    conn.send(("ready", None))
    while True:
        try:
            message = conn.recv()
        except (EOFError, KeyboardInterrupt):
            return
        if message is None:
            return
        index, payload = message
        _CURRENT["conn"] = conn
        _CURRENT["index"] = index
        try:
            conn.send(("ok", (index, task(payload))))
        except BaseException as exc:  # noqa: BLE001 - one bad molecule must not stop the run
            conn.send(("error", (index, f"{type(exc).__name__}: {exc}")))
        finally:
            _CURRENT["conn"] = None
            _CURRENT["index"] = -1


@dataclass
class _Slot:
    conn: Connection
    process: Any
    state: str = _STARTING
    index: int = -1
    deadline: float = 0.0
    init_failures: int = 0
    progress: Any = None
    dispatched_at: float = 0.0


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one task, in input order."""

    index: int
    value: Any
    status: str
    detail: str = ""
    partial: Any = None
    elapsed_s: float = 0.0


class KillableWorkerPool:
    """Run a task per payload, forcibly killing workers that overrun.

    ``initializer(*initargs)`` runs once inside each worker process and must
    return the callable that handles a payload.  Keeping expensive, badly
    picklable state (feature factories, pharmacophores) behind that callable
    means it is built once per worker instead of being shipped per task.
    """

    def __init__(
        self,
        n_workers: int,
        initializer: Callable[..., Callable[[Any], Any]],
        initargs: Sequence[Any] = (),
        timeout: float | None = None,
        start_method: str | None = None,
    ) -> None:
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        self._n_workers = n_workers
        self._initializer = initializer
        self._initargs = tuple(initargs)
        self._timeout = timeout
        self._context = mp.get_context(start_method or preferred_start_method())
        self._slots: list[_Slot] = []
        self._kills = 0

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "KillableWorkerPool":
        self._slots = [self._spawn() for _ in range(self._n_workers)]
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _spawn(self) -> _Slot:
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_loop,
            args=(child_conn, self._initializer, self._initargs),
            daemon=True,
        )
        process.start()
        # The child owns its end; holding it in the parent would stop EOF from
        # ever being observed when the child dies.
        child_conn.close()
        return _Slot(conn=parent_conn, process=process)

    def _replace(self, slot: _Slot, init_failures: int = 0) -> _Slot:
        try:
            slot.conn.close()
        except OSError:
            pass
        if slot.process.is_alive():
            slot.process.kill()
        slot.process.join(timeout=5)
        fresh = self._spawn()
        fresh.init_failures = init_failures
        return fresh

    def close(self) -> None:
        for slot in self._slots:
            try:
                if slot.process.is_alive():
                    slot.conn.send(None)
            except (OSError, BrokenPipeError, ValueError):
                pass
        for slot in self._slots:
            slot.process.join(timeout=2)
            if slot.process.is_alive():
                slot.process.kill()
                slot.process.join(timeout=2)
            try:
                slot.conn.close()
            except OSError:
                pass
        self._slots = []

    @property
    def kills(self) -> int:
        """How many workers were killed for overrunning their deadline."""
        return self._kills

    # -- dispatch ----------------------------------------------------------

    def map_ordered(self, payloads: Iterable[Any]) -> Iterator[TaskResult]:
        """Yield one :class:`TaskResult` per payload, in input order."""
        if not self._slots:
            raise RuntimeError("the pool must be used as a context manager")

        source: Iterator[tuple[int, Any]] = enumerate(payloads)
        exhausted = False
        buffered: dict[int, TaskResult] = {}
        next_emit = 0

        while True:
            # Hand work to every idle slot.
            while not exhausted:
                idle = next(
                    (
                        position
                        for position, slot in enumerate(self._slots)
                        if slot.state == _IDLE
                    ),
                    None,
                )
                if idle is None:
                    break
                try:
                    index, payload = next(source)
                except StopIteration:
                    exhausted = True
                    break
                slot = self._slots[idle]
                try:
                    slot.conn.send((index, payload))
                except (OSError, BrokenPipeError, ValueError):
                    # The worker died between results; retry it on a fresh one.
                    self._slots[idle] = self._replace(slot)
                    source = _prepend((index, payload), source)
                    continue
                slot.state = _BUSY
                slot.index = index
                slot.progress = None
                slot.dispatched_at = monotonic()
                slot.deadline = (
                    monotonic() + self._timeout if self._timeout else float("inf")
                )

            while next_emit in buffered:
                yield buffered.pop(next_emit)
                next_emit += 1

            busy = any(slot.state == _BUSY for slot in self._slots)
            if exhausted and not busy and not buffered:
                return

            open_conns = [
                slot.conn for slot in self._slots if slot.state in (_BUSY, _STARTING)
            ]
            if open_conns:
                for conn in wait(open_conns, timeout=_POLL_INTERVAL):
                    position, slot = self._find(conn)
                    if slot is None:
                        continue
                    result = self._receive(position, slot)
                    if result is not None:
                        buffered[result.index] = result

            self._enforce_deadlines(buffered)

    def _find(self, conn: Connection) -> tuple[int, _Slot | None]:
        for position, slot in enumerate(self._slots):
            if slot.conn is conn:
                return position, slot
        return -1, None

    def _receive(self, position: int, slot: _Slot) -> TaskResult | None:
        """Read one message from a worker, replacing it if it has died."""
        try:
            kind, payload = slot.conn.recv()
        except (EOFError, OSError, ValueError):
            # The process vanished (segfault, OOM kill, ...).
            if slot.state == _BUSY:
                failed = TaskResult(
                    slot.index,
                    None,
                    Status.ERROR,
                    "worker process died",
                    partial=slot.progress,
                    elapsed_s=monotonic() - slot.dispatched_at,
                )
                self._slots[position] = self._replace(slot)
                return failed
            self._slots[position] = self._replace(slot, slot.init_failures)
            return None

        if kind == "init_error":
            if slot.init_failures >= 1:
                raise WorkerInitError(str(payload))
            self._slots[position] = self._replace(slot, slot.init_failures + 1)
            return None
        if kind == "ready":
            slot.state = _IDLE
            return None
        if kind == "progress":
            progress_index, value = payload
            if progress_index == slot.index:
                slot.progress = value
            return None

        index, body = payload
        elapsed = monotonic() - slot.dispatched_at
        slot.state = _IDLE
        slot.index = -1
        if kind == "ok":
            return TaskResult(index, body, Status.OK, elapsed_s=elapsed)
        return TaskResult(index, None, Status.ERROR, str(body), elapsed_s=elapsed)

    def _enforce_deadlines(self, buffered: dict[int, TaskResult]) -> None:
        if not self._timeout:
            return
        now = monotonic()
        for position, slot in enumerate(self._slots):
            if slot.state != _BUSY or now < slot.deadline:
                continue
            buffered[slot.index] = TaskResult(
                slot.index,
                None,
                Status.TIMEOUT,
                f"exceeded {self._timeout:g}s",
                partial=slot.progress,
                elapsed_s=now - slot.dispatched_at,
            )
            self._kills += 1
            self._slots[position] = self._replace(slot)


def _prepend(item: tuple[int, Any], rest: Iterator[tuple[int, Any]]) -> Iterator[tuple[int, Any]]:
    yield item
    yield from rest
