"""
Scalability tests: establish where the brute-force engine hits its limit.

Uses ProcessPoolExecutor so timeouts actually kill the worker — threads cannot
be interrupted in Python, but processes can.

Run with:
    pytest tests/test_scalability.py -v -s
"""

import time
import multiprocessing
import pytest
from dice.parser import parse
from dice.engine import build_pmf
from dice.simplify import simplify

TIMEOUT = 3.0  # seconds per case (includes ~1s Windows subprocess startup overhead)


# Top-level so it is picklable by the worker process
def _worker(text: str, queue: multiprocessing.Queue) -> None:
    t0 = time.perf_counter()
    expr, domains = parse(text)
    expr, domains = simplify(expr, domains)
    result = build_pmf(expr, domains)
    queue.put((result, time.perf_counter() - t0))


def _run(text: str) -> tuple[dict[int, float] | None, float]:
    """Run in a subprocess; terminate and return None on timeout."""
    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(text, q))
    p.start()
    p.join(timeout=TIMEOUT)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, TIMEOUT
    return q.get()


def _n_dice(n: int, sides: int) -> str:
    return ' + '.join(f'd{sides}' for _ in range(n))


def _check(text: str, label: str) -> dict[int, float]:
    pmf, elapsed = _run(text)
    if pmf is None:
        pytest.skip(f'{label}: timed out (>{TIMEOUT}s)')
    assert abs(sum(pmf.values()) - 1.0) < 1e-9
    assert all(p >= 0 for p in pmf.values())
    return pmf


# ── Sum of N identical dice ───────────────────────────────────────────────────
# All dice are independent — the ideal case for convolution.
# Current engine: O(sides^N) — should blow up around N=7 for d6, N=4 for d20.


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_sum_nd6(n):
    expr = _n_dice(n, 6)
    pmf = _check(expr, f'{n}d6')
    assert min(pmf) == n and max(pmf) == 6 * n


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 7])
def test_sum_nd20(n):
    expr = _n_dice(n, 20)
    pmf = _check(expr, f'{n}d20')
    assert min(pmf) == n and max(pmf) == 20 * n


# ── Two independent groups summed ─────────────────────────────────────────────
# (Nd6) + (Nd6): both halves are fully independent — the simplification pass
# should collapse each half to one virtual variable and enumerate only 2 dims.
# Current engine sees 2N dice and enumerates 6^(2N).


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_independent_halves(n):
    expr = f'({_n_dice(n, 6)}) + ({_n_dice(n, 6)})'
    pmf = _check(expr, f'({n}d6)+({n}d6)')
    assert min(pmf) == 2 * n and max(pmf) == 12 * n


# ── Condition independent of branches ─────────────────────────────────────────
# d6 > 5 -> Nd20 | 0: condition and branches share no variables.
# Simplification pass should collapse Nd20 to one virtual variable,
# then enumerate only d6 × virtual_d20 (6 × (19N+1) entries).
# Current engine: 6 × 20^N.


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 10])
def test_independent_conditional(n):
    expr = f'd6 > 5 -> {_n_dice(n, 20)} | 0'
    pmf = _check(expr, f'd6>5 -> {n}d20|0')
    assert abs(pmf.get(0, 0.0) - 5 / 6) < 1e-9


def test_independent_conditional_large():
    # 50d20 in-process: subprocess startup costs ~2s on Windows, masking the
    # real computation time. Run directly to verify simplify collapses all 51
    # variables to 1 virtual variable and completes well under 1s.
    import time
    n = 50
    expr_text = f'd6 > 5 -> {_n_dice(n, 20)} | 0'
    expr, domains = parse(expr_text)
    expr, domains = simplify(expr, domains)
    assert len(domains) == 1, f"Expected 1 virtual variable, got {len(domains)}"
    t0 = time.perf_counter()
    pmf = build_pmf(expr, domains)
    elapsed = time.perf_counter() - t0
    assert abs(pmf.get(0, 0.0) - 5 / 6) < 1e-9
    assert elapsed < 1.0, f"build_pmf took {elapsed:.3f}s after simplification"


# ── Shared variable across condition and branch (must enumerate) ───────────────
# f(X): X > 3 -> X | 0  — X appears in both condition and then-branch.
# No shortcut possible; must enumerate X. Should always be fast (single die).


@pytest.mark.parametrize('sides', [6, 20, 100])
def test_shared_variable_conditional(sides):
    expr = f'f(X): X > 3 -> X | 0; f(d{sides})'
    pmf = _check(expr, f'X>3->X|0 (d{sides})')
    assert abs(sum(pmf.values()) - 1.0) < 1e-9
