"""run_analysis's pool: results in task order, the slowest fits dispatched first, no executor.

The paper gate (`test_paper_numbers.py`) checks that the refit's outputs stay bit-identical;
these check the scheduling it relies on, in a second, on stand-in tasks.
"""

from __future__ import annotations

import ast
import multiprocessing
import os

import pytest

from conftest import ANALYSIS


def _echo(task: tuple[str, tuple]) -> tuple[int, int]:
    """A stand-in unit: (its task index, the pid of the worker that ran it)."""
    return task[1][0], os.getpid()


@pytest.fixture
def ra(monkeypatch: pytest.MonkeyPatch):
    import run_analysis

    monkeypatch.setattr(run_analysis, "_unit", _echo)  # forked workers see the stand-in
    return run_analysis


def test_results_come_back_in_task_order(ra) -> None:
    laws = ["Kaplan", "Parametric-Epoch", "Chinchilla", "Droppo-Elibol"] * 25
    tasks = [("fc", (i, "csv", law)) for i, law in enumerate(laws)]
    with multiprocessing.get_context("fork").Pool(4) as pool:
        out = ra._pool_map(pool, tasks)
    assert [i for i, _ in out] == list(range(len(tasks)))


def test_slowest_laws_are_dispatched_first(ra, monkeypatch: pytest.MonkeyPatch) -> None:
    laws = [
        "Kaplan",
        "Chinchilla",
        "Parametric-Epoch",
        "Droppo-Elibol",
        "Kaplan",
        "Parametric-Epoch",
    ]
    tasks = [("sv", (i, "csv", law)) for i, law in enumerate(laws)]
    sent: list[list[tuple[str, tuple]]] = []

    class Recorder:
        def imap(self, fn, items, chunksize):
            sent.append(list(items))
            return map(fn, sent[-1])

    out = ra._pool_map(Recorder(), tasks)
    assert [t[1][2] for t in sent[0]] == [
        "Parametric-Epoch",
        "Parametric-Epoch",
        "Droppo-Elibol",
        "Chinchilla",
        "Kaplan",
        "Kaplan",
    ]
    assert [t[1][0] for t in sent[0]][:2] == [2, 5]  # ties keep their task order
    assert [i for i, _ in out] == list(range(len(tasks)))


def test_no_concurrent_futures_executor() -> None:
    """ProcessPoolExecutor.map can deadlock in submit() on Python 3.10 (CPython gh-105829)."""
    tree = ast.parse((ANALYSIS / "run_analysis.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "ProcessPoolExecutor" not in names
