"""conftest.run on a hung command: prints every process's stack, and leaves no process behind."""

from __future__ import annotations

import os
import time

import pytest

from conftest import run

HANG = r"""
import faulthandler, os, signal, sys, time
faulthandler.register(signal.SIGUSR1, all_threads=True)
child = os.fork()
def stuck_in_the_child(): time.sleep(600)
def stuck_in_the_driver(): time.sleep(600)
open(sys.argv[1], "a").write(f"{os.getpid()} ")
if child == 0:
    stuck_in_the_child()
stuck_in_the_driver()
"""


def test_timeout_dumps_stacks_and_kills_the_group(tmp_path):
    pids = tmp_path / "pids"
    with pytest.raises(pytest.fail.Exception) as e:
        run(["python", "-c", HANG, pids], timeout=3)
    msg = str(e.value)
    assert "timed out after 3 s" in msg
    assert "stuck_in_the_driver" in msg and "stuck_in_the_child" in msg, msg
    time.sleep(0.5)
    alive = []
    for pid in map(int, pids.read_text().split()):
        try:
            os.kill(pid, 0)
            alive.append(pid)
        except ProcessLookupError:
            pass
    assert not alive, alive


def test_a_normal_command_still_passes(tmp_path):
    assert run(["python", "-c", "print('ok')"]).stdout == "ok\n"
