"""Verifier for gw-glitch-search.

The deliverable is the program /root/solver/search.py. harness.py simulates 72 held-out
segments at check time from the same forward model as the development segments (the draw
is fixed by the program's bytes), invokes the program once per segment as an unprivileged
account with resource limits, and scores what it prints. The truth of every held-out
segment is known by construction (the injection parameters) and never reaches the program.

Every failure message names each segment that misses (file name and what it contains)
and the size of the miss against the 25 ms / 5 % requirement; the tests never stop on a
missing or malformed value.
"""
import json
import os
import pwd
from pathlib import Path

import pytest

import harness as H

LOG_DIR = Path("/logs/verifier")


def _fmt(items, limit=30):
    items = list(items)
    more = f"; ... and {len(items) - limit} more" if len(items) > limit else ""
    return "; ".join(items[:limit]) + more


@pytest.fixture(scope="session")
def graded():
    try:
        pwd.getpwnam(H.RUNNER)
    except KeyError:
        pytest.fail(f"verifier image is missing the '{H.RUNNER}' account; refusing to run untrusted code")
    try:
        program_bytes = H.read_program()
    except AssertionError as exc:
        pytest.fail(str(exc))
    seed, report = H.evaluate(program_bytes)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / "metrics.json").write_text(json.dumps(
        dict(seed=seed, n=len(report), passed=all(r["ok"] for r in report), held_out=report),
        indent=1, default=str))
    for r in report:
        print(f"{r['name']} [{r['kind']}] ok={r['ok']} rc={r['rc']} {r['elapsed_s']}s "
              f"signal={r['signal']} t={r['t']} mc={r['mc']} true=({r['true_signal']}, {r['true_t']}, {r['true_mc']})")
    return report


def test_program_file():
    """/root/solver/search.py exists as a regular file (not a symbolic link), is not empty
    and is at most 1,000,000 bytes long."""
    H.read_program()


def test_every_invocation_completes(graded):
    """On every held-out segment the program exits with status 0 within 45 s wall and CPU
    time, without exceeding the other limits, and writes at most 1 MiB to stdout."""
    bad = []
    for r in graded:
        if r["completed"]:
            continue
        if r["timed_out"]:
            why = f"timed out after {H.WALL_S} s wall time"
        elif r["too_big"]:
            why = "wrote more than 1 MiB to stdout"
        elif r["rc"] is not None and r["rc"] < 0:
            why = (f"killed by signal {-r['rc']} after {r['elapsed_s']} s (the limits are {H.CPU_S} s CPU "
                   f"time, 2 GiB address space, 64 processes and 256 MiB per file)")
        else:
            why = f"exit status {r['rc']}"
        tail = f" (stderr ends: {r['stderr_tail'][-200:]})" if r["stderr_tail"] else ""
        bad.append(f"{r['name']} ({r['describe']}): {why}{tail}")
    assert not bad, f"{len(bad)} of {len(graded)} invocations did not complete - " + _fmt(bad)


def test_output_format(graded):
    """For every completed invocation, stdout is UTF-8 text of exactly the header
    signal,t_merger,chirp_mass and one row: signal 0 with two empty fields, or signal 1
    with t_merger and chirp_mass written as plain decimal numbers (no sign or exponent).
    Lines may end with \n or \r\n."""
    bad = [f"{r['name']}: {r['format_problem']}" for r in graded if r["completed"] and r["format_problem"]]
    assert not bad, f"{len(bad)} invocations printed output that breaks the format - " + _fmt(bad)


def test_classification(graded):
    """Every held-out segment is classified correctly (signal 1 exactly when it contains
    an injected inspiral). Segments whose invocation failed or printed a malformed row
    count as not classified."""
    bad = []
    for r in graded:
        if r["class_ok"]:
            continue
        got = "no valid row" if r["signal"] is None else f"signal {r['signal']}"
        want = "contains a signal" if r["true_signal"] else "contains no signal"
        bad.append(f"{r['name']} {want} ({r['describe']}), program gave {got}")
    assert not bad, f"{len(bad)} of {len(graded)} segments misclassified - " + _fmt(bad)


def test_merger_times(graded):
    """For every held-out segment that contains a signal, t_merger is within 25 ms of the
    true H1 coalescence time."""
    bad = []
    for r in graded:
        if r["time_ok"]:
            continue
        if r["dt_s"] is None:
            bad.append(f"{r['name']} contains a signal ({r['describe']}) but no t_merger was reported")
        else:
            bad.append(f"{r['name']} ({r['describe']}): t_merger {r['t']:.4f} s is off by "
                       f"{r['dt_s'] * 1000:.1f} ms (allowed 25 ms)")
    assert not bad, f"{len(bad)} merger times outside the requirement - " + _fmt(bad)


def test_chirp_masses(graded):
    """For every held-out segment that contains a signal, chirp_mass is within 5 % of the
    true detector-frame chirp mass."""
    bad = []
    for r in graded:
        if r["mass_ok"]:
            continue
        if r["dmc_rel"] is None:
            bad.append(f"{r['name']} contains a signal ({r['describe']}) but no chirp_mass was reported")
        else:
            bad.append(f"{r['name']} ({r['describe']}): chirp_mass {r['mc']:.4f} is off by "
                       f"{r['dmc_rel'] * 100:.2f} % (allowed 5 %)")
    assert not bad, f"{len(bad)} chirp masses outside the requirement - " + _fmt(bad)
