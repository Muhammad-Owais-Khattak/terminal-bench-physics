"""Held-out draw, sandboxed invocation of the submitted program, output parsing and scoring.

The held-out segments are simulated here, at check time, by gwfamily.draw_set() from the
same forward model as the development segments. Their truth is known by construction and
never reaches the program: only the segment files are handed over.
"""
import hashlib
import os
import pwd
import re
import resource
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path

import gwfamily as gf

PROGRAM = Path("/root/solver/search.py")
RUNNER = "runner"
HELD_OUT_COUNT = 72                      # three blocks of the 24-segment mix
SALT = b"turing/gw-glitch-search/held-out/v1"
WALL_S = 45
CPU_S = 45
MEM_BYTES = 2 * 1024 ** 3
NPROC = 64
FSIZE = 256 * 1024 ** 2
STDOUT_CAP = 1024 * 1024
PROGRAM_CAP = 1_000_000
TIME_TOL_S = 0.025                       # |t - t_true| <= 25 ms
MC_REL_TOL = 0.05                        # |Mc / Mc_true - 1| <= 5 %
HEADER = "signal,t_merger,chirp_mass"
PLAIN = re.compile(r"[0-9]+(\.[0-9]+)?")  # used with fullmatch


def seed_for(program_bytes):
    return int.from_bytes(hashlib.sha256(SALT + program_bytes).digest()[:8], "big")


def read_program(path=PROGRAM):
    """Bytes of the submitted program. It must be a regular file (not a symlink) of at
    most PROGRAM_CAP bytes; it is opened without following links and without blocking."""
    try:
        st = os.lstat(path)
    except OSError:
        raise AssertionError(f"missing program file {path}") from None
    if not stat.S_ISREG(st.st_mode):
        raise AssertionError(f"{path} must be a regular file (not a symbolic link, directory or FIFO)")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise AssertionError(f"{path} must be a regular file")
        data = b""
        while len(data) <= PROGRAM_CAP:
            chunk = os.read(fd, PROGRAM_CAP + 1 - len(data))
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    if len(data) == 0:
        raise AssertionError(f"{path} is empty")
    if len(data) > PROGRAM_CAP:
        raise AssertionError(f"{path} is larger than {PROGRAM_CAP} bytes")
    return data


# ------------------------------------------------------------------ sandbox
def _limits(uid, gid):
    def apply():
        os.setsid()
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
        resource.setrlimit(resource.RLIMIT_NPROC, (NPROC, NPROC))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FSIZE, FSIZE))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.setgroups([])
        os.setgid(gid)
        os.setuid(uid)
    return apply


def install_program(program_bytes):
    """Root-owned, read-only copy of the program that the runner account can read."""
    d = Path(tempfile.mkdtemp(prefix="prog-", dir="/tmp"))
    p = d / "search.py"
    p.write_bytes(program_bytes)
    os.chmod(p, 0o444)
    os.chmod(d, 0o555)
    return p


def stage_inputs(items):
    """Write the held-out segments under their opaque names into a read-only directory."""
    d = Path(tempfile.mkdtemp(prefix="in-", dir="/tmp"))
    for it in items:
        gf.save_segment(d / it["name"], it)
        os.chmod(d / it["name"], 0o444)
    os.chmod(d, 0o555)
    return d


def run_sandboxed(program, input_path):
    """Invoke `python3 program input_path` as the unprivileged runner account."""
    pw = pwd.getpwnam(RUNNER)
    work = Path(tempfile.mkdtemp(prefix="run-", dir="/tmp"))
    os.chown(work, pw.pw_uid, pw.pw_gid)
    os.chmod(work, 0o700)
    out_f = tempfile.TemporaryFile()
    err_f = tempfile.TemporaryFile()
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work),
           "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    python = shutil.which("python3") or "/usr/local/bin/python3"
    t0 = time.monotonic()
    proc = subprocess.Popen([python, str(program), str(input_path)], cwd=work, env=env,
                            stdin=subprocess.DEVNULL, stdout=out_f, stderr=err_f,
                            preexec_fn=_limits(pw.pw_uid, pw.pw_gid), close_fds=True)
    rc, timed_out = None, False
    try:
        rc = proc.wait(timeout=WALL_S)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if rc is None:
            proc.wait()
    elapsed = time.monotonic() - t0
    out_f.seek(0)
    raw = out_f.read(STDOUT_CAP + 1)
    err_f.seek(0, os.SEEK_END)
    err_f.seek(max(0, err_f.tell() - 400))
    err_tail = err_f.read().decode("utf-8", "replace").strip().replace("\n", " | ")
    out_f.close()
    err_f.close()
    shutil.rmtree(work, ignore_errors=True)
    return dict(rc=rc, timed_out=timed_out, elapsed_s=round(elapsed, 2),
                stdout=raw[:STDOUT_CAP], too_big=len(raw) > STDOUT_CAP, stderr_tail=err_tail)


# ---------------------------------------------------------------- scoring
def parse(raw):
    """(row, problem): row is (signal, t_merger, chirp_mass) with floats or None."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "stdout is not valid UTF-8"
    # a line ends with \n or \r\n; text after the last \n is a final line without a
    # line ending and is kept unchanged (so a lone trailing \r stays in its field)
    parts = text.split("\n")
    tail = parts.pop()
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in parts]
    if tail != "":
        lines.append(tail)
    if len(lines) != 2:
        return None, f"stdout must be exactly 2 lines (header and one row), got {len(lines)}: {text[:200]!r}"
    if lines[0] != HEADER:
        return None, f"first line must be exactly {HEADER!r}, got {lines[0][:100]!r}"
    f = lines[1].split(",")
    if len(f) != 3:
        return None, f"row must have 3 comma-separated fields, got {lines[1][:100]!r}"
    sig, t, mc = f
    if sig == "0":
        if t != "" or mc != "":
            return None, f"signal 0 needs empty t_merger and chirp_mass, got {lines[1][:100]!r}"
        return (0, None, None), None
    if sig != "1":
        return None, f"signal must be 0 or 1, got {sig[:20]!r}"
    for name, v in (("t_merger", t), ("chirp_mass", mc)):
        if not PLAIN.fullmatch(v):
            return None, f"{name} must be a plain decimal number (digits, optional decimal point and digits), got {v[:30]!r}"
    return (1, float(t), float(mc)), None


def score(items, runs):
    """One verdict per held-out segment, computed only from what the program printed."""
    report = []
    for it, run in zip(items, runs):
        tr = it["truth"]
        completed = run["rc"] == 0 and not run["timed_out"] and not run["too_big"]
        row, problem = parse(run["stdout"]) if completed else (None, None)
        r = dict(name=it["name"], kind=tr["kind"], describe=gf.describe(tr),
                 true_signal=tr["signal"], true_t=tr["t_merger"], true_mc=tr["chirp_mass"],
                 rc=run["rc"], timed_out=run["timed_out"], too_big=run["too_big"],
                 elapsed_s=run["elapsed_s"], stderr_tail=run["stderr_tail"],
                 completed=completed, format_problem=problem,
                 signal=None if row is None else row[0],
                 t=None if row is None else row[1], mc=None if row is None else row[2])
        r["class_ok"] = r["signal"] == tr["signal"]
        r["dt_s"] = abs(r["t"] - tr["t_merger"]) if (tr["signal"] and r["t"] is not None) else None
        r["dmc_rel"] = abs(r["mc"] / tr["chirp_mass"] - 1.0) if (tr["signal"] and r["mc"] is not None) else None
        r["time_ok"] = (not tr["signal"]) or (r["dt_s"] is not None and r["dt_s"] <= TIME_TOL_S)
        r["mass_ok"] = (not tr["signal"]) or (r["dmc_rel"] is not None and r["dmc_rel"] <= MC_REL_TOL)
        r["ok"] = completed and problem is None and r["class_ok"] and r["time_ok"] and r["mass_ok"]
        report.append(r)
    return report


def evaluate(program_bytes, seed=None, n=HELD_OUT_COUNT):
    """Draw the held-out set for these program bytes (or an explicit seed), invoke the
    program once per segment in the sandbox and score every invocation."""
    seed = seed_for(program_bytes) if seed is None else seed
    items = gf.draw_set(seed, n)
    prog = install_program(program_bytes)
    in_dir = stage_inputs(items)
    try:
        runs = [run_sandboxed(prog, in_dir / it["name"]) for it in items]
    finally:
        shutil.rmtree(in_dir, ignore_errors=True)
        shutil.rmtree(prog.parent, ignore_errors=True)
    return seed, score(items, runs)
