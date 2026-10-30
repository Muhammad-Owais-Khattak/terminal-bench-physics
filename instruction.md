Write a program that decides whether a two-detector gravitational-wave strain segment contains a real compact binary inspiral and, if it does, measures the signal's coalescence time and chirp mass.

Twenty-four segments are in `/root/data/` for development, from `segment_00.npz` to `segment_23.npz`; no true values are provided for them. Each file has three arrays: `H1` and `L1` (float64 strain of the two LIGO detectors, as recorded) and `sample_rate` (2048.0 Hz). Every segment is 16 s long and time is measured in seconds from the first sample of the segment.

A segment contains at most one real signal. All real signals are non-spinning inspirals generated with the TaylorF2 model (3.5PN phase, Newtonian amplitude), starting at 30 Hz and ending at the ISCO frequency of the total mass. Masses are detector-frame. A real signal is present in both detectors, with its own amplitude and phase in each and an arrival-time difference between L1 and H1 consistent with the light-travel time between the two LIGO sites. As in real detector data, the noise level is not exactly constant, and the strain also contains short non-Gaussian instrumental transients (glitches), in one detector or in both; glitches are not signals.

The deliverable is `/root/solver/search.py`, a regular file (not a symbolic link) of at most 1,000,000 bytes, invoked as `python3 /root/solver/search.py <segment_path>` with one absolute path to one segment file (same three arrays) per invocation. It must exit with status 0 and print exactly two lines of UTF-8 text to stdout, the header

```
signal,t_merger,chirp_mass
```

followed by one row for the segment:

1. `signal`: `1` if the segment contains a real inspiral signal as described above, otherwise `0`
2. `t_merger`: coalescence time of the signal at the H1 detector (the tc of the TaylorF2 phase), in seconds from the segment start, as a plain decimal number
3. `chirp_mass`: detector-frame chirp mass in solar masses, as a plain decimal number

If `signal` is `0`, leave `t_merger` and `chirp_mass` empty (the row is `0,,`). Fields are not quoted and contain no spaces, and numbers are plain decimals such as `11.5034` (no sign or exponent). Lines may end with `\n` or `\r\n`. Nothing else may be written to stdout; stderr is ignored.

The program must handle one invocation for each of 72 further segments generated from the same population as the development segments: same detectors, noise, sample rate and length, and the same kinds, proportions and parameter ranges of signals and glitches. The draw is fixed by the bytes of the program file, so the same file always receives the same segments. Each segment is handed over under a random file name ending in `.npz` in a read-only directory. For every one of the 72 segments, the invocation must complete within the limits below with exit status 0 and the stdout described above, `signal` must be correct, and if the segment contains a signal, `t_merger` must be within 25 ms of the true value (absolute difference) and `chirp_mass` within 5% of the true value (relative difference); if any segment misses even one of these, the program does not satisfy the requirements.

The invocations happen in a separate container, not this one:

- Python 3.11 at `/usr/local/bin/python3` with `numpy==1.26.4`, `scipy==1.13.1` and the standard library. Anything you install here is absent there.
- `/root/solver/search.py` is the only file carried over, as a read-only copy. `/root/data/` is not present, and the checker's own files are not readable.
- Each invocation is a fresh process run as an unprivileged user with no network access, empty stdin, and an empty writable temporary directory as its working directory. Its environment holds exactly `PATH=/usr/local/bin:/usr/bin:/bin`, `HOME` and `TMPDIR` set to that working directory, `LANG=C.UTF-8`, `PYTHONHASHSEED=0`, `PYTHONDONTWRITEBYTECODE=1`, and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to `1`.
- Limits per invocation: 45 s wall time, 45 s CPU time, 2 GiB address space, 64 processes, 256 MiB per written file, no core dumps, and 1 MiB of stdout.

Your analysis must fit within 1 CPU and 2 GB of memory.

You have 18000 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task.
