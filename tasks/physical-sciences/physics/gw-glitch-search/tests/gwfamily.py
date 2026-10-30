"""Forward model of the two-detector segments (verifier side only, never shipped to the agent).

The development segments in environment/data and the held-out segments simulated at
check time come from the same function, make_segment(), with the same parameter ranges.

CLI (reproduces the development set bit for bit with seed 2027):
    python3 gwfamily.py <out_dir> <truth_csv> <n_segments> <seed>
"""
import csv
import hashlib
import os
import sys

import numpy as np

from gwphys import (FS, N, FREQS, aligo_psd, colored_noise, chirp_mass, taylorf2,
                    sine_gaussian_td, opt_snr_fd, fd_from_td, td_from_fd)

PSD = aligo_psd(FREQS)
MAX_LIGHT_DELAY = 0.010          # H1-L1 light travel time bound [s]
# composition of every block of 24 segments
MIX = (["signal"] * 5 + ["signal+glitch"] * 8 + ["glitch"] * 4
       + ["glitch-pair"] * 4 + ["noise"] * 3)


def draw_signal(rng):
    while True:
        m1, m2 = rng.uniform(6.0, 16.0, 2)
        m1, m2 = max(m1, m2), min(m1, m2)
        mc = chirp_mass(m1, m2)
        if 6.5 <= mc <= 12.0:
            return m1, m2, mc


def add_glitch(rng, x_by_det, det, t0, snr):
    f0 = rng.uniform(40.0, 250.0)
    Q = rng.uniform(3.0, 12.0)
    g = sine_gaussian_td(t0, f0, Q, rng.uniform(0, 2 * np.pi))
    g *= snr / opt_snr_fd(fd_from_td(g), PSD)
    x_by_det[det] += g
    return dict(det=det, t0=t0, f0=f0, Q=Q, snr=snr)


def make_segment(rng, kind):
    t = np.arange(N) / FS
    x = {}
    env = {}
    for det in ("H1", "L1"):
        # slow drift of the broadband noise level, as in a real interferometer
        # (sensitivity changes of a few per cent over minutes): relative
        # amplitude 3-7 %, period 180-600 s, random phase.
        A = rng.uniform(0.15, 0.35) / 5.0
        T0 = rng.uniform(6.0, 20.0) * 30.0
        ph = rng.uniform(0, 2 * np.pi)
        env[det] = lambda tt, A=A, T0=T0, ph=ph: 1.0 + A * np.sin(2 * np.pi * tt / T0 + ph)
        x[det] = colored_noise(rng, PSD) * env[det](t)
    truth = dict(kind=kind, signal=0, t_merger=None, chirp_mass=None, glitches=[])
    tc = None
    if kind in ("signal", "signal+glitch"):
        m1, m2, mc = draw_signal(rng)
        tc = rng.uniform(8.0, 13.0)
        dt = rng.uniform(-MAX_LIGHT_DELAY, MAX_LIGHT_DELAY)          # L1 - H1 delay
        net = rng.uniform(12.0, 18.0)
        ratio = rng.uniform(0.7, 1.3)                                 # L1/H1 amplitude
        hH = taylorf2(FREQS, m1, m2, tc, rng.uniform(0, 2 * np.pi))
        hL = taylorf2(FREQS, m1, m2, tc + dt, rng.uniform(0, 2 * np.pi))
        sH, sL = opt_snr_fd(hH, PSD), opt_snr_fd(hL, PSD)
        rhoH = net / np.sqrt(1 + ratio**2)
        # SNR is defined against the LOCAL noise level at the merger
        x["H1"] += td_from_fd(hH * rhoH / sH) * env["H1"](tc)
        x["L1"] += td_from_fd(hL * rhoH * ratio / sL) * env["L1"](tc + dt)
        truth.update(signal=1, t_merger=round(tc, 4), chirp_mass=round(mc, 4),
                     m1=m1, m2=m2, dt=dt, net_snr=net)
    if kind == "signal+glitch":
        # loud glitch shortly BEFORE the merger, inside the signal's in-band duration
        det = rng.choice(["H1", "L1"])
        t0 = tc - rng.uniform(0.25, 0.8)
        truth["glitches"].append(add_glitch(rng, x, det, t0, rng.uniform(25.0, 80.0)))
    if kind in ("glitch", "glitch-pair"):
        n_g = 1 if kind != "glitch-pair" else 2
        for _ in range(n_g):
            while True:
                t0 = rng.uniform(3.0, 14.0)
                if tc is None or abs(t0 - tc) > 1.0:
                    break
            det = rng.choice(["H1", "L1"])
            truth["glitches"].append(add_glitch(rng, x, det, t0, rng.uniform(15.0, 60.0)))
        if kind == "glitch-pair":
            # same glitch morphology in both detectors but with a non-physical delay
            g0 = truth["glitches"][0]
            other = "L1" if g0["det"] == "H1" else "H1"
            delay = rng.choice([-1, 1]) * rng.uniform(0.03, 0.2)
            g = sine_gaussian_td(g0["t0"] + delay, g0["f0"], g0["Q"], 0.0)
            g *= g0["snr"] / opt_snr_fd(fd_from_td(g), PSD)
            x[other] += g
    return x, truth


def kinds_for(rng, n_seg):
    kinds = (MIX * ((n_seg + len(MIX) - 1) // len(MIX)))[:n_seg]
    rng.shuffle(kinds)
    return kinds


def describe(truth):
    """Short plain-language description of a segment, used in failure messages."""
    k = truth["kind"]
    if k == "signal":
        return f"inspiral, network SNR {truth['net_snr']:.1f}"
    if k == "signal+glitch":
        g = truth["glitches"][0]
        return (f"inspiral, network SNR {truth['net_snr']:.1f}, with a SNR {g['snr']:.0f} glitch in "
                f"{g['det']} {truth['t_merger'] - g['t0']:.2f} s before the merger")
    if k == "glitch":
        return f"no signal; one glitch in {truth['glitches'][0]['det']}"
    if k == "glitch-pair":
        return "no signal; glitches in both detectors with a non-physical delay"
    return "no signal; noise only"


def draw_set(seed, n_seg):
    """n_seg segments for one seed: list of dicts with name, H1, L1 and truth.

    File names are opaque (derived from the seed and the index) so that nothing
    about a segment can be read from its name."""
    rng = np.random.default_rng(seed)
    out = []
    for i, kind in enumerate(kinds_for(rng, n_seg)):
        x, truth = make_segment(rng, kind)
        name = hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()[:16] + ".npz"
        out.append(dict(name=name, H1=x["H1"].astype(np.float64), L1=x["L1"].astype(np.float64),
                        truth=truth))
    return out


def save_segment(path, item):
    np.savez(path, H1=item["H1"], L1=item["L1"], sample_rate=np.float64(FS))


def main():
    out_dir, truth_csv, n_seg, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, it in enumerate(draw_set(seed, n_seg)):
        sid = f"segment_{i:02d}"
        save_segment(os.path.join(out_dir, sid + ".npz"), it)
        t = it["truth"]
        rows.append(dict(segment_id=sid, signal=t["signal"],
                         t_merger="" if t["t_merger"] is None else t["t_merger"],
                         chirp_mass="" if t["chirp_mass"] is None else t["chirp_mass"]))
    with open(truth_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
