"""Reference program: glitch-robust two-detector matched-filter search.

usage: python3 search.py <segment.npz>
Prints the CSV header signal,t_merger,chirp_mass and one row for the segment.
"""
import sys
import numpy as np
from scipy.signal import welch
from scipy.ndimage import maximum_filter1d, label
from scipy.optimize import minimize

MSUN_S = 4.925490947e-6
GAMMA_E = 0.5772156649015329
F_LOW, F_HIGH = 30.0, 1000.0
MAX_DELAY = 0.010
EDGE = 1.0                # ignore first/last second (filter wrap-around)
GATE_Z = 6.0              # whitened-amplitude gating threshold
GATE_HALF = 0.25          # gate half-width [s]
DET_THR = 8.5              # network re-weighted SNR threshold
N_CHI_BINS = 16
BIN = 256                  # candidate time bins of 0.125 s
N_CAND = 10                # loudest bins re-weighted with chi-squared
CAND_MIN = 6.0             # network SNR below this is never a candidate


# ---------------------------------------------------------------- waveform
def taylorf2(f, mc, eta):
    """Non-spinning TaylorF2 (3.5PN phase, Newtonian amplitude), tc = 0, phic = 0."""
    M = mc / eta ** 0.6 * MSUN_S
    f_isco = 1.0 / (6.0 ** 1.5 * np.pi * M)
    h = np.zeros_like(f, dtype=complex)
    band = (f >= F_LOW) & (f <= f_isco)
    fb = f[band]
    v = (np.pi * M * fb) ** (1.0 / 3.0)
    pi = np.pi
    p2 = 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta)
    p3 = -16.0 * pi
    p4 = 10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta ** 2)
    p5 = pi * (38645.0 / 756.0 - 65.0 / 9.0 * eta) * (1.0 + 3.0 * np.log(v * np.sqrt(6.0)))
    p6 = (11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * pi ** 2 - 6848.0 / 21.0 * GAMMA_E
          + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * pi ** 2)
          + 76055.0 / 1728.0 * eta ** 2 - 127825.0 / 1296.0 * eta ** 3
          - 6848.0 / 21.0 * np.log(4.0 * v))
    p7 = pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta ** 2)
    psi = 3.0 / (128.0 * eta * v ** 5) * (1 + p2 * v ** 2 + p3 * v ** 3 + p4 * v ** 4
                                           + p5 * v ** 5 + p6 * v ** 6 + p7 * v ** 7)
    lo = np.clip((fb - F_LOW) / 2.0, 0, 1)
    hi = np.clip((f_isco - fb) / 5.0, 0, 1)
    taper = (0.5 - 0.5 * np.cos(np.pi * lo)) * (0.5 - 0.5 * np.cos(np.pi * hi))
    h[band] = fb ** (-7.0 / 6.0) * taper * np.exp(-1j * (-pi / 4 + psi))
    return h


# ---------------------------------------------------------------- data conditioning
class Detector:
    def __init__(self, x, fs):
        self.fs, self.n = fs, len(x)
        self.f = np.fft.rfftfreq(self.n, 1 / fs)
        self.df = self.f[1] - self.f[0]
        self.band = (self.f >= F_LOW) & (self.f <= F_HIGH)
        self.gated = np.zeros(self.n, bool)
        x = np.asarray(x, float)
        # 1) remove the slow drift of the noise level (non-stationary noise)
        self.psd = self._psd(x)
        x = x / self._envelope(self._whiten(x))
        # 2) gate loud transients tightly, then re-estimate the PSD
        self.psd = self._psd(x)
        x = self._gate(x)
        self.psd = self._psd(x)
        self.dtilde = np.fft.rfft(x) / fs

    def _psd(self, x):
        f, p = welch(x, fs=self.fs, nperseg=int(4 * self.fs), average="median")
        return np.interp(self.f, f, p)

    def _whiten(self, x):
        wf = np.fft.rfft(x) / np.sqrt(self.psd)
        wf[(self.f < 25) | (self.f > 900)] = 0
        w = np.fft.irfft(wf, n=self.n)
        return w / (np.median(np.abs(w)) / 0.6745)

    def _envelope(self, w):
        """Slowly varying noise amplitude from a running robust std of the whitened data."""
        step = int(0.25 * self.fs)
        win = int(1.0 * self.fs)
        centers = np.arange(win // 2, self.n - win // 2, step)
        sig = np.array([np.median(np.abs(w[c - win // 2:c + win // 2])) / 0.6745 for c in centers])
        k = 9                                                  # ~2 s running median: ignores short glitches
        pad = np.pad(sig, k // 2, mode="edge")
        sig = np.array([np.median(pad[i:i + k]) for i in range(len(sig))])
        env = np.interp(np.arange(self.n), centers, sig)
        return env / np.median(env)

    def _gate(self, x):
        """Zero out loud transients with a tight, data-driven gate (keeps nearby signal)."""
        w = self._whiten(x)
        a = np.abs(w)
        loud = a > GATE_Z
        if not loud.any():
            return x
        join = int(0.02 * self.fs)
        mask3 = maximum_filter1d((a > 3.0).astype(np.uint8), size=2 * join + 1).astype(bool)
        lab, nlab = label(mask3)
        keep = np.unique(lab[loud])
        core = np.isin(lab, keep[keep > 0])
        pad = int(0.03 * self.fs)
        core = maximum_filter1d(core.astype(np.uint8), size=2 * pad + 1).astype(float)
        taper = int(0.03 * self.fs)
        smooth = np.convolve(core, np.ones(taper) / taper, mode="same")
        win = 1.0 - np.clip(smooth, 0, 1)
        self.gated = win < 0.999
        return x * win

    def snr_series(self, h):
        """Complex matched-filter SNR time series and template norm."""
        w = np.zeros(self.n, complex)
        q = np.zeros_like(h)
        q[self.band] = self.dtilde[self.band] * np.conj(h[self.band]) / self.psd[self.band]
        w[:len(q)] = q
        z = 4 * self.df * self.n * np.fft.ifft(w)
        sigma = np.sqrt(4 * self.df * np.sum(np.abs(h[self.band]) ** 2 / self.psd[self.band]))
        return z / sigma, sigma

    def chisq(self, h, idx):
        """Allen chi-squared with N_CHI_BINS equal-power frequency bins at sample idx."""
        hb = np.where(self.band, h, 0)
        p = np.abs(hb) ** 2 / self.psd
        cum = np.cumsum(p) / p.sum()
        edges = np.searchsorted(cum, np.linspace(0, 1, N_CHI_BINS + 1)[1:-1])
        edges = np.concatenate([[0], edges, [len(h)]])
        sigma = np.sqrt(4 * self.df * p.sum())
        t = idx / self.fs
        ph = np.exp(2j * np.pi * self.f * t)
        integrand = 4 * self.df * self.dtilde * np.conj(hb) / self.psd * ph / sigma
        zs = np.array([integrand[a:b].sum() for a, b in zip(edges[:-1], edges[1:])])
        z = zs.sum()
        return N_CHI_BINS * np.sum(np.abs(zs - z / N_CHI_BINS) ** 2)


def newsnr(rho, chisq):
    r = chisq / (2 * N_CHI_BINS - 2)
    return rho if r <= 1 else rho / ((1 + r ** 3) / 2) ** (1 / 6)


# ---------------------------------------------------------------- search
def network_stat(dets, h, valid):
    zH, _ = dets[0].snr_series(h)
    zL, _ = dets[1].snr_series(h)
    lag = int(round(MAX_DELAY * dets[0].fs))
    aL = maximum_filter1d(np.abs(zL), size=2 * lag + 1)
    net = np.sqrt(np.abs(zH) ** 2 + aL ** 2)
    net[~valid] = 0
    return net, zH, zL


def analyse(path):
    d = np.load(path)
    fs = float(d["sample_rate"])
    dets = [Detector(d["H1"], fs), Detector(d["L1"], fs)]
    n = dets[0].n
    valid = np.ones(n, bool)
    valid[:int(EDGE * fs)] = valid[-int(EDGE * fs):] = False
    lag = int(round(MAX_DELAY * fs))
    # a candidate time is unusable if either detector is gated near it
    for det in dets:
        valid &= ~maximum_filter1d(det.gated.astype(np.uint8), size=2 * lag + 1).astype(bool)
    f = dets[0].f

    # loudest network SNR in every 0.125 s time bin, over the whole bank: a glitch that
    # is louder than a nearby signal then cannot hide the signal's own trigger
    nb = n // BIN
    bin_val = np.zeros(nb)
    bin_arg = [None] * nb
    for mc in np.exp(np.arange(np.log(6.0), np.log(18.0), 0.004)):
        for eta in (0.17, 0.20, 0.23, 0.25):
            h = taylorf2(f, mc, eta)
            net, zH, zL = network_stat(dets, h, valid)
            nr = net[:nb * BIN].reshape(nb, BIN)
            vmax = nr.max(axis=1)
            for b in np.nonzero(vmax > bin_val)[0]:
                bin_val[b] = vmax[b]
                bin_arg[b] = (mc, eta, b * BIN + int(np.argmax(nr[b])))

    # re-weight the loudest candidates with chi-squared in each detector
    stats = {}
    for b in np.argsort(bin_val)[::-1][:N_CAND]:
        if bin_arg[b] is None or bin_val[b] < CAND_MIN:
            break
        mc, eta, i = bin_arg[b]
        h = taylorf2(f, mc, eta)
        zH, _ = dets[0].snr_series(h)
        zL, _ = dets[1].snr_series(h)
        j = i - lag + int(np.argmax(np.abs(zL[i - lag:i + lag + 1])))
        stats[b] = np.hypot(newsnr(abs(zH[i]), dets[0].chisq(h, i)),
                            newsnr(abs(zL[j]), dets[1].chisq(h, j)))
    if not stats or max(stats.values()) < DET_THR:
        return dict(signal=0)
    # the detection is the candidate with the largest re-weighted SNR; its parameters
    # are refined from the loudest signal-consistent trigger in the same or a
    # neighbouring time bin (the re-weighted maximum can sit on a side lobe)
    b0 = max(stats, key=stats.get)
    near = [b for b in stats if abs(b - b0) <= 1 and stats[b] >= 0.9 * stats[b0]]
    mc, eta, i = bin_arg[max(near, key=lambda b: bin_val[b])]

    # refine (mc, eta) by maximising the network SNR near the trigger
    win = np.zeros(n, bool)
    win[max(i - int(0.05 * fs), 0):i + int(0.05 * fs)] = True
    win &= valid

    def neg(p):
        m, e = p
        if not (4.0 < m < 25.0 and 0.10 < e <= 0.25):
            return 0.0
        net, _, _ = network_stat(dets, taylorf2(f, m, e), win)
        return -net.max()

    res = minimize(neg, [mc, eta], method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-4, initial_simplex=[[mc, eta], [mc * 1.004, eta], [mc, eta - 0.02]]))
    mc, eta = res.x
    net, _, _ = network_stat(dets, taylorf2(f, mc, eta), win)
    i = int(np.argmax(net))
    return dict(signal=1, t_merger=i / fs, chirp_mass=mc)


def main():
    r = analyse(sys.argv[1])
    if r["signal"]:
        row = f"1,{r['t_merger']:.4f},{r['chirp_mass']:.4f}"
    else:
        row = "0,,"
    sys.stdout.write("signal,t_merger,chirp_mass\n" + row + "\n")


if __name__ == "__main__":
    main()
