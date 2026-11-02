"""Shared physics for the data generator (author side only, never shipped to the agent)."""
import numpy as np

MSUN_S = 4.925490947e-6      # G*Msun/c^3 in seconds
EULER_GAMMA = 0.5772156649015329

FS = 2048.0                  # sample rate [Hz]
T_SEG = 16.0                 # segment length [s]
N = int(FS * T_SEG)
F_LOW = 30.0                 # signal starts at 30 Hz
FREQS = np.fft.rfftfreq(N, 1.0 / FS)
DF = FREQS[1] - FREQS[0]


def aligo_psd(f):
    """Analytic fit to the Advanced LIGO zero-detuned high-power design PSD
    (Ajith 2011, PRD 84, 084037, eq. 4.7). One-sided, strain^2/Hz."""
    f = np.asarray(f, dtype=float)
    fc = np.clip(f, 10.0, None)            # flatten below 10 Hz to keep noise finite
    x = fc / 245.4
    s = 1e-48 * (0.0152 * x**-4 + 0.2935 * x**(9.0 / 4) + 2.7951 * x**1.5
                 - 6.5080 * x**0.75 + 17.7622)
    return s


def colored_noise(rng, psd):
    """Gaussian noise with one-sided PSD `psd` sampled on FREQS."""
    white = rng.standard_normal(N)
    wf = np.fft.rfft(white)
    wf *= np.sqrt(psd * FS / 2.0)
    return np.fft.irfft(wf, n=N)


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def taylorf2(f, m1, m2, tc, phic):
    """Non-spinning TaylorF2: 3.5PN phase, Newtonian amplitude (arbitrary scale).
    Returns complex FD strain on f (zero outside [F_LOW, f_isco])."""
    M = (m1 + m2) * MSUN_S
    eta = m1 * m2 / (m1 + m2) ** 2
    f_isco = 1.0 / (6.0 ** 1.5 * np.pi * M)
    h = np.zeros_like(f, dtype=complex)
    band = (f >= F_LOW) & (f <= f_isco)
    fb = f[band]
    v = (np.pi * M * fb) ** (1.0 / 3.0)
    pi = np.pi
    p0 = 1.0
    p2 = 20.0 / 9.0 * (743.0 / 336.0 + 11.0 / 4.0 * eta)
    p3 = -16.0 * pi
    p4 = 10.0 * (3058673.0 / 1016064.0 + 5429.0 / 1008.0 * eta + 617.0 / 144.0 * eta**2)
    p5 = pi * (38645.0 / 756.0 - 65.0 / 9.0 * eta) * (1.0 + 3.0 * np.log(v * np.sqrt(6.0)))
    p6 = (11583231236531.0 / 4694215680.0 - 640.0 / 3.0 * pi**2 - 6848.0 / 21.0 * EULER_GAMMA
          + eta * (-15737765635.0 / 3048192.0 + 2255.0 / 12.0 * pi**2)
          + 76055.0 / 1728.0 * eta**2 - 127825.0 / 1296.0 * eta**3
          - 6848.0 / 21.0 * np.log(4.0 * v))
    p7 = pi * (77096675.0 / 254016.0 + 378515.0 / 1512.0 * eta - 74045.0 / 756.0 * eta**2)
    psi = 3.0 / (128.0 * eta * v**5) * (p0 + p2 * v**2 + p3 * v**3 + p4 * v**4
                                         + p5 * v**5 + p6 * v**6 + p7 * v**7)
    phase = 2 * pi * fb * tc - phic - pi / 4 + psi
    amp = fb ** (-7.0 / 6.0)
    # short cosine tapers at both band edges to suppress ringing
    lo = np.clip((fb - F_LOW) / 2.0, 0, 1)
    hi = np.clip((f_isco - fb) / 5.0, 0, 1)
    taper = (0.5 - 0.5 * np.cos(np.pi * lo)) * (0.5 - 0.5 * np.cos(np.pi * hi))
    h[band] = amp * taper * np.exp(-1j * phase)
    return h


def sine_gaussian_td(t0, f0, Q, phi):
    t = np.arange(N) / FS
    tau = Q / (2 * np.pi * f0)
    return np.exp(-((t - t0) ** 2) / (2 * tau**2)) * np.sin(2 * np.pi * f0 * (t - t0) + phi)


def opt_snr_fd(hf, psd):
    band = FREQS >= 20.0
    return np.sqrt(4.0 * DF * np.sum(np.abs(hf[band]) ** 2 / psd[band]))


def fd_from_td(x):
    return np.fft.rfft(x) / FS          # continuous-FT normalisation


def td_from_fd(hf):
    return np.fft.irfft(hf * FS, n=N)
