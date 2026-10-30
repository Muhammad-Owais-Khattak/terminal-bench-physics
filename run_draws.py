"""Run a program through the real harness (sandbox, limits, scoring) on held-out draws.

usage (inside the verifier image, as root):
    python3 /tests/evidence/run_draws.py <program.py> [seed ...]
Without seeds it uses the program's own draw (the one the verifier uses for these bytes).
Prints, for every draw, the number of segments that miss and each miss with its size.
"""
import sys

sys.path.insert(0, "/tests")
sys.path.insert(0, __file__.rsplit("/", 2)[0])

import harness as H  # noqa: E402


def main():
    prog = sys.argv[1]
    data = open(prog, "rb").read()
    seeds = [int(s) for s in sys.argv[2:]] or [None]
    for s in seeds:
        seed, rep = H.evaluate(data, seed=s)
        sig = [r for r in rep if r["true_signal"]]
        dts = [r["dt_s"] for r in sig if r["dt_s"] is not None and r["class_ok"]]
        dms = [r["dmc_rel"] for r in sig if r["dmc_rel"] is not None and r["class_ok"]]
        bad = [r for r in rep if not r["ok"]]
        print(f"{prog} seed {seed}: {len(bad)} of {len(rep)} segments miss; "
              f"misclassified {sum(not r['class_ok'] for r in rep)}, "
              f"time > 25 ms {sum(not r['time_ok'] for r in rep if r['class_ok'])}, "
              f"mass > 5% {sum(not r['mass_ok'] for r in rep if r['class_ok'])}; "
              f"worst correct-class dt {max(dts) * 1000 if dts else float('nan'):.1f} ms, "
              f"dMc {max(dms) * 100 if dms else float('nan'):.2f} %; "
              f"slowest invocation {max(r['elapsed_s'] for r in rep):.1f} s", flush=True)
        for r in bad:
            print(f"    {r['name']} [{r['describe']}] rc={r['rc']} signal={r['signal']} "
                  f"t={r['t']} mc={r['mc']} true=({r['true_t']}, {r['true_mc']}) "
                  f"{r['format_problem'] or ''}", flush=True)


if __name__ == "__main__":
    main()
