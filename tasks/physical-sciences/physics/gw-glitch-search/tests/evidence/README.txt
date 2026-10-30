tests/evidence/ - index (verifier image only)

run_draws.py
    Runs a program through the real harness (same sandbox, limits and scoring as
    the verifier) on held-out draws and prints every miss with its size.
    Inside the verifier image, as root:
        python3 /tests/evidence/run_draws.py <program.py>            (its own draw)
        python3 /tests/evidence/run_draws.py <program.py> 101 102 103 (explicit seeds)
controls/standard_pipeline.py
    Negative control in program form: global median-Welch PSD, fixed +-0.25 s
    gates, Allen chi-squared on the loudest trigger. Misses real signals that sit
    next to a glitch.
controls/no_gating_no_chi2.py
    Negative control in program form: plain template-bank search with no glitch
    handling. Misclassifies glitch segments.

The forward model is tests/gwfamily.py (with tests/gwphys.py):
    python3 /tests/gwfamily.py <out_dir> <truth.csv> 24 2027
reproduces environment/data bit for bit and writes its injection table.

Example (laptop):
    docker build -t gw-tests tests/
    docker run --rm -v "$PWD/solution:/s:ro" gw-tests python3 /tests/evidence/run_draws.py /s/search.py
