# Same base and the same numpy/scipy pins as environment/Dockerfile: the submitted program
# runs here, so its runtime matches the agent's container.
FROM python:3.11.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends bash && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy==1.26.4 scipy==1.13.1 pytest==8.4.1 pytest-json-ctrf==0.3.5

# Unprivileged account that executes the submitted program
RUN useradd --system --no-create-home --shell /usr/sbin/nologin runner

# Parent of the declared artifact and the log directory
RUN mkdir -p /root/solver /logs/verifier

COPY . /tests/

# The forward model, the harness and the tests are readable by root only
RUN chown -R root:root /tests && chmod -R go-rwx /tests && chmod 700 /logs/verifier
