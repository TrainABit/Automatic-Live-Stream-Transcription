# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LST_STT_MODELS_DIR=/models

# ffmpeg decodes every source; ca-certificates lets the HTTPS clients verify peers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# An unprivileged user owns the working directory and the model cache.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin lst \
    && mkdir -p /data /models \
    && chown lst:lst /data /models

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[local]"

USER lst
WORKDIR /data
VOLUME ["/data", "/models"]

ENTRYPOINT ["lst"]
CMD ["--help"]
