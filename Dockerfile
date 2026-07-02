# Q-GEOCompress Valohai runtime image
# Build: docker build -t qgeocompress:latest .
# Smoke: docker run --rm qgeocompress:latest pytest -q
# Push to your registry and set `image: your-registry/qgeocompress:latest` in valohai.yaml

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY datasets/dota128 ./datasets/dota128

RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONUNBUFFERED=1
ENV VALOHAI_OUTPUTS_DIR=/valohai/outputs

HEALTHCHECK CMD pytest -q || exit 1

CMD ["pytest", "-q"]
