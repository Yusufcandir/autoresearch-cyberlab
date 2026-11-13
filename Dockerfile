# AutoResearch-CyberLab application image (API, dashboard and worker).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv==0.13.7 \
 && useradd --create-home --uid 10001 cyberlab

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY packages packages
COPY services services
COPY plugins plugins
RUN uv sync --locked --no-dev && chown -R cyberlab /app

USER cyberlab
EXPOSE 8100
CMD ["cyberlab", "serve", "--host", "0.0.0.0", "--port", "8100"]
