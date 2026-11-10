FROM python:3.15-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1
WORKDIR /app

# Dependencies first, in their own layer: rebuilding after a code change doesn't reinstall them
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --extra otel --no-install-project

COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra otel

ENV PATH="/app/.venv/bin:$PATH"
USER nobody
CMD ["frigate-decisions", "--config", "/config/config.yml"]
