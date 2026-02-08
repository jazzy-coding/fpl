# Stage 1: Build/Dependencies
FROM python:3.12-slim-bookworm AS builder

# Install system tools needed for building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies into a virtualenv
# We use --no-dev to keep the image lean
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --no-dev

# Stage 2: Runtime
FROM python:3.12-slim-bookworm

# We STILL need gcc/g++ in the final image because PyTensor 
# compiles code AT RUNTIME when you call pm.sample()
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment and code from builder
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Ensure the virtualenv is used
ENV PATH="/app/.venv/bin:$PATH"
# PyTensor configuration for Cloud Run
ENV PYTENSOR_FLAGS="device=cpu,force_device=True,compile.timeout=120"

# Run the module
CMD ["python", "-m", "src.main"]