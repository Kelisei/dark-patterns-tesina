# Builder stage
FROM python:3.12.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
# Install dependencies first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application and install
COPY . .
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.12.11-slim

ENV USER=api
ENV HOME=/app
WORKDIR ${HOME}

# Copy only the application and the virtual environment from builder
COPY --from=builder /app /app

# Expose the virtual environment to the system PATH
ENV PATH="/app/.venv/bin:$PATH"

# Create a non-root user
RUN addgroup --system ${USER} --gid 1000 && adduser -u 1000 --gid 1000 ${USER} 
RUN chown -R ${USER}:${USER} /app
USER ${USER} 

EXPOSE 5000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]