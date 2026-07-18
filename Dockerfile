# Slim multi-stage image for the Musaeus API. Keys come from the environment at
# runtime (never baked into a layer). Build: docker build -t musaeus .
FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[server]"

FROM python:3.12-slim
WORKDIR /app
COPY --from=build /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=build /usr/local/bin /usr/local/bin
COPY --from=build /app/src ./src
EXPOSE 8000
# Serve the streaming API. Provide keys with `docker run -e ANTHROPIC_API_KEY=...`.
CMD ["uvicorn", "musaeus.server:app", "--host", "0.0.0.0", "--port", "8000"]
