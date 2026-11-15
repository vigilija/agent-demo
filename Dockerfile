# Small image, non-root user, secrets passed in at runtime (never baked into the image).
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Build the database at start-up (needs the API key), then start the API.
CMD ["sh", "-c", "python ingest.py && uvicorn app.api:app --host 0.0.0.0 --port 8000"]
