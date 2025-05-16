# Dockerfile
FROM python:3.10-slim

RUN pip install --no-cache-dir fastapi uvicorn[standard] transformers torch mlflow evidently

WORKDIR /app
COPY app.py /app/

ENV PORT=8080
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

