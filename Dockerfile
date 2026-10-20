FROM python:3.12-slim

WORKDIR /app

# xgboost needs libgomp on linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY static ./static

# model files aren't in git so train during the build
RUN mkdir -p artifacts && python src/train.py

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
