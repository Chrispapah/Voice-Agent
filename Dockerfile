FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY templates/ templates/
ENV PYTHONPATH=/app/src

EXPOSE 3000

CMD ["sh", "-c", "uvicorn ai_sdr_agent.app:create_app --host 0.0.0.0 --port ${PORT:-3000} --factory"]
