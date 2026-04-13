FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY templates/ templates/
ENV PYTHONPATH=/app/src

EXPOSE 3000

CMD ["uvicorn", "ai_sdr_agent.app:create_app", "--host", "0.0.0.0", "--port", "3000", "--factory"]
