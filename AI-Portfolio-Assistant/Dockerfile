FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# LLM_BASE_URL / LLM_API_KEY / LLM_MODEL are injected at runtime (optional).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
