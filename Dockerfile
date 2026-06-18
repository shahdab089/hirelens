# Application Autopsy — Hugging Face Spaces (Docker SDK) image.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; pypdf/python-docx are pure-python.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HF Spaces serves on port 7860. Write the SQLite db to a writable dir.
ENV APP_DB_PATH=/tmp/applications.db
EXPOSE 7860

CMD ["uvicorn", "app_web:app", "--host", "0.0.0.0", "--port", "7860"]
