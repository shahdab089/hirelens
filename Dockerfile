# Hirelens — Docker image for Hugging Face Spaces and Render.
FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; pypdf/python-docx are pure-python.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Write the SQLite db to a writable dir on either host.
ENV APP_DB_PATH=/tmp/applications.db
EXPOSE 7860

# Bind to the host-provided $PORT (Render) and fall back to 7860 (HF Spaces).
# Shell form so ${PORT} is expanded at runtime.
CMD uvicorn app_web:app --host 0.0.0.0 --port ${PORT:-7860}
