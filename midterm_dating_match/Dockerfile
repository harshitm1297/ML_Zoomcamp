FROM python:3.13-slim

WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY features.py predict.py model.bin ./
EXPOSE 9696
CMD ["sh", "-c", "exec uvicorn predict:app --host 0.0.0.0 --port ${PORT:-9696}"]
