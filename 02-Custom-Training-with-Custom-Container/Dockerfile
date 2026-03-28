FROM python:3.10-slim

WORKDIR /app

COPY trainer.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "trainer.py"]





