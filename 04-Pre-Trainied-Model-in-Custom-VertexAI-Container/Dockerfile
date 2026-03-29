FROM python:3.10-slim

# Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model + app
COPY model.joblib .
COPY app.py .

# Expose port
EXPOSE 8080

# Start API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]