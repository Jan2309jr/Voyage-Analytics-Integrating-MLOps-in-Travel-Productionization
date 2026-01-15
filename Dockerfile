# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY artifacts/ artifacts/
COPY src/ src/

# Expose Flask port
EXPOSE 5000

# Run the Flask API
CMD ["python", "api/app.py"]
