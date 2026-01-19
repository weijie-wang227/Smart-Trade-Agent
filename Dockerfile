FROM python:3.11-slim

# 1. Set working directory
WORKDIR /app

# 2. Install system deps (needed for sentence-transformers + Chroma)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Copy dependency files first (for Docker layer caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the app
COPY . .

# 6. Create directory for Chroma persistence (if using local persist)
RUN mkdir -p /app/chroma_db

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Start FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
