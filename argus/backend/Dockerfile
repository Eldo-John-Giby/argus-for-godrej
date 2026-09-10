FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV
# (libgl1-mesa-glx was removed in Debian 12; libgl1 is the replacement)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# python-multipart: required by FastAPI for form/video-upload endpoints.
# Separate layer so editing requirements.txt never forces a torch re-download.
RUN pip install --no-cache-dir python-multipart==0.0.12

COPY . .

RUN mkdir -p /app/storage/clips /app/storage/keyframes

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
