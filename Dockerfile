# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HF_HOME=/code/.hf_cache

# Set the working directory in the container
WORKDIR /code

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and pre-install lightweight PyTorch CPU wheel
# This avoids downloading 2.5GB of heavy CUDA packages from PyPI that cause ReadTimeoutError
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 torch --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file and install dependencies with increased timeout & retries
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries 10 -r requirements.txt

# Copy the model download script and pre-download the model
COPY download_models.py .
RUN python download_models.py

# Copy the rest of the application code
COPY . .

# Adjust permissions for Hugging Face non-root user (UID 1000)
RUN chmod -R 777 /code

# Expose the port the app runs on (Hugging Face expects 7860)
EXPOSE 7860

# Command to run the application using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
