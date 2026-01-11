# 1. Base Image: Python 3.10
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
# 'libgl1-mesa-glx' is required by Matplotlib/OpenCV to render images
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your application code
COPY Inference.py .

# 6. Create the results directory (optional, ensures it exists)
RUN mkdir -p inference_results

# 7. Network Configuration
# Expose the port defined in your launch() command
EXPOSE 7890
# Allow Gradio to listen on all interfaces
ENV GRADIO_SERVER_NAME="0.0.0.0"

# 8. Start the app
CMD ["python", "Inference.py"]