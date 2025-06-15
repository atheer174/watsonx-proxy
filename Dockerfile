# Base image with Python runtime
FROM python:3.10-slim AS base

# Create a non-root user
RUN addgroup --system appgroup && adduser --system appuser --ingroup appgroup

WORKDIR /app

# Copy only what's needed
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Ensure runtime files owned by non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the listening port
EXPOSE 8080

# Launch command
CMD ["python", "app.py"]
