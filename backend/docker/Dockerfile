# Use an official Python runtime as a parent image for the build stage
FROM python:3.11.2-slim as builder

# Set the working directory
WORKDIR /usr/src/app

# Ensure that Python outputs are sent straight to terminal without buffering
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc-dev g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the local code to the container's working directory
COPY . .

# Use an official Python runtime as a parent image for the final stage
FROM python:3.11.2-slim

# Create and set the working directory
WORKDIR /usr/src/app

# Copy the built Python dependencies and application code from the builder stage
COPY --from=builder /usr/src/app /usr/src/app

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Copy entrypoint script into the container
COPY entrypoint.sh /usr/src/app/entrypoint.sh

# Give execution rights on the entrypoint script
RUN chmod +x /usr/src/app/entrypoint.sh

# Define environment variable
ENV MODEL_PATH /usr/src/app/models

# Run the application
CMD ["/usr/src/app/entrypoint.sh"]
