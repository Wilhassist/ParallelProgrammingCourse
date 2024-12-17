# Use python-slim as the base image
FROM python:3.12.7-slim-bookworm

# Install necessary tools for C++ development and Conan
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    gcc \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the entire current directory (all files and subfolders) to the working directory
COPY . .

# Install dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Automatically detect a Conan profile based on the system
RUN conan profile detect --force

# Build Release configuration
RUN conan install . -of . --build=missing -c tools.system.package_manager:mode=install \
    -s build_type=Release

RUN conan install . -of . --build=missing -c tools.system.package_manager:mode=install \
    -s build_type=Debug

RUN find . -type d \( -name "cmake" -o -name "src" -o -name "include" -o -name "tests" \) \
    -exec rm -rf {} + && \
    find . -maxdepth 1 -type f -delete
    
# Default command
WORKDIR /app
CMD ["/bin/bash"]