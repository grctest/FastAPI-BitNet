FROM python:3.10

WORKDIR /code

COPY ./app /code

# Clone BitNet with submodules directly into /code (ensures all files and submodules are present)
RUN git clone --recursive https://github.com/microsoft/BitNet.git /tmp/BitNet && \
    cp -r /tmp/BitNet/* /code && \
    rm -rf /tmp/BitNet

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    lsb-release \
    software-properties-common \
    gnupg \
    cmake \
    clang \
    && bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt && \
    pip install "fastapi[standard]" "uvicorn[standard]" httpx fastapi-mcp

# (Optional) Run your setup_env.py if needed
RUN python /code/setup_env.py -md /code/models/BitNet-b1.58-2B-4T -q i2_s

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]