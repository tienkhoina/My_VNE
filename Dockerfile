# -------- Base --------
FROM python:3.10-slim

WORKDIR /app
COPY . .

# tránh interactive prompt
ENV DEBIAN_FRONTEND=noninteractive

# system deps cần cho scipy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# upgrade pip
RUN pip install --upgrade pip

# -------- Install torch CPU --------
RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# -------- Install PyTorch Geometric CPU --------
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    pip install torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    pip install torch-cluster -f https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.2.0+cpu.html && \
    pip install torch-geometric

# -------- Other dependencies --------
RUN pip install \
    omegaconf \
    hydra-core \
    networkx \
    numpy \
    scipy \
    pandas \
    matplotlib \
    tqdm \
    pyyaml

# -------- Install virne --------
RUN pip install -e .

ENV PYTHONPATH=/app

CMD ["bash"]