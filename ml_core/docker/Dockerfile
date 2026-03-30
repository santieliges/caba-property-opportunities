FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    JUPYTER_PORT=8888

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    gfortran \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pipelines/requirements.txt /tmp/pipelines-requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/pipelines-requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]
