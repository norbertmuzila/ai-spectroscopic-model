# =============================================================================
#  AI Spectroscopic Model - hosted demo image
#
#  This builds the *demo* deployment: the full backend, engine and dashboard,
#  running against the built-in simulator.
#
#  It deliberately cannot read your spectrometer. A container in a datacentre
#  has no USB bus, and no amount of configuration changes that - the hardware
#  path has to run on the machine the USB4000 is plugged into. Use this image
#  to share the interface and the model; use `python run.py` locally to measure.
#
#  Build from the project root:
#      docker build -t spectral-console .
#      docker run -p 8000:8000 spectral-console
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SPECTRO_DEMO=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Build tools for scipy/scikit-learn wheels that need them, then removed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so the layer caches across code edits. seabreeze and the
# USB backend are dropped here: they only exist to reach hardware that a
# container cannot see, and they pull in native libraries for nothing.
COPY requirements.txt .
RUN grep -viE '^(seabreeze|libusb-package|pyusb)' requirements.txt > requirements-cloud.txt \
    && pip install --no-cache-dir -r requirements-cloud.txt \
    && apt-get purge -y build-essential && apt-get autoremove -y

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY run.py ./

# Build the spectral library and train the model at image-build time so the
# container starts instantly rather than spending minutes on first request.
RUN mkdir -p data/library data/models data/sessions data/reports data/incoming \
    && python scripts/train.py --samples 220

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/status',timeout=4)"

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
