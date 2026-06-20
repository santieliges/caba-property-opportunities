FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

ENV DISPLAY=:99 \
    SCREEN_WIDTH=1440 \
    SCREEN_HEIGHT=900 \
    SCREEN_DEPTH=24 \
    VNC_PORT=5900 \
    NOVNC_PORT=6080 \
    HEADLESS=0

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        fluxbox \
        novnc \
        tini \
        websockify \
        x11-utils \
        x11vnc \
        xvfb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /tmp/scraper-requirements.txt
RUN pip install --no-cache-dir -r /tmp/scraper-requirements.txt \
    && pip install --no-cache-dir debugpy pytest==7.3.2

COPY . /app/scraper_service
COPY scripts/start-desktop.sh /usr/local/bin/start-desktop
RUN chmod +x /usr/local/bin/start-desktop

EXPOSE 5678 6080

ENTRYPOINT ["tini", "-g", "--", "/usr/local/bin/start-desktop"]

# Comando por defecto para depuración (cambiar el script según necesidad)
#CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "scraper_service/scrape_caba_bulk_listings.py"]
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client"]
