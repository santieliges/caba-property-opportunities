FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

WORKDIR /app

COPY requirements.txt /tmp/scraper-requirements.txt
RUN pip install --no-cache-dir -r /tmp/scraper-requirements.txt

# Instalar debugpy para depuración
RUN pip install debugpy

COPY . /app/scraper_service

# Exponer puerto de depuración
EXPOSE 5678

# Comando por defecto para depuración (cambiar el script según necesidad)
#CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "scraper_service/scrap_new_data.py"]
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client"]
RUN pip install --no-cache-dir pytest==7.3.2
