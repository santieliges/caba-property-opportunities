FROM mcr.microsoft.com/playwright/python:v1.58.0-jammy

WORKDIR /app

COPY scraper_service/requirements.txt /tmp/scraper-requirements.txt
RUN pip install --no-cache-dir -r /tmp/scraper-requirements.txt

COPY scraper_service /app/scraper_service

CMD ["python", "-m", "scraper_service.update_data"]
