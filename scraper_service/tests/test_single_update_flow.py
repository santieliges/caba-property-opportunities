from pathlib import Path
import sys
import asyncio
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scraper_service.routine_job.routine_job import RoutineJob
from scraper_service.scraper.argenprop_scraper import ArgenPropScraper
from scraper_service.storage.storage import CSVStorage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.updater import Updater
from scraper_service.updater.dataSource import ScrappingDataSource
from scraper_service.updater.samplers import NormalSampler, PoissonSampler


def test_single_department_update_flow(tmp_path):
    entry_id = 17568094
    url = "https://www.argenprop.com/departamento-en-venta-en-boedo-1-ambiente--17568094"
    csv_path = tmp_path / "single_update.csv"

    # Simular un CSV de entrada con un solo registro, igual que en producción.
    storage = CSVStorage(str(csv_path))
    storage.insert(
        {"id": entry_id, "url": url, "image_url": None, "imagen_path": None},
        valid_from=datetime.now(),
    )

    scraper = ArgenPropScraper(
        headless=False,
        url_base="https://www.argenprop.com",
        download_images=False,
        use_api_details=False,
    )
    sync = Synchronizer(storage=storage)
    updater = Updater(
        synchronizer=sync,
        data_source=ScrappingDataSource(scraper),
    )
    job = RoutineJob(storage=storage, scraper=scraper, updater=updater, synchronizer=sync)

    result = asyncio.run(
        job.fetch_and_sync_data(
            batch_size_sampler=PoissonSampler(lam=1),
            batch_delay_sampler=NormalSampler(mean=0, std=0),
            max_entries=1,
            max_concurrency=1,
        )
    )

    assert result == {"processed": 1, "closed": 0, "failed": 0, "total": 1}

    updated = storage.get_by_id(entry_id)
    assert updated is not None, "El registro no quedó en el storage después del update."
    assert updated["pozo"] == 1, f"Se esperaba pozo=1 para el aviso {entry_id}, pero quedó {updated['pozo']}"
    assert updated["informacion_adicional"], "No se obtuvo informacion_adicional en el update."
