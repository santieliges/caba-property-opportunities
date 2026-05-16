import asyncio
import os
import traceback
from pathlib import Path

from scraper_service.routine_job.routine_job import RoutineJob
from scraper_service.scraper.argenprop_scraper import ArgenPropScraper
from scraper_service.storage.storage import CSVStorage
from scraper_service.sync.sync import Synchronizer
from scraper_service.updater.updater import Updater


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def clean_barrio_slug(barrio: str) -> str:
    return barrio.strip().lower().replace(" ", "-")


def build_m2_query(min_m2: int | None = None, max_m2: int | None = None) -> str:
    if min_m2 is not None and max_m2 is not None:
        return f"{min_m2}-{max_m2}-m2-cubiertos"
    if min_m2 is not None:
        return f"desde-{min_m2}-m2-cubiertos"
    if max_m2 is not None:
        return f"hasta-{max_m2}-m2-cubiertos"
    return ""


def build_argenprop_url(barrio: str, min_m2: int | None = None, max_m2: int | None = None, orden: str = "orden-masnuevos") -> str:
    slug = clean_barrio_slug(barrio)
    query_parts = []
    range_query = build_m2_query(min_m2, max_m2)
    if range_query:
        query_parts.append(range_query)
    if orden:
        query_parts.append(orden)

    query = f"?{'&'.join(query_parts)}" if query_parts else ""
    return f"https://www.argenprop.com/departamentos/venta/{slug}{query}"


async def run_job(url, csv_path, n_pages=5, max_existing_hits=30):
    scraper = ArgenPropScraper(
        headless=env_flag("HEADLESS", True),
        url_base=url,
        download_images=False,
    )
    storage = CSVStorage(csv_path)
    sync_data = Synchronizer(storage=storage)
    updater = Updater()

    routine_job = RoutineJob(
        storage=storage,
        scraper=scraper,
        updater=updater,
        synchronizer=sync_data,
    )
    try:
        await routine_job.fetch_and_sync_new_listings(
            n_pages=n_pages,
            delay_s=3,
            jitter_s=10,
            max_existing_hits=max_existing_hits,
        )
    except Exception:
        print(f"Error running job for {url}")
        traceback.print_exc()


async def run_jobs_for_barrio(barrio: str, ranges, csv_path: str, n_pages: int = 20, max_existing_hits: int = 30):
    for min_m2, max_m2 in ranges:
        url = build_argenprop_url(barrio=barrio, min_m2=min_m2, max_m2=max_m2)
        print(f"Iniciando scraping para {barrio}: {min_m2 or 0} - {max_m2 or '∞'} m2 -> {url}")
        await run_job(url=url, csv_path=csv_path, n_pages=n_pages, max_existing_hits=max_existing_hits)


async def main():
    csv_path = str(RAW_DATA_DIR / "arg_venta_data.csv")
    for barrio in BARRIOS_CABA:
        ranges = [
            (10, 20),
            (20, 40),
            (40, 60),
            (60, 80),
            (80, 100),
            (100, None),
        ]
        await run_jobs_for_barrio(barrio=barrio, ranges=ranges, csv_path=csv_path, n_pages=30, max_existing_hits=30)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BARRIOS_CABA = [
            "Agronomía",
            "Almagro",
            "Balvanera",
            "Barracas",
            "Belgrano",
            "Boedo",
            "Caballito",
            "Chacarita",
            "Coghlan",
            "Colegiales",
            "Constitución",
            "Flores",
            "Floresta",
            "La Boca",
            "La Paternal",
            "Liniers",
            "Mataderos",
            "Monte Castro",
            "Monserrat",
            "Nueva Pompeya",
            "Nuñez",
            "Palermo",
            "Parque Avellaneda",
            "Parque Chacabuco",
            "Parque Chas",
            "Parque Patricios",
            "Puerto Madero",
            "Recoleta",
            "Retiro",
            "Saavedra",
            "San Cristobal",
            "San Nicolas",
            "San Telmo",
            "Velez Sarsfield",
            "Versalles",
            "Villa Crespo",
            "Villa del Parque",
            "Villa Devoto",
            "Villa General Mitre",
            "Villa Lugano",
            "Villa Luro",
            "Villa Ortuzar",
            "Villa Pueyrredon",
            "Villa Real",
            "Villa Riachuelo",
            "Villa Santa Rita",
            "Villa Soldati",
            "Villa Urquiza"
        ]
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

    asyncio.run(main())
