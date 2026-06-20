import argparse
import asyncio
import os
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from scraper_service.routine_job.routine_job import RoutineJob
from scraper_service.scraper.argenprop_scraper import ArgenPropScraper
from scraper_service.storage.storage import CSVStorage
from scraper_service.sync.sync import Synchronizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CSV_PATH = RAW_DATA_DIR / "arg_venta_data.csv"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def ensure_newest_order(url: str) -> str:
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)

    if any(key == "orden-masnuevos" for key, _ in query_items):
        return url

    raw_query = parts.query
    if "orden-masnuevos" in raw_query:
        return url

    if raw_query:
        new_query = f"{raw_query}&orden-masnuevos"
    else:
        new_query = "orden-masnuevos"

    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


async def run_job(
    *,
    url: str,
    csv_path: str,
    n_pages: int,
    max_existing_hits: int,
    delay_s: float,
    jitter_s: float,
):
    storage = CSVStorage(csv_path)
    existing_ids_before = {
        int(row["id"])
        for _, row in storage.get_all().iterrows()
        if row.get("id") is not None
    }

    scraper = ArgenPropScraper(
        headless=env_flag("HEADLESS", False),
        url_base=url,
        download_images=False,
    )
    sync = Synchronizer(storage=storage)
    job = RoutineJob(
        storage=storage,
        scraper=scraper,
        synchronizer=sync,
    )

    await job.fetch_and_sync_new_listings(
        n_pages=n_pages,
        delay_s=delay_s,
        jitter_s=jitter_s,
        max_existing_hits=max_existing_hits,
    )

    active_ids_after = {
        int(row["id"])
        for _, row in storage.get_all().iterrows()
        if row.get("id") is not None
    }
    new_ids = active_ids_after - existing_ids_before
    return {
        "new_ids_count": len(new_ids),
        "new_ids_sample": sorted(new_ids)[:10],
        "csv_path": csv_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scrapea avisos desde una URL de Argenprop ordenada por mas nuevos "
            "y corta al encontrar avisos que ya existen en el CSV."
        )
    )
    parser.add_argument(
        "url",
        help="URL de Argenprop a scrapear. Si falta el orden por mas nuevos, se agrega automaticamente.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_PATH),
        help=f"CSV de destino. Default: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--n-pages",
        type=int,
        default=30,
        help="Maximo de paginas a recorrer antes de cortar. Default: 30.",
    )
    parser.add_argument(
        "--max-existing-hits",
        type=int,
        default=3,
        help="Cantidad de avisos ya conocidos para detener el scraping. Default: 1.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=3.0,
        help="Delay base entre avisos. Default: 3.0.",
    )
    parser.add_argument(
        "--jitter-s",
        type=float,
        default=10.0,
        help="Jitter aleatorio entre avisos. Default: 10.0.",
    )
    return parser


async def main():
    args = build_parser().parse_args()
    url = ensure_newest_order(args.url)

    if url != args.url:
        print(f"URL ajustada para ordenar por mas nuevos: {url}")

    print(
        "Iniciando scraping incremental de avisos nuevos. "
        f"Corta al llegar a {args.max_existing_hits} aviso(s) ya existente(s)."
    )
    result = await run_job(
        url=url,
        csv_path=args.csv_path,
        n_pages=args.n_pages,
        max_existing_hits=args.max_existing_hits,
        delay_s=args.delay_s,
        jitter_s=args.jitter_s,
    )
    print(
        f"Finalizado. Nuevos ids activos agregados: {result['new_ids_count']}. "
        f"Muestra: {result['new_ids_sample']}"
    )
    print(f"CSV actualizado: {result['csv_path']}")


if __name__ == "__main__":
    asyncio.run(main())
