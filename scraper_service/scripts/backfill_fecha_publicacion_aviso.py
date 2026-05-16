import argparse
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scraper_service.update_data import RAW_DATA_DIR, backfill_active_field_from_aviso


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfillea fecha_publicacion_aviso_dt sobre los registros vigentes sin versionarlos.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(RAW_DATA_DIR / "arg_venta_data.csv"),
        help="Path al CSV a actualizar.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Segundos de espera entre requests a Sosiva.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        help="Limita la cantidad de registros vigentes a procesar.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Si se pasa, reescribe la columna aunque ya tenga valor.",
    )
    args = parser.parse_args()

    asyncio.run(
        backfill_active_field_from_aviso(
            csv_path=args.csv_path,
            aviso_field="FechaPublicacionAviso_dt",
            target_field="fecha_publicacion_aviso_dt",
            only_if_missing=not args.overwrite,
            delay_s=args.delay,
            max_entries=args.max_entries,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
