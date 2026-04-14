from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union


DEFAULT_STATE_ORDER = {
    "Excelente": 5,
    "Muy Bueno": 4,
    "Bueno": 3,
    "Regular": 2,
    "A Refaccionar": 1,
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_path: Path
    output_path: Path
    convert_from_currency: str
    convert_mode: str
    jitter_duplicate_coordinates: bool = False


def _load_official_dollar_rate(dollar_path: Path) -> float:
    df_dolar = pd.read_csv(dollar_path)
    row = df_dolar.loc[df_dolar["tipo_dolar"] == "oficial", "valor"]
    if row.empty or pd.isna(row.iloc[0]):
        raise ValueError("No se encontro un valor valido para dolar oficial.")
    return float(row.iloc[0])


def _build_geodataframe(df: pd.DataFrame, barrios_path: Path) -> gpd.GeoDataFrame:
    df = df.copy()
    gdf_points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitud"], df["latitud"]),
        crs="EPSG:4326",
    )

    barrios = gpd.read_file(barrios_path)
    barrios = barrios.to_crs(gdf_points.crs)

    gdf = gpd.sjoin(
        gdf_points,
        barrios[["nombre", "comuna", "geometry"]],
        how="left",
        predicate="within",
    ).rename(columns={"nombre": "barrio"})

    # Mantener solo observaciones dentro de CABA.
    caba_polygon = unary_union(barrios.geometry)
    gdf = gdf[gdf.geometry.within(caba_polygon)].copy()
    return gdf


def _jitter_duplicate_coordinates(
    df: pd.DataFrame,
    *,
    lon_col: str = "longitud",
    lat_col: str = "latitud",
    jitter_degrees: float = 1e-5,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.copy()
    dup_mask = df.duplicated(subset=[lon_col, lat_col], keep=False)
    if not dup_mask.any():
        return df

    rng = np.random.default_rng(random_state)
    dup_count = int(dup_mask.sum())
    df.loc[dup_mask, lon_col] = (
        df.loc[dup_mask, lon_col].astype(float)
        + rng.normal(0.0, jitter_degrees, size=dup_count)
    )
    df.loc[dup_mask, lat_col] = (
        df.loc[dup_mask, lat_col].astype(float)
        + rng.normal(0.0, jitter_degrees, size=dup_count)
    )
    return df


def _apply_fixed_cleaning_rules(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    gdf = gdf[gdf["valido_hasta"].isna()]
    gdf = gdf[gdf["precio"].notna()]
    gdf = gdf[gdf["area_m2_total"].notna()]
    gdf = gdf[gdf["ambientes"].notna()]
    gdf = gdf[gdf["latitud"].notna() & gdf["longitud"].notna()]
    gdf = gdf[gdf["barrio"].notna()]
    gdf = gdf[gdf["precio"] > 0]
    gdf = gdf[gdf["area_m2_total"] > 0]

    gdf["expensas"] = gdf["expensas"].fillna(0)
    gdf["banos"] = gdf["banos"].fillna(1)
    gdf["cocheras"] = gdf["cocheras"].fillna(0)
    gdf["area_m2_descubierta"] = gdf["area_m2_descubierta"].fillna(0)
    gdf["antiguedad"] = gdf["antiguedad"].apply(_normalize_antiguedad)

    gdf["estado_num"] = gdf["estado"].map(DEFAULT_STATE_ORDER)
    return gdf


def _normalize_antiguedad(value: float | int | str | None) -> float | None:
    current_year = datetime.now().year
    if pd.isna(value):
        return np.nan

    try:
        value = float(value)
    except Exception:
        return np.nan

    if 1800 <= value <= current_year:
        return current_year - value
    if value < 0 or value > 300:
        return np.nan
    return value


def _convert_prices(
    gdf: gpd.GeoDataFrame,
    *,
    from_currency: str,
    official_dollar_rate: float,
    mode: str,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    mask = gdf["moneda"] == from_currency

    if mode == "divide":
        gdf.loc[mask, "precio"] = gdf.loc[mask, "precio"] / official_dollar_rate
    elif mode == "multiply":
        gdf.loc[mask, "precio"] = gdf.loc[mask, "precio"] * official_dollar_rate
    else:
        raise ValueError(f"Modo de conversion no soportado: {mode}")

    gdf = gdf[gdf["precio"] > 0].copy()
    gdf["log_precio"] = np.log(gdf["precio"])
    gdf["precio_sobre_m2"] = gdf["precio"] / gdf["area_m2_total"]
    return gdf


def build_processed_dataset(
    *,
    config: DatasetConfig,
    barrios_path: str | Path,
    dollar_path: str | Path,
    verbose: bool = True,
) -> pd.DataFrame:
    raw_path = Path(config.raw_path)
    output_path = Path(config.output_path)
    barrios_path = Path(barrios_path)
    dollar_path = Path(dollar_path)

    stage_counts = []

    df = pd.read_csv(raw_path)
    stage_counts.append(("raw", len(df)))

    if config.jitter_duplicate_coordinates:
        df = _jitter_duplicate_coordinates(df)
        stage_counts.append(("after_jitter", len(df)))

    gdf = _build_geodataframe(df, barrios_path)
    stage_counts.append(("after_spatial_join", len(gdf)))

    gdf = _apply_fixed_cleaning_rules(gdf)
    stage_counts.append(("after_clean_rules", len(gdf)))

    official_rate = _load_official_dollar_rate(dollar_path)
    gdf = _convert_prices(
        gdf,
        from_currency=config.convert_from_currency,
        official_dollar_rate=official_rate,
        mode=config.convert_mode,
    )
    stage_counts.append(("after_currency_convert", len(gdf)))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = pd.DataFrame(gdf.drop(columns="geometry"))
    df_out.to_csv(output_path, index=False)

    if verbose:
        print(f"[build_processed_dataset:{config.name}] registros por etapa:")
        for name, count in stage_counts:
            print(f"  - {name}: {count}")

    return df_out


def build_venta_processed_dataset(
    *,
    raw_path: str | Path = "scraper_service/storage/data/arg_venta_data.csv",
    output_path: str | Path = "scraper_service/storage/data/arg_venta_caba_processed.csv",
    barrios_path: str | Path = "barrios.geojson",
    dollar_path: str | Path = "scraper_service/storage/data/dolar_hoy.csv",
) -> pd.DataFrame:
    return build_processed_dataset(
        config=DatasetConfig(
            name="venta",
            raw_path=Path(raw_path),
            output_path=Path(output_path),
            convert_from_currency="ARS",
            convert_mode="divide",
            jitter_duplicate_coordinates=False,  # No jitter en preprocessing; hacerlo en notebooks si se necesita
        ),
        barrios_path=barrios_path,
        dollar_path=dollar_path,
    )


def build_alquiler_processed_dataset(
    *,
    raw_path: str | Path = "scraper_service/storage/data/arg_alquiler_data.csv",
    output_path: str | Path = "scraper_service/storage/data/arg_alquiler_caba_processed.csv",
    barrios_path: str | Path = "barrios.geojson",
    dollar_path: str | Path = "scraper_service/storage/data/dolar_hoy.csv",
) -> pd.DataFrame:
    return build_processed_dataset(
        config=DatasetConfig(
            name="alquiler",
            raw_path=Path(raw_path),
            output_path=Path(output_path),
            convert_from_currency="USD",
            convert_mode="multiply"
        ),
        barrios_path=barrios_path,
        dollar_path=dollar_path,
    )
