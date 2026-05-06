from .ambito_dolar_scraper import AmbitoDolarScraper
from .argenprop_scraper import ArgenPropScraper, InmuebleData
from .scraper_base import BaseScraper
from .SosivaApiClient import SosivaApiClient

__all__ = [
    "AmbitoDolarScraper",
    "ArgenPropScraper",
    "BaseScraper",
    "InmuebleData",
    "SosivaApiClient",
]
