from pathlib import Path
import re
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scraper_service.scraper.SosivaApiClient import (
    SosivaApiClient,
    detect_pozo,
    map_aviso_to_inmueble_fields,
)

def test_detect_pozo_with_direct_keyword():
    aviso = {"InformacionAdicional_t": "Departamento de 2 ambientes en pozo con entrega estimada para 2027."}
    assert detect_pozo(aviso) == 1


def test_detect_pozo_with_payment_plan():
    aviso = {
        "InformacionAdicional_t": "Anticipo de 35.000 U$D + 24 cuotas de 1.200 USD. Valor total: 63.800 USD."
    }
    assert detect_pozo(aviso) == 1


def test_detect_pozo_without_clear_signals():
    aviso = {
        "InformacionAdicional_t": "Semipiso de 3 ambientes reciclado, listo para mudarse, con balcón corrido."
    }
    assert detect_pozo(aviso) == 0


def test_map_aviso_to_inmueble_fields_exposes_pozo():
    aviso = {
        "MontoOperacion_i": 95000,
        "MonedaSimbolo_t": "USD",
        "InformacionAdicional_t": "Preventa con financiación en cuotas y posesión estimada en 18 meses.",
    }

    mapped = map_aviso_to_inmueble_fields(aviso)

    assert mapped["pozo"] == 1
    assert mapped["informacion_adicional"] == aviso["InformacionAdicional_t"]

# tests/test_pozo_links_integration.py

POZO_LINKS = [
    "https://www.argenprop.com/departamento-en-venta-en-boedo-1-ambiente--17568094",
    "https://www.argenprop.com/departamento-en-venta-en-villa-crespo-2-ambientes--17947256",
    "https://www.argenprop.com/departamento-en-venta-en-villa-urquiza-2-ambientes--18038929"
]



def extract_argenprop_id(url: str) -> int:
    match = re.search(r"--(\d+)$", url)
    assert match, f"No pude extraer ID de ArgenProp desde: {url}"
    return int(match.group(1))

@pytest.mark.parametrize("link", POZO_LINKS)
def test_specific_argenprop_links_are_detected_as_pozo(link):
    aviso_id = extract_argenprop_id(link)

    client = SosivaApiClient()
    response = client.get_aviso(aviso_id)

    assert response.status_code == 200
    assert response.json_data is not None

    mapped = map_aviso_to_inmueble_fields(response.json_data)

    assert mapped["pozo"] == 1, {
        "link": link,
        "aviso_id": aviso_id,
        "informacion_adicional": mapped.get("informacion_adicional"),
        "estado": mapped.get("estado"),
        "estado_edificio": mapped.get("estado_edificio"),
    }
