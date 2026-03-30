import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class SosivaResponse:
    status_code: int
    url: str
    json_data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None


class SosivaApiClient:
    def __init__(
        self,
        *,
        base_url: str = "https://api.sosiva451.com",
        headers: Optional[Dict[str, str]] = None,
        timeout_s: int = 20,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout_s = timeout_s

    def get_aviso(self, aviso_id: int) -> SosivaResponse:
        url = f"{self.base_url}/Avisos/{aviso_id}"
        response = requests.get(url, headers=self.headers, timeout=self.timeout_s)

        try:
            data = response.json()
            if isinstance(data, dict):
                return SosivaResponse(
                    status_code=response.status_code,
                    url=url,
                    json_data=data,
                )
            return SosivaResponse(
                status_code=response.status_code,
                url=url,
                json_data={"_non_dict_json": data},
            )
        except Exception:
            return SosivaResponse(status_code=response.status_code, url=url, text=response.text)


def map_aviso_to_inmueble_fields(aviso: Dict[str, Any]) -> Dict[str, Any]:
    moneda = aviso.get("MonedaSimbolo_t")
    precio = aviso.get("MontoOperacion_i")
    if precio is None:
        precio = aviso.get("MontoNormalizado_d")

    expensas = aviso.get("Expensas_i")
    area_total = aviso.get("SuperficieTotal_d")
    area_cubierta = aviso.get("SuperficieCubierta_d")
    area_desc = aviso.get("SuperficieDesCubierta_d")
    if area_desc is None:
        area_desc = aviso.get("SuperficieDescubierta_d")

    ambientes = aviso.get("CantidadAmbientes_i")
    dormitorios = aviso.get("CantidadDormitorios_i")
    banos = aviso.get("CantidadBanos_i") or aviso.get("CantidadBaños_i")

    cocheras = aviso.get("Cocheras_i")
    if cocheras is None and "Cocheras_b" in aviso:
        cocheras = 1 if aviso.get("Cocheras_b") else 0

    lat = aviso.get("Direccion_Latitud_d")
    lon = aviso.get("Direccion_Longitud_d")

    return {
        "precio": precio,
        "moneda": moneda,
        "expensas": expensas,
        "area_m2_cubierta": area_cubierta,
        "area_m2_descubierta": area_desc,
        "area_m2_total": area_total,
        "antiguedad": aviso.get("Antiguedad_i"),
        "estado_edificio": aviso.get("EstadoEdificio_t"),
        "ambientes": ambientes,
        "dormitorios": dormitorios,
        "banos": banos,
        "estado": aviso.get("Estado_t"),
        "disposicion": aviso.get("Disposicion_t"),
        "orientacion": aviso.get("Orientacion_t"),
        "cocheras": cocheras,
        "latitud": lat,
        "longitud": lon,
    }


def load_headers_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError('Headers JSON debe ser un objeto {"Header": "valor", ...}')
    return {str(key): str(value) for key, value in data.items()}
