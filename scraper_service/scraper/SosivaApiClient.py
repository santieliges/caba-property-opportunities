import json
import re
import unicodedata
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


def get_aviso_field_value(aviso: Dict[str, Any], field_path: str) -> Any:
    value: Any = aviso
    for part in field_path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


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

    antiguedad_raw = get_antiguedad(aviso=aviso)
    antiguedad = _parse_antiguedad(antiguedad_raw)
    estado_edificio = aviso.get("EstadoEdificio_t")
    estado = aviso.get("Estado_t")
    disposicion = aviso.get("Disposicion_t")
    orientacion = aviso.get("Orientacion_t")
    informacion_adicional = aviso.get("InformacionAdicional_t")
    pozo = detect_pozo(aviso)
    fecha_publicacion_aviso_dt = aviso.get("FechaPublicacionAviso_dt")
    fecha_modificacion_aviso_dt = aviso.get("FechaModificacionAviso_dt")
    fecha_modificacion_puntos_dt = aviso.get("FechaModificacionPuntos_dt")

    return {
        "precio": precio,
        "moneda": moneda,
        "expensas": expensas,
        "area_m2_cubierta": area_cubierta,
        "area_m2_descubierta": area_desc,
        "area_m2_total": area_total,
        "antiguedad": antiguedad,
        "estado_edificio": estado_edificio,
        "ambientes": ambientes,
        "dormitorios": dormitorios,
        "banos": banos,
        "estado": estado,
        "disposicion": disposicion,
        "orientacion": orientacion,
        "cocheras": cocheras,
        "latitud": lat,
        "longitud": lon,
        "informacion_adicional": informacion_adicional,
        "pozo": pozo,
        "fecha_publicacion_aviso_dt": fecha_publicacion_aviso_dt,
        "fecha_modificacion_aviso_dt": fecha_modificacion_aviso_dt,
        "fecha_modificacion_puntos_dt": fecha_modificacion_puntos_dt,
    }



def load_headers_from_json(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError('Headers JSON debe ser un objeto {"Header": "valor", ...}')
    return {str(key): str(value) for key, value in data.items()}

### helpers

def get_antiguedad(aviso):
    # 1. DatosComunes (mejor fuente)
    for item in aviso.get("DatosComunes_s", []):
        if item.get("TipoDatoComun") == "ANTIGUEDAD":
            return item.get("Valor")

    # 2. Secciones (fallback)
    for sec in aviso.get("Secciones_s", {}).get("Secciones", []):
        if sec.get("Nombre") == "Características":
            for item in sec.get("Items", []):
                if item.get("Nombre") in ["Antiguedad", "Antigüedad"]:
                    return item.get("Valor")

    # 3. campo directo (último fallback)
    return aviso.get("Antiguedad_i")


def _parse_antiguedad(value):
    """Convierte valores tipo "28 años" o "A Estrenar" a enteros razonables.

    - Si hay dígitos, devuelve ese número (int).
    - Si dice "A Estrenar" o similar, devuelve 0 (o None si prefieres); elegimos 0 para conservar señal.
    - Si no se puede interpretar, devuelve None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        digits = ''.join(ch for ch in value if ch.isdigit())
        if digits:
            return int(digits)
        if value.strip().lower() in {"a estrenar", "estrenar", "a estrenar."}:
            return 0
    return None


def detect_pozo(aviso: Dict[str, Any]) -> int:
    text_parts = [
        aviso.get("InformacionAdicional_t"),
        aviso.get("Titulo_t"),
        aviso.get("Subtitulo_t"),
        aviso.get("EstadoEdificio_t"),
        aviso.get("Estado_t"),
    ]
    text = " ".join(part for part in text_parts if isinstance(part, str) and part.strip())
    normalized_text = _normalize_text(text)

    if not normalized_text:
        return 0

    strong_patterns = [
        r"\ben pozo\b",
        r"\bpozo\b",
        r"\bpreventa\b",
        r"\bfideicomiso\b",
        r"\ben construccion\b",
        r"\bentrega estimada\b",
        r"\bfecha de entrega\b",
        r"\bposesion\b",
        r"\bunidades? en desarrollo\b",
    ]
    payment_patterns = [
        r"\banticipo\b.{0,80}\bcuotas?\b",
        r"\bcuotas?\b.{0,80}\banticipo\b",
        r"\bsaldo\b.{0,80}\bcuotas?\b",
        r"\bvalor total\b",
        r"\bfinanciacion\b",
        r"\bfinanciado\b",
    ]
    currency_count = len(re.findall(r"(?:u\$s|usd|\$)\s*\d", normalized_text))

    score = 0
    if any(re.search(pattern, normalized_text) for pattern in strong_patterns):
        score += 2
    if any(re.search(pattern, normalized_text) for pattern in payment_patterns):
        score += 2
    if currency_count >= 2:
        score += 1

    # Un aviso con "pozo" o "preventa" solo ya suele ser suficiente; si no, pedimos
    # una segunda senal fuerte como esquema de cuotas o multiples montos.
    return 1 if score >= 2 else 0


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    without_accents = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return without_accents.lower()
