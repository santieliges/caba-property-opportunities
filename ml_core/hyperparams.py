from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _to_jsonable(value):
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(val)
            for key, val in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def snapshot_model_config(model, extra=None):
    payload = {
        "model_class": model.__class__.__name__,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for attr_name in (
        "best_params_",
        "rf_params",
        "kriging_params",
        "gwr_params",
        "sar_params",
    ):
        attr_value = getattr(model, attr_name, None)
        if attr_value is not None:
            payload[attr_name] = _to_jsonable(attr_value)

    if getattr(model, "bw_", None) is not None:
        payload["selected_bw"] = _to_jsonable(model.bw_)

    if getattr(model, "k_", None) is not None:
        payload["selected_k"] = _to_jsonable(model.k_)
    elif getattr(model, "k", None) is not None:
        payload["selected_k"] = _to_jsonable(model.k)

    if extra:
        payload.update(_to_jsonable(extra))

    return payload


def save_model_config(model, path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = snapshot_model_config(model, extra=extra)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def load_model_config(path, default=None):
    path = Path(path)
    if not path.exists():
        return default

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)

