
import pandas as pd
from pathlib import Path
from storage.storage import Storage
DTYPES = {
    "id": "Int64",
    "precio": "Int64",
    "moneda": "string",
    "expensas": "Int64",
    "tipo_unidad": "string",
    "area_m2_total": "Float64",
    "ambientes": "Int64",
    "banos": "Int64",
    "cocheras": "Int64",
    "latitud": "Float64",
    "longitud": "Float64",
}

class CSVStorage(Storage):
    def __init__(self, path: str):
        self.path = Path(path)
        self.df = self.load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def _normalize_entry(self, entry: dict) -> dict:
        out = {}
        for k, v in entry.items():
            if k in ["precio", "expensas", "ambientes", "banos", "cocheras"]:
                out[k] = None if v is None else int(v)
            elif k in ["area_m2_total", "latitud", "longitud"]:
                out[k] = None if v is None else float(v)
            else:
                out[k] = v
        return out

    def load(self):
        if self.path.exists():
            df = pd.read_csv(
                self.path,
                parse_dates=["valido_desde", "valido_hasta"],
                dtype={k: v for k, v in DTYPES.items() if k != "id"}
            )
            df["id"] = df["id"].astype("int")
            return df.set_index("idx")
        else:
            df = pd.DataFrame(columns=["id","valido_desde", "valido_hasta"])
            df.index.name = "idx"
            return df


    def _next_idx(self) -> int:
        if self.df.empty:
            return 1
        return int(self.df.index.max()) + 1

    def save(self, new_path: str = None):
        if new_path:
            self.path = Path(new_path)
        self.df.reset_index().to_csv(self.path, index=False)

    def get_all(self) -> pd.DataFrame:
        return self.df.copy()

    def get_by_id(self, entry_id, only_active=True):
        mask = self.df["id"] == entry_id
        if only_active:
            mask &= self.df["valido_hasta"].isna()

        rows = self.df.loc[mask]
        if rows.empty:
            return None

        return rows.iloc[0].to_dict()

    def exists(self, entry_id) -> bool:
        return (
            (self.df["id"] == entry_id) &
            (self.df["valido_hasta"].isna())
        ).any()


    def insert(self, entry: dict, valid_from):
        entry = self._normalize_entry(entry.copy())
        entry["valido_desde"] = valid_from
        entry["valido_hasta"] = pd.NaT
        idx = self._next_idx()
        self.df.loc[idx, entry.keys()] = entry.values()



    def close(self, entry_id, valid_to):
        mask = (
            (self.df["id"] == entry_id) &
            (self.df["valido_hasta"].isna())
        )
        self.df.loc[mask, "valido_hasta"] = valid_to

    def update(self, entry_id, updates: dict, valid_from):
        mask_current = (
            (self.df["id"] == entry_id) &
            (self.df["valido_hasta"].isna())
        )

        if not mask_current.any():
            raise ValueError(f"No existe registro activo para id={entry_id}")

        old_entry = self.df.loc[mask_current].iloc[0]

        # cerrar versión anterior
        self.df.loc[mask_current, "valido_hasta"] = valid_from
        new_entry = old_entry.to_dict()
        new_entry.update(updates)
        self.insert(new_entry, valid_from)

    def delete(self, entry_id, valid_to):
        self.close(entry_id, valid_to)


    def apply_to_column(self, column, func):
        self.df[column] = self.df[column].apply(func)

    def fillna(self, column, fill_value):
        self.df[column] = self.df[column].fillna(fill_value)

    def dropna(self, column):
        self.df = self.df.dropna(subset=[column])
