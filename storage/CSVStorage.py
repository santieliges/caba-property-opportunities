
import pandas as pd
from pathlib import Path
from storage.storage import Storage

class CSVStorage(Storage):
    def __init__(self, path: str):
        self.path = Path(path)
        self.df = self.load()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()

    def load(self):
        if self.path.exists():
            return pd.read_csv(
                self.path,
                parse_dates=["valido_desde", "valido_hasta"]
            ).set_index("idx")
        else:
            df = pd.DataFrame(columns=["valido_desde", "valido_hasta"])
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
        return self.df.loc[mask].copy()

    def exists(self, entry_id) -> bool:
        return (
            (self.df["id"] == entry_id) &
            (self.df["valido_hasta"].isna())
        ).any()


    def insert(self, entry: dict, valid_from):
        entry = entry.copy()
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
