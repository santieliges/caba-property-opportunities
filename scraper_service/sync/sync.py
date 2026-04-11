from datetime import datetime

import pandas as pd

from storage.storage import Storage


class Synchronizer:
    def __init__(
        self,
        storage: Storage,
        BUSINESS_FIELDS={"precio", "moneda", "ambientes", "expensas", "latitud", "longitud", "antiguedad"},
    ):
        self.storage = storage
        self.BUSINESS_FIELDS = BUSINESS_FIELDS

    def sync_data(self, new_df: pd.DataFrame):
        for entry_id, row in new_df.to_dict(orient="index").items():
            self.sync_entry(entry_id, row)
        self.storage.save()

    def sync_entry(self, entry_id, entry: dict | None):
        now = datetime.now()

        if entry is None:
            if self.storage.exists(entry_id):
                self.storage.close(entry_id=entry_id, valid_to=now)
            return

        if not self.storage.exists(entry_id):
            self.storage.insert(entry=entry, valid_from=now)
            return

        old_entry = self.storage.get_by_id(entry_id)
        if self._has_changed(old_entry, entry):
            self.storage.close(entry_id=entry_id, valid_to=now)
            self.storage.insert(entry=entry, valid_from=now)

    def _has_changed(self, old, new: dict) -> bool:
        for field in self.BUSINESS_FIELDS:
            old_val = self._normalize(old.get(field))
            new_val = self._normalize(new.get(field))
            if old_val != new_val:
                return True
        return False

    def get_entry_by_id(self, entry_id):
        return self.storage.get_by_id(entry_id)

    def existing_ids(self):
        data = self.storage.get_all()
        return data.index if hasattr(data, "index") else None

    def _normalize(self, value):
        if pd.isna(value) or value in ("", b""):
            return None

        if isinstance(value, float) and value.is_integer():
            return int(value)

        if isinstance(value, str):
            value = value.strip()
            if value.replace(".", "", 1).isdigit():
                return float(value) if "." in value else int(value)
            return value

        if isinstance(value, float):
            return round(value, 5)

        return value
