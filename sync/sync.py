import pandas as pd
from storage.storage import Storage
from datetime import datetime


class Synchronizer:
    def __init__(self, storage: Storage):
        self.storage = storage

    # ----------- API pública -----------

    def sync_data(self, new_df: pd.DataFrame):
        """
        Sincroniza un DataFrame completo contra el storage.
        """
        for entry_id, row in new_df.iterrows():
            self.sync_entry(entry_id, row.to_dict())

        self.storage.save()

    def sync_entry(self, entry_id, entry: dict | None):
        """
        Sincroniza una sola entidad.

        entry:
        - dict  -> estado actual
        - None  -> la entidad ya no existe (cerrar)
        """
        now = datetime.now()

        # Caso: no hay estado actual → cerrar si existe
        if entry is None:
            if self.storage.exists(entry_id):
                self.storage.close(entry_id=entry_id, valid_to=now)
            return

        # Caso: nuevo registro
        if not self.storage.exists(entry_id):
            self.storage.insert(
                entry=entry,
                valid_from=now
            )
            return

        # Caso: posible actualización
        old_entry = self.storage.get_by_id(entry_id)

        if self._has_changed(old_entry, entry):
            self.storage.close(entry_id=entry_id, valid_to=now)
            self.storage.insert(
                entry=entry,
                valid_from=now
            )

    # ----------- Lógica interna -----------

    def _has_changed(self, old, new: dict) -> bool:
        for k, v in new.items():
            if k == "id":
                continue

            if k not in old.index:
                return True

            if pd.isna(old[k]) and pd.isna(v):
                continue

            if old[k] != v:
                return True

        return False

    # ----------- Helpers -----------

    def get_entry_by_id(self, entry_id):
        return self.storage.get_by_id(entry_id)

    def existing_ids(self):
        data = self.storage.get_all()
        return data.index if hasattr(data, "index") else None
