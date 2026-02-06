import pandas as pd
from storage.storage import Storage
from datetime import datetime


class Synchronizer:
    def __init__(self, storage: Storage, BUSINESS_FIELDS = {"precio","moneda","ambientes","expensas","latitud","longitud" }):
        self.storage = storage
        self.BUSINESS_FIELDS = BUSINESS_FIELDS 
 

    # ----------- API pública -----------

    def sync_data(self, new_df: pd.DataFrame):
        """
        Sincroniza un DataFrame completo contra el storage.
        """
        for entry_id, row in new_df.to_dict(orient="index").items():
            self.sync_entry(entry_id, row)


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
        for field in self.BUSINESS_FIELDS:
            old_val = self._normalize(old.get(field))
            new_val = self._normalize(new.get(field))

            if old_val != new_val:
                return True

        return False


    # ----------- Helpers -----------

    def get_entry_by_id(self, entry_id):
        return self.storage.get_by_id(entry_id)

    def existing_ids(self):
        data = self.storage.get_all()
        return data.index if hasattr(data, "index") else None
    
    def _normalize(self, v):
        if pd.isna(v) or v in ("", b""):
            return None

        if isinstance(v, float) and v.is_integer():
            return int(v)

        if isinstance(v, str):
            v = v.strip()
            if v.replace(".", "", 1).isdigit():
                return float(v) if "." in v else int(v)
            return v


        if isinstance(v, float):
            return round(v, 5)
    
        return v
