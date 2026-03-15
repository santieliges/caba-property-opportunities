"""
Compat: CSVStorage se movió a storage/storage.py.
Este archivo se mantiene para no romper imports existentes.
"""

from storage.storage import CSVStorage  # noqa: F401

