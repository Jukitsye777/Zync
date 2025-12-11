# src/check_db.py
import os
from config import DB_PATH

abs_path = os.path.abspath(DB_PATH)
print("DB_PATH (relative):", DB_PATH)
print("DB_PATH (absolute):", abs_path)
print("Exists:", os.path.exists(abs_path))
if os.path.exists(abs_path):
    print("Size (bytes):", os.path.getsize(abs_path))
