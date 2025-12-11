# src/inspect_new_db.py
import sqlite3
from pathlib import Path

# adjust if your DB is elsewhere
DB_PATH = Path("output_description/vlm_analysis/vlm_results.db")

print("Using DB:", DB_PATH.resolve())

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

print("\n=== Some rows from keyframes ===")
cur.execute("SELECT id, video_name, frame_index, frame_path, clip_id FROM keyframes LIMIT 10;")
for row in cur.fetchall():
    print(row)

print("\n=== Some rows from descriptions ===")
cur.execute("SELECT id, video_name, frame_index, clip_id, substr(description, 1, 80) || '...' FROM descriptions LIMIT 10;")
for row in cur.fetchall():
    print(row)

print("\n=== Distinct video_name values in descriptions ===")
cur.execute("SELECT DISTINCT video_name FROM descriptions LIMIT 10;")
for row in cur.fetchall():
    print(row[0])

conn.close()
