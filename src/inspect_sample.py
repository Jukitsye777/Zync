# src/inspect_sample.py
import sqlite3
DB="output_description/vlm_analysis/vlm_results.db"
conn = sqlite3.connect(DB)
cur = conn.cursor()
print("KEYFRAMES (first 20):")
for row in cur.execute("SELECT rowid, * FROM keyframes LIMIT 20"):
    print(row)
print("\nDESCRIPTIONS (first 20):")
for row in cur.execute("SELECT rowid, * FROM descriptions LIMIT 20"):
    print(row)
conn.close()
