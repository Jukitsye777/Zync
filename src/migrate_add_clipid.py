# src/migrate_clipid_robust.py
import sqlite3
import os
import re
import shutil
from pathlib import Path

DB_REL = "output_description/vlm_analysis/vlm_results.db"
DB = Path(DB_REL)
if not DB.exists():
    raise SystemExit(f"DB not found at {DB}")

BACKUP = DB.with_suffix(".backup.db")
print("Backing up DB to:", BACKUP)
shutil.copy2(DB, BACKUP)

conn = sqlite3.connect(str(DB))
cur = conn.cursor()

def table_exists(name):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,))
    return cur.fetchone() is not None

def get_columns(table):
    cur.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]

print("\nFound tables:", [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()])

# Print schemas for quick manual check
for t in ['keyframes', 'descriptions']:
    if table_exists(t):
        cols = get_columns(t)
        print(f"\nPRAGMA table_info({t}): {cols}")
    else:
        print(f"\nTable {t} not found.")

# Add clip_id if missing
def add_col_if_missing(table, col, col_type="INTEGER"):
    cols = get_columns(table)
    if col in cols:
        print(f"{table}.{col} exists")
        return False
    print(f"Adding column {col} to {table}")
    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
    conn.commit()
    return True

if table_exists("keyframes"):
    add_col_if_missing("keyframes", "clip_id")
if table_exists("descriptions"):
    add_col_if_missing("descriptions", "clip_id")

# Helper to try to populate clip_id in keyframes using regex on video_name or frame_path
def populate_keyframes_from_video_name():
    print("\nProcessing keyframes to extract clip_id from video_name or frame_path...")
    updated = 0
    pattern = re.compile(r"_subclip_(\d+)")
    cur.execute("SELECT rowid, video_name, frame_path, clip_id FROM keyframes")
    rows = cur.fetchall()
    for rowid, vname, fpath, existing in rows:
        if existing not in (None, ''):
            continue
        cid = None
        if vname:
            m = pattern.search(vname)
            if m:
                cid = int(m.group(1))
        # try from frame_path (e.g., ...subclip_5_frame_90.jpg)
        if cid is None and fpath:
            m = pattern.search(fpath)
            if m:
                cid = int(m.group(1))
        if cid is not None:
            cur.execute("UPDATE keyframes SET clip_id=? WHERE rowid=?", (cid, rowid))
            updated += 1
    conn.commit()
    print(f"  Updated {updated} keyframe rows using name/path parsing.")
    return updated

# Helper to populate descriptions.clip_id by joining keys (when possible)
def populate_descriptions_from_keyframes():
    print("\nPropagating clip_id from keyframes to descriptions via matching (video_name + frame_index or image name)...")
    updated = 0
    # Try join on video_name AND frame_index if those columns exist
    key_cols = get_columns("keyframes")
    desc_cols = get_columns("descriptions")

    # If descriptions has video_name and frame_index use join
    if "video_name" in desc_cols and "frame_index" in desc_cols:
        cur.execute("""
            UPDATE descriptions
            SET clip_id = (
                SELECT k.clip_id FROM keyframes k
                WHERE k.video_name = descriptions.video_name AND k.frame_index = descriptions.frame_index
                LIMIT 1
            )
            WHERE (descriptions.clip_id IS NULL OR descriptions.clip_id = '')
        """)
        updated = conn.total_changes
        conn.commit()
        print(f"  Updated {updated} description rows by joining video_name & frame_index.")
        return updated

    # If descriptions has image_name or image_path, try matching on filename part
    possible_desc_image_cols = [c for c in desc_cols if c in ("image_name","image_path","frame_path","image")]
    if possible_desc_image_cols:
        desc_col = possible_desc_image_cols[0]
        # match by filename substring
        cur.execute(f"SELECT rowid, {desc_col} FROM descriptions")
        rows = cur.fetchall()
        for rowid, val in rows:
            if not val:
                continue
            filename = Path(val).name
            # try joining to keyframes by frame_path containing filename
            cur.execute("SELECT clip_id FROM keyframes WHERE frame_path LIKE ? LIMIT 1", (f"%{filename}%",))
            r = cur.fetchone()
            if r and r[0] is not None:
                cur.execute("UPDATE descriptions SET clip_id=? WHERE rowid=?", (r[0], rowid))
                updated += 1
        conn.commit()
        print(f"  Updated {updated} description rows by matching image filename to keyframes.")
        return updated

    print("  Could not auto-populate descriptions.clip_id — no matching columns found.")
    return 0

# Run population steps
populate_keyframes_from_video_name()
populate_descriptions_from_keyframes()

# Final counts
def count_missing(table):
    if not table_exists(table):
        return 0
    cols = get_columns(table)
    if "clip_id" not in cols:
        return 0
    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE clip_id IS NULL OR clip_id = ''")
    return cur.fetchone()[0]

print("\nFinished migration steps.")
print("Remaining missing clip_id rows:")
print("  keyframes:", count_missing("keyframes"))
print("  descriptions:", count_missing("descriptions"))

print("\nIf rows remain missing, inspect sample rows with:")
print(f"  sqlite3 {DB} \"SELECT * FROM keyframes LIMIT 10;\"")
print(f"  sqlite3 {DB} \"SELECT * FROM descriptions LIMIT 10;\"")

conn.close()
