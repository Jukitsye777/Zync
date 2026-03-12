# db_helper.py
import sqlite3
from setup_db import DB_NAME

def get_connection():
    return sqlite3.connect(DB_NAME)

def get_or_create_scene(scene_name):
    conn = get_connection()
    c = conn.cursor()

    c.execute("SELECT id FROM scenes WHERE scene_name = ?", (scene_name,))
    row = c.fetchone()

    if row:
        conn.close()
        return row[0]

    c.execute("INSERT INTO scenes (scene_name) VALUES (?)", (scene_name,))
    conn.commit()
    scene_id = c.lastrowid
    conn.close()
    return scene_id


def insert_keyframe(scene_id, keyframe_name, timestamp, description):
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        INSERT INTO keyframes (scene_id, keyframe_name, timestamp, description)
        VALUES (?, ?, ?, ?)
    """, (scene_id, keyframe_name, timestamp, description))

    conn.commit()
    conn.close()
