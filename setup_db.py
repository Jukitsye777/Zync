# setup_db.py
import sqlite3

DB_NAME = "zync_data.db"

def create_tables():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Stores each scene/video
    c.execute("""
    CREATE TABLE IF NOT EXISTS scenes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_name TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Stores keyframe-level descriptions
    c.execute("""
    CREATE TABLE IF NOT EXISTS keyframes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id INTEGER,
        keyframe_name TEXT,
        timestamp REAL,
        description TEXT,
        FOREIGN KEY(scene_id) REFERENCES scenes(id)
    )
    """)

    conn.commit()
    conn.close()
    print("✅ Database setup complete!")


if __name__ == "__main__":
    create_tables()
