import os
import sqlite3
import cv2
from datetime import datetime

DB_PATH = "output_description/vlm_analysis/vlm_results.db"
FRAME_DIR = "output_description/vlm_analysis/keyframes"
INPUT_VIDEO_DIR = "input_videos"

os.makedirs(FRAME_DIR, exist_ok=True)

# ---------------------------
# CREATE DATABASE (No duplicates)
# ---------------------------
def create_database():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS keyframes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        frame_path TEXT,
        UNIQUE(video_name, frame_index)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS descriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        description TEXT,
        UNIQUE(video_name, frame_index)
    );
    """)

    conn.commit()
    conn.close()
    print("✅ Database ready.")

# ---------------------------
# Extract Keyframes
# ---------------------------
def extract_keyframes(video_path):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved = 0

    create_database()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 30 == 0:
            frame_file = f"{FRAME_DIR}/{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(frame_file, frame)

            # Insert only if NOT a duplicate
            cur.execute("""
                INSERT OR IGNORE INTO keyframes (video_name, frame_index, frame_path)
                VALUES (?, ?, ?)
            """, (video_name, frame_count, frame_file))

            saved += 1

        frame_count += 1

    conn.commit()
    conn.close()
    cap.release()

    print(f"🎞 Extracted {saved} keyframes for {video_name}")
    return True

# ---------------------------
# Store Descriptions (NO duplicates)
# ---------------------------
def store_description(video_name, frame_index, description):
    create_database()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO descriptions (video_name, frame_index, description)
        VALUES (?, ?, ?)
    """, (video_name, frame_index, description))

    conn.commit()
    conn.close()

# ---------------------------
# MAIN VLM Processing
# ---------------------------
def process_video(video_path, description_function):
    extract_keyframes(video_path)

    video_name = os.path.basename(video_path)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT frame_index, frame_path FROM keyframes WHERE video_name=?", (video_name,))
    frames = cur.fetchall()
    conn.close()

    for frame_index, frame_path in frames:
        desc = description_function(frame_path)
        store_description(video_name, frame_index, desc)

    print("✨ All descriptions stored!")


# Dummy test description generator
def dummy_description(frame_path):
    return f"Description for {os.path.basename(frame_path)}"

# Run directly
if __name__ == "__main__":
    print("🔍 Scanning for videos in input_videos/...")

    for filename in os.listdir(INPUT_VIDEO_DIR):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            video_path = os.path.join(INPUT_VIDEO_DIR, filename)
            print(f"\n🎥 Processing: {filename}")
            process_video(video_path, dummy_description)

    print("\n✅ All videos processed!")

