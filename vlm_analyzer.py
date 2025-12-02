import os
import sqlite3
import cv2
from datetime import datetime
from pathlib import Path
import requests
import base64
import time

# ---------------------------
# CONFIG
# ---------------------------
DB_PATH = "output_description/vlm_analysis/vlm_results.db"
FRAME_DIR = "output_description/vlm_analysis/keyframes"
DESCRIPTION_DIR = "output_description/descriptions"
INPUT_VIDEO_DIR = "input_videos"

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "llava"

os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(DESCRIPTION_DIR, exist_ok=True)

# ---------------------------
# DATABASE SETUP
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
# EXTRACT KEYFRAMES
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

        # Save every 30th frame
        if frame_count % 30 == 0:
            frame_file = f"{FRAME_DIR}/{video_name}_frame_{frame_count}.jpg"
            cv2.imwrite(frame_file, frame)

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
# STORE DESCRIPTION
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
# VLM DESCRIPTION GENERATOR
# ---------------------------
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def generate_description(frame_path, prompt=None):
    if prompt is None:
        prompt = """Analyze this keyframe in detail. Include scene, objects, setting, actions, colors, text, and technical details."""

    base64_image = encode_image_to_base64(frame_path)
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "images": [base64_image],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40
        }
    }

    try:
        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            desc = result.get("response", "No description generated")
        else:
            desc = f"Failed: HTTP {response.status_code}"
    except Exception as e:
        desc = f"Failed: {str(e)}"

    # Save description as a text file in descriptions folder
    frame_name = Path(frame_path).name
    desc_file = Path(DESCRIPTION_DIR) / f"{frame_name}.txt"
    with open(desc_file, "w", encoding="utf-8") as f:
        f.write(desc)

    return desc

# ---------------------------
# PROCESS VIDEO
# ---------------------------
def process_video(video_path):
    extract_keyframes(video_path)
    video_name = os.path.basename(video_path)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT frame_index, frame_path FROM keyframes WHERE video_name=?", (video_name,))
    frames = cur.fetchall()
    conn.close()

    for frame_index, frame_path in frames:
        print(f"🧠 Generating description for {frame_path}...")
        desc = generate_description(frame_path)
        store_description(video_name, frame_index, desc)

    print("✨ All descriptions stored!")

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    print("🔍 Scanning for videos in input_videos/...")

    for filename in os.listdir(INPUT_VIDEO_DIR):
        if filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            video_path = os.path.join(INPUT_VIDEO_DIR, filename)
            print(f"\n🎥 Processing: {filename}")
            process_video(video_path)

    print("\n✅ All videos processed!")
