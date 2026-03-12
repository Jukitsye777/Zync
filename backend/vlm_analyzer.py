import os
import cv2
from pathlib import Path
import base64
import requests
import tempfile
from typing import List, Dict, Optional
import time
import re
import numpy as np

from my_supabase_helper import insert_keyframe, insert_description, fetch_keyframes, description_exists

# ---------------------------
# CONFIG
# ---------------------------
FRAME_DIR = "output_description/vlm_analysis/keyframes"
INPUT_VIDEO_DIR = "input_videos"

os.makedirs(FRAME_DIR, exist_ok=True)

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "moondream"

PREPROCESS_SIZE = (640, 480)
DENOISE_STRENGTH = 5


# ---------------------------
# SIMPLE SCENE CHANGE DETECTOR
# (fixes clip_id issue)
# ---------------------------
def detect_scene_change(prev_frame, curr_frame, threshold=30):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.mean(diff)

    return score > threshold


# ---------------------------
# PREPROCESS IMAGE
# ---------------------------
def preprocess_image(img_bytes: bytes) -> bytes:

    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return img_bytes

        img = cv2.resize(img, PREPROCESS_SIZE, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.fastNlMeansDenoisingColored(
            img, None,
            h=DENOISE_STRENGTH,
            hColor=DENOISE_STRENGTH,
            templateWindowSize=7,
            searchWindowSize=21
        )

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        if not success:
            return img_bytes

        return buffer.tobytes()

    except Exception as e:
        print(f"⚠ Preprocessing error: {e}")
        return img_bytes


# ---------------------------
# DOWNLOAD IMAGE
# ---------------------------
def download_image_to_temp(url: str) -> str:

    resp = requests.get(url, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"Failed to download image from Supabase: {url}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(resp.content)
    tmp.close()

    return tmp.name


# ---------------------------
# GENERATE DESCRIPTION
# ---------------------------
def generate_description(frame_path: str) -> Optional[str]:

    try:

        if frame_path.startswith("http://") or frame_path.startswith("https://"):

            resp = requests.get(frame_path, timeout=30)

            if resp.status_code != 200:
                print(f"❌ Cannot download frame → {frame_path}")
                return None

            img_bytes = resp.content

        else:

            if not os.path.exists(frame_path):
                print(f"❌ Missing frame: {frame_path}")
                return None

            with open(frame_path, "rb") as f:
                img_bytes = f.read()

        img_bytes = preprocess_image(img_bytes)

        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "model": MODEL_NAME,
            "prompt": "Describe this image in detail.",
            "images": [img_b64],
            "stream": False
        }

        # retry logic for Ollama
        for attempt in range(3):

            try:

                response = requests.post(
                    f"{OLLAMA_HOST}/api/generate",
                    json=payload,
                    timeout=180
                )

                if response.status_code != 200:
                    print("❌ Ollama returned error:", response.text)
                    return None

                result = response.json()
                print("🧠 Ollama response received")

                return result.get("response", "").strip()

            except requests.exceptions.RequestException:

                print(f"⚠ Ollama retry {attempt+1}")
                time.sleep(3)

        return None

    except Exception as e:
        print(f"❌ Ollama error for {frame_path}: {e}")
        return None


# ---------------------------
# EXTRACT KEYFRAMES
# ---------------------------
def extract_keyframes(video_path: str, video_name: str = None, clip_id: int = 1, frame_skip: int = 10) -> List[Dict]:

    os.makedirs(FRAME_DIR, exist_ok=True)

    if video_name is None:
        video_name = Path(video_path).stem

    existing_keyframes = fetch_keyframes(video_name)

    if existing_keyframes:

        print(f"⏭️ Keyframes already exist for {video_name} ({len(existing_keyframes)} frames)")
        print("📊 Skipping keyframe extraction")

        return existing_keyframes

    print(f"🎬 Extracting keyframes for {video_name}...")

    cap = cv2.VideoCapture(video_path)

    keyframes_info = []

    frame_index = 0
    clip_id = 1
    prev_frame = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if prev_frame is not None:

            if detect_scene_change(prev_frame, frame):
                clip_id += 1
                print(f"🎬 Scene change detected → clip {clip_id}")

        if frame_index % frame_skip == 0:

            frame_filename = f"{video_name}_frame_{frame_index}.jpg"
            frame_path = os.path.join(FRAME_DIR, frame_filename)

            cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

            public_url = insert_keyframe(video_name, frame_index, frame_path, clip_id)

            success = bool(public_url)

            if success:
                print(f"✅ Uploaded frame {frame_index} (clip {clip_id})")
            else:
                print(f"❌ Failed to upload frame {frame_index}")

            keyframes_info.append({
                "frame_index": frame_index,
                "frame_path": frame_path,
                "uploaded_url": public_url,
                "status": success,
                "clip_id": clip_id
            })

            time.sleep(0.15)

        prev_frame = frame
        frame_index += 1

    cap.release()

    print(f"📊 Extracted {len(keyframes_info)} new keyframes")

    return keyframes_info


# ---------------------------
# PROCESS VIDEO
# ---------------------------
def process_video(video_name: str):

    keyframes = fetch_keyframes(video_name)

    if not keyframes:
        print(f"❌ No keyframes found for {video_name}")
        return

    print(f"📝 Processing descriptions for {video_name}...")

    skipped = 0
    generated = 0

    for frame in keyframes:

        frame_index = frame.get("frame_index")
        frame_path = frame.get("frame_path")
        clip_id = frame.get("clip_id", 1)

        if not frame_path:
            continue

        if description_exists(video_name, frame_index):

            print(f"⏭️ Skipping frame {frame_index} - description already exists")
            skipped += 1
            continue

        if frame_path.startswith("http://") or frame_path.startswith("https://"):

            try:
                local_frame = download_image_to_temp(frame_path)
            except Exception as e:
                print(f"⚠ Could not download keyframe → {frame_path} | {e}")
                continue

        else:
            local_frame = frame_path

        description = generate_description(local_frame)

        if not description:
            continue

        success = insert_description(video_name, frame_index, description, clip_id)

        if success:
            print(f"✅ Description inserted for frame {frame_index}")
            generated += 1
        else:
            print(f"⚠ Description already exists → frame {frame_index}")

        time.sleep(1.5)

    print(f"📊 Summary: {generated} generated, {skipped} skipped")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    if not os.path.exists(INPUT_VIDEO_DIR):
        print(f"❌ Missing input folder: {INPUT_VIDEO_DIR}")
        exit()

    videos = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

    if not videos:
        print("❌ No videos found in input_videos/")
        exit()

    for video in videos:

        full_path = os.path.join(INPUT_VIDEO_DIR, video)

        extract_keyframes(full_path, frame_skip=10)

        process_video(Path(full_path).stem)

    print("\n✅ All videos processed!")