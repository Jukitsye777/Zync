import os
import cv2
from pathlib import Path
import base64
import requests

from supabase_helper import insert_keyframe, insert_description, fetch_keyframes
import requests
import tempfile

def download_image_to_temp(url: str):
    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Failed to download image from Supabase: {url}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name

# ---------------------------
# CONFIG
# ---------------------------
FRAME_DIR = "output_description/vlm_analysis/keyframes"
INPUT_VIDEO_DIR = "input_videos"

os.makedirs(FRAME_DIR, exist_ok=True)

OLLAMA_HOST = "http://localhost:11434"
MODEL_NAME = "moondream"




# ---------------------------
# GENERATE DESCRIPTION (Ollama)
# ---------------------------
import time
def generate_description(frame_path):
    try:
        # Case 1: frame_path is a Supabase public URL → download it
        if frame_path.startswith("http://") or frame_path.startswith("https://"):
            resp = requests.get(frame_path)
            if resp.status_code != 200:
                print(f"❌ Cannot download frame → {frame_path}")
                return None
            img_bytes = resp.content

        # Case 2: local file path (fallback)
        else:
            if not os.path.exists(frame_path):
                print(f"❌ Missing frame: {frame_path}")
                return None
            with open(frame_path, "rb") as f:
                img_bytes = f.read()

        # Encode image to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        payload = {
            "model": MODEL_NAME,
            "prompt": "Describe this image in detail.",
            "images": [img_b64],
            "stream": False
        }

        response = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()

        return result.get("response", "").strip()

    except Exception as e:
        print(f"❌ Ollama error for {frame_path}: {e}")
        return None





def extract_keyframes(video_path):
    video_name = Path(video_path).stem
    frames_per_clip = 5
    saved = 0

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % 30 == 0:  # 1 frame every 30 frames
            clip_id = saved // frames_per_clip + 1
            unique_video_name = f"{video_name}_subclip_{clip_id}"

            frame_filename = f"{unique_video_name}_frame_{frame_count}.jpg"
            local_path = os.path.abspath(os.path.join(FRAME_DIR, frame_filename))

            cv2.imwrite(local_path, frame)

            # Insert metadata
            insert_keyframe(unique_video_name, frame_count, local_path, clip_id)

            saved += 1

        frame_count += 1

    cap.release()
    print(f"🎞 Extracted {saved} keyframes → {video_name}")


# ---------------------------
# PROCESS VIDEO (generate descriptions)
# ---------------------------
def process_video(video_path):
    base_name = Path(video_path).stem
    print(f"🎥 Processing video: {base_name}")

    keyframes = fetch_keyframes(base_name)
    if not keyframes:
        print(f"❌ No keyframes found for {base_name}")
        return

    print(f"📝 Generating descriptions for {len(keyframes)} frames...")

    for frame in keyframes:
        frame_index = frame["frame_index"]
        frame_path = frame["frame_path"]
        clip_id = frame["clip_id"]
        vid_name = frame["video_name"]
        if frame_path.startswith("http://") or frame_path.startswith("https://"):
            try:
                local_frame = download_image_to_temp(frame_path)
            except Exception as e:
                print(f"⚠ Could not download keyframe → {frame_path} | {e}")
                continue
        else:
            if not os.path.exists(frame_path):
                print(f"⚠ Missing local frame file → {frame_path}")
                continue
            local_frame = frame_path

        description = generate_description(local_frame)
        if not description:
            print(f"⚠ No description for frame {frame_index}")
            continue

        insert_description(vid_name, frame_index, description, clip_id)

    print(f"✅ Completed → {base_name}")


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    if not os.path.exists(INPUT_VIDEO_DIR):
        print(f"❌ Missing input folder: {INPUT_VIDEO_DIR}")
        exit()

    videos = [f for f in os.listdir(INPUT_VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    if not videos:
        print("❌ No video files in input_videos/")
        exit()

    for video in videos:
        full_path = os.path.join(INPUT_VIDEO_DIR, video)
        extract_keyframes(full_path)
        process_video(full_path)

    print("\n✅ All videos processed!")
