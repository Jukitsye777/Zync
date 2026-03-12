from supabase import create_client, Client
import os
from typing import Optional, List, Dict
import time

# ---------------------------
# SUPABASE SETUP
# ---------------------------

SUPABASE_URL = "https://cfxhiibvphwuycrpjssp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNmeGhpaWJ2cGh3dXljcnBqc3NwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NTQzMDIwMSwiZXhwIjoyMDgxMDA2MjAxfQ.9sb3Aw7PdTMEUDE4U7_bRAoqbTeYjDotJW42Km1Fj_E"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------
# INSERT KEYFRAME
# ---------------------------
def insert_keyframe(video_name: str, frame_index: int, frame_path: str, clip_id: int) -> Optional[str]:
    try:
        # 1️⃣ Check if frame already exists
        existing = supabase.table("keyframes") \
            .select("frame_path") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .execute()
        if existing.data:
            return existing.data[0]["frame_path"]

        # 2️⃣ Upload frame to Supabase Storage
        filename = os.path.basename(frame_path)
        storage_path = f"{video_name}/{filename}"
        bucket = supabase.storage.from_("keyframe")  # ✅ FIXED: removed ()

        with open(frame_path, "rb") as f:
            data = f.read()

        bucket.upload(storage_path, data, file_options={"content-type": "image/jpeg"})

        # 3️⃣ Get public URL
        public_url = bucket.get_public_url(storage_path)

        # 4️⃣ Insert metadata in DB
        supabase.table("keyframes").insert({
            "video_name": video_name,
            "frame_index": frame_index,
            "frame_path": public_url,
            "clip_id": clip_id
        }).execute()

        print(f"✅ Keyframe saved → {video_name} frame {frame_index}")
        time.sleep(0.05)
        return public_url

    except Exception as e:
        print(f"❌ Failed inserting keyframe {video_name} frame {frame_index}: {e}")
        return None

# ---------------------------
# INSERT DESCRIPTION
# ---------------------------
def insert_description(video_name: str, frame_index: int, description: str, clip_id: int) -> bool:
    try:
        existing = supabase.table("descriptions") \
            .select("id") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .execute()
        if existing.data:
            print(f"⚠ Description exists → {video_name} frame {frame_index}")
            return True

        supabase.table("descriptions").insert({
            "video_name": video_name,
            "frame_index": frame_index,
            "clip_id": clip_id,
            "description": description
        }).execute()

        return True

    except Exception as e:
        print(f"❌ Failed inserting description {video_name} frame {frame_index}: {e}")
        return False

# ---------------------------
# FETCH KEYFRAMES (sorted)
# ---------------------------
def fetch_keyframes(video_prefix: str) -> List[Dict]:
    try:
        result = supabase.table("keyframes") \
            .select("*") \
            .ilike("video_name", f"{video_prefix}%") \
            .order("frame_index", desc=False) \
            .execute()
        return result.data if result.data else []
    except Exception as e:
        print(f"❌ Error fetching keyframes for '{video_prefix}': {e}")
        return []

# ---------------------------
# CHECK IF DESCRIPTION EXISTS
# ---------------------------
def description_exists(video_name: str, frame_index: int) -> bool:
    try:
        result = supabase.table("descriptions") \
            .select("id") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .execute()
        return bool(result.data)
    except Exception as e:
        print(f"⚠ Error checking description: {e}")
        return False