from supabase import create_client, Client
import os
from typing import Optional, List, Dict

# ---------------------------
# SUPABASE SETUP
# ---------------------------

SUPABASE_URL = "https://cfxhiibvphwuycrpjssp.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNmeGhpaWJ2cGh3dXljcnBqc3NwIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NTQzMDIwMSwiZXhwIjoyMDgxMDA2MjAxfQ.9sb3Aw7PdTMEUDE4U7_bRAoqbTeYjDotJW42Km1Fj_E"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------
# INSERT KEYFRAME
# ---------------------------
def insert_keyframe(video_name: str, frame_index: int, frame_path: str, clip_id: int):
    try:
        # avoid duplicates
        existing = supabase.table("keyframes") \
            .select("id") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .execute()

        if existing.data:
            print(f"⚠ Keyframe already exists → {video_name} frame {frame_index}")
            return

        data = {
            "video_name": video_name,
            "frame_index": frame_index,
            "frame_path": frame_path,
            "clip_id": clip_id
        }

        supabase.table("keyframes").insert(data).execute()

    except Exception as e:
        print(f"❌ Failed inserting keyframe {video_name} frame {frame_index}: {e}")


# ---------------------------
# INSERT DESCRIPTION
# ---------------------------
def insert_description(video_name: str, frame_index: int, description: str, clip_id: int):
    try:
        existing = supabase.table("descriptions") \
            .select("id") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .execute()

        if existing.data:
            print(f"⚠ Description exists → {video_name} frame {frame_index}")
            return

        data = {
            "video_name": video_name,
            "frame_index": frame_index,
            "clip_id": clip_id,
            "description": description
        }

        supabase.table("descriptions").insert(data).execute()

    except Exception as e:
        print(f"❌ Failed inserting description {video_name} frame {frame_index}: {e}")


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
