# src/supabase_helper.py
import os
from typing import List, Dict, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

# load .env from repo root (create this file, see README below)
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment (.env).")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_keyframe(video_name: str, frame_index: int, frame_path: str, clip_id: int) -> bool:
    try:
        # avoid duplicates by video_name + frame_index
        existing = supabase.table("keyframes") \
            .select("id") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .limit(1) \
            .execute()

        if existing.data:
            return False

        data = {
            "video_name": video_name,
            "frame_index": frame_index,
            "frame_path": frame_path,
            "clip_id": clip_id
        }
        supabase.table("keyframes").insert(data).execute()
        return True
    except Exception as e:
        print(f"[supabase] insert_keyframe error: {e}")
        return False


def insert_description(video_name: str, frame_index: int, description: str, clip_id: int) -> bool:
    try:
        existing = supabase.table("descriptions") \
            .select("id") \
            .eq("video_name", video_name) \
            .eq("frame_index", frame_index) \
            .limit(1) \
            .execute()

        if existing.data:
            return False

        data = {
            "video_name": video_name,
            "frame_index": frame_index,
            "clip_id": clip_id,
            "description": description
        }
        supabase.table("descriptions").insert(data).execute()
        return True
    except Exception as e:
        print(f"[supabase] insert_description error: {e}")
        return False


def fetch_keyframes(video_prefix: str) -> List[Dict]:
    """
    Returns list of keyframe rows sorted by frame_index for video_prefix.
    Expected columns: id, video_name, frame_index, frame_path, clip_id, created_at (if present)
    """
    try:
        # using ilike for prefix matching (case-insensitive)
        resp = supabase.table("keyframes") \
            .select("*") \
            .ilike("video_name", f"{video_prefix}%") \
            .order("frame_index", desc=False) \
            .execute()
        return resp.data or []
    except Exception as e:
        print(f"[supabase] fetch_keyframes error: {e}")
        return []


def fetch_descriptions(video_prefix: str) -> List[Dict]:
    """
    Returns list of description rows for a video prefix.
    Expected columns: id, video_name, frame_index, description, clip_id, created_at
    """
    try:
        resp = supabase.table("descriptions") \
            .select("*") \
            .ilike("video_name", f"{video_prefix}%") \
            .order("frame_index", desc=False) \
            .execute()
        return resp.data or []
    except Exception as e:
        print(f"[supabase] fetch_descriptions error: {e}")
        return []
