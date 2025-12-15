# src/db_reader.py
"""
Supabase DB reader for ZYNC (hanna branch).

Provides:
- get_all_descriptions_for_video(video_stem)
    -> returns list of tuples (video_name, frame_index, frame_path, description, clip_id)
- get_all_videos()
    -> returns list of video stems (e.g., "classroom")
- get_keyframes_for_video_prefix(video_prefix)
    -> returns list of dict rows from keyframes table (useful for debugging)

Environment:
- SUPABASE_URL and SUPABASE_KEY must be set in environment or in a .env file.
  Example .env:
    SUPABASE_URL=https://xxxxx.supabase.co
    SUPABASE_KEY=eyJ...secret...
"""

import os
from typing import List, Tuple, Dict, Optional
from supabase import create_client, Client
from dotenv import load_dotenv
from pathlib import Path

# Load .env if present (safe to keep only for local dev)
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase URL/KEY not found. Set SUPABASE_URL and SUPABASE_KEY in environment or .env")

# create a Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def _norm_video_stem(video_stem: str) -> str:
    """
    Helper: normalize user input video stem to what the DB uses (no trailing whitespace).
    """
    return video_stem.strip()


def get_all_descriptions_for_video(video_stem: str) -> List[Tuple[str, int, str, str, Optional[int]]]:
    """
    Fetches descriptions joined with keyframe metadata from Supabase.

    Returns list of tuples:
      (video_name, frame_index, frame_path, description, clip_id)

    The query uses `video_name` ILIKE '<video_stem>%', ordered by frame_index asc.

    Notes:
    - If your Supabase table/column names differ, adapt the `select()` fields.
    - If there are many rows, this will fetch all pages (pagination handled).
    """
    video_stem = _norm_video_stem(video_stem)
    like_pattern = f"{video_stem}%"  # matches video_name starting with the stem

    try:
        # Using supabase client to query the joined information requires two separate queries
        # because PostgREST (used by supabase) doesn't support server-side joins across unrelated tables
        # unless you created foreign-key relationships and rpc. We'll pull descriptions and keyframes
        # and coalesce them client-side by video_name + frame_index.
        #
        # Approach:
        # 1) fetch descriptions where video_name ILIKE video_stem%
        # 2) fetch keyframes where video_name ILIKE video_stem%
        # 3) merge them in Python by (video_name, frame_index)

        # fetch descriptions
        desc_res = supabase.table("descriptions") \
            .select("video_name, frame_index, clip_id, description") \
            .ilike("video_name", like_pattern) \
            .order("frame_index", desc=False) \
            .execute()
        desc_rows = desc_res.data or []

        # fetch keyframes
        kf_res = supabase.table("keyframes") \
            .select("video_name, frame_index, frame_path, clip_id") \
            .ilike("video_name", like_pattern) \
            .order("frame_index", desc=False) \
            .execute()
        kf_rows = kf_res.data or []

        # build a lookup for keyframes by (video_name, frame_index)
        kf_lookup: Dict[Tuple[str, int], Dict] = {}
        for r in kf_rows:
            key = (r.get("video_name"), int(r.get("frame_index")))
            kf_lookup[key] = r

        merged: List[Tuple[str, int, str, str, Optional[int]]] = []

        # Prefer description rows as primary; try to augment with frame_path from keyframes
        for r in desc_rows:
            vname = r.get("video_name")
            fi = int(r.get("frame_index"))
            desc_text = r.get("description") or ""
            clip_id = r.get("clip_id")

            kf = kf_lookup.get((vname, fi))
            frame_path = kf.get("frame_path") if kf else None
            # If keyframe contained clip_id and descriptions didn't, coalesce
            if not clip_id and kf:
                clip_id = kf.get("clip_id")

            merged.append((vname, fi, frame_path, desc_text, int(clip_id) if clip_id is not None else None))

        # If descriptions table has missing rows, optionally include keyframes without descriptions
        # (useful in early pipeline stages)
        for key, kf in kf_lookup.items():
            vname, fi = key
            # if not already included
            if not any(mv == vname and mfi == fi for mv, mfi, *_ in merged):
                merged.append((vname, fi, kf.get("frame_path"), "", int(kf.get("clip_id") if kf.get("clip_id") is not None else None)))

        # Sort by frame_index ascending to keep consistent order
        merged.sort(key=lambda x: x[1])
        return merged

    except Exception as e:
        print(f"❌ Error fetching descriptions for '{video_stem}': {e}")
        return []


def get_all_videos() -> List[str]:
    """
    Returns a list of distinct video stems present in keyframes (or descriptions).
    Example: if video_name stored as 'classroom_subclip_1', returns ['classroom', ...].
    """
    try:
        res = supabase.table("keyframes").select("video_name").execute()
        rows = res.data or []
        stems = set()
        for r in rows:
            name = r.get("video_name", "")
            # attempt to split by _subclip_ to get base
            if "_subclip_" in name:
                stem = name.split("_subclip_")[0]
            else:
                # fallback: take filename stem without extension
                stem = Path(name).stem if name else ""
            if stem:
                stems.add(stem)
        return sorted(list(stems))
    except Exception as e:
        print(f"❌ Error fetching video stems: {e}")
        return []


def get_keyframes_for_video_prefix(video_prefix: str) -> List[Dict]:
    """
    Utility: returns raw keyframe rows for debugging or preview.
    """
    try:
        res = supabase.table("keyframes") \
            .select("*") \
            .ilike("video_name", f"{video_prefix}%") \
            .order("frame_index", desc=False) \
            .execute()
        return res.data or []
    except Exception as e:
        print(f"❌ Error fetching keyframes for '{video_prefix}': {e}")
        return []
