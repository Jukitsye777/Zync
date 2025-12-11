# clip_selector.py
"""
This module maps the selected keyframe rows to actual clip time ranges and
normalizes the output for the editor. It expects either:
 - a separate JSON file with mapping frame_index -> (start, end) per video_stem
 OR
 - the clip_id stored in DB corresponds to subclip index and you can compute times
If Vision Engineer has provided a metadata JSON, use that.
"""

import json
from pathlib import Path

METADATA_PATH = "data/clip_metadata.json"   # ask Vision Engineer to provide or generate

def load_metadata(path=METADATA_PATH):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        # file missing or empty -> just return empty dict
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # file exists but has invalid JSON
        return {}

def map_selected_to_clips(selected, metadata=None):
    """
    Input: selected list from SimilarityFilter (clip_id, video_name, frame_index, score)
    Output: list of dicts with clip_id, video_name, start_time, end_time, frame_index, score
    """
    if metadata is None:
        metadata = load_metadata()

    result = []
    for item in selected:
        video_stem = item["video_name"].split("_subclip_")[0]
        clip_id = item["clip_id"]
        # metadata keyed by video_stem, then by clip_id or subclip name
        video_meta = metadata.get(video_stem, {})
        clip_key = str(clip_id) if str(clip_id) in video_meta else f"subclip_{clip_id}"
        clip_info = video_meta.get(clip_key)
        if clip_info:
            result.append({
                "clip_id": clip_id,
                "video_name": video_stem,
                "start_time": clip_info.get("start"),
                "end_time": clip_info.get("end"),
                "frame_index": item["frame_index"],
                "score": item["score"]
            })
        else:
            # fallback: keep clip_id without start/end so editor can still retrieve using subclip name
            result.append({
                "clip_id": clip_id,
                "video_name": item["video_name"],
                "start_time": None,
                "end_time": None,
                "frame_index": item["frame_index"],
                "score": item["score"]
            })
    # deduplicate by clip_id (keep highest score) — though SimilarityFilter already did
    unique = {}
    for r in result:
        key = (r["video_name"], r["clip_id"])
        if key not in unique or r["score"] > unique[key]["score"]:
            unique[key] = r
    return list(unique.values())
