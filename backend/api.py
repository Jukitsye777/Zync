import sys
sys.path.insert(0, "hanna_rep/src")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import re
import json
import requests as http_requests
from vlm_analyzer import extract_keyframes, process_video
from hanna_rep.src.similarity_filter import SimilarityFilter
from post_processing import merge_clips_from_urls
from fastapi.staticfiles import StaticFiles
from edit_config import parse_edit_prompt, describe_settings

app = FastAPI()
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.middleware("http")
async def no_cache_outputs(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/outputs/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_DIR = "input_video"
os.makedirs(INPUT_DIR, exist_ok=True)

VALID_MUSIC   = ["None", "Cinematic", "Upbeat", "Sad", "Energetic"]
VALID_FILTERS = ["None", "Sunset", "Happy", "Sad", "Dramatic", "Vintage",
                 "Night", "Cinematic", "Black & White", "Teal & Orange", "Fade", "Neon"]
VALID_TRANS   = ["none", "fade", "wipe", "zoom", "flash", "glitch", "blur", "dip"]


# ---------- SINGLE OLLAMA CALL: caption + settings together ----------
def analyze_prompt_with_ollama(prompt: str) -> dict:
    """
    Single llama3.2 call that returns BOTH caption and settings.
    Avoids running Ollama twice simultaneously.
    Returns: {"caption": "...", "filter": "...", "music": "...", "transition": "..."}
    """
    try:
        response = http_requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": (
                    "You are a video editing AI. Analyze this scene description and return editing settings + caption.\n"
                    "Scene: " + prompt + "\n\n"
                    "Return ONLY a JSON object with these exact keys:\n"
                    "- caption: 2-5 word evocative reel title (poetic, emotional, NOT literal)\n"
                    "- filter: one of [None, Sunset, Happy, Sad, Dramatic, Vintage, Night, Cinematic, Black & White, Teal & Orange, Fade, Neon]\n"
                    "- music: one of [None, Cinematic, Upbeat, Sad, Energetic]\n"
                    "- transition: one of [none, fade, wipe, zoom, flash, glitch, blur, dip]\n\n"
                    "Caption rules: poetic, evocative, NOT the object names. Optional emoji at end.\n"
                    "yellow bus stairway -> Every Road Taken\n"
                    "sad plants -> Quietly Blooming\n"
                    "golden hour beach -> Chasing the Light\n\n"
                    "Filter/music rules — match MOOD and FEELING:\n"
                    "joyful/excited/fun/cheerful/happy -> Happy + Upbeat\n"
                    "sad/lonely/grief/heartbreak/melancholic/gloomy -> Sad + Sad\n"
                    "intense/powerful/dark/gritty/bold -> Dramatic + Cinematic\n"
                    "dreamy/soft/hazy/peaceful/calm -> Fade + Cinematic\n"
                    "nostalgic/retro/vintage/throwback/classic -> Vintage + Cinematic\n"
                    "night/midnight/cold/dark blue -> Night + Cinematic\n"
                    "futuristic/neon/cyber/synthwave/glowing -> Neon + Upbeat\n"
                    "golden/warm/sunset/sunrise -> Sunset + Cinematic\n"
                    "cinematic/film/epic/dramatic -> Cinematic + Cinematic\n"
                    "default transition: fade\n\n"
                    "Example: {\"caption\": \"Every Road Taken\", \"filter\": \"Sad\", \"music\": \"Sad\", \"transition\": \"fade\"}\n"
                    "Reply with ONLY the JSON. No explanation, no markdown, no backticks."
                ),
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "num_predict": 60,
                },
            },
            timeout=30,
        )
        raw = response.json().get("response", "").strip()
        match = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            result = {}
            caption = data.get("caption", "").strip().strip('"').strip("'")
            if caption and len(caption) <= 40:
                result["caption"] = caption
            if data.get("filter")     in VALID_FILTERS: result["filter"]     = data["filter"]
            if data.get("music")      in VALID_MUSIC:   result["music"]      = data["music"]
            if data.get("transition") in VALID_TRANS:   result["transition"] = data["transition"]
            print(f"  🧠 Ollama result: {result}")
            return result
        print(f"  ⚠️ Could not parse Ollama JSON from: {raw[:80]}")
        return {}
    except Exception as e:
        print(f"  ⚠️ Ollama analysis failed: {e}")
        return {}


# ---------- HELPERS ----------
def download_video(video_url: str) -> str:
    filename = video_url.split("/")[-1]
    local_path = os.path.join(INPUT_DIR, filename)
    print("Downloading video to:", local_path)
    response = http_requests.get(video_url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path


def search_clips_for_video(sf: SimilarityFilter, video_name: str, prompt: str) -> list:
    stems_to_try = [video_name]
    name_without_ext = video_name.replace(".mp4", "").replace(".mov", "").replace(".avi", "")
    if name_without_ext != video_name:
        stems_to_try.append(name_without_ext)

    for stem in stems_to_try:
        try:
            clips = sf.score_and_select(video_stem=stem, user_prompt=prompt, match_mode="any")
            if clips:
                print(f"  ✅ Found {len(clips)} clip(s) using stem='{stem}'")
                return clips
            print(f"  ⚠️  No clips found using stem='{stem}', trying next...")
        except Exception as e:
            print(f"  ⚠️  search failed for stem='{stem}' — {e}")

    print(f"  ❌ No clips found for video '{video_name}'")
    return []


# ---------- MODELS ----------
class Video(BaseModel):
    id: str
    name: str
    url: str
    duration: Optional[float] = None
    class Config:
        extra = "ignore"

class VideosPayload(BaseModel):
    videos: List[Video]

class ProcessPayload(BaseModel):
    prompt: str
    videos: Optional[List[Video]] = None
    video_url: Optional[str] = None
    video_name: Optional[str] = None
    class Config:
        extra = "ignore"

class EditPromptPayload(BaseModel):
    prompt: str
    videos: Optional[List[Video]] = None
    class Config:
        extra = "ignore"


# ---------- ROUTES ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/videos")
def receive_videos(payload: VideosPayload):
    print("Received videos:", payload.videos)
    results = []
    for video in payload.videos:
        try:
            temp_path = download_video(video.url)
            keyframes_info = extract_keyframes(temp_path, video_name=video.name, clip_id=1)
            uploaded_frames = sum(1 for k in keyframes_info if k["status"])
            failed_frames = sum(1 for k in keyframes_info if not k["status"])
            print(f"Extracted {len(keyframes_info)} keyframes for {video.name}")
            process_video(video.name)
            os.remove(temp_path)
            results.append({
                "video_name": video.name,
                "total_frames": len(keyframes_info),
                "uploaded_frames": uploaded_frames,
                "failed_frames": failed_frames,
            })
        except Exception as e:
            print("Error processing video:", str(e))
            results.append({"video_name": video.name, "error": str(e)})
    return {"status": "processed", "results": results}


@app.post("/videos/process")
def process_video_route(payload: ProcessPayload):
    print(f"📝 User prompt: {payload.prompt}")
    if payload.videos and len(payload.videos) > 0:
        video_names = [v.name for v in payload.videos]
        video_url_map = {v.name: v.url for v in payload.videos}
        video_duration_map = {v.name: v.duration for v in payload.videos}
    elif payload.video_name:
        video_names = [payload.video_name]
        video_url_map = {payload.video_name: payload.video_url}
        video_duration_map = {payload.video_name: None}
    else:
        return {"status": "error", "message": "No videos provided"}

    sf = SimilarityFilter()
    all_clips = []
    for video_name in video_names:
        clips = search_clips_for_video(sf, video_name, payload.prompt)
        for clip in clips:
            clip["video_name"] = video_name
            clip["video_url"] = video_url_map.get(video_name, "")
            clip["video_duration"] = video_duration_map.get(video_name) or 11
        all_clips.extend(clips)

    all_clips.sort(key=lambda c: c.get("score", 0), reverse=True)
    return {"status": "processed", "selected_clips": all_clips}


@app.post("/videos/edit-prompt")
def edit_prompt_route(payload: EditPromptPayload):
    print(f"\n🎬 EDIT-PROMPT: '{payload.prompt}'")

    # 1. Parse editing settings from prompt keywords (edit_config)
    edit_settings = parse_edit_prompt(payload.prompt)
    summary = describe_settings(edit_settings)
    print(f"  Parsed settings: {summary}")

    # 2. Single Ollama call — gets caption + filter + music + transition together
    ollama_result = analyze_prompt_with_ollama(payload.prompt)
    if ollama_result.get("caption"):
        edit_settings["overlayText"] = ollama_result["caption"]
    for key in ["filter", "music", "transition"]:
        if ollama_result.get(key):
            edit_settings[key] = ollama_result[key]
    if not ollama_result:
        print("  ℹ️ Ollama unavailable — frontend inference will handle settings")

    # 4. Run semantic clip matching
    selected_clips = []
    if payload.videos:
        video_url_map = {v.name: v.url for v in payload.videos}
        video_duration_map = {v.name: v.duration for v in payload.videos}
        sf = SimilarityFilter()
        for video in payload.videos:
            print(f"\n  🔍 Searching in video: '{video.name}'")
            clips = search_clips_for_video(sf, video.name, payload.prompt)
            for clip in clips:
                clip["video_name"] = video.name
                clip["video_url"] = video_url_map.get(video.name, "")
                clip["video_duration"] = video_duration_map.get(video.name) or 11
            selected_clips.extend(clips)
            print(f"  → {len(clips)} clip(s) added from '{video.name}'")

        selected_clips.sort(key=lambda c: c.get("clip_id", 0))

    # 5. Sanitize music — remove invalid options
    if edit_settings.get("music") and edit_settings["music"] not in VALID_MUSIC:
        print(f"  ⚠️ Removing invalid music '{edit_settings['music']}'")
        del edit_settings["music"]

    print(f"\n✅ Total clips: {len(selected_clips)}, ids: {sorted(set(c.get('clip_id') for c in selected_clips))}")
    return {
        "status": "processed",
        "selected_clips": selected_clips,
        "edit_settings": edit_settings,
        "settings_summary": summary,
    }


# ── Merge: raw JSON to avoid Pydantic 422 ─────────────────────────────────────
@app.post("/videos/merge")
async def merge_videos(request: Request):
    try:
        body = await request.json()
        print(f"📦 MERGE body type: {type(body)}")

        if isinstance(body, dict):
            clips         = body.get("clips", [])
            transition    = body.get("transition", "fade")
            mute_original = body.get("muteOriginal", False)
        elif isinstance(body, list):
            clips         = body
            transition    = "fade"
            mute_original = False
        else:
            return {"status": "error", "message": f"Unexpected body type: {type(body)}"}

        print(f"  clips={len(clips)}, transition='{transition}', muteOriginal={mute_original}")
        if clips:
            print(f"  First clip keys: {list(clips[0].keys())}")

    except Exception as e:
        print(f"❌ Body parse error: {e}")
        return {"status": "error", "message": f"Body parse error: {e}"}

    try:
        output_file = os.path.join(OUTPUT_DIR, "merged_output.mp4")
        urls      = [c["videoUrl"] for c in clips]
        trims     = [(c["trimStart"], c["trimEnd"]) for c in clips]
        fade_ins  = [c.get("fadeIn", 0.0) or 0.0 for c in clips]
        fade_outs = [c.get("fadeOut", 0.0) or 0.0 for c in clips]

        merge_clips_from_urls(
            video_urls=urls,
            trims=trims,
            output_path=output_file,
            fade_ins=fade_ins,
            fade_outs=fade_outs,
            transition=transition,
            mute_original=mute_original,
        )
        print("✅ Merge completed:", output_file)
        return {"status": "merged", "output_file": "merged_output.mp4"}

    except Exception as e:
        import traceback
        print("❌ MERGE ERROR:", str(e))
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


# ── Export: raw JSON ──────────────────────────────────────────────────────────
@app.post("/videos/export")
async def export_video(request: Request):
    try:
        body = await request.json()
        clips         = body.get("clips", [])
        transition    = body.get("transition", "fade")
        filter_name   = body.get("filter", "None")
        overlay_text  = body.get("overlayText", "")
        music_name    = body.get("music", "None")
        music_volume  = body.get("musicVolume", 0.4)
        mute_original = body.get("muteOriginal", False)
        brightness    = body.get("brightness", 100)
        contrast      = body.get("contrast", 100)
        aspect_ratio  = body.get("aspectRatio", "original")
        caption_x     = body.get("captionX", 50.0)
        caption_y     = body.get("captionY", 85.0)
        crop_offset   = body.get("cropOffset", 50)
        playback_rate = body.get("playbackRate", 1.0)

        print(f"EXPORT — filter={filter_name}, transition={transition}, music={music_name}, clips={len(clips)}")

    except Exception as e:
        print(f"❌ Export body parse error: {e}")
        return {"status": "error", "message": str(e)}

    try:
        output_file = os.path.join(OUTPUT_DIR, "export_output.mp4")
        urls      = [c["videoUrl"] for c in clips]
        trims     = [(c["trimStart"], c["trimEnd"]) for c in clips]
        fade_ins  = [c.get("fadeIn", 0.0) or 0.0 for c in clips]
        fade_outs = [c.get("fadeOut", 0.0) or 0.0 for c in clips]

        merge_clips_from_urls(
            video_urls=urls,
            trims=trims,
            output_path=output_file,
            filter_name=filter_name,
            overlay_text=overlay_text,
            music_name=music_name,
            music_volume=music_volume,
            mute_original=mute_original,
            brightness=brightness,
            contrast=contrast,
            aspect_ratio=aspect_ratio,
            caption_x=caption_x,
            caption_y=caption_y,
            crop_offset=crop_offset,
            fade_ins=fade_ins,
            fade_outs=fade_outs,
            playback_rate=playback_rate,
            transition=transition,
        )
        print("✅ Export completed:", output_file)
        return {"status": "exported", "output_file": "export_output.mp4"}

    except Exception as e:
        import traceback
        print("❌ EXPORT ERROR:", str(e))
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
