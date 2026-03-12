import sys
sys.path.insert(0, "hanna_rep/src")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
from vlm_analyzer import extract_keyframes, process_video
from hanna_rep.src.similarity_filter import SimilarityFilter
from post_processing import merge_clips_from_urls
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INPUT_DIR = "input_video"
os.makedirs(INPUT_DIR, exist_ok=True)

# ---------- HELPERS ----------
def download_video(video_url: str) -> str:
    filename = video_url.split("/")[-1]
    local_path = os.path.join(INPUT_DIR, filename)
    print("Downloading video to:", local_path)
    response = requests.get(video_url, stream=True)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return local_path

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

class AssembleClip(BaseModel):
    id: str
    name: str
    videoUrl: str
    trimStart: float
    trimEnd: float
    fadeIn: Optional[float] = 0.0
    fadeOut: Optional[float] = 0.0

class ExportEffects(BaseModel):
    clips: List[AssembleClip]
    filter: Optional[str] = "None"
    overlayText: Optional[str] = ""
    music: Optional[str] = "None"
    musicVolume: Optional[float] = 0.4
    muteOriginal: Optional[bool] = False
    brightness: Optional[int] = 100
    contrast: Optional[int] = 100
    aspectRatio: Optional[str] = "original"
    captionX: Optional[float] = 50.0
    captionY: Optional[float] = 85.0
    cropOffset: Optional[int] = 50
    playbackRate: Optional[float] = 1.0

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
            print(f"Downloaded video: {temp_path}")

            keyframes_info = extract_keyframes(temp_path, video_name=video.name, clip_id=1)
            uploaded_frames = sum(1 for k in keyframes_info if k["status"])
            failed_frames = sum(1 for k in keyframes_info if not k["status"])
            print(f"Extracted {len(keyframes_info)} keyframes for {video.name}")

            process_video(video.name)
            print(f"Processed video: {video.name}")

            os.remove(temp_path)
            print("Temporary file deleted")

            results.append({
                "video_name": video.name,
                "total_frames": len(keyframes_info),
                "uploaded_frames": uploaded_frames,
                "failed_frames": failed_frames
            })

        except Exception as e:
            print("Error processing video:", str(e))
            results.append({"video_name": video.name, "error": str(e)})

    return {"status": "processed", "results": results}


@app.post("/videos/process")
def process_video_route(payload: ProcessPayload):
    print(f"📝 User prompt: {payload.prompt}")
    print(f"📦 Videos received: {[v.name for v in payload.videos] if payload.videos else 'none'}")

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

    print(f"🔍 Searching across {len(video_names)} video(s): {video_names}")

    sf = SimilarityFilter()
    all_clips = []

    for video_name in video_names:
        base_name = video_name.replace(".mp4", "")
        try:
            clips = sf.score_and_select(
                video_stem=base_name,
                user_prompt=payload.prompt,
                match_mode="any"
            )
            for clip in clips:
                clip["video_name"] = video_name
                clip["video_url"]  = video_url_map.get(video_name, "")
                clip["video_duration"] = video_duration_map.get(video_name) or 11
            all_clips.extend(clips)
            print(f"  ✅ {base_name}: {len(clips)} clips found")
        except Exception as e:
            print(f"  ⚠️ {base_name}: search failed — {e}")

    all_clips.sort(key=lambda c: c.get("score", 0), reverse=True)
    print(f"✅ Total clips found across all videos: {len(all_clips)}")

    return {
        "status": "processed",
        "selected_clips": all_clips
    }


@app.post("/videos/merge")
def merge_videos(clips: List[AssembleClip]):
    """Basic merge with no effects. Use /videos/export for full effects."""
    print("MERGE ENDPOINT HIT")
    try:
        output_file = os.path.join(OUTPUT_DIR, "merged_output.mp4")
        urls  = [clip.videoUrl for clip in clips]
        trims = [(clip.trimStart, clip.trimEnd) for clip in clips]
        merge_clips_from_urls(video_urls=urls, trims=trims, output_path=output_file)
        print("Merge completed:", output_file)
        return {"status": "merged", "output_file": "merged_output.mp4"}
    except Exception as e:
        print("MERGE ERROR:", str(e))
        return {"status": "error", "message": str(e)}


@app.post("/videos/export")
def export_video(payload: ExportEffects):
    print("EXPORT ENDPOINT HIT")
    print(f"  filter={payload.filter}, music={payload.music}, "
          f"overlay='{payload.overlayText}', speed={payload.playbackRate}")
    print(f"  clips: {len(payload.clips)}")

    try:
        output_file = os.path.join(OUTPUT_DIR, "export_output.mp4")

        urls      = [c.videoUrl  for c in payload.clips]
        trims     = [(c.trimStart, c.trimEnd) for c in payload.clips]
        fade_ins  = [c.fadeIn  or 0.0 for c in payload.clips]
        fade_outs = [c.fadeOut or 0.0 for c in payload.clips]

        merge_clips_from_urls(
            video_urls=urls,
            trims=trims,
            output_path=output_file,
            filter_name=payload.filter       or "None",
            overlay_text=payload.overlayText or "",
            music_name=payload.music         or "None",
            music_volume=payload.musicVolume or 0.4,
            mute_original=payload.muteOriginal or False,
            brightness=payload.brightness    or 100,
            contrast=payload.contrast        or 100,
            aspect_ratio=payload.aspectRatio or "original",
            caption_x=payload.captionX if payload.captionX is not None else 50.0,
            caption_y=payload.captionY if payload.captionY is not None else 85.0,
            crop_offset=payload.cropOffset if payload.cropOffset is not None else 50,
            fade_ins=fade_ins,
            fade_outs=fade_outs,
            playback_rate=payload.playbackRate or 1.0,
        )

        print("Export completed:", output_file)
        return {"status": "exported", "output_file": "export_output.mp4"}

    except Exception as e:
        import traceback
        print("EXPORT ERROR:", str(e))
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
