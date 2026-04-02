#!/usr/bin/env python3
"""
EditConfig — central configuration for the Zync editing pipeline.

Load from a JSON file or build programmatically.
Every module (transition_applier, text_overlay, color_grader,
speed_controller, audio_processor) reads from this single config so
all editing decisions live in one place.

JSON example (edit_config.json):
{
  "clips": [
    {"clip_path": "output_clips/video_scene_001.mp4", "speed": 1.0},
    {"clip_path": "output_clips/video_scene_002.mp4", "speed": 0.5,
     "freeze_at": 3.2, "freeze_duration": 1.5}
  ],
  "transition": "crossfade",
  "transition_duration": 0.8,
  "text_overlays": [
    {
      "text": "Summer Highlights 2024",
      "start_time": 0.5,
      "duration": 3.0,
      "position": "center",
      "font_size": 64,
      "font_color": "white",
      "bold": true,
      "animation": "fade_in"
    },
    {
      "text": "By Zync AI",
      "start_time": 0.5,
      "duration": 3.0,
      "position": "lower_third",
      "font_size": 32,
      "font_color": "white",
      "bold": false,
      "animation": "none"
    }
  ],
  "color_grade": {
    "brightness": 1.05,
    "contrast": 1.1,
    "saturation": 1.2,
    "temperature": "warm",
    "vignette": true,
    "grain": 0.15,
    "lut_file": null
  },
  "background_music": "music/lofi_chill.mp3",
  "music_volume": 0.35,
  "duck_audio": true,
  "duck_level": 0.15,
  "output_resolution": "1920x1080",
  "output_fps": 30,
  "output_path": "final_video.mp4"
}
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# ── Sub-configs ───────────────────────────────────────────────────────────────

@dataclass
class ClipEdit:
    """Per-clip editing parameters."""
    clip_path: str
    trim_start: float = 0.0          # Seconds into the source clip to start
    trim_end: Optional[float] = None  # Seconds into the source clip to end (None = full)
    speed: float = 1.0               # 0.25–4.0. Values <1 = slow-mo, >1 = fast
    freeze_at: Optional[float] = None        # Source timestamp to freeze
    freeze_duration: float = 0.0     # How many seconds to hold the freeze frame


@dataclass
class TextOverlay:
    """A single text overlay drawn onto the video."""
    text: str
    start_time: float = 0.0          # Seconds into final video when text appears
    duration: float = 4.0            # How long the text stays on screen
    position: str = "center"         # center | bottom | top | lower_third | upper_third
    font_size: int = 48
    font_color: str = "white"        # CSS colour name or 0xRRGGBB
    outline_color: str = "black"
    outline_width: int = 2
    bold: bool = False
    animation: str = "none"          # none | fade_in | fade_out | fade_both | typewriter


@dataclass
class ColorGrade:
    """Global colour-grading settings applied to the whole final video."""
    brightness: float = 1.0          # 0.5 (dark) – 1.5 (bright). 1.0 = no change
    contrast: float = 1.0            # 0.5 – 1.5
    saturation: float = 1.0          # 0.0 (greyscale) – 2.0 (vivid)
    temperature: str = "neutral"     # warm | cool | neutral
    vignette: bool = False           # Dark circular edge overlay
    grain: float = 0.0               # 0.0 (none) – 1.0 (heavy film grain)
    lut_file: Optional[str] = None   # Path to a .cube LUT file


@dataclass
class EditConfig:
    """Master config consumed by every editing module."""
    clips: List[ClipEdit] = field(default_factory=list)

    # Transitions
    transition: str = "cut"          # cut | crossfade | fadeblack | fadewhite |
                                     # wipeleft | wiperight | wipeup | wipedown |
                                     # dissolve | zoomin | slideleft | slideright
    transition_duration: float = 0.5 # Seconds of overlap for xfade transitions

    # Text
    text_overlays: List[TextOverlay] = field(default_factory=list)

    # Colour
    color_grade: Optional[ColorGrade] = None

    # Audio
    background_music: Optional[str] = None   # Path to MP3/WAV
    music_volume: float = 0.35               # 0.0–1.0
    duck_audio: bool = True                  # Lower music when speech is detected
    duck_level: float = 0.15                 # How quiet music gets while ducking (0–1)

    # Output
    output_resolution: str = "1920x1080"
    output_fps: int = 30
    output_path: str = "final_video.mp4"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_from_json(json_path: str) -> EditConfig:
    """Load an EditConfig from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = [ClipEdit(**c) for c in data.get("clips", [])]

    raw_grade = data.get("color_grade")
    color_grade = ColorGrade(**raw_grade) if raw_grade else None

    raw_overlays = data.get("text_overlays", [])
    text_overlays = [TextOverlay(**t) for t in raw_overlays]

    return EditConfig(
        clips=clips,
        transition=data.get("transition", "cut"),
        transition_duration=float(data.get("transition_duration", 0.5)),
        text_overlays=text_overlays,
        color_grade=color_grade,
        background_music=data.get("background_music"),
        music_volume=float(data.get("music_volume", 0.35)),
        duck_audio=bool(data.get("duck_audio", True)),
        duck_level=float(data.get("duck_level", 0.15)),
        output_resolution=data.get("output_resolution", "1920x1080"),
        output_fps=int(data.get("output_fps", 30)),
        output_path=data.get("output_path", "final_video.mp4"),
    )


def save_to_json(config: EditConfig, json_path: str):
    """Serialise an EditConfig back to a JSON file."""
    data = asdict(config)
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved edit config → {json_path}")


# ── Prompt parser ─────────────────────────────────────────────────────────────

# Keyword maps used by parse_prompt_to_config()
_TRANSITION_KEYWORDS = {
    "crossfade": ["crossfade", "cross fade", "dissolve", "blend"],
    "fadeblack": ["fade to black", "fade black", "blackout"],
    "fadewhite": ["fade to white", "fade white", "whiteout"],
    "wipeleft":  ["wipe left", "swipe left"],
    "wiperight": ["wipe right", "swipe right"],
    "zoomin":    ["zoom in", "zoom transition"],
    "slideleft": ["slide left"],
    "slideright":["slide right"],
    "cut":       ["hard cut", "jump cut", "no transition", "direct cut"],
}

_TEMPERATURE_KEYWORDS = {
    "warm":    ["warm", "golden", "sunset", "cozy", "orange", "summery"],
    "cool":    ["cool", "cold", "blue", "winter", "clinical", "icy"],
    "neutral": ["neutral", "natural", "realistic", "balanced"],
}

_STYLE_PRESETS = {
    # style_name: ColorGrade kwargs
    "cinematic": dict(contrast=1.2, saturation=0.85, brightness=0.95, vignette=True, grain=0.1),
    "vibrant":   dict(saturation=1.5, contrast=1.1, brightness=1.05),
    "vintage":   dict(saturation=0.7, contrast=0.9, brightness=0.95, grain=0.3, temperature="warm"),
    "dramatic":  dict(contrast=1.4, saturation=0.6, brightness=0.9, vignette=True),
    "bright":    dict(brightness=1.15, contrast=1.05, saturation=1.1),
    "dark":      dict(brightness=0.85, contrast=1.2, vignette=True),
    "bw":        dict(saturation=0.0, contrast=1.2),
    "warm":      dict(temperature="warm", saturation=1.1, brightness=1.05),
    "cool":      dict(temperature="cool", saturation=0.95, brightness=1.0),
}


def parse_prompt_to_config(
    prompt: str,
    ordered_clips: List[str],
    output_path: str = "final_video.mp4",
) -> EditConfig:
    """
    Build an EditConfig by parsing keywords from a free-text prompt.

    The prompt can contain instructions like:
      "cinematic style, crossfade transitions, slow motion on action clips,
       add title 'Summer 2024' at the start, warm colour grade, lo-fi music,
       1080p output"

    Args:
        prompt:        Free-text editing instruction.
        ordered_clips: Ordered list of clip paths (from prompt_matcher).
        output_path:   Where to write the final video.

    Returns:
        A populated EditConfig.
    """
    low = prompt.lower()

    # ── Transition ──────────────────────────────────────────────────────────
    transition = "cut"
    for name, keywords in _TRANSITION_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            transition = name
            break

    # Duration hint — "0.5 second transition" / "1s fade"
    dur_match = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec)?\s*(?:transition|fade|dissolve)", low)
    transition_duration = float(dur_match.group(1)) if dur_match else 0.6

    # ── Speed ───────────────────────────────────────────────────────────────
    default_speed = 1.0
    if re.search(r"slow[\s-]?mo(?:tion)?|slow down|half speed", low):
        default_speed = 0.5
    elif re.search(r"fast[\s-]?forward|speed up|double speed|2x", low):
        default_speed = 2.0
    elif re.search(r"timelapse|time[\s-]?lapse|4x", low):
        default_speed = 4.0

    clips = [ClipEdit(clip_path=p, speed=default_speed) for p in ordered_clips]

    # ── Colour grade ────────────────────────────────────────────────────────
    grade_kwargs: dict = {}

    # Named style preset takes priority
    for style, kwargs in _STYLE_PRESETS.items():
        if style in low:
            grade_kwargs.update(kwargs)
            break

    # Temperature override
    for temp, keywords in _TEMPERATURE_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            grade_kwargs["temperature"] = temp
            break

    # Numeric brightness hint — "increase brightness", "reduce brightness"
    if re.search(r"increase brightness|brighter|lighten", low):
        grade_kwargs.setdefault("brightness", 1.15)
    elif re.search(r"decrease brightness|darker|darken", low):
        grade_kwargs.setdefault("brightness", 0.85)

    if re.search(r"high contrast|more contrast|contrasty", low):
        grade_kwargs.setdefault("contrast", 1.3)
    if re.search(r"vignette", low):
        grade_kwargs["vignette"] = True
    if re.search(r"grain|grainy|film grain", low):
        grade_kwargs.setdefault("grain", 0.25)
    if re.search(r"black\s*(?:and|&)\s*white|grayscale|greyscale|monochrome", low):
        grade_kwargs["saturation"] = 0.0

    color_grade = ColorGrade(**grade_kwargs) if grade_kwargs else None

    # ── Text overlays ───────────────────────────────────────────────────────
    text_overlays: List[TextOverlay] = []

    # Quoted title — e.g. add title "My Video" or title: My Video
    title_match = re.search(
        r'(?:title|heading|intro text)[:\s]+["\']?([^"\']+?)["\']?'
        r'(?:\s*(?:at|for|duration)|\s*$|,)',
        low
    )
    if title_match:
        title_text = title_match.group(1).strip().title()
        text_overlays.append(TextOverlay(
            text=title_text,
            start_time=0.5,
            duration=3.5,
            position="center",
            font_size=72,
            font_color="white",
            bold=True,
            animation="fade_both",
        ))

    # Lower third — e.g. "lower third: John Doe"
    lt_match = re.search(
        r'lower[\s-]third[:\s]+["\']?([^"\']+?)["\']?(?:\s*,|\s*$)',
        low
    )
    if lt_match:
        text_overlays.append(TextOverlay(
            text=lt_match.group(1).strip().title(),
            start_time=1.0,
            duration=3.0,
            position="lower_third",
            font_size=36,
            font_color="white",
            bold=False,
            animation="fade_in",
        ))

    # Watermark / credit at end
    if re.search(r"watermark|credit|made with|created by", low):
        credit_match = re.search(
            r'(?:watermark|credit|made with|created by)[:\s]+["\']?([^"\']+?)["\']?'
            r'(?:\s*,|\s*$)',
            low
        )
        credit_text = credit_match.group(1).strip() if credit_match else "Made with Zync"
        text_overlays.append(TextOverlay(
            text=credit_text,
            start_time=-1,      # Special value: 2 s before end (handled in text_overlay.py)
            duration=2.5,
            position="bottom",
            font_size=28,
            font_color="white",
            animation="fade_both",
        ))

    # ── Audio ───────────────────────────────────────────────────────────────
    music_map = {
        "lofi":      "music/lofi_chill.mp3",
        "lo-fi":     "music/lofi_chill.mp3",
        "chill":     "music/lofi_chill.mp3",
        "cinematic": "music/cinematic.mp3",
        "epic":      "music/cinematic.mp3",
        "upbeat":    "music/upbeat.mp3",
        "energetic": "music/upbeat.mp3",
        "happy":     "music/upbeat.mp3",
    }
    background_music = None
    for kw, path in music_map.items():
        if kw in low:
            background_music = path
            break

    music_volume = 0.35
    vol_match = re.search(r"music\s+(?:volume\s+)?(?:at\s+)?(\d{1,3})\s*%", low)
    if vol_match:
        music_volume = int(vol_match.group(1)) / 100

    duck_audio = "no duck" not in low and "no music duck" not in low

    # ── Resolution ──────────────────────────────────────────────────────────
    resolution = "1920x1080"
    if "4k" in low or "2160" in low:
        resolution = "3840x2160"
    elif "720p" in low or "1280x720" in low:
        resolution = "1280x720"
    elif "vertical" in low or "portrait" in low or "9:16" in low:
        resolution = "1080x1920"
    elif "square" in low or "1:1" in low:
        resolution = "1080x1080"

    fps = 30
    if "60fps" in low or "60 fps" in low:
        fps = 60
    elif "24fps" in low or "24 fps" in low or "cinematic" in low:
        fps = 24

    return EditConfig(
        clips=clips,
        transition=transition,
        transition_duration=transition_duration,
        text_overlays=text_overlays,
        color_grade=color_grade,
        background_music=background_music,
        music_volume=music_volume,
        duck_audio=duck_audio,
        output_resolution=resolution,
        output_fps=fps,
        output_path=output_path,
    )


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick smoke-test of the prompt parser
    sample_clips = [
        "output_clips/video_scene_001.mp4",
        "output_clips/video_scene_002.mp4",
        "output_clips/video_scene_003.mp4",
    ]
    sample_prompt = (
        'cinematic style, crossfade transitions 0.8s, slow motion, '
        'add title "Summer Highlights 2024", lower third: John Doe, '
        'warm colour grade, vignette, lofi music at 30%, 1080p output'
    )
    cfg = parse_prompt_to_config(sample_prompt, sample_clips)
    save_to_json(cfg, "edit_config_preview.json")
    print("Generated config saved to edit_config_preview.json")
