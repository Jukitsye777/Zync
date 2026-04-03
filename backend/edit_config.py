#!/usr/bin/env python3
"""
edit_config.py — dataclasses for the editing pipeline + a natural language
prompt parser that returns settings compatible with the Zync frontend.

Used by the /videos/edit-prompt endpoint in api.py.

Example prompts
---------------
  "sad emotional video, warm golden colour grade, lofi music at 30%,
   slow motion, add title 'Missing You', fade to black transitions"

  "cinematic travel highlights, crossfade transitions, upbeat music,
   title 'Tokyo 2024', 9:16 vertical, high contrast"

  "black and white documentary style, no music, dramatic lighting,
   lower third text 'Interview', normal speed"
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Dataclasses (used by audio_processor, color_grader, etc.) ────────────────

@dataclass
class ClipEdit:
    clip_path: str
    trim_start: float = 0.0
    trim_end: Optional[float] = None
    speed: float = 1.0
    freeze_at: Optional[float] = None
    freeze_duration: float = 0.0

@dataclass
class TextOverlay:
    text: str
    start_time: float = 0.0
    duration: float = 4.0
    position: str = "center"
    font_size: int = 48
    font_color: str = "white"
    outline_color: str = "black"
    outline_width: int = 2
    bold: bool = False
    animation: str = "none"

@dataclass
class ColorGrade:
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature: str = "neutral"
    vignette: bool = False
    grain: float = 0.0
    lut_file: Optional[str] = None

@dataclass
class EditConfig:
    clips: List[ClipEdit] = field(default_factory=list)
    transition: str = "cut"
    transition_duration: float = 0.5
    text_overlays: List[TextOverlay] = field(default_factory=list)
    color_grade: Optional[ColorGrade] = None
    background_music: Optional[str] = None
    music_volume: float = 0.35
    duck_audio: bool = True
    duck_level: float = 0.15
    output_resolution: str = "1920x1080"
    output_fps: int = 30
    output_path: str = "final_video.mp4"


# ── Keyword maps ──────────────────────────────────────────────────────────────

_FILTER_KEYWORDS = {
    "Black & White": [
        "black and white", "black & white", "b&w", "grayscale",
        "greyscale", "monochrome", "desaturated",
    ],
    "Cinematic": [
        "cinematic", "dramatic", "film look", "movie look",
        "hollywood", "teal and orange",
    ],
    "Warm": [
        "warm", "golden", "sunset", "cozy", "orange tones",
        "warm tones", "golden hour", "amber",
    ],
}

_MUSIC_KEYWORDS = {
    "Lo-fi": ["lofi", "lo-fi", "lo fi", "chill", "relaxing", "calm", "mellow", "sad"],
    "Cinematic": ["cinematic music", "orchestral", "epic music", "dramatic music",
                  "score", "instrumental"],
    "Upbeat": ["upbeat", "energetic", "happy", "fun", "hype", "fast music",
               "pop", "party"],
}

_TRANSITION_KEYWORDS = {
    "fadeblack":   ["fade to black", "fade black", "blackout", "fade out to black"],
    "crossfade":   ["crossfade", "cross fade", "dissolve", "blend"],
    "wipeleft":    ["wipe left", "swipe left"],
    "wiperight":   ["wipe right", "swipe right"],
    "slideleft":   ["slide in", "slide left", "slide transition"],
    "slideright":  ["slide right"],
    "zoomin":      ["zoom in", "zoom transition"],
    "fadewhite":   ["fade to white", "fade white", "whiteout"],
}


def parse_edit_prompt(prompt: str) -> dict:
    """
    Parse a free-text editing prompt and return a dict of frontend settings.

    Returned keys (all optional — only set if detected):
        filter        str   "Warm" | "Cinematic" | "Black & White"
        music         str   "Lo-fi" | "Cinematic" | "Upbeat"
        musicVolume   float 0.0 – 1.0
        muteOriginal  bool
        speed         float 0.25 – 4.0
        brightness    int   50 – 150  (100 = no change)
        contrast      int   50 – 150
        aspectRatio   str   "16:9" | "9:16" | "1:1" | "original"
        overlayText   str   caption / title text extracted from prompt
        transition    str   "fadeblack" | "crossfade" | "wipeleft" | ...

    Args:
        prompt: The user's full natural language prompt string.

    Returns:
        dict of detected settings (empty dict if nothing was detected).
    """
    low = prompt.lower()
    settings = {}

    # ── Filter / colour grade ─────────────────────────────────────────────
    for filter_name, keywords in _FILTER_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            settings["filter"] = filter_name
            break

    # ── Music ─────────────────────────────────────────────────────────────
    for music_name, keywords in _MUSIC_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            settings["music"] = music_name
            break

    # Music volume  — "music at 30%" / "music volume 50%"
    vol_match = re.search(
        r"music\s+(?:at\s+|volume\s+)?(\d{1,3})\s*%", low
    )
    if vol_match:
        settings["musicVolume"] = int(vol_match.group(1)) / 100

    # Mute original audio
    if re.search(r"\b(mute|no original audio|music only|silence original)\b", low):
        settings["muteOriginal"] = True

    # ── Speed ─────────────────────────────────────────────────────────────
    if re.search(r"slow[\s-]?mo(?:tion)?|slow down|half speed|0\.5x", low):
        settings["speed"] = 0.5
    elif re.search(r"(?:fast[\s-]?forward|speed up|double speed|2x)\b", low):
        settings["speed"] = 2.0
    elif re.search(r"timelapse|time[\s-]?lapse|4x", low):
        settings["speed"] = 4.0

    # Speed ramp hint (numeric) — "1.5x speed" / "speed 0.75"
    speed_match = re.search(
        r"(?:speed\s+)?(\d+(?:\.\d+)?)\s*x\s+speed|speed\s+(\d+(?:\.\d+)?)\s*x", low
    )
    if speed_match and "speed" not in settings:
        val = float(speed_match.group(1) or speed_match.group(2))
        if 0.25 <= val <= 4.0:
            settings["speed"] = val

    # ── Brightness / contrast ─────────────────────────────────────────────
    if re.search(r"\b(brighter|increase brightness|lighten|more light)\b", low):
        settings["brightness"] = 115
    elif re.search(r"\b(darker|dim|decrease brightness|darken|moody)\b", low):
        settings["brightness"] = 85

    if re.search(r"\b(high contrast|contrasty|more contrast|punchy)\b", low):
        settings["contrast"] = 130
    elif re.search(r"\b(low contrast|soft|matte|flat)\b", low):
        settings["contrast"] = 80

    # ── Aspect ratio ─────────────────────────────────────────────────────
    if re.search(r"\b(vertical|portrait|9:16|tiktok|reels|shorts)\b", low):
        settings["aspectRatio"] = "9:16"
    elif re.search(r"\b(square|1:1|instagram post)\b", low):
        settings["aspectRatio"] = "1:1"
    elif re.search(r"\b(16:9|widescreen|landscape|youtube|horizontal)\b", low):
        settings["aspectRatio"] = "16:9"

    # ── Caption / title text ──────────────────────────────────────────────
    # Match:  title "My Text"  /  text "My Text"  /  caption 'My Text'
    #         add title My Text  /  text: My Text
    title_match = re.search(
        r'(?:title|caption|text|overlay|add text|label)'
        r'[\s:]+["\']([^"\']{1,60})["\']',
        prompt,
        re.IGNORECASE,
    )
    if title_match:
        settings["overlayText"] = title_match.group(1).strip()
    else:
        # Unquoted: "title My Video Title" (take up to 5 words after keyword)
        unquoted = re.search(
            r'(?:add\s+)?(?:title|caption)\s+([A-Za-z0-9 ]{2,40}?)(?:\s*,|\s*$)',
            prompt,
            re.IGNORECASE,
        )
        if unquoted:
            settings["overlayText"] = unquoted.group(1).strip()

    # ── Transition ────────────────────────────────────────────────────────
    for trans_name, keywords in _TRANSITION_KEYWORDS.items():
        if any(kw in low for kw in keywords):
            settings["transition"] = trans_name
            break

    return settings


def describe_settings(settings: dict) -> str:
    """Return a human-readable summary of the parsed settings."""
    if not settings:
        return "No editing keywords detected — using current settings."

    parts = []
    if "filter" in settings:
        parts.append(f"Filter: {settings['filter']}")
    if "music" in settings:
        vol = settings.get("musicVolume", 0.4)
        parts.append(f"Music: {settings['music']} @ {int(vol * 100)}%")
    if "speed" in settings:
        parts.append(f"Speed: {settings['speed']}x")
    if "brightness" in settings:
        parts.append(f"Brightness: {settings['brightness']}%")
    if "contrast" in settings:
        parts.append(f"Contrast: {settings['contrast']}%")
    if "aspectRatio" in settings:
        parts.append(f"Ratio: {settings['aspectRatio']}")
    if "overlayText" in settings:
        parts.append(f"Caption: \"{settings['overlayText']}\"")
    if "transition" in settings:
        parts.append(f"Transition: {settings['transition']}")
    if settings.get("muteOriginal"):
        parts.append("Mute original audio")

    return " · ".join(parts)
