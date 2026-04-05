from moviepy.editor import (
    VideoFileClip, AudioFileClip, CompositeAudioClip,
    concatenate_videoclips
)
import numpy as np
import os
import shutil
import requests
import subprocess
from moviepy.config import get_setting

TEMP_DIR = "temp_clips"

CAPTION_FONT       = "Arial"
CAPTION_SIZE       = 36
CAPTION_COLOR      = "white"
CAPTION_BORDER     = 2
CAPTION_BORDER_CLR = "black"

# ── Map our transition names to FFmpeg xfade transition names ─────────────────
XFADE_MAP = {
    "fade":   "fade",
    "wipe":   "wipeleft",
    "zoom":   "zoomin",
    "flash":  "fadewhite",
    "glitch": "pixelize",
    "blur":   "radial",
    "dip":    "fade",       # dip to black = fade through black
    "none":   None,         # hard cut — no xfade
}
XFADE_DURATION = 0.4        # seconds per transition


def apply_filter(clip, filter_name: str):
    if not filter_name or filter_name == "None":
        return clip
    f = filter_name.strip().lower()
    if f == "black & white":
        return clip.fx(__import__('moviepy.video.fx.blackwhite', fromlist=['blackwhite']).blackwhite)
    if f == "warm":
        def warm_filter(frame):
            frame = frame.astype(np.float32)
            frame[:, :, 0] = np.clip(frame[:, :, 0] * 1.15, 0, 255)
            frame[:, :, 1] = np.clip(frame[:, :, 1] * 1.05, 0, 255)
            frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.88, 0, 255)
            return frame.astype(np.uint8)
        return clip.fl_image(warm_filter)
    if f == "cinematic":
        def cinematic_filter(frame):
            frame = frame.astype(np.float32)
            frame = np.clip((frame - 128) * 1.2 + 128, 0, 255)
            frame = np.clip(frame * 0.92, 0, 255)
            gray = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
            gray = gray[:, :, np.newaxis]
            frame = np.clip(frame * 0.85 + gray * 0.15, 0, 255)
            return frame.astype(np.uint8)
        return clip.fl_image(cinematic_filter)
    return clip


def apply_clip_fades(clip, fade_in: float, fade_out: float):
    MAX_FADE = 0.5
    fi = min(fade_in, MAX_FADE)
    fo = min(fade_out, MAX_FADE)
    if fi > 0:
        clip = clip.fadein(fi)
    if fo > 0:
        clip = clip.fadeout(fo)
    return clip


def apply_audio_fades(clip, fade_in: float, fade_out: float):
    MAX_FADE = 0.5
    fi = min(fade_in, MAX_FADE)
    fo = min(fade_out, MAX_FADE)
    if clip.audio is None:
        return clip
    if fi > 0:
        clip = clip.audio_fadein(fi)
    if fo > 0:
        clip = clip.audio_fadeout(fo)
    return clip


def download_video(url, filename):
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with open(filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def get_ffmpeg():
    try:
        return get_setting("FFMPEG_BINARY")
    except Exception:
        return "ffmpeg"


def apply_xfade_transitions(input_paths: list, output_path: str, transition: str) -> bool:
    """
    Use FFmpeg xfade filter to apply transitions between pre-trimmed clip files.
    Returns True if successful, False if it should fall back to simple concat.
    """
    xfade_name = XFADE_MAP.get(transition)
    if not xfade_name or len(input_paths) < 2:
        return False

    ffmpeg_bin = get_ffmpeg()
    n = len(input_paths)

    # Get durations of each clip so we can calculate xfade offsets
    durations = []
    for path in input_paths:
        result = subprocess.run(
            [ffmpeg_bin, "-i", path, "-f", "null", "-"],
            capture_output=True, text=True
        )
        dur = 0.0
        for line in result.stderr.split("\n"):
            if "Duration:" in line:
                try:
                    t = line.split("Duration:")[1].split(",")[0].strip()
                    h, m, s = t.split(":")
                    dur = float(h) * 3600 + float(m) * 60 + float(s)
                except Exception:
                    pass
        durations.append(dur)

    if any(d == 0 for d in durations):
        print("⚠️ Could not determine clip durations for xfade — falling back to concat")
        return False

    # Build FFmpeg filter_complex for N clips with xfade between each pair
    # Input flags
    inputs = []
    for p in input_paths:
        inputs += ["-i", p]

    # Build filter_complex
    # Each xfade offset = sum of previous clip durations minus accumulated transition overlap
    filter_parts = []
    current_label = "[0:v]"
    audio_labels = ["[0:a]"]
    accumulated_offset = 0.0

    for i in range(1, n):
        accumulated_offset += durations[i - 1] - XFADE_DURATION
        next_label = f"[v{i}]" if i < n - 1 else "[vout]"
        filter_parts.append(
            f"{current_label}[{i}:v]xfade=transition={xfade_name}"
            f":duration={XFADE_DURATION:.2f}"
            f":offset={accumulated_offset:.2f}{next_label}"
        )
        current_label = next_label
        audio_labels.append(f"[{i}:a]")

    # Audio: concat all audio streams
    audio_concat = "".join(audio_labels) + f"concat=n={n}:v=0:a=1[aout]"
    filter_parts.append(audio_concat)

    filter_complex = ";".join(filter_parts)

    cmd = [ffmpeg_bin, "-y"] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-codec:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-codec:a", "aac",
        "-movflags", "+faststart",
        output_path,
    ]

    print(f"🎬 Applying '{transition}' transitions via FFmpeg xfade...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ xfade failed:\n{result.stderr[-600:]}")
        return False

    print(f"✅ Transitions applied successfully")
    return True


def merge_clips_from_urls(
    video_urls: list,
    trims: list,
    output_path: str = "final_output.mp4",
    filter_name: str = "None",
    overlay_text: str = "",
    music_name: str = "None",
    music_volume: float = 0.4,
    mute_original: bool = False,
    brightness: int = 100,
    contrast: int = 100,
    aspect_ratio: str = "original",
    caption_x: float = 50.0,
    caption_y: float = 85.0,
    crop_offset: int = 50,
    fade_ins: list = None,
    fade_outs: list = None,
    playback_rate: float = 1.0,
    transition: str = "fade",
):
    os.makedirs(TEMP_DIR, exist_ok=True)
    fade_ins  = fade_ins  or [0.0] * len(video_urls)
    fade_outs = fade_outs or [0.0] * len(video_urls)

    local_paths = []
    trimmed_paths = []
    clips = []

    try:
        # ── 1. Download ───────────────────────────────────────────────────────
        for index, url in enumerate(video_urls):
            local_path = os.path.join(TEMP_DIR, f"clip_{index}.mp4")
            download_video(url, local_path)
            local_paths.append(local_path)

        # ── 2. Load, trim, filter, fades — write trimmed clips to disk ────────
        for index, path in enumerate(local_paths):
            clip = VideoFileClip(path)
            start, end = trims[index]
            clip = clip.subclip(start, end)
            if mute_original:
                clip = clip.without_audio()
            clip = apply_filter(clip, filter_name)
            fi = fade_ins[index]  if index < len(fade_ins)  else 0.0
            fo = fade_outs[index] if index < len(fade_outs) else 0.0
            clip = apply_clip_fades(clip, fi, fo)
            clip = apply_audio_fades(clip, fi, fo)
            clips.append(clip)

        # ── 3. Normalize resolution ───────────────────────────────────────────
        if clips:
            target_w = clips[0].w
            target_h = clips[0].h
            target_w += target_w % 2
            target_h += target_h % 2

            normalized = []
            for c in clips:
                if c.w != target_w or c.h != target_h:
                    scale = max(target_w / c.w, target_h / c.h)
                    new_w = int(c.w * scale) + (int(c.w * scale) % 2)
                    new_h = int(c.h * scale) + (int(c.h * scale) % 2)
                    c = c.resize((new_w, new_h))
                    t = crop_offset / 100.0
                    x_start = int((new_w - target_w) * t)
                    y_start = int((new_h - target_h) * t)
                    c = c.crop(x1=x_start, y1=y_start,
                               x2=x_start + target_w, y2=y_start + target_h)
                normalized.append(c)
            clips = normalized

        # ── 4. Write each trimmed clip to disk for xfade processing ───────────
        for index, clip in enumerate(clips):
            trimmed_path = os.path.join(TEMP_DIR, f"trimmed_{index}.mp4")
            clip.write_videofile(
                trimmed_path,
                codec="libx264", audio_codec="aac",
                fps=24, preset="fast",
                ffmpeg_params=["-crf", "23"],
                logger=None,
            )
            trimmed_paths.append(trimmed_path)

        for clip in clips:
            try: clip.close()
            except: pass
        clips = []

        # ── 5. Apply transitions via FFmpeg xfade ─────────────────────────────
        intermediate = output_path.replace(".mp4", "_raw.mp4")
        xfade_ok = False

        if len(trimmed_paths) > 1 and transition and transition != "none":
            xfade_ok = apply_xfade_transitions(trimmed_paths, intermediate, transition)

        if not xfade_ok:
            # Fall back: simple concat with MoviePy
            print(f"📎 Using simple concatenation (transition='{transition}')")
            reload_clips = [VideoFileClip(p) for p in trimmed_paths]
            final_video = concatenate_videoclips(reload_clips, method="compose")
            final_video.write_videofile(
                intermediate,
                codec="libx264", audio_codec="aac",
                fps=24, preset="fast",
                ffmpeg_params=["-crf", "23"],
            )
            final_video.close()
            for c in reload_clips:
                try: c.close()
                except: pass

        # ── 6. Resolve music path ─────────────────────────────────────────────
        resolved_music_path = None
        print(f"🎵 Music requested: '{music_name}'")
        if music_name and music_name != "None":
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            music_filename = music_name.lower().replace(' ', '_').replace('-', '_') + ".mp3"
            resolved_music_path = os.path.join(BASE_DIR, "music", music_filename)
            print(f"🎵 Music path: {resolved_music_path}")
            print(f"🎵 File exists: {os.path.exists(resolved_music_path)}")
            if not os.path.exists(resolved_music_path):
                print(f"⚠️ Music file not found — skipping")
                resolved_music_path = None

        # ── 7. FFmpeg post-pass: aspect ratio, brightness, speed, caption ─────
        needs_pass = (
            bool(overlay_text and overlay_text.strip())
            or brightness != 100
            or contrast != 100
            or (aspect_ratio and aspect_ratio != "original")
            or playback_rate != 1.0
        )

        if needs_pass:
            ffmpeg_bin = get_ffmpeg()
            vf_parts = []

            if aspect_ratio and aspect_ratio != "original":
                t = crop_offset / 100.0
                if aspect_ratio == "16:9":
                    crop_f = f"crop=iw:iw*9/16:0:(ih-iw*9/16)*{t:.4f}"
                elif aspect_ratio == "9:16":
                    crop_f = f"crop=ih*9/16:ih:(iw-ih*9/16)*{t:.4f}:0"
                elif aspect_ratio == "1:1":
                    crop_f = (f"crop=min(iw\\,ih):min(iw\\,ih):"
                              f"(iw-min(iw\\,ih))*{t:.4f}:(ih-min(iw\\,ih))*{t:.4f}")
                else:
                    crop_f = None
                if crop_f:
                    vf_parts.append(crop_f)

            eq_contrast   = contrast / 100.0
            eq_brightness = (brightness - 100) / 100.0
            if brightness != 100 or contrast != 100:
                vf_parts.append(f"eq=brightness={eq_brightness:.2f}:contrast={eq_contrast:.2f}")

            if playback_rate != 1.0:
                pts = 1.0 / playback_rate
                vf_parts.append(f"setpts={pts:.4f}*PTS")

            if overlay_text and overlay_text.strip():
                safe = (overlay_text
                    .replace("\\", "\\\\")
                    .replace("'",  "\u2019")
                    .replace(":",  "\\:")
                    .replace("%",  "\\%"))
                fx = f"(w*{caption_x/100:.4f}-text_w/2)"
                fy = f"(h*{caption_y/100:.4f}-text_h/2)"
                vf_parts.append(
                    f"drawtext=text='{safe}':font='{CAPTION_FONT}'"
                    f":fontsize={CAPTION_SIZE}:fontcolor={CAPTION_COLOR}"
                    f":borderw={CAPTION_BORDER}:bordercolor={CAPTION_BORDER_CLR}"
                    f":x={fx}:y={fy}"
                )

            vf_string = ",".join(vf_parts) if vf_parts else "null"

            af_parts = []
            if playback_rate != 1.0:
                rate = playback_rate
                while rate > 2.0:
                    af_parts.append("atempo=2.0"); rate /= 2.0
                while rate < 0.5:
                    af_parts.append("atempo=0.5"); rate /= 2.0
                af_parts.append(f"atempo={rate:.4f}")

            cmd = [get_ffmpeg(), "-y", "-i", intermediate, "-vf", vf_string]
            if af_parts:
                cmd += ["-af", ",".join(af_parts)]
            cmd += ["-codec:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-codec:a", "aac", "-movflags", "+faststart", output_path]

            print(f"🎬 FFmpeg post-pass...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ Post-pass failed:\n{result.stderr[-800:]}")
                shutil.copy(intermediate, output_path)
            else:
                print("✅ Post-pass complete")
        else:
            ffmpeg_bin = get_ffmpeg()
            cmd = [ffmpeg_bin, "-y", "-i", intermediate,
                   "-codec", "copy", "-movflags", "+faststart", output_path]
            r = subprocess.run(cmd, capture_output=True)
            if r.returncode != 0:
                shutil.move(intermediate, output_path)

        # ── 8. Mix music ──────────────────────────────────────────────────────
        if resolved_music_path:
            ffmpeg_bin = get_ffmpeg()
            music_out = output_path.replace(".mp4", "_music.mp4")
            cmd = [
                ffmpeg_bin, "-y",
                "-i", output_path,
                "-stream_loop", "-1",
                "-i", resolved_music_path,
                "-filter_complex",
                f"[0:a]aformat=fltp:44100:stereo,volume=1.0[va];"
                f"[1:a]aformat=fltp:44100:stereo,volume={music_volume:.2f}[ma];"
                "[va][ma]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
                "-codec:v", "copy",
                "-codec:a", "aac",
                "-movflags", "+faststart",
                music_out,
            ]
            print("🎵 Mixing music...")
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                shutil.move(music_out, output_path)
                print("✅ Music mixed in")
            else:
                print(f"⚠️ Music mix failed: {r.stderr[-400:]}")
                if os.path.exists(music_out):
                    os.remove(music_out)

        return output_path

    finally:
        for clip in clips:
            try: clip.close()
            except: pass
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        intermediate = output_path.replace(".mp4", "_raw.mp4")
        if os.path.exists(intermediate):
            try: os.remove(intermediate)
            except: pass
