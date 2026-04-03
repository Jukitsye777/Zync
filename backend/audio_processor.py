#!/usr/bin/env python3
"""
AudioProcessor — mixes background music into a video, with optional
automatic audio ducking (lowering music volume when speech is present).

Features
--------
- Mix a background music track (MP3/WAV/AAC) under the video's original audio
- Loop the music track automatically if it is shorter than the video
- Fade the music in at the start and out at the end
- Audio ducking: lower music volume during the whole video (simple version)
  True sidechain ducking requires complex FFmpeg graphs; we use a simpler
  approach: lower music to duck_level for the full duration.
- Volume control for both music and original audio

Usage
-----
    from audio_processor import AudioProcessor
    from edit_config import EditConfig

    config = EditConfig(
        ...
        background_music="music/lofi_chill.mp3",
        music_volume=0.35,
        duck_audio=True,
        duck_level=0.15,
    )
    ap = AudioProcessor()
    output = ap.apply(input_video="colour_graded.mp4", config=config)
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from edit_config import EditConfig


class AudioProcessor:
    """Mixes background music into a video."""

    def apply(
        self,
        input_video: str,
        config: EditConfig,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Mix background music (if configured) into the video.

        Args:
            input_video:  Path to the video that already has all visual effects.
            config:       EditConfig with audio settings.
            output_path:  Output path. Defaults to <stem>_audio.mp4.

        Returns:
            Path to the video with audio mixed.
        """
        if not config.background_music:
            print("No background music configured — skipping audio mix.")
            return input_video

        music_path = config.background_music
        if not Path(music_path).exists():
            print(f"WARNING: Music file not found: {music_path} — skipping.")
            return input_video

        if output_path is None:
            p = Path(input_video)
            output_path = str(p.parent / (p.stem + "_audio" + p.suffix))

        video_duration = self._get_duration(input_video)
        print(f"Mixing audio: {Path(music_path).name} "
              f"(vol={config.music_volume:.0%}, "
              f"duck={config.duck_audio})")

        if config.duck_audio:
            return self._mix_with_ducking(
                input_video, music_path, video_duration, config, output_path
            )
        else:
            return self._mix_simple(
                input_video, music_path, video_duration, config, output_path
            )

    # ── Simple mix (no ducking) ───────────────────────────────────────────

    def _mix_simple(
        self,
        video: str,
        music: str,
        duration: float,
        config: EditConfig,
        output: str,
    ) -> str:
        """
        Straightforward amix: original audio at full volume + music at music_volume.
        Music is looped to fit the video duration and faded in/out.
        """
        fade_dur = min(2.0, duration * 0.05)  # 2 s fade or 5% of video

        # Audio filter chain for the music track:
        #   1. aloop=-1: loop indefinitely
        #   2. atrim=end=duration: cut at video end
        #   3. volume: scale to music_volume
        #   4. afade in + out
        music_af = (
            f"aloop=-1:size=2e+09,"
            f"atrim=end={duration:.3f},"
            f"volume={config.music_volume:.4f},"
            f"afade=t=in:st=0:d={fade_dur:.2f},"
            f"afade=t=out:st={max(0, duration - fade_dur):.2f}:d={fade_dur:.2f}"
        )

        # amix inputs: [0:a] = original video audio, [1:a] = processed music
        filter_complex = (
            f"[1:a]{music_af}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-i", music,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output,
        ]
        self._run(cmd, "simple audio mix")
        print(f"Audio mixed → {output}")
        return output

    # ── Mix with ducking ─────────────────────────────────────────────────

    def _mix_with_ducking(
        self,
        video: str,
        music: str,
        duration: float,
        config: EditConfig,
        output: str,
    ) -> str:
        """
        Mix with simple ducking: music plays at duck_level (much quieter)
        during the full video, so speech is always clearly audible.

        A full sidechain compressor approach is complex and fragile across
        FFmpeg versions; the duck_level approach is reliable and practical.
        """
        fade_dur = min(2.0, duration * 0.05)
        duck_vol = config.duck_level  # e.g. 0.15 = 15% of music_volume

        # Music stays at duck_level the whole time (speech always wins)
        effective_volume = config.music_volume * duck_vol

        music_af = (
            f"aloop=-1:size=2e+09,"
            f"atrim=end={duration:.3f},"
            f"volume={effective_volume:.4f},"
            f"afade=t=in:st=0:d={fade_dur:.2f},"
            f"afade=t=out:st={max(0, duration - fade_dur):.2f}:d={fade_dur:.2f}"
        )

        filter_complex = (
            f"[1:a]{music_af}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-i", video,
            "-i", music,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            output,
        ]
        self._run(cmd, "ducked audio mix")
        print(f"Audio mixed (ducked) → {output}")
        return output

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_duration(path: str) -> float:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        try:
            return float(r.stdout.strip())
        except ValueError:
            return 60.0

    @staticmethod
    def _run(cmd: List[str], label: str = ""):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FFmpeg error ({label}):\n{r.stderr[-2000:]}")
            raise RuntimeError(f"FFmpeg audio processing failed: {label}")


# ── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python audio_processor.py <input_video> <music_file> <output_video> [volume]")
        print("  volume: 0.0–1.0, default 0.35")
        sys.exit(1)

    inp   = sys.argv[1]
    music = sys.argv[2]
    outp  = sys.argv[3]
    vol   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35

    cfg = EditConfig(
        clips=[],
        background_music=music,
        music_volume=vol,
        duck_audio=True,
        duck_level=0.15,
    )
    ap = AudioProcessor()
    result = ap.apply(inp, cfg, outp)
    print(f"Output: {result}")
