#!/usr/bin/env python3
"""
TransitionApplier — joins an ordered list of video clips with smooth
transitions using FFmpeg's xfade filter.

Supported transitions (maps to FFmpeg xfade names):
  cut        — instant hard cut (no xfade, plain concat)
  crossfade  — classic cross-dissolve
  fadeblack  — fade through black
  fadewhite  — fade through white
  dissolve   — pixel-level dissolve
  wipeleft   — left-to-right wipe
  wiperight  — right-to-left wipe
  wipeup     — bottom-to-top wipe
  wipedown   — top-to-bottom wipe
  slideleft  — new clip slides in from right
  slideright — new clip slides in from left
  zoomin     — zoom-in reveal of next clip

Usage:
    from transition_applier import TransitionApplier
    from edit_config import EditConfig, ClipEdit

    config = EditConfig(
        clips=[ClipEdit("a.mp4"), ClipEdit("b.mp4"), ClipEdit("c.mp4")],
        transition="crossfade",
        transition_duration=0.8,
        output_path="merged.mp4",
    )
    ta = TransitionApplier()
    ta.apply(config)
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Optional

from edit_config import EditConfig, ClipEdit

# Map friendly names → FFmpeg xfade transition names
XFADE_MAP = {
    "crossfade":  "dissolve",
    "fadeblack":  "fade",       # fade through black
    "fadewhite":  "fadewhite",
    "dissolve":   "pixelize",
    "wipeleft":   "wipeleft",
    "wiperight":  "wiperight",
    "wipeup":     "wipeup",
    "wipedown":   "wipedown",
    "slideleft":  "slideleft",
    "slideright": "slideright",
    "zoomin":     "zoomin",
    "cut":        None,          # plain concat, no xfade
}


class TransitionApplier:
    """Joins clips with FFmpeg xfade transitions."""

    def __init__(self, temp_dir: str = "temp_transitions"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)

    # ── Public entry point ────────────────────────────────────────────────

    def apply(self, config: EditConfig) -> str:
        """
        Merge all clips in config.clips with the specified transition.

        Returns:
            Path to the merged output file.
        """
        clip_paths = [c.clip_path for c in config.clips if Path(c.clip_path).exists()]

        if not clip_paths:
            raise FileNotFoundError("No valid clip paths found in config.clips")

        if len(clip_paths) == 1:
            print("Only one clip — copying directly to output.")
            import shutil
            shutil.copy(clip_paths[0], config.output_path)
            return config.output_path

        xfade_name = XFADE_MAP.get(config.transition, None)

        if xfade_name is None or config.transition == "cut":
            print(f"Transition: hard cut — using fast concat")
            return self._concat_cut(clip_paths, config)
        else:
            print(f"Transition: {config.transition} ({xfade_name}) @ {config.transition_duration}s")
            return self._concat_xfade(clip_paths, xfade_name, config)

    # ── Hard cut (no transition) ─────────────────────────────────────────

    def _concat_cut(self, clip_paths: List[str], config: EditConfig) -> str:
        """Fast FFmpeg concat via a filelist — no re-encoding if formats match."""
        filelist = self.temp_dir / "filelist.txt"
        with open(filelist, "w", encoding="utf-8") as f:
            for p in clip_paths:
                f.write(f"file '{Path(p).resolve()}'\n")

        w, h = config.output_resolution.split("x")
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(filelist),
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-r", str(config.output_fps),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            config.output_path,
        ]
        self._run(cmd, "concat cut")
        return config.output_path

    # ── xfade transitions ────────────────────────────────────────────────

    def _concat_xfade(
        self,
        clip_paths: List[str],
        xfade_name: str,
        config: EditConfig,
    ) -> str:
        """
        Build an FFmpeg filter_complex that chains xfade between every pair
        of consecutive clips.

        Strategy:
          1. Get duration of each clip via ffprobe.
          2. Build a filter_complex with scale+setsar for each input, then
             chain xfade filters: [v0][v1]xfade...[xf01], [xf01][v2]xfade...[xf012] …
          3. Do the same for audio with acrossfade.
        """
        durations = [self._get_duration(p) for p in clip_paths]
        dur = config.transition_duration
        w, h = config.output_resolution.split("x")

        # ── Build filter_complex ──────────────────────────────────────────
        parts: List[str] = []

        # Scale + normalise each input
        for i in range(len(clip_paths)):
            parts.append(
                f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={config.output_fps}[sv{i}]"
            )

        # Chain xfade for video
        # offset = cumulative duration up to clip i, minus transition overlap
        offset = durations[0] - dur
        parts.append(f"[sv0][sv1]xfade=transition={xfade_name}:"
                     f"duration={dur}:offset={offset:.4f}[xfv1]")

        prev_label = "xfv1"
        for i in range(2, len(clip_paths)):
            offset += durations[i - 1] - dur
            new_label = f"xfv{i}"
            parts.append(
                f"[{prev_label}][sv{i}]xfade=transition={xfade_name}:"
                f"duration={dur}:offset={offset:.4f}[{new_label}]"
            )
            prev_label = new_label

        # Chain acrossfade for audio
        for i in range(len(clip_paths)):
            parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[sa{i}]")

        audio_offset = durations[0] - dur
        parts.append(
            f"[sa0][sa1]acrossfade=d={dur}:c1=tri:c2=tri[xfa1]"
        )
        prev_audio = "xfa1"
        for i in range(2, len(clip_paths)):
            new_audio = f"xfa{i}"
            parts.append(f"[{prev_audio}][sa{i}]acrossfade=d={dur}:c1=tri:c2=tri[{new_audio}]")
            prev_audio = new_audio

        filter_complex = "; ".join(parts)

        # ── Assemble FFmpeg command ───────────────────────────────────────
        cmd = ["ffmpeg", "-y"]
        for p in clip_paths:
            cmd += ["-i", p]
        cmd += [
            "-filter_complex", filter_complex,
            "-map", f"[{prev_label}]",
            "-map", f"[{prev_audio}]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            config.output_path,
        ]
        self._run(cmd, f"xfade ({xfade_name})")
        return config.output_path

    # ── Helpers ──────────────────────────────────────────────────────────

    def _get_duration(self, path: str) -> float:
        """Return clip duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 5.0  # Fallback

    def _run(self, cmd: List[str], label: str):
        """Run an FFmpeg command and raise on failure."""
        print(f"  Running FFmpeg: {label} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FFmpeg stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"FFmpeg failed during: {label}")
        print(f"  Done: {label}")

    def cleanup(self):
        """Remove temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


# ── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from edit_config import ClipEdit

    if len(sys.argv) < 3:
        print("Usage: python transition_applier.py <transition> <clip1> <clip2> [clip3 ...] [output.mp4]")
        print("       Transitions: cut crossfade fadeblack wipeleft wiperight zoomin slideleft")
        sys.exit(1)

    transition = sys.argv[1]
    *input_clips, output = sys.argv[2:]

    # If last argument looks like a clip (not .mp4 as output), handle it
    if not output.endswith(".mp4"):
        input_clips.append(output)
        output = "output_with_transitions.mp4"

    cfg = EditConfig(
        clips=[ClipEdit(p) for p in input_clips],
        transition=transition,
        transition_duration=0.7,
        output_path=output,
    )
    ta = TransitionApplier()
    result = ta.apply(cfg)
    print(f"\nOutput: {result}")
