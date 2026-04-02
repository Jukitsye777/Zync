#!/usr/bin/env python3
"""
SpeedController — processes per-clip speed changes and freeze frames,
producing a modified clip ready for the merge step.

Capabilities
------------
- Global speed change per clip  (0.25x slow-mo → 4x fast-forward)
  Audio is pitch-corrected with FFmpeg's atempo filter (0.5x–2.0x range;
  larger/smaller factors are chained automatically).
- Freeze frame: hold a specific timestamp as a still image for N seconds,
  then continue playing from that point.

Usage
-----
    from speed_controller import SpeedController
    from edit_config import EditConfig, ClipEdit

    config = EditConfig(
        clips=[
            ClipEdit("scene_001.mp4", speed=1.0),
            ClipEdit("scene_002.mp4", speed=0.5),                       # slow-mo
            ClipEdit("scene_003.mp4", speed=2.0),                       # fast-forward
            ClipEdit("scene_004.mp4", speed=1.0, freeze_at=3.2, freeze_duration=1.5),
        ],
        ...
    )
    sc = SpeedController(output_dir="temp_speed")
    processed_paths = sc.process_all(config)
    # processed_paths is a list of output file paths in the same order as config.clips
"""

import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from edit_config import EditConfig, ClipEdit


class SpeedController:
    """Applies per-clip speed changes and freeze frames."""

    def __init__(self, output_dir: str = "temp_speed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def process_all(self, config: EditConfig) -> List[str]:
        """
        Process every clip in config.clips.

        Returns a list of output file paths (same length and order as
        config.clips), ready to pass into TransitionApplier.
        """
        results: List[str] = []
        for i, clip_edit in enumerate(config.clips):
            src = clip_edit.clip_path
            if not Path(src).exists():
                print(f"  WARNING: clip not found — {src}")
                results.append(src)
                continue

            out = str(self.output_dir / f"speed_{i:03d}_{Path(src).stem}.mp4")

            # No-op? Just copy so downstream always gets a consistent path.
            if (clip_edit.speed == 1.0 and
                    clip_edit.freeze_at is None and
                    clip_edit.trim_start == 0.0 and
                    clip_edit.trim_end is None):
                shutil.copy(src, out)
                results.append(out)
                continue

            # Trim first (if requested), then freeze, then speed
            working = src

            if clip_edit.trim_start > 0 or clip_edit.trim_end is not None:
                working = self._trim(working, clip_edit, i)

            if clip_edit.freeze_at is not None and clip_edit.freeze_duration > 0:
                working = self._freeze(working, clip_edit, i)

            if clip_edit.speed != 1.0:
                working = self._change_speed(working, clip_edit.speed, i)

            # Move final working file to output path
            shutil.move(working, out)
            results.append(out)
            print(f"  Processed clip {i+1}/{len(config.clips)}: {Path(src).name}")

        return results

    # ── Trim ─────────────────────────────────────────────────────────────

    def _trim(self, src: str, clip_edit: ClipEdit, idx: int) -> str:
        """Trim clip to [trim_start, trim_end]."""
        out = str(self.output_dir / f"_trim_{idx:03d}.mp4")
        dur = self._get_duration(src)
        end = clip_edit.trim_end if clip_edit.trim_end is not None else dur

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(clip_edit.trim_start),
            "-to", str(end),
            "-i", src,
            "-c", "copy",
            out,
        ]
        self._run(cmd, f"trim clip {idx}")
        return out

    # ── Freeze frame ─────────────────────────────────────────────────────

    def _freeze(self, src: str, clip_edit: ClipEdit, idx: int) -> str:
        """
        Split clip at freeze_at, insert a still frame for freeze_duration
        seconds, then continue from freeze_at.

        Implementation:
          part_a  = clip from 0 → freeze_at
          still   = single frame at freeze_at, looped for freeze_duration s
          part_b  = clip from freeze_at → end
          concat all three
        """
        freeze_t = clip_edit.freeze_at
        freeze_d = clip_edit.freeze_duration

        part_a = str(self.output_dir / f"_frz_a_{idx:03d}.mp4")
        still  = str(self.output_dir / f"_frz_s_{idx:03d}.mp4")
        part_b = str(self.output_dir / f"_frz_b_{idx:03d}.mp4")
        out    = str(self.output_dir / f"_frz_{idx:03d}.mp4")

        # Part A: 0 → freeze_at
        self._run([
            "ffmpeg", "-y", "-i", src,
            "-t", str(freeze_t), "-c", "copy", part_a,
        ], f"freeze part A {idx}")

        # Still: extract one frame at freeze_at, loop it
        self._run([
            "ffmpeg", "-y",
            "-ss", str(freeze_t), "-i", src,
            "-vframes", "1", "-loop", "1",
            "-t", str(freeze_d),
            "-vf", "fps=30",
            "-c:v", "libx264", "-an",
            still,
        ], f"freeze still {idx}")

        # Part B: freeze_at → end
        self._run([
            "ffmpeg", "-y",
            "-ss", str(freeze_t), "-i", src,
            "-c", "copy", part_b,
        ], f"freeze part B {idx}")

        # Concat A + still + B
        filelist = str(self.output_dir / f"_frz_list_{idx:03d}.txt")
        with open(filelist, "w") as f:
            for p in [part_a, still, part_b]:
                f.write(f"file '{Path(p).resolve()}'\n")

        self._run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", filelist,
            "-c:v", "libx264", "-c:a", "aac",
            out,
        ], f"freeze concat {idx}")

        # Clean up intermediates
        for tmp in [part_a, still, part_b, filelist]:
            Path(tmp).unlink(missing_ok=True)

        return out

    # ── Speed change ─────────────────────────────────────────────────────

    def _change_speed(self, src: str, speed: float, idx: int) -> str:
        """
        Apply playback speed change.

        Video: setpts=1/speed*PTS  (e.g. speed=2 → PTS halved → plays 2x faster)
        Audio: atempo supports 0.5–2.0; for factors outside this range we
               chain multiple atempo filters.
        """
        out = str(self.output_dir / f"_spd_{idx:03d}.mp4")

        pts_factor = 1.0 / speed
        video_filter = f"setpts={pts_factor:.6f}*PTS"
        audio_filter = self._atempo_chain(speed)

        cmd = [
            "ffmpeg", "-y", "-i", src,
            "-vf", video_filter,
            "-af", audio_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            out,
        ]
        self._run(cmd, f"speed {speed}x on clip {idx}")
        return out

    @staticmethod
    def _atempo_chain(speed: float) -> str:
        """
        Build a chained atempo filter string.
        atempo only accepts values in [0.5, 2.0], so for extreme speeds
        we chain multiple filters (e.g. speed=4.0 → atempo=2.0,atempo=2.0).
        """
        if speed <= 0:
            speed = 0.5

        filters = []
        remaining = speed

        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0

        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5

        filters.append(f"atempo={remaining:.4f}")
        return ",".join(filters)

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
            return 10.0

    @staticmethod
    def _run(cmd: List[str], label: str = ""):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  FFmpeg error ({label}):\n{r.stderr[-1500:]}")
            raise RuntimeError(f"FFmpeg speed control failed: {label}")

    def cleanup(self):
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)


# ── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python speed_controller.py <input_video> <output_video> <speed>")
        print("  speed: 0.5 = half speed (slow-mo), 2.0 = double speed, etc.")
        print("  Example: python speed_controller.py clip.mp4 out.mp4 0.5")
        sys.exit(1)

    inp   = sys.argv[1]
    outp  = sys.argv[2]
    spd   = float(sys.argv[3])

    cfg = EditConfig(clips=[ClipEdit(clip_path=inp, speed=spd)])
    sc  = SpeedController()
    paths = sc.process_all(cfg)
    shutil.copy(paths[0], outp)
    print(f"Output: {outp}")
