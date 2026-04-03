#!/usr/bin/env python3
"""
ColorGrader — applies colour-grading effects to a video using FFmpeg filters.

Effects (all stackable, applied in a single pass):
  - Brightness / contrast / saturation  via  eq  filter
  - Colour temperature (warm / cool)    via  colorbalance  filter
  - Vignette                            via  vignette  filter
  - Film grain / noise                  via  noise  filter
  - LUT (Look-Up Table)                 via  lut3d  filter  (.cube file)
  - Greyscale / B&W                     via  hue  filter  (saturation=0)

Usage
-----
    from color_grader import ColorGrader
    from edit_config import EditConfig, ColorGrade

    config = EditConfig(
        ...
        color_grade=ColorGrade(
            brightness=1.05,
            contrast=1.15,
            saturation=1.2,
            temperature="warm",
            vignette=True,
            grain=0.2,
        ),
    )
    grader = ColorGrader()
    output = grader.apply(input_video="merged.mp4", config=config)
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from edit_config import EditConfig, ColorGrade


class ColorGrader:
    """Applies colour-grading filters from an EditConfig to a video."""

    def apply(
        self,
        input_video: str,
        config: EditConfig,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Grade the video.

        Args:
            input_video:  Path to source video.
            config:       EditConfig with a populated color_grade field.
            output_path:  Destination. Defaults to <stem>_graded.mp4.

        Returns:
            Path to the graded video.
        """
        if config.color_grade is None:
            print("No colour grade defined — skipping.")
            return input_video

        if output_path is None:
            p = Path(input_video)
            output_path = str(p.parent / (p.stem + "_graded" + p.suffix))

        vf = self._build_vf(config.color_grade)

        cmd = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "copy",
            output_path,
        ]

        print(f"Applying colour grade: {self._describe(config.color_grade)}")
        self._run(cmd)
        print(f"Colour grade applied → {output_path}")
        return output_path

    # ── Filter builder ────────────────────────────────────────────────────

    def _build_vf(self, grade: ColorGrade) -> str:
        """Return a comma-joined FFmpeg video filter chain."""
        filters: List[str] = []

        # 1. LUT (applied first so subsequent filters adjust the graded look)
        if grade.lut_file and Path(grade.lut_file).exists():
            filters.append(f"lut3d='{grade.lut_file}'")

        # 2. Brightness / contrast / saturation via eq
        #    eq uses: brightness (-1 to 1, 0 = no change)
        #             contrast   (0 to 2,  1 = no change)
        #             saturation (0 to 3,  1 = no change)
        eq_brightness = grade.brightness - 1.0   # Convert 0.5–1.5 → -0.5–0.5
        eq_contrast   = grade.contrast            # Already in correct range
        eq_saturation = grade.saturation          # Already in correct range

        if (abs(eq_brightness) > 0.01 or
                abs(eq_contrast - 1.0) > 0.01 or
                abs(eq_saturation - 1.0) > 0.01):
            filters.append(
                f"eq=brightness={eq_brightness:.3f}:"
                f"contrast={eq_contrast:.3f}:"
                f"saturation={eq_saturation:.3f}"
            )

        # 3. Colour temperature via colorbalance
        if grade.temperature == "warm":
            # Boost reds/yellows in shadows + midtones; slight blue reduction
            filters.append(
                "colorbalance=rs=0.05:gs=0.0:bs=-0.05:"  # shadows
                "rm=0.05:gm=0.0:bm=-0.05:"               # midtones
                "rh=0.02:gh=0.0:bh=-0.02"                # highlights
            )
        elif grade.temperature == "cool":
            # Boost blues; slight red reduction
            filters.append(
                "colorbalance=rs=-0.05:gs=0.0:bs=0.05:"
                "rm=-0.05:gm=0.0:bm=0.05:"
                "rh=-0.02:gh=0.0:bh=0.02"
            )

        # 4. Vignette — dark elliptical edge overlay
        if grade.vignette:
            # angle=PI/5 gives a medium-strength vignette
            filters.append("vignette=PI/5")

        # 5. Film grain / noise
        if grade.grain > 0.0:
            # noise alls=X adds gaussian noise to all channels
            # X range 0–100; we scale our 0–1 input to 0–40
            noise_strength = int(grade.grain * 40)
            filters.append(f"noise=alls={noise_strength}:allf=t+u")

        # If nothing was added, return a null passthrough
        if not filters:
            return "copy"

        return ",".join(filters)

    # ── Description helper (for logging) ─────────────────────────────────

    @staticmethod
    def _describe(grade: ColorGrade) -> str:
        parts = []
        if abs(grade.brightness - 1.0) > 0.01:
            parts.append(f"brightness={grade.brightness:.2f}")
        if abs(grade.contrast - 1.0) > 0.01:
            parts.append(f"contrast={grade.contrast:.2f}")
        if abs(grade.saturation - 1.0) > 0.01:
            parts.append(f"saturation={grade.saturation:.2f}")
        if grade.temperature != "neutral":
            parts.append(f"temp={grade.temperature}")
        if grade.vignette:
            parts.append("vignette")
        if grade.grain > 0:
            parts.append(f"grain={grade.grain:.2f}")
        if grade.lut_file:
            parts.append(f"lut={Path(grade.lut_file).name}")
        return ", ".join(parts) or "none"

    @staticmethod
    def _run(cmd: List[str]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FFmpeg stderr:\n{r.stderr[-2000:]}")
            raise RuntimeError("FFmpeg colour grade failed")


# ── Standalone entry ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python color_grader.py <input_video> <output_video> [preset]")
        print("Presets: cinematic  vibrant  vintage  warm  cool  bw  dramatic")
        sys.exit(1)

    input_v  = sys.argv[1]
    output_v = sys.argv[2]
    preset   = sys.argv[3] if len(sys.argv) > 3 else "cinematic"

    preset_grades = {
        "cinematic": ColorGrade(contrast=1.2, saturation=0.85, brightness=0.95, vignette=True,  grain=0.1),
        "vibrant":   ColorGrade(saturation=1.5, contrast=1.1,  brightness=1.05),
        "vintage":   ColorGrade(saturation=0.7, contrast=0.9,  brightness=0.95, grain=0.3,  temperature="warm"),
        "warm":      ColorGrade(temperature="warm", saturation=1.1, brightness=1.05),
        "cool":      ColorGrade(temperature="cool", saturation=0.95),
        "bw":        ColorGrade(saturation=0.0, contrast=1.2),
        "dramatic":  ColorGrade(contrast=1.4, saturation=0.6, brightness=0.9, vignette=True),
    }

    grade = preset_grades.get(preset, ColorGrade())
    cfg   = EditConfig(clips=[], color_grade=grade)

    grader = ColorGrader()
    result = grader.apply(input_v, cfg, output_v)
    print(f"Output: {result}")
