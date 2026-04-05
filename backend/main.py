#!/usr/bin/env python3
"""
Zync — Main execution file for the full video editing pipeline.

Pipeline order
--------------
  Step 1  Video Dissection       video_dissector.py   → scene clips
  Step 2  Keyframe Extraction    keyframe_extractor.py → JPEG frames
  Step 3  VLM Analysis           vlm_analyzer.py       → text descriptions
  Step 4  Prompt Matching        prompt_matcher.py     → ordered clip list
  Step 5  Speed Control          speed_controller.py   → per-clip speed / freeze
  Step 6  Transitions + Merge    transition_applier.py → single merged video
  Step 7  Colour Grade           color_grader.py       → graded video
  Step 8  Text Overlays          text_overlay.py       → titles / captions
  Step 9  Audio Mix              audio_processor.py    → final video with music

Run modes
---------
  python main.py                          # batch-process all videos in input_videos/
  python main.py video.mp4               # process a single video file
  python main.py video.mp4 edit.json     # single video + custom edit config JSON
  python main.py video.mp4 "prompt text" # single video + parse editing prompt

Example editing prompt
----------------------
  "cinematic style, crossfade transitions 0.8s, slow motion,
   add title 'Summer Highlights 2024', lower third: John Doe,
   warm colour grade, vignette, lofi music at 30%, 1080p 24fps output"

Prompt keywords (what you can mention)
---------------------------------------
  TRANSITIONS:  crossfade / fade to black / fade to white / wipe left /
                wipe right / zoom in / slide left / hard cut
  SPEED:        slow motion / slow mo / half speed / fast forward /
                speed up / 2x / timelapse / 4x
  COLOUR STYLE: cinematic / vibrant / vintage / warm / cool / dramatic /
                black and white / grayscale / bright / dark
  COLOUR ADJ:   increase brightness / darker / high contrast /
                vignette / grain / film grain
  TEXT:         title "My Video Title"  /  lower third: Name Here  /
                watermark: Brand Name
  MUSIC:        lofi / lo-fi / chill / cinematic / upbeat / energetic
                music volume 30%  /  no duck
  OUTPUT:       1080p / 720p / 4K / vertical / portrait / square /
                24fps / 30fps / 60fps
"""

import os
import sys
import shutil
from pathlib import Path

from video_dissector import VideoDissector
from keyframe_extractor import KeyframeExtractor, process_after_dissection
from vlm_analyzer import analyze_after_keyframe_extraction
from prompt_matcher import PromptMatcher

from edit_config import EditConfig, ClipEdit, parse_prompt_to_config, load_from_json
from speed_controller import SpeedController
from transition_applier import TransitionApplier
from color_grader import ColorGrader
from text_overlay import TextOverlayProcessor
from audio_processor import AudioProcessor


# ── Directory layout ──────────────────────────────────────────────────────────
INPUT_DIR          = Path("input_videos")
OUTPUT_CLIPS_DIR   = Path("output_clips")
OUTPUT_DESC_DIR    = Path("output_description")
OUTPUT_FINAL_DIR   = Path("output_final")
TEMP_DIR           = Path("temp_editing")


def _ensure_dirs():
    for d in [INPUT_DIR, OUTPUT_CLIPS_DIR, OUTPUT_DESC_DIR, OUTPUT_FINAL_DIR, TEMP_DIR]:
        d.mkdir(exist_ok=True)


# ── Steps 1–3: Dissect → Keyframes → VLM ─────────────────────────────────────

def run_analysis_pipeline(video_file: Path) -> dict:
    """
    Run Steps 1–3 on a single video file.

    Returns:
        keyframe_results dict  (clip_name → list[keyframe_path])
    """
    print(f"\n{'='*60}")
    print(f"  VIDEO: {video_file.name}")
    print(f"{'='*60}")

    # Step 1: Dissect into scenes
    print("\nStep 1 — Dissecting video into scenes...")
    dissector = VideoDissector(INPUT_DIR, OUTPUT_CLIPS_DIR)
    clips = dissector.dissect_video(str(video_file))
    if not clips:
        print(f"  No clips created from {video_file.name}")
        return {}
    print(f"  Created {len(clips)} clips")

    # Step 2: Extract keyframes
    print("\nStep 2 — Extracting keyframes...")
    keyframe_results = process_after_dissection(
        dissector_output_clips=clips,
        output_description_dir=str(OUTPUT_DESC_DIR),
        keyframe_interval=3.0,
    )
    total_kf = sum(len(v) for v in keyframe_results.values())
    print(f"  Extracted {total_kf} keyframes from {len(clips)} clips")

    # Step 3: VLM analysis
    print("\nStep 3 — Analysing keyframes with VLM (Ollama)...")
    vlm_results = analyze_after_keyframe_extraction(
        keyframe_results=keyframe_results,
        output_description_dir=str(OUTPUT_DESC_DIR),
        ollama_host="http://localhost:11434",
        model_name="llava",
    )
    total_analyzed = sum(len(v) for v in vlm_results.values())
    print(f"  Analysed {total_analyzed} keyframes")

    return keyframe_results


# ── Step 4: Prompt matching ───────────────────────────────────────────────────

def run_prompt_matching(
    prompt: str,
    vlm_analysis_dir: Path,
    max_clips: int = None,
    min_score: float = 0.1,
) -> list:
    """
    Match a natural-language prompt against VLM descriptions.

    Returns:
        Ordered list of clip file paths.
    """
    print(f"\nStep 4 — Matching prompt to clips...")
    print(f"  Prompt: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")

    matcher = PromptMatcher(str(vlm_analysis_dir), prompt_file=None)
    matcher.user_prompt = prompt
    matcher.load_vlm_descriptions()

    ordered_names = matcher.generate_clip_order(
        min_score=min_score, max_clips=max_clips
    )

    # Resolve clip names → actual file paths
    clip_paths = []
    for name in ordered_names:
        # Try to find the matching file in OUTPUT_CLIPS_DIR
        matches = list(OUTPUT_CLIPS_DIR.glob(f"{name}*.mp4"))
        if matches:
            clip_paths.append(str(matches[0]))
        else:
            # Fallback: use name as-is (might already be a path)
            if Path(name).exists():
                clip_paths.append(name)

    print(f"  Resolved {len(clip_paths)} clip path(s)")
    return clip_paths


# ── Steps 5–9: Edit pipeline ──────────────────────────────────────────────────

def run_edit_pipeline(config: EditConfig) -> str:
    """
    Run Steps 5–9 using the provided EditConfig.

    Returns:
        Path to the final output video.
    """
    current_video: str = None

    # Step 5: Speed control (per-clip)
    print("\nStep 5 — Applying speed changes...")
    sc = SpeedController(output_dir=str(TEMP_DIR / "speed"))
    processed_clip_paths = sc.process_all(config)

    # Update config.clips to point at the processed versions
    for i, path in enumerate(processed_clip_paths):
        config.clips[i].clip_path = path

    # Step 6: Transitions + Merge
    print("\nStep 6 — Merging clips with transitions...")
    merge_out = str(TEMP_DIR / "merged.mp4")
    config_for_merge = EditConfig(
        clips=config.clips,
        transition=config.transition,
        transition_duration=config.transition_duration,
        output_resolution=config.output_resolution,
        output_fps=config.output_fps,
        output_path=merge_out,
    )
    ta = TransitionApplier(temp_dir=str(TEMP_DIR / "transitions"))
    current_video = ta.apply(config_for_merge)
    print(f"  Merged → {current_video}")

    # Step 7: Colour grade
    print("\nStep 7 — Applying colour grade...")
    grade_out = str(TEMP_DIR / "graded.mp4")
    cg = ColorGrader()
    current_video = cg.apply(current_video, config, grade_out)

    # Step 8: Text overlays
    print("\nStep 8 — Burning text overlays...")
    text_out = str(TEMP_DIR / "text.mp4")
    tp = TextOverlayProcessor()
    current_video = tp.apply(current_video, config, text_out)

    # Step 9: Audio mix
    print("\nStep 9 — Mixing audio...")
    audio_out = str(TEMP_DIR / "audio.mp4")
    ap = AudioProcessor()
    current_video = ap.apply(current_video, config, audio_out)

    # Copy final result to output_final/
    OUTPUT_FINAL_DIR.mkdir(exist_ok=True)
    final_path = config.output_path
    shutil.copy(current_video, final_path)
    print(f"\n  Final video → {final_path}")
    return final_path


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main():
    """Batch-process all videos in input_videos/."""
    _ensure_dirs()

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    video_files = [
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {INPUT_DIR}/")
        print(f"Place .mp4 / .mov files there and re-run.")
        return

    print(f"Zync Processing Pipeline")
    print(f"Found {len(video_files)} video file(s) in {INPUT_DIR}/")

    # Default editing prompt (can be overridden per-video)
    default_prompt = (
        "Select the most visually interesting scenes. "
        "Apply cinematic colour grade, crossfade transitions, "
        "add title 'Zync Highlights', lofi music at 30%."
    )

    stats = {"success": 0, "fail": 0}

    for video_file in video_files:
        try:
            # Steps 1–3
            keyframe_results = run_analysis_pipeline(video_file)
            if not keyframe_results:
                stats["fail"] += 1
                continue

            # Step 4: Prompt match
            vlm_dir = OUTPUT_DESC_DIR / "vlm_analysis"
            ordered_clips = run_prompt_matching(
                prompt=default_prompt,
                vlm_analysis_dir=vlm_dir,
            )

            if not ordered_clips:
                print("  No clips matched — using all clips in order")
                ordered_clips = sorted(str(p) for p in OUTPUT_CLIPS_DIR.glob("*.mp4"))

            # Build edit config from prompt
            out_path = str(OUTPUT_FINAL_DIR / f"{video_file.stem}_final.mp4")
            config = parse_prompt_to_config(default_prompt, ordered_clips, out_path)

            # Steps 5–9
            final = run_edit_pipeline(config)
            print(f"\n  Done: {final}")
            stats["success"] += 1

        except Exception as e:
            print(f"\n  Error processing {video_file.name}: {e}")
            stats["fail"] += 1

    print(f"\n{'='*60}")
    print(f"Pipeline complete — Success: {stats['success']}  Failed: {stats['fail']}")
    print(f"Final videos saved to: {OUTPUT_FINAL_DIR}/")


def process_single_video(
    video_path: str,
    editing_prompt: str = None,
    config_json: str = None,
):
    """
    Process one video file through the complete pipeline.

    Args:
        video_path:      Path to the input video.
        editing_prompt:  Free-text editing instructions (parsed by edit_config).
        config_json:     Path to a pre-built edit_config.json (overrides prompt).
    """
    _ensure_dirs()

    video_file = Path(video_path)
    if not video_file.exists():
        print(f"File not found: {video_path}")
        return

    # Steps 1–3: Analysis
    keyframe_results = run_analysis_pipeline(video_file)
    if not keyframe_results:
        print("Analysis pipeline produced no clips.")
        return

    # Step 4: Prompt match
    vlm_dir = OUTPUT_DESC_DIR / "vlm_analysis"
    prompt = editing_prompt or (
        "Select the best scenes. Apply cinematic style, crossfade transitions, "
        "lofi music at 30%."
    )
    ordered_clips = run_prompt_matching(prompt=prompt, vlm_analysis_dir=vlm_dir)

    if not ordered_clips:
        print("  No clips matched — using all output clips in order")
        ordered_clips = sorted(str(p) for p in OUTPUT_CLIPS_DIR.glob("*.mp4"))

    # Build / load edit config
    out_path = str(OUTPUT_FINAL_DIR / f"{video_file.stem}_final.mp4")
    if config_json and Path(config_json).exists():
        print(f"\nLoading edit config from {config_json}")
        config = load_from_json(config_json)
        config.clips = [ClipEdit(clip_path=p) for p in ordered_clips]
        config.output_path = out_path
    else:
        config = parse_prompt_to_config(prompt, ordered_clips, out_path)

    # Steps 5–9: Edit
    final = run_edit_pipeline(config)

    print(f"\nFinal video: {final}")
    return final


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Batch mode
        main()

    elif len(sys.argv) == 2:
        # Single video, default prompt
        process_single_video(sys.argv[1])

    elif len(sys.argv) == 3:
        arg = sys.argv[2]
        if arg.endswith(".json"):
            # Single video + config JSON
            process_single_video(sys.argv[1], config_json=arg)
        else:
            # Single video + inline prompt
            process_single_video(sys.argv[1], editing_prompt=arg)

    else:
        print(__doc__)
