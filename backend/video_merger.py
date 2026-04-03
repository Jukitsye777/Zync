#!/usr/bin/env python3
"""
Script to merge ordered video clips into a single final video
"""

import subprocess
import json
from pathlib import Path
from typing import List, Optional
import os
from datetime import datetime


class VideoMerger:
    def __init__(self, output_filename: str = "final_video.mp4"):
        """
        Initialize VideoMerger
        
        Args:
            output_filename: Name of the final merged video
        """
        self.output_filename = output_filename
        self.clips = []
        
    def check_ffmpeg(self) -> bool:
        """Check if FFmpeg is installed and available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True)
            if result.returncode == 0:
                print("✅ FFmpeg is installed")
                return True
            return False
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install FFmpeg:")
            print("   - Ubuntu/Debian: sudo apt install ffmpeg")
            print("   - macOS: brew install ffmpeg")
            print("   - Windows: Download from https://ffmpeg.org/")
            return False
    
    def load_clips_from_filelist(self, filelist_path: str) -> List[str]:
        """
        Load clip paths from FFmpeg filelist.txt
        
        Args:
            filelist_path: Path to filelist.txt
        
        Returns:
            List of clip paths
        """
        clips = []
        try:
            with open(filelist_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("file '") and line.endswith("'"):
                        # Extract path from "file 'path'" format
                        clip_path = line[6:-1]  # Remove "file '" and trailing "'"
                        clips.append(clip_path)
            
            print(f"✅ Loaded {len(clips)} clips from filelist")
            self.clips = clips
            return clips
        except Exception as e:
            print(f"❌ Error loading filelist: {e}")
            return []
    
    def load_clips_from_directory(self, clips_dir: str, pattern: str = "*.mp4") -> List[str]:
        """
        Load clips from a directory in alphabetical order
        
        Args:
            clips_dir: Directory containing clips
            pattern: Glob pattern for clip files
        
        Returns:
            List of clip paths
        """
        clips_dir = Path(clips_dir)
        clips = sorted(clips_dir.glob(pattern))
        self.clips = [str(c) for c in clips]
        
        print(f"✅ Found {len(self.clips)} clips in {clips_dir}")
        return self.clips
    
    def merge_videos_concat_demuxer(self, 
                                    filelist_path: str, 
                                    output_path: str,
                                    codec: str = "copy") -> bool:
        """
        Merge videos using FFmpeg concat demuxer (fastest, no re-encoding)
        All clips must have same codec, resolution, and frame rate
        
        Args:
            filelist_path: Path to filelist.txt
            output_path: Path for output video
            codec: 'copy' for no re-encoding, or codec name for re-encoding
        
        Returns:
            True if successful
        """
        print(f"\n🎬 Merging videos using concat demuxer...")
        print(f"📋 Using filelist: {filelist_path}")
        print(f"💾 Output: {output_path}")
        
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', filelist_path,
            '-c', codec,
            '-y',  # Overwrite output file
            output_path
        ]
        
        try:
            print(f"\n⚙️  Running FFmpeg command...")
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  check=True)
            
            if Path(output_path).exists():
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                print(f"✅ Video merged successfully!")
                print(f"📦 File size: {size_mb:.2f} MB")
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr}")
            return False
    
    def merge_videos_filter_complex(self, 
                                    clips: List[str], 
                                    output_path: str,
                                    resolution: str = "1920x1080",
                                    fps: int = 30) -> bool:
        """
        Merge videos using FFmpeg filter_complex (handles different formats)
        Re-encodes all clips to same format
        
        Args:
            clips: List of clip paths
            output_path: Path for output video
            resolution: Target resolution (WxH)
            fps: Target frame rate
        
        Returns:
            True if successful
        """
        print(f"\n🎬 Merging videos using filter_complex...")
        print(f"🔧 Target resolution: {resolution}, FPS: {fps}")
        print(f"💾 Output: {output_path}")
        
        # Build FFmpeg command
        cmd = ['ffmpeg']
        
        # Add input files
        for clip in clips:
            cmd.extend(['-i', clip])
        
        # Build filter_complex string
        filter_parts = []
        for i in range(len(clips)):
            filter_parts.append(f"[{i}:v]scale={resolution},setsar=1,fps={fps}[v{i}];")
        
        concat_inputs = ''.join([f"[v{i}]" for i in range(len(clips))])
        filter_complex = ''.join(filter_parts) + f"{concat_inputs}concat=n={len(clips)}:v=1:a=0[outv]"
        
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-y',
            output_path
        ])
        
        try:
            print(f"\n⚙️  Running FFmpeg command (this may take a while)...")
            result = subprocess.run(cmd, 
                                  capture_output=True, 
                                  text=True,
                                  check=True)
            
            if Path(output_path).exists():
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                print(f"✅ Video merged successfully!")
                print(f"📦 File size: {size_mb:.2f} MB")
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr}")
            return False
    
    def merge_with_audio_concat(self,
                               clips: List[str],
                               output_path: str) -> bool:
        """
        Merge videos with audio using concat protocol (preserves audio)
        
        Args:
            clips: List of clip paths
            output_path: Path for output video
        
        Returns:
            True if successful
        """
        print(f"\n🎬 Merging videos with audio preservation...")
        
        # Create intermediate files
        temp_dir = Path("temp_merge")
        temp_dir.mkdir(exist_ok=True)
        
        # Convert all clips to same format first
        intermediate_files = []
        for i, clip in enumerate(clips):
            intermediate = temp_dir / f"clip_{i:03d}.ts"
            cmd = [
                'ffmpeg',
                '-i', clip,
                '-c', 'copy',
                '-bsf:v', 'h264_mp4toannexb',
                '-f', 'mpegts',
                '-y',
                str(intermediate)
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, check=True)
                intermediate_files.append(str(intermediate))
                print(f"  ✓ Converted clip {i+1}/{len(clips)}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to convert clip {i+1}")
                return False
        
        # Concatenate all intermediate files
        concat_str = '|'.join(intermediate_files)
        cmd = [
            'ffmpeg',
            '-i', f'concat:{concat_str}',
            '-c', 'copy',
            '-bsf:a', 'aac_adtstoasc',
            '-y',
            output_path
        ]
        
        try:
            print(f"\n⚙️  Merging all clips...")
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Clean up intermediate files
            for f in intermediate_files:
                Path(f).unlink()
            temp_dir.rmdir()
            
            if Path(output_path).exists():
                size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                print(f"✅ Video merged successfully with audio!")
                print(f"📦 File size: {size_mb:.2f} MB")
                return True
            return False
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e}")
            return False
    
    def get_video_info(self, video_path: str) -> dict:
        """Get video information using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return json.loads(result.stdout)
        except:
            return {}


def merge_clips(clips_source: str,
               output_path: str = "final_video.mp4",
               method: str = "auto",
               resolution: Optional[str] = None,
               fps: Optional[int] = None) -> bool:
    """
    Main function to merge video clips
    
    Args:
        clips_source: Path to filelist.txt or directory containing clips
        output_path: Path for the merged output video
        method: Merging method ('auto', 'concat', 'filter', 'audio')
                'auto' - automatically choose best method
                'concat' - fastest, no re-encoding (requires same format)
                'filter' - re-encode to handle different formats
                'audio' - preserve audio tracks
        resolution: Target resolution for filter method (e.g., "1920x1080")
        fps: Target frame rate for filter method
    
    Returns:
        True if successful
    """
    print("🎥 Starting video merge process...")
    
    merger = VideoMerger(output_path)
    
    # Check FFmpeg installation
    if not merger.check_ffmpeg():
        return False
    
    # Determine if clips_source is a filelist or directory
    source_path = Path(clips_source)
    
    if source_path.is_file() and source_path.name.endswith('.txt'):
        # It's a filelist
        print(f"📋 Loading clips from filelist: {clips_source}")
        clips = merger.load_clips_from_filelist(clips_source)
        filelist_path = clips_source
    elif source_path.is_dir():
        # It's a directory
        print(f"📁 Loading clips from directory: {clips_source}")
        clips = merger.load_clips_from_directory(clips_source)
        
        # Create temporary filelist
        filelist_path = source_path / "temp_filelist.txt"
        with open(filelist_path, 'w') as f:
            for clip in clips:
                f.write(f"file '{Path(clip).resolve()}'\n")
    else:
        print(f"❌ Invalid clips source: {clips_source}")
        return False
    
    if not clips:
        print("❌ No clips found to merge")
        return False
    
    print(f"\n📊 Clips to merge: {len(clips)}")
    for i, clip in enumerate(clips[:5], 1):  # Show first 5
        print(f"  {i}. {Path(clip).name}")
    if len(clips) > 5:
        print(f"  ... and {len(clips) - 5} more")
    
    # Choose merging method
    success = False
    
    if method == "auto" or method == "concat":
        # Try concat demuxer first (fastest)
        print(f"\n🚀 Attempting fast merge (no re-encoding)...")
        success = merger.merge_videos_concat_demuxer(filelist_path, output_path, codec="copy")
        
        if not success and method == "auto":
            print(f"\n⚠️  Fast merge failed. Trying re-encoding method...")
            method = "filter"
    
    if (method == "filter" or (method == "auto" and not success)):
        # Use filter_complex with re-encoding
        if not resolution:
            resolution = "1920x1080"
        if not fps:
            fps = 30
        
        success = merger.merge_videos_filter_complex(clips, output_path, resolution, fps)
    
    elif method == "audio":
        # Use audio-preserving concat
        success = merger.merge_with_audio_concat(clips, output_path)
    
    # Clean up temp filelist if created
    if source_path.is_dir() and filelist_path.exists():
        filelist_path.unlink()
    
    if success:
        print(f"\n🎉 Video merging complete!")
        print(f"📹 Final video: {output_path}")
        
        # Get video duration
        output_path_obj = Path(output_path)
        if output_path_obj.exists():
            size_mb = output_path_obj.stat().st_size / (1024 * 1024)
            print(f"📦 File size: {size_mb:.2f} MB")
            print(f"📂 Location: {output_path_obj.resolve()}")
    else:
        print(f"\n❌ Video merging failed")
    
    return success


def merge_from_ordered_clips(ordered_clips_dir: str = "ordered_clips",
                            output_path: str = "final_video.mp4",
                            method: str = "auto") -> bool:
    """
    Convenience function to merge clips from ordered_clips directory
    
    Args:
        ordered_clips_dir: Directory with ordered clips
        output_path: Output video path
        method: Merging method
    
    Returns:
        True if successful
    """
    ordered_dir = Path(ordered_clips_dir)
    
    # Look for filelist.txt first
    filelist = ordered_dir / "filelist.txt"
    
    if filelist.exists():
        print(f"✅ Found filelist.txt in {ordered_clips_dir}")
        return merge_clips(str(filelist), output_path, method)
    else:
        print(f"📁 Using directory: {ordered_clips_dir}")
        return merge_clips(ordered_clips_dir, output_path, method)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        clips_source = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "final_video.mp4"
        method = sys.argv[3] if len(sys.argv) > 3 else "auto"
        
        success = merge_clips(clips_source, output_path, method)
        
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        print("Usage: python video_merger.py <clips_source> [output_path] [method]")
        print("\nArguments:")
        print("  clips_source  - Path to filelist.txt or directory with clips")
        print("  output_path   - Output video filename (default: final_video.mp4)")
        print("  method        - Merge method: auto|concat|filter|audio (default: auto)")
        print("\nExamples:")
        print("  python video_merger.py ordered_clips/filelist.txt final.mp4")
        print("  python video_merger.py ordered_clips final.mp4 auto")
        print("  python video_merger.py ordered_clips final.mp4 filter")                                                                                                                