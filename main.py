#!/usr/bin/env python3
"""
Main execution file for video dissection program
"""

import os
import sys
from pathlib import Path
from video_dissector import VideoDissector

def main():
    # Create necessary directories
    input_dir = Path("input_videos")
    output_dir = Path("output_clips")
    
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize the video dissector
    dissector = VideoDissector(input_dir, output_dir)
    
    # Get all video files from input directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        print(f"Please place your video files (.mp4, .avi, .mov, .mkv, .wmv) in the {input_dir} folder")
        return
    
    print(f"Found {len(video_files)} video file(s):")
    for video_file in video_files:
        print(f"  - {video_file.name}")
    
    # Process each video file
    for video_file in video_files:
        print(f"\nProcessing: {video_file.name}")
        try:
            clips = dissector.dissect_video(str(video_file))
            print(f"Successfully created {len(clips)} clips from {video_file.name}")
        except Exception as e:
            print(f"Error processing {video_file.name}: {str(e)}")
    
    print(f"\nAll clips saved to: {output_dir}")

if __name__ == "__main__":
    main()