#!/usr/bin/env python3
"""
Video dissector module for scene detection and clip generation
"""

import cv2
import numpy as np
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
from pathlib import Path
import os

class VideoDissector:
    def __init__(self, input_dir, output_dir, threshold=5.0, min_scene_length=2.0):
        """
        Initialize the video dissector
        
        Args:
            input_dir (str): Directory containing input videos
            output_dir (str): Directory to save output clips
            threshold (float): Scene change detection threshold (0-100)
            min_scene_length (float): Minimum scene length in seconds
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
    
    def detect_scenes(self, video_path):
        """
        Detect scene changes in a video using frame difference analysis
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of scene change timestamps
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        scene_changes = [0.0]  # Start with beginning of video
        prev_frame = None
        frame_idx = 0
        
        print(f"Analyzing {frame_count} frames at {fps:.2f} FPS...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale for comparison
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray_frame)
                mean_diff = np.mean(diff)
                
                # If difference exceeds threshold, mark as scene change
                if mean_diff > self.threshold:
                    timestamp = frame_idx / fps
                    
                    # Only add if it's far enough from the last scene change
                    if not scene_changes or (timestamp - scene_changes[-1]) >= self.min_scene_length:
                        scene_changes.append(timestamp)
                        print(f"Scene change detected at {timestamp:.2f}s (diff: {mean_diff:.2f})")
            
            prev_frame = gray_frame
            frame_idx += 1
            
            # Progress indicator
            if frame_idx % 30 == 0:
                progress = (frame_idx / frame_count) * 100
                print(f"Progress: {progress:.1f}%", end='\r')
        
        cap.release()
        
        # Add end of video
        duration = frame_count / fps
        scene_changes.append(duration)
        
        print(f"\nFound {len(scene_changes) - 1} scenes")
        return scene_changes
    
    def create_clips(self, video_path, scene_timestamps):
        """
        Create video clips based on scene timestamps with error handling
        
        Args:
            video_path (str): Path to the original video
            scene_timestamps (list): List of scene change timestamps
            
        Returns:
            list: List of created clip file paths
        """
        video_clip = VideoFileClip(video_path)
        video_name = Path(video_path).stem
        created_clips = []
        
        print(f"\nCreating {len(scene_timestamps) - 1} clips from detected scenes...")
        
        for i in range(len(scene_timestamps) - 1):
            start_time = scene_timestamps[i]
            end_time = scene_timestamps[i + 1]
            
            # Skip very short clips
            if end_time - start_time < self.min_scene_length:
                print(f"Skipping short clip {i+1}: {start_time:.1f}s - {end_time:.1f}s (too short: {end_time - start_time:.1f}s)")
                continue
            
            try:
                # Create clip
                clip = video_clip.subclip(start_time, end_time)
                
                # Generate output filename
                clip_filename = f"{video_name}_scene_{i+1:03d}_{start_time:.1f}s-{end_time:.1f}s.mp4"
                clip_path = self.output_dir / clip_filename
                
                # Write clip to file with error handling
                print(f"Creating clip {i+1}/{len(scene_timestamps) - 1}: {start_time:.1f}s - {end_time:.1f}s ({end_time - start_time:.1f}s)")
                
                # Use more robust encoding settings
                clip.write_videofile(
                    str(clip_path),
                    verbose=False,
                    logger=None,
                    codec='libx264',
                    audio_codec='aac',
                    temp_audiofile='temp-audio.m4a',
                    remove_temp=True
                )
                
                clip.close()
                created_clips.append(str(clip_path))
                print(f"✅ Successfully created: {clip_filename}")
                
            except Exception as e:
                print(f"❌ Error creating clip {i+1}: {str(e)}")
                # Try alternative method without audio
                try:
                    print(f"Retrying clip {i+1} without audio...")
                    clip = video_clip.subclip(start_time, end_time)
                    clip_filename = f"{video_name}_scene_{i+1:03d}_{start_time:.1f}s-{end_time:.1f}s_no_audio.mp4"
                    clip_path = self.output_dir / clip_filename
                    
                    clip.write_videofile(
                        str(clip_path),
                        verbose=False,
                        logger=None,
                        codec='libx264',
                        audio=False  # Skip audio to avoid errors
                    )
                    
                    clip.close()
                    created_clips.append(str(clip_path))
                    print(f"✅ Successfully created (no audio): {clip_filename}")
                    
                except Exception as e2:
                    print(f"❌ Failed completely for clip {i+1}: {str(e2)}")
                    continue
        
        video_clip.close()
        return created_clips
    
    def dissect_video(self, video_path):
        """
        Main function to dissect a video into clips
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of created clip file paths
        """
        print(f"Starting dissection of: {video_path}")
        
        # Detect scene changes
        scene_timestamps = self.detect_scenes(video_path)
        
        print(f"\nScene timestamps detected:")
        for i, timestamp in enumerate(scene_timestamps):
            print(f"  Scene {i}: {timestamp:.2f}s")
        
        if len(scene_timestamps) <= 2:
            print("No scene changes detected. Creating single clip.")
            # Get video duration properly
            temp_clip = VideoFileClip(video_path)
            duration = temp_clip.duration
            temp_clip.close()
            scene_timestamps = [0.0, duration]
        
        # Create clips for all detected scenes
        clips = self.create_clips(video_path, scene_timestamps)
        
        print(f"\n🎬 Video dissection complete!")
        print(f"📊 Expected clips: {len(scene_timestamps) - 1}")
        print(f"✅ Successfully created: {len(clips)} clips")
        
        if len(clips) < len(scene_timestamps) - 1:
            print(f"⚠️  Some clips failed to create. Check error messages above.")
        
        return clips