#!/usr/bin/env python3
"""
Keyframe extractor module for extracting frames from video clips
"""

import cv2
import numpy as np
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip
from pathlib import Path
import os

class KeyframeExtractor:
    def __init__(self, input_dir, output_dir, interval=3.0):
        """
        Initialize the keyframe extractor
        
        Args:
            input_dir (str): Directory containing input video clips
            output_dir (str): Directory to save output keyframes
            interval (float): Interval between keyframes in seconds
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.interval = interval
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_keyframes(self, video_path):
        """
        Extract keyframes from a single video clip
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of saved keyframe file paths
        """
        video_path = Path(video_path)
        video_name = video_path.stem
        
        # Create subdirectory for this clip's keyframes
        clip_output_dir = self.output_dir / video_name
        clip_output_dir.mkdir(exist_ok=True)
        
        print(f"Extracting keyframes from: {video_name}")
        
        try:
            # Load video using moviepy for better format support
            video_clip = VideoFileClip(str(video_path))
            duration = video_clip.duration
            fps = video_clip.fps
            
            print(f"Video duration: {duration:.2f}s, FPS: {fps:.2f}")
            
            keyframe_paths = []
            keyframe_count = 0
            
            # Calculate timestamps for keyframes
            timestamps = np.arange(0, duration, self.interval)
            
            print(f"Extracting {len(timestamps)} keyframes at {self.interval}s intervals...")
            
            for i, timestamp in enumerate(timestamps):
                # Ensure we don't exceed video duration
                if timestamp >= duration:
                    timestamp = duration - 0.1  # Small buffer from end
                
                try:
                    # Extract frame at timestamp
                    frame = video_clip.get_frame(timestamp)
                    
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Generate keyframe filename
                    keyframe_filename = f"{video_name}_keyframe_{i+1:03d}_{timestamp:.1f}s.jpg"
                    keyframe_path = clip_output_dir / keyframe_filename
                    
                    # Save keyframe
                    success = cv2.imwrite(str(keyframe_path), frame_bgr, 
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
                    
                    if success:
                        keyframe_paths.append(str(keyframe_path))
                        keyframe_count += 1
                        print(f"✅ Keyframe {i+1}/{len(timestamps)}: {timestamp:.1f}s -> {keyframe_filename}")
                    else:
                        print(f"❌ Failed to save keyframe at {timestamp:.1f}s")
                        
                except Exception as e:
                    print(f"❌ Error extracting keyframe at {timestamp:.1f}s: {str(e)}")
                    continue
            
            video_clip.close()
            
            print(f"✅ Successfully extracted {keyframe_count} keyframes from {video_name}")
            return keyframe_paths
            
        except Exception as e:
            print(f"❌ Error processing video {video_name}: {str(e)}")
            
            # Fallback to OpenCV method
            print(f"🔄 Trying OpenCV fallback for {video_name}...")
            return self._extract_keyframes_opencv(video_path)
    
    def _extract_keyframes_opencv(self, video_path):
        """
        Fallback method using OpenCV for keyframe extraction
        
        Args:
            video_path (Path): Path to the video file
            
        Returns:
            list: List of saved keyframe file paths
        """
        video_name = video_path.stem
        clip_output_dir = self.output_dir / video_name
        clip_output_dir.mkdir(exist_ok=True)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Could not open video file: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        print(f"OpenCV fallback - Duration: {duration:.2f}s, FPS: {fps:.2f}")
        
        keyframe_paths = []
        keyframe_count = 0
        
        # Calculate frame intervals
        frame_interval = int(fps * self.interval)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract keyframe at intervals
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / fps
                
                # Generate keyframe filename
                keyframe_num = frame_idx // frame_interval + 1
                keyframe_filename = f"{video_name}_keyframe_{keyframe_num:03d}_{timestamp:.1f}s.jpg"
                keyframe_path = clip_output_dir / keyframe_filename
                
                # Save keyframe
                success = cv2.imwrite(str(keyframe_path), frame,
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if success:
                    keyframe_paths.append(str(keyframe_path))
                    keyframe_count += 1
                    print(f"✅ Keyframe {keyframe_num}: {timestamp:.1f}s -> {keyframe_filename}")
                else:
                    print(f"❌ Failed to save keyframe at {timestamp:.1f}s")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"✅ OpenCV fallback extracted {keyframe_count} keyframes from {video_name}")
        return keyframe_paths
    
    def process_clips(self):
        """
        Process all video clips in the input directory
        
        Returns:
            dict: Dictionary mapping clip names to their keyframe paths
        """
        # Find all video files in input directory
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(self.input_dir.glob(f'*{ext}'))
            video_files.extend(self.input_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"❌ No video files found in {self.input_dir}")
            return {}
        
        print(f"🎬 Found {len(video_files)} video clips to process")
        
        results = {}
        
        for i, video_file in enumerate(video_files, 1):
            print(f"\n📹 Processing clip {i}/{len(video_files)}: {video_file.name}")
            
            try:
                keyframe_paths = self.extract_keyframes(video_file)
                results[video_file.stem] = keyframe_paths
                
            except Exception as e:
                print(f"❌ Failed to process {video_file.name}: {str(e)}")
                results[video_file.stem] = []
        
        return results
    
    def extract_from_clips(self, clip_paths=None):
        """
        Extract keyframes from a list of specific clip paths
        
        Args:
            clip_paths (list): List of video clip file paths
            
        Returns:
            dict: Dictionary mapping clip names to their keyframe paths
        """
        if clip_paths is None:
            return self.process_clips()
        
        print(f"🎬 Processing {len(clip_paths)} specified clips")
        
        results = {}
        
        for i, clip_path in enumerate(clip_paths, 1):
            clip_path = Path(clip_path)
            
            if not clip_path.exists():
                print(f"❌ Clip not found: {clip_path}")
                continue
            
            print(f"\n📹 Processing clip {i}/{len(clip_paths)}: {clip_path.name}")
            
            try:
                keyframe_paths = self.extract_keyframes(clip_path)
                results[clip_path.stem] = keyframe_paths
                
            except Exception as e:
                print(f"❌ Failed to process {clip_path.name}: {str(e)}")
                results[clip_path.stem] = []
        
        return results
    
    def generate_summary_report(self, results):
        """
        Generate a summary report of keyframe extraction
        
        Args:
            results (dict): Results from keyframe extraction
        """
        total_clips = len(results)
        total_keyframes = sum(len(paths) for paths in results.values())
        successful_clips = sum(1 for paths in results.values() if paths)
        
        print(f"\n📊 KEYFRAME EXTRACTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total clips processed: {total_clips}")
        print(f"Successfully processed: {successful_clips}")
        print(f"Failed clips: {total_clips - successful_clips}")
        print(f"Total keyframes extracted: {total_keyframes}")
        print(f"Average keyframes per clip: {total_keyframes/successful_clips:.1f}" if successful_clips > 0 else "N/A")
        print(f"Keyframe interval: {self.interval} seconds")
        print(f"Output directory: {self.output_dir}")
        
        if successful_clips > 0:
            print(f"\n📁 Created folders:")
            for clip_name, paths in results.items():
                if paths:
                    folder_path = self.output_dir / clip_name
                    print(f"  {folder_path} ({len(paths)} keyframes)")


# Example usage and integration with video dissector
def process_after_dissection(dissector_output_clips, output_description_dir, keyframe_interval=3.0):
    """
    Process clips created by VideoDissector to extract keyframes
    
    Args:
        dissector_output_clips (list): List of clip paths from VideoDissector
        output_description_dir (str): Directory to save keyframes
        keyframe_interval (float): Interval between keyframes in seconds
    
    Returns:
        dict: Results from keyframe extraction
    """
    print(f"🔗 Starting keyframe extraction after video dissection")
    print(f"Input clips: {len(dissector_output_clips)}")
    print(f"Output directory: {output_description_dir}")
    print(f"Keyframe interval: {keyframe_interval}s")
    
    # Initialize keyframe extractor
    # Note: We pass a dummy input_dir since we're providing specific clip paths
    extractor = KeyframeExtractor(
        input_dir="dummy",  # Not used when providing specific paths
        output_dir=output_description_dir,
        interval=keyframe_interval
    )
    
    # Extract keyframes from dissector output clips
    results = extractor.extract_from_clips(dissector_output_clips)
    
    # Generate summary report
    extractor.generate_summary_report(results)
    
    return results


if __name__ == "__main__":
    # Example standalone usage
    input_clips_dir = "input_clips"
    output_description_dir = "output_description"
    
    extractor = KeyframeExtractor(input_clips_dir, output_description_dir, interval=3.0)
    results = extractor.process_clips()
    extractor.generate_summary_report(results)