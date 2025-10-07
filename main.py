#!/usr/bin/env python3
"""
Main execution file for video dissection and keyframe extraction program
"""

import os
import sys
from pathlib import Path
from video_dissector import VideoDissector
from keyframe_extractor import KeyframeExtractor, process_after_dissection
from vlm_analyzer import analyze_after_keyframe_extraction

def main():
    # Create necessary directories
    input_dir = Path("input_videos")
    output_clips_dir = Path("output_clips")
    output_description_dir = Path("output_description")
    
    input_dir.mkdir(exist_ok=True)
    output_clips_dir.mkdir(exist_ok=True)
    output_description_dir.mkdir(exist_ok=True)
    
    print("🎬 Video Processing Pipeline Starting...")
    print(f"📁 Input videos: {input_dir}")
    print(f"📁 Output clips: {output_clips_dir}")
    print(f"📁 Output keyframes: {output_description_dir}")
    
    # Initialize the video dissector
    dissector = VideoDissector(input_dir, output_clips_dir)
    
    # Get all video files from input directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_dir.glob(f"*{ext}"))
        video_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not video_files:
        print(f"❌ No video files found in {input_dir}")
        print(f"Please place your video files ({', '.join(video_extensions)}) in the {input_dir} folder")
        return
    
    print(f"\n🎥 Found {len(video_files)} video file(s):")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file.name}")
    
    # Track overall statistics
    total_clips_created = 0
    total_keyframes_extracted = 0
    total_keyframes_analyzed = 0
    successful_videos = 0
    failed_videos = 0
    
    # Process each video file
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"🎬 PROCESSING VIDEO {i}/{len(video_files)}: {video_file.name}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Video Dissection
            print(f"\n📹 Step 1: Dissecting video into scenes...")
            clips = dissector.dissect_video(str(video_file))
            
            if clips:
                print(f"✅ Successfully created {len(clips)} clips from {video_file.name}")
                total_clips_created += len(clips)
                
                # Step 2: Keyframe Extraction
                print(f"\n🖼️  Step 2: Extracting keyframes from clips...")
                keyframe_results = process_after_dissection(
                    dissector_output_clips=clips,
                    output_description_dir=str(output_description_dir),
                    keyframe_interval=3.0
                )
                
                # Count total keyframes for this video
                video_keyframes = sum(len(paths) for paths in keyframe_results.values())
                total_keyframes_extracted += video_keyframes
                
                print(f"✅ Extracted {video_keyframes} keyframes from {len(clips)} clips")
                
                # Step 3: VLM Analysis
                print(f"\n🤖 Step 3: Analyzing keyframes with VLM (Ollama Llava)...")
                vlm_results = analyze_after_keyframe_extraction(
                    keyframe_results=keyframe_results,
                    output_description_dir=str(output_description_dir),
                    ollama_host="http://localhost:11434",
                    model_name="llava"
                )
                
                # Count total analyzed keyframes
                video_analyzed_keyframes = sum(
                    len(clip_results) for clip_results in vlm_results.values()
                )
                total_keyframes_analyzed += video_analyzed_keyframes
                
                print(f"✅ Analyzed {video_analyzed_keyframes} keyframes with detailed descriptions")
                successful_videos += 1
                
            else:
                print(f"❌ No clips were created from {video_file.name}")
                failed_videos += 1
                
        except Exception as e:
            print(f"❌ Error processing {video_file.name}: {str(e)}")
            failed_videos += 1
            continue
        
        print(f"✅ Completed processing: {video_file.name}")
    
    # Final Summary Report
    print(f"\n{'='*80}")
    print(f"🎊 FINAL PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Videos processed: {len(video_files)}")
    print(f"✅ Successful: {successful_videos}")
    print(f"❌ Failed: {failed_videos}")
    print(f"🎬 Total clips created: {total_clips_created}")
    print(f"🖼️  Total keyframes extracted: {total_keyframes_extracted}")
    print(f"🤖 Total keyframes analyzed: {total_keyframes_analyzed}")
    print(f"📁 Clips saved to: {output_clips_dir}")
    print(f"📁 Keyframes saved to: {output_description_dir}")
    print(f"📁 VLM analysis saved to: {output_description_dir}/vlm_analysis")
    
    if successful_videos > 0:
        avg_clips_per_video = total_clips_created / successful_videos
        avg_keyframes_per_video = total_keyframes_extracted / successful_videos
        avg_analyzed_per_video = total_keyframes_analyzed / successful_videos
        print(f"📈 Average clips per video: {avg_clips_per_video:.1f}")
        print(f"📈 Average keyframes per video: {avg_keyframes_per_video:.1f}")
        print(f"📈 Average analyzed keyframes per video: {avg_analyzed_per_video:.1f}")
    
    print(f"\n🎉 Processing pipeline completed!")


def process_single_video(video_path, output_clips_dir="output_clips", output_description_dir="output_description"):
    """
    Process a single video file through the complete pipeline
    
    Args:
        video_path (str): Path to the video file
        output_clips_dir (str): Directory to save clips
        output_description_dir (str): Directory to save keyframes
    
    Returns:
        tuple: (clips_created, keyframes_extracted, keyframes_analyzed)
    """
    video_path = Path(video_path)
    output_clips_dir = Path(output_clips_dir)
    output_description_dir = Path(output_description_dir)
    
    # Create output directories
    output_clips_dir.mkdir(exist_ok=True)
    output_description_dir.mkdir(exist_ok=True)
    
    print(f"🎬 Processing single video: {video_path.name}")
    
    # Initialize dissector
    dissector = VideoDissector("dummy", output_clips_dir)
    
    try:
        # Step 1: Video Dissection
        print(f"📹 Step 1: Dissecting video...")
        clips = dissector.dissect_video(str(video_path))
        
        if not clips:
            print(f"❌ No clips created from {video_path.name}")
            return 0, 0, 0
        
        print(f"✅ Created {len(clips)} clips")
        
        # Step 2: Keyframe Extraction
        print(f"🖼️  Step 2: Extracting keyframes...")
        keyframe_results = process_after_dissection(
            dissector_output_clips=clips,
            output_description_dir=str(output_description_dir),
            keyframe_interval=3.0
        )
        
        total_keyframes = sum(len(paths) for paths in keyframe_results.values())
        print(f"✅ Extracted {total_keyframes} keyframes")
        
        # Step 3: VLM Analysis
        print(f"🤖 Step 3: Analyzing keyframes with VLM...")
        # Create dummy keyframe_results dict for single video processing
        keyframe_results = {"single_video": [str(p) for p in output_description_dir.glob("**/*.jpg")]}
        
        vlm_results = analyze_after_keyframe_extraction(
            keyframe_results=keyframe_results,
            output_description_dir=str(output_description_dir),
            ollama_host="http://localhost:11434",
            model_name="llava"
        )
        
        total_analyzed = sum(len(clip_results) for clip_results in vlm_results.values())
        print(f"✅ Analyzed {total_analyzed} keyframes")
        
        return len(clips), total_keyframes, total_analyzed
        
    except Exception as e:
        print(f"❌ Error processing {video_path.name}: {str(e)}")
        return 0, 0


if __name__ == "__main__":
    # Check if a specific video file is provided as argument
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        if Path(video_file).exists():
            clips, keyframes, analyzed = process_single_video(video_file)
            print(f"\n🎉 Single video processing complete!")
            print(f"📊 Clips created: {clips}")
            print(f"🖼️  Keyframes extracted: {keyframes}")
            print(f"🤖 Keyframes analyzed: {analyzed}")
        else:
            print(f"❌ Video file not found: {video_file}")
    else:
        # Run the main batch processing
        main()