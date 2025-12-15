# test_db_access.py
"""
Diagnostic script to check what videos are in your database
and test access to a specific video.
"""

from db_reader import supabase, get_all_descriptions_for_video, get_keyframes_for_video_prefix
import sys

def list_all_videos():
    """Show all unique video names in database"""
    print("=" * 60)
    print("CHECKING KEYFRAMES TABLE")
    print("=" * 60)
    
    kf_res = supabase.table("keyframes").select("video_name").execute()
    kf_videos = set(row['video_name'] for row in kf_res.data)
    print(f"Found {len(kf_videos)} unique video names in keyframes:")
    for vname in sorted(kf_videos):
        print(f"  - {vname}")
    
    print("\n" + "=" * 60)
    print("CHECKING DESCRIPTIONS TABLE")
    print("=" * 60)
    
    desc_res = supabase.table("descriptions").select("video_name").execute()
    desc_videos = set(row['video_name'] for row in desc_res.data)
    print(f"Found {len(desc_videos)} unique video names in descriptions:")
    for vname in sorted(desc_videos):
        print(f"  - {vname}")
    
    # Find videos in keyframes but not in descriptions
    missing_desc = kf_videos - desc_videos
    if missing_desc:
        print("\n⚠️  WARNING: Videos with keyframes but NO descriptions:")
        for vname in sorted(missing_desc):
            print(f"  - {vname}")
    
    return sorted(kf_videos.union(desc_videos))

def test_video_access(video_stem):
    """Test accessing a specific video"""
    print("\n" + "=" * 60)
    print(f"TESTING ACCESS TO: {video_stem}")
    print("=" * 60)
    
    # Test keyframes
    keyframes = get_keyframes_for_video_prefix(video_stem)
    print(f"\n✓ Keyframes found: {len(keyframes)}")
    if keyframes:
        print(f"  Sample: {keyframes[0]}")
    else:
        print("  ❌ No keyframes found")
    
    # Test descriptions
    descriptions = get_all_descriptions_for_video(video_stem)
    print(f"\n✓ Descriptions found: {len(descriptions)}")
    if descriptions:
        print(f"  Sample: {descriptions[0]}")
    else:
        print("  ❌ No descriptions found")
    
    # Check for partial matches
    print(f"\n🔍 Checking for partial matches with '{video_stem}'...")
    all_videos = supabase.table("keyframes").select("video_name").execute()
    matches = [v['video_name'] for v in all_videos.data if video_stem.lower() in v['video_name'].lower()]
    if matches:
        print(f"  Found {len(matches)} videos containing '{video_stem}':")
        for match in sorted(set(matches))[:10]:  # show first 10 unique matches
            print(f"    - {match}")
    else:
        print(f"  No videos contain '{video_stem}'")

def main():
    print("\n🔍 ZYNC Database Diagnostic Tool\n")
    
    # List all videos
    all_videos = list_all_videos()
    
    # Test specific video if provided
    if len(sys.argv) > 1:
        test_video = sys.argv[1]
        test_video_access(test_video)
    else:
        print("\n" + "=" * 60)
        print("To test a specific video, run:")
        print(f"  python test_db_access.py <video_name>")
        print("=" * 60)

if __name__ == "__main__":
    main()