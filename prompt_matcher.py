#!/usr/bin/env python3
"""
Script to match user prompts with VLM-analyzed clip descriptions
and determine the best clip ordering based on semantic similarity
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import re


class PromptMatcher:
    def __init__(self, vlm_analysis_dir: str, prompt_file: str):
        """
        Initialize the PromptMatcher
        
        Args:
            vlm_analysis_dir: Directory containing VLM analysis JSON files
            prompt_file: Path to text file containing user prompt
        """
        self.vlm_analysis_dir = Path(vlm_analysis_dir)
        self.prompt_file = Path(prompt_file)
        self.user_prompt = ""
        self.clip_descriptions = {}
        
    def load_user_prompt(self) -> str:
        """Load user prompt from text file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                self.user_prompt = f.read().strip()
            print(f"✅ Loaded user prompt: '{self.user_prompt[:100]}...'")
            return self.user_prompt
        except Exception as e:
            print(f"❌ Error loading prompt file: {e}")
            return ""
    
    def load_vlm_descriptions(self) -> Dict[str, List[Dict]]:
        """Load all VLM analysis JSON files"""
        json_files = list(self.vlm_analysis_dir.glob("*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {self.vlm_analysis_dir}")
            return {}
        
        print(f"📂 Found {len(json_files)} VLM analysis files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    clip_name = json_file.stem
                    self.clip_descriptions[clip_name] = data
            except Exception as e:
                print(f"❌ Error loading {json_file.name}: {e}")
        
        print(f"✅ Loaded descriptions for {len(self.clip_descriptions)} clips")
        return self.clip_descriptions
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        # Remove common stop words and extract meaningful words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were',
                      'this', 'that', 'these', 'those', 'be', 'been', 'being'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords
    
    def calculate_similarity_score(self, prompt: str, description: str) -> float:
        """
        Calculate similarity score between prompt and clip description
        Simple keyword matching approach
        """
        prompt_keywords = set(self.extract_keywords(prompt))
        desc_keywords = set(self.extract_keywords(description))
        
        if not prompt_keywords:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(prompt_keywords & desc_keywords)
        union = len(prompt_keywords | desc_keywords)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def score_clip(self, clip_name: str, clip_data: List[Dict]) -> Tuple[float, Dict]:
        """
        Score a single clip based on how well it matches the user prompt
        
        Returns:
            tuple: (average_score, best_keyframe_info)
        """
        if not clip_data:
            return 0.0, {}
        
        scores = []
        best_keyframe = None
        best_score = 0.0
        
        for keyframe in clip_data:
            description = keyframe.get('description', '')
            score = self.calculate_similarity_score(self.user_prompt, description)
            scores.append(score)
            
            if score > best_score:
                best_score = score
                best_keyframe = keyframe
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        return avg_score, best_keyframe if best_keyframe else {}
    
    def rank_clips(self) -> List[Dict]:
        """
        Rank all clips based on relevance to user prompt
        
        Returns:
            List of dictionaries with clip info sorted by relevance
        """
        ranked_clips = []
        
        for clip_name, clip_data in self.clip_descriptions.items():
            avg_score, best_keyframe = self.score_clip(clip_name, clip_data)
            
            ranked_clips.append({
                'clip_name': clip_name,
                'score': avg_score,
                'best_keyframe': best_keyframe,
                'total_keyframes': len(clip_data)
            })
        
        # Sort by score in descending order
        ranked_clips.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_clips
    
    def generate_clip_order(self, min_score: float = 0.1, max_clips: int = None) -> List[str]:
        """
        Generate ordered list of clip names based on relevance
        
        Args:
            min_score: Minimum similarity score to include clip
            max_clips: Maximum number of clips to include (None = all)
        
        Returns:
            List of clip names in order
        """
        ranked_clips = self.rank_clips()
        
        # Filter by minimum score
        filtered_clips = [c for c in ranked_clips if c['score'] >= min_score]
        
        # Limit number of clips if specified
        if max_clips:
            filtered_clips = filtered_clips[:max_clips]
        
        clip_order = [clip['clip_name'] for clip in filtered_clips]
        
        print(f"\n🎯 Generated clip order ({len(clip_order)} clips):")
        for i, clip in enumerate(filtered_clips, 1):
            print(f"  {i}. {clip['clip_name']} (score: {clip['score']:.3f})")
            if clip['best_keyframe']:
                desc = clip['best_keyframe'].get('description', '')[:80]
                print(f"     Best match: {desc}...")
        
        return clip_order
    
    def save_clip_order(self, output_file: str, min_score: float = 0.1, max_clips: int = None):
        """
        Save the ordered clip list to a JSON file
        
        Args:
            output_file: Path to save the ordered clips JSON
            min_score: Minimum similarity score to include clip
            max_clips: Maximum number of clips to include
        """
        ranked_clips = self.rank_clips()
        filtered_clips = [c for c in ranked_clips if c['score'] >= min_score]
        
        if max_clips:
            filtered_clips = filtered_clips[:max_clips]
        
        output_data = {
            'user_prompt': self.user_prompt,
            'total_clips_analyzed': len(self.clip_descriptions),
            'clips_selected': len(filtered_clips),
            'min_score_threshold': min_score,
            'ordered_clips': filtered_clips
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved clip order to: {output_file}")
        return output_data


def match_prompt_to_clips(prompt_file: str, 
                          vlm_analysis_dir: str,
                          output_file: str = "clip_order.json",
                          min_score: float = 0.1,
                          max_clips: int = None) -> List[str]:
    """
    Main function to match user prompt with clips and generate ordering
    
    Args:
        prompt_file: Path to user prompt text file
        vlm_analysis_dir: Directory with VLM analysis JSON files
        output_file: Where to save the clip order
        min_score: Minimum similarity score threshold
        max_clips: Maximum clips to include (None = all above threshold)
    
    Returns:
        List of ordered clip names
    """
    print("🔍 Starting prompt matching process...")
    
    matcher = PromptMatcher(vlm_analysis_dir, prompt_file)
    
    # Load data
    if not matcher.load_user_prompt():
        print("❌ Failed to load user prompt")
        return []
    
    if not matcher.load_vlm_descriptions():
        print("❌ Failed to load VLM descriptions")
        return []
    
    # Generate and save ordering
    matcher.save_clip_order(output_file, min_score, max_clips)
    clip_order = matcher.generate_clip_order(min_score, max_clips)
    
    print(f"\n✅ Prompt matching complete!")
    return clip_order


if __name__ == "__main__":
    import sys
    
    # Example usage
    if len(sys.argv) > 2:
        prompt_file = sys.argv[1]
        vlm_dir = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "clip_order.json"
        
        clip_order = match_prompt_to_clips(prompt_file, vlm_dir, output_file)
        
    else:
        print("Usage: python prompt_matcher.py <prompt_file> <vlm_analysis_dir> [output_file]")
        print("\nExample:")
        print("  python prompt_matcher.py user_prompt.txt output_description/vlm_analysis clip_order.json")