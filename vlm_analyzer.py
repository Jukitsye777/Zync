#!/usr/bin/env python3
"""
VLM (Vision Language Model) analyzer for generating detailed descriptions of keyframes
using Ollama Llava
"""

import requests
import json
import base64
from pathlib import Path
import time
import csv
from datetime import datetime
import os

class VLMAnalyzer:
    def __init__(self, output_dir, ollama_host="http://localhost:11434", model_name="llava"):
        """
        Initialize the VLM analyzer
        
        Args:
            output_dir (str): Directory containing keyframes to analyze
            ollama_host (str): Ollama server URL
            model_name (str): Name of the Llava model to use
        """
        self.output_dir = Path(output_dir)
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = model_name
        self.analysis_results = {}
        
        # Create analysis output directory
        self.analysis_output_dir = self.output_dir / "vlm_analysis"
        self.analysis_output_dir.mkdir(exist_ok=True)
        
        print(f"🤖 VLM Analyzer initialized")
        print(f"📁 Keyframes directory: {self.output_dir}")
        print(f"🔗 Ollama host: {self.ollama_host}")
        print(f"🧠 Model: {self.model_name}")
    
    def check_ollama_connection(self):
        """
        Check if Ollama server is running and model is available
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=10)
            if response.status_code != 200:
                print(f"❌ Ollama server not responding at {self.ollama_host}")
                return False
            
            # Check if the model is available
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if not any(self.model_name in name for name in model_names):
                print(f"❌ Model '{self.model_name}' not found in Ollama")
                print(f"Available models: {model_names}")
                print(f"To install Llava, run: ollama pull {self.model_name}")
                return False
            
            print(f"✅ Ollama connection successful, model '{self.model_name}' available")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to connect to Ollama: {str(e)}")
            print(f"Make sure Ollama is running: ollama serve")
            return False
    
    def encode_image_to_base64(self, image_path):
        """
        Encode image to base64 for Ollama API
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding image {image_path}: {str(e)}")
            return None
    
    def analyze_keyframe(self, image_path, custom_prompt=None):
        """
        Send keyframe to Llava for analysis
        
        Args:
            image_path (str): Path to the keyframe image
            custom_prompt (str): Custom prompt for analysis
            
        Returns:
            dict: Analysis result with description and metadata
        """
        if custom_prompt is None:
            custom_prompt = """Analyze this keyframe from a video in detail. Provide a comprehensive description including:

1. **Scene Overview**: What is happening in this frame?
2. **Objects and Elements**: List and describe all visible objects, people, animals, text, or UI elements
3. **Setting and Environment**: Describe the location, time of day, weather, lighting conditions
4. **Actions and Movement**: What actions or movements can be inferred?
5. **Colors and Visual Style**: Dominant colors, visual style, quality, composition
6. **Text Content**: Any visible text, subtitles, or written content
7. **Technical Aspects**: Image quality, resolution, any technical issues
8. **Context Clues**: Any clues about what type of content this might be (movie, tutorial, game, etc.)

Be specific, detailed, and objective in your description."""

        # Encode image
        base64_image = self.encode_image_to_base64(image_path)
        if not base64_image:
            return None
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "prompt": custom_prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Lower temperature for more consistent descriptions
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            print(f"🧠 Analyzing: {Path(image_path).name}...", end=" ")
            start_time = time.time()
            
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json=payload,
                timeout=120  # 2 minute timeout for complex images
            )
            
            if response.status_code == 200:
                result = response.json()
                analysis_time = time.time() - start_time
                
                analysis_result = {
                    'image_path': str(image_path),
                    'image_name': Path(image_path).name,
                    'description': result.get('response', 'No description generated'),
                    'analysis_time': analysis_time,
                    'timestamp': datetime.now().isoformat(),
                    'model_used': self.model_name,
                    'prompt_used': custom_prompt,
                    'success': True
                }
                
                print(f"✅ ({analysis_time:.1f}s)")
                return analysis_result
            else:
                print(f"❌ API Error: {response.status_code}")
                return {
                    'image_path': str(image_path),
                    'image_name': Path(image_path).name,
                    'description': f'Analysis failed: HTTP {response.status_code}',
                    'analysis_time': 0,
                    'timestamp': datetime.now().isoformat(),
                    'model_used': self.model_name,
                    'success': False
                }
                
        except requests.exceptions.Timeout:
            print(f"❌ Timeout")
            return {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'description': 'Analysis failed: Request timeout',
                'analysis_time': 0,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'success': False
            }
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            return {
                'image_path': str(image_path),
                'image_name': Path(image_path).name,
                'description': f'Analysis failed: {str(e)}',
                'analysis_time': 0,
                'timestamp': datetime.now().isoformat(),
                'model_used': self.model_name,
                'success': False
            }
    
    def analyze_clip_keyframes(self, clip_dir):
        """
        Analyze all keyframes in a clip directory
        
        Args:
            clip_dir (Path): Path to clip directory containing keyframes
            
        Returns:
            list: List of analysis results for all keyframes
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        keyframes = []
        
        for ext in image_extensions:
            keyframes.extend(clip_dir.glob(f'*{ext}'))
            keyframes.extend(clip_dir.glob(f'*{ext.upper()}'))
        
        if not keyframes:
            print(f"❌ No keyframes found in {clip_dir}")
            return []
        
        # Sort keyframes by name for proper order
        keyframes.sort()
        
        print(f"🎬 Analyzing {len(keyframes)} keyframes in {clip_dir.name}")
        
        clip_results = []
        
        for i, keyframe_path in enumerate(keyframes, 1):
            print(f"  📸 Keyframe {i}/{len(keyframes)}: ", end="")
            
            result = self.analyze_keyframe(keyframe_path)
            if result:
                clip_results.append(result)
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
        
        return clip_results
    
    def save_analysis_results(self, clip_name, results):
        """
        Save analysis results to various formats
        
        Args:
            clip_name (str): Name of the clip
            results (list): List of analysis results
        """
        if not results:
            return
        
        clip_analysis_dir = self.analysis_output_dir / clip_name
        clip_analysis_dir.mkdir(exist_ok=True)
        
        # Save as JSON (detailed format)
        json_file = clip_analysis_dir / f"{clip_name}_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save as CSV (tabular format)
        csv_file = clip_analysis_dir / f"{clip_name}_analysis.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if results:
                fieldnames = ['image_name', 'description', 'analysis_time', 'timestamp', 'success']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    writer.writerow({
                        'image_name': result['image_name'],
                        'description': result['description'],
                        'analysis_time': result['analysis_time'],
                        'timestamp': result['timestamp'],
                        'success': result['success']
                    })
        
        # Save as readable text report
        txt_file = clip_analysis_dir / f"{clip_name}_report.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"VLM Analysis Report for {clip_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total keyframes analyzed: {len(results)}\n")
            f.write(f"Successful analyses: {sum(1 for r in results if r['success'])}\n")
            f.write(f"Model used: {self.model_name}\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"KEYFRAME {i}: {result['image_name']}\n")
                f.write(f"{'-'*50}\n")
                f.write(f"Analysis Time: {result['analysis_time']:.1f}s\n")
                f.write(f"Status: {'✅ Success' if result['success'] else '❌ Failed'}\n")
                f.write(f"Description:\n{result['description']}\n\n")
        
        print(f"📄 Analysis saved to {clip_analysis_dir}")
    
    def analyze_all_keyframes(self):
        """
        Analyze all keyframes in all clip directories
        
        Returns:
            dict: Complete analysis results organized by clip
        """
        if not self.check_ollama_connection():
            return {}
        
        # Find all clip directories
        clip_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name != "vlm_analysis"]
        
        if not clip_dirs:
            print(f"❌ No clip directories found in {self.output_dir}")
            return {}
        
        print(f"🎬 Found {len(clip_dirs)} clip directories to analyze")
        
        all_results = {}
        total_keyframes_analyzed = 0
        total_successful = 0
        start_time = time.time()
        
        for i, clip_dir in enumerate(clip_dirs, 1):
            print(f"\n📁 Processing clip directory {i}/{len(clip_dirs)}: {clip_dir.name}")
            
            clip_results = self.analyze_clip_keyframes(clip_dir)
            
            if clip_results:
                all_results[clip_dir.name] = clip_results
                self.save_analysis_results(clip_dir.name, clip_results)
                
                successful_count = sum(1 for r in clip_results if r['success'])
                total_keyframes_analyzed += len(clip_results)
                total_successful += successful_count
                
                print(f"✅ Completed {clip_dir.name}: {successful_count}/{len(clip_results)} successful")
            else:
                print(f"❌ No keyframes analyzed in {clip_dir.name}")
        
        # Generate overall summary
        total_time = time.time() - start_time
        self.generate_overall_summary(all_results, total_keyframes_analyzed, total_successful, total_time)
        
        return all_results
    
    def generate_overall_summary(self, all_results, total_keyframes, total_successful, total_time):
        """
        Generate overall summary report
        
        Args:
            all_results (dict): All analysis results
            total_keyframes (int): Total keyframes analyzed
            total_successful (int): Total successful analyses
            total_time (float): Total processing time
        """
        summary_file = self.analysis_output_dir / "analysis_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("VLM ANALYSIS COMPLETE SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model used: {self.model_name}\n")
            f.write(f"Total processing time: {total_time/60:.1f} minutes\n\n")
            
            f.write("STATISTICS:\n")
            f.write(f"- Clip directories processed: {len(all_results)}\n")
            f.write(f"- Total keyframes analyzed: {total_keyframes}\n")
            f.write(f"- Successful analyses: {total_successful}\n")
            f.write(f"- Failed analyses: {total_keyframes - total_successful}\n")
            f.write(f"- Success rate: {(total_successful/total_keyframes*100):.1f}%\n")
            f.write(f"- Average time per keyframe: {total_time/total_keyframes:.1f}s\n\n")
            
            f.write("CLIP BREAKDOWN:\n")
            for clip_name, results in all_results.items():
                successful = sum(1 for r in results if r['success'])
                f.write(f"- {clip_name}: {successful}/{len(results)} keyframes\n")
        
        print(f"\n📊 ANALYSIS COMPLETE!")
        print(f"📄 Summary saved to: {summary_file}")
        print(f"📁 All results saved to: {self.analysis_output_dir}")


def analyze_after_keyframe_extraction(keyframe_results, output_description_dir, ollama_host="http://localhost:11434", model_name="llava"):
    """
    Analyze keyframes using VLM after keyframe extraction completes
    
    Args:
        keyframe_results (dict): Results from keyframe extraction
        output_description_dir (str): Directory containing keyframes
        ollama_host (str): Ollama server URL
        model_name (str): Llava model name
    
    Returns:
        dict: VLM analysis results
    """
    print(f"\n🤖 Starting VLM analysis after keyframe extraction")
    print(f"📊 Clips to analyze: {len(keyframe_results)}")
    print(f"📊 Total keyframes: {sum(len(paths) for paths in keyframe_results.values())}")
    
    # Initialize VLM analyzer
    analyzer = VLMAnalyzer(
        output_dir=output_description_dir,
        ollama_host=ollama_host,
        model_name=model_name
    )
    
    # Analyze all keyframes
    vlm_results = analyzer.analyze_all_keyframes()
    
    return vlm_results


if __name__ == "__main__":
    # Example standalone usage
    output_description_dir = "output_description"
    
    analyzer = VLMAnalyzer(output_description_dir)
    results = analyzer.analyze_all_keyframes()
    
    print(f"\n🎉 Analysis complete! Check the results in {analyzer.analysis_output_dir}")