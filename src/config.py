# config.py
DB_PATH = "output_description/vlm_analysis/vlm_results.db"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
SIMILARITY_THRESHOLD = 0.40   # tune this: 0.65-0.80 typical start
TOP_K_PER_CLIP = 2            # keep top scoring keyframe per clip
OUTPUT_JSON = "selected_clips.json"
