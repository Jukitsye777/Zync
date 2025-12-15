from sentence_transformers import SentenceTransformer, util
from config import EMBED_MODEL, SIMILARITY_THRESHOLD, TOP_K_PER_CLIP
from db_reader import get_all_descriptions_for_video
import json
import numpy as np


class SimilarityFilter:
    def __init__(self, model_name=EMBED_MODEL, threshold=SIMILARITY_THRESHOLD):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def split_prompt(self, prompt: str):
        """Split prompt into semantic clauses using semicolon separator."""
        clauses = [c.strip() for c in prompt.split(";") if c.strip()]
        return clauses if clauses else [prompt.strip()]

    def score_and_select(self, video_stem, user_prompt, match_mode="any"):
        """
        Match modes:
        - "any": Return frames matching ANY clause (OR logic)
        - "all_in_one": Each frame must match ALL clauses (strict AND)
        - "all_distributed": ALL clauses must be matched, but across different frames (recommended)
        """
        rows = get_all_descriptions_for_video(video_stem)
        if not rows:
            print(f"No descriptions found for video_stem={video_stem}")
            return []

        descriptions = [row[3] if row[3] else "" for row in rows]
        desc_embs = self.model.encode(descriptions, convert_to_tensor=True)

        clauses = self.split_prompt(user_prompt)
        print(f"Processing {len(clauses)} clauses: {clauses}")
        
        clause_embs = self.model.encode(clauses, convert_to_tensor=True)
        sim_matrix = util.cos_sim(clause_embs, desc_embs).cpu().numpy()

        selected = []

        if match_mode == "any":
            # OR logic: frame matches if it satisfies ANY clause
            for i in range(len(descriptions)):
                max_score = sim_matrix[:, i].max()
                if float(max_score) >= self.threshold:
                    video_name, frame_index, frame_path, desc, clip_id = rows[i]
                    # Find which clause matched
                    best_clause_idx = sim_matrix[:, i].argmax()
                    selected.append({
                        "video_name": video_name,
                        "frame_index": frame_index,
                        "image_path": frame_path,
                        "clip_id": clip_id,
                        "score": float(max_score),
                        "matched_clause": clauses[best_clause_idx],
                        "clause_index": int(best_clause_idx)
                    })

        elif match_mode == "all_in_one":
            # Strict AND: each frame must match ALL clauses
            for i in range(len(descriptions)):
                min_score = sim_matrix[:, i].min()
                if float(min_score) >= self.threshold:
                    video_name, frame_index, frame_path, desc, clip_id = rows[i]
                    selected.append({
                        "video_name": video_name,
                        "frame_index": frame_index,
                        "image_path": frame_path,
                        "clip_id": clip_id,
                        "score": float(min_score)
                    })

        elif match_mode == "all_distributed":
            # ALL clauses must be satisfied, but across different frames
            clause_coverage = {i: [] for i in range(len(clauses))}
            
            # For each frame, find its best matching clause
            for i in range(len(descriptions)):
                scores = sim_matrix[:, i]
                best_clause_idx = scores.argmax()
                best_score = scores[best_clause_idx]
                
                if float(best_score) >= self.threshold:
                    video_name, frame_index, frame_path, desc, clip_id = rows[i]
                    clause_coverage[best_clause_idx].append({
                        "video_name": video_name,
                        "frame_index": frame_index,
                        "image_path": frame_path,
                        "clip_id": clip_id,
                        "score": float(best_score),
                        "matched_clause": clauses[best_clause_idx],
                        "clause_index": int(best_clause_idx)
                    })
            
            # Check if all clauses are covered
            uncovered = [i for i, matches in clause_coverage.items() if not matches]
            if uncovered:
                print(f"Warning: Clauses not matched: {[clauses[i] for i in uncovered]}")
                return []
            
            # Collect all matches
            for matches in clause_coverage.values():
                selected.extend(matches)

        if not selected:
            print(f"No frames matched the criteria with threshold {self.threshold}")
            return []

        # Deduplicate per clip, keeping top scores
        grouped = {}
        for item in selected:
            cid = item["clip_id"]
            grouped.setdefault(cid, []).append(item)

        final = []
        for cid, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            final.extend(items_sorted[:TOP_K_PER_CLIP])

        print(f"Selected {len(final)} frames from {len(grouped)} clips")
        return sorted(final, key=lambda x: x["score"], reverse=True)

    def save_output(self, selected, output_path="selected_clips.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        print(f"Output saved to {output_path}")
        return output_path