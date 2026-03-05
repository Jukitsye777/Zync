from sentence_transformers import SentenceTransformer, util
from config import EMBED_MODEL, SIMILARITY_THRESHOLD, TOP_K_PER_CLIP, EXCLUDE_THRESHOLD
from db_reader import get_all_descriptions_for_video
import json
import numpy as np


class SimilarityFilter:
    def __init__(self, model_name=EMBED_MODEL, threshold=SIMILARITY_THRESHOLD):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.exclude_threshold = EXCLUDE_THRESHOLD

    def split_prompt(self, prompt: str):
        """
        Split prompt into semantic clauses.
        Supports multiple delimiters (checked in order):
        1. " | " (pipe with spaces) - explicit separator
        2. ";" (semicolon) - fallback
        3. No delimiter - treat as single clause
        """
        # Try pipe separator first (most explicit)
        if " | " in prompt:
            clauses = [c.strip() for c in prompt.split("|") if c.strip()]
            return clauses if clauses else [prompt.strip()]
        
        # Fallback to semicolon
        elif ";" in prompt:
            clauses = [c.strip() for c in prompt.split(";") if c.strip()]
            return clauses if clauses else [prompt.strip()]
        
        # Single clause
        else:
            return [prompt.strip()]

    def score_and_select(self, video_stem, user_prompt, match_mode="any", exclude_prompt=None):
        """
        Match modes:
        - "any": Return frames matching ANY clause (OR logic)
        - "all_in_one": Each frame must match ALL clauses (strict AND)
        - "all_distributed": ALL clauses must be matched, but across different frames (recommended)
        
        use_query_expansion: If True, automatically expands queries with synonyms
        exclude_prompt: Optional string of scenes to exclude (e.g., "rainy weather | nighttime")
        """
        rows = get_all_descriptions_for_video(video_stem)
        if not rows:
            print(f"No descriptions found for video_stem={video_stem}")
            return []

        descriptions = [row[3] if row[3] else "" for row in rows]
        desc_embs = self.model.encode(descriptions, convert_to_tensor=True)

        clauses = self.split_prompt(user_prompt)
        
        
        print(f"Processing {len(clauses)} clauses: {clauses[:3]}{'...' if len(clauses) > 3 else ''}")
        
        clause_embs = self.model.encode(clauses, convert_to_tensor=True)
        sim_matrix = util.cos_sim(clause_embs, desc_embs).cpu().numpy()
        
        # Handle exclusions if provided
        exclude_matrix = None
        if exclude_prompt:
            exclude_clauses = self.split_prompt(exclude_prompt)
            print(f"Processing {len(exclude_clauses)} exclusion clauses: {exclude_clauses}")
            exclude_embs = self.model.encode(exclude_clauses, convert_to_tensor=True)
            exclude_matrix = util.cos_sim(exclude_embs, desc_embs).cpu().numpy()

        selected = []

        if match_mode == "any":
            # OR logic: frame matches if it satisfies ANY clause
            for i in range(len(descriptions)):
                max_score = sim_matrix[:, i].max()
                
                # Check exclusions
                #if exclude_matrix is not None:
                  #  exclude_score = exclude_matrix[:, i].max()
                   # if exclude_score >= self.threshold:
                        # This frame matches an exclusion - skip it
                     #   continue
                exclude_score = 0

                if exclude_matrix is not None:
                    exclude_score = exclude_matrix[:, i].max()

                # Apply penalty
                final_score = max_score - 0.6 * exclude_score

                if float(final_score) >= self.threshold:
                #if float(max_score) >= self.threshold:
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
                # Check exclusions first
                if exclude_matrix is not None:
                    exclude_score = exclude_matrix[:, i].max()
                    if exclude_score >= self.exclude_threshold:
                        continue
                
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
                # Check exclusions first
                if exclude_matrix is not None:
                    exclude_score = exclude_matrix[:, i].max()
                    if exclude_score >= self.threshold:
                        continue
                
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