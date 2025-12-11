# similarity_filter.py
from sentence_transformers import SentenceTransformer, util
from config import EMBED_MODEL, SIMILARITY_THRESHOLD, TOP_K_PER_CLIP
from db_reader import get_all_descriptions_for_video
import json

class SimilarityFilter:
    def __init__(self, model_name=EMBED_MODEL, threshold=SIMILARITY_THRESHOLD):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold

    def score_and_select(self, video_stem, user_prompt):
        """
        Returns list of dicts like:
        [
          {
            "video_name": str,    # e.g. "classroom_subclip_1"
            "frame_index": int,   # keyframe index (0, 30, 60, ...)
            "image_path": str,    # path to keyframe image (from keyframes table)
            "clip_id": int,       # subclip id from DB
            "score": float        # similarity score
          },
          ...
        ]

        One subclip can have multiple keyframes; we keep up to TOP_K_PER_CLIP
        keyframes per subclip based on score.
        """
        rows = get_all_descriptions_for_video(video_stem)
        if not rows:
            print(f"No descriptions found for video_stem={video_stem}")
            return []

        # rows: (video_name, frame_index, frame_path, description, clip_id)
        descriptions = [row[3] if row[3] is not None else "" for row in rows]

        # encode
        desc_embs = self.model.encode(descriptions, convert_to_tensor=True)
        prompt_emb = self.model.encode(user_prompt, convert_to_tensor=True)

        sims = util.cos_sim(prompt_emb, desc_embs)[0].cpu().numpy()

        selected = []
        for i, score in enumerate(sims):
            if float(score) >= self.threshold:
                video_name, frame_index, image_path, desc, clip_id = rows[i]
                selected.append({
                    "video_name": video_name,
                    "frame_index": frame_index,
                    "image_path": image_path,
                    "clip_id": clip_id,
                    "score": float(score)
                })

        if not selected:
            return []

        # group by subclip (each clip_id / video_name = one subclip)
        grouped = {}
        for item in selected:
            cid = item["clip_id"]  # or item["video_name"]
            if cid not in grouped:
                grouped[cid] = []
            grouped[cid].append(item)

        deduped = []
        for cid, items in grouped.items():
            # keep top-K keyframes per subclip
            items_sorted = sorted(items, key=lambda x: x["score"], reverse=True)
            top_items = items_sorted[:TOP_K_PER_CLIP]
            deduped.extend(top_items)

        # globally sort by score
        deduped = sorted(deduped, key=lambda x: x["score"], reverse=True)
        return deduped

    def save_output(self, selected, output_path="selected_clips.json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        return output_path
