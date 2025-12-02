import os
import nltk
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

VLM_DIR = "output_description"
REFERENCE_DIR = "reference_captions"    # <-- Place ground-truth captions here

def extract_description_from_txt(filepath):
    """Extracts the description section from a VLM .txt report."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if "Description:" not in content:
            return None  # No usable text found

        desc_part = content.split("Description:")[1]
        desc = desc_part.strip()
        return desc

    except Exception as e:
        print(f"[ERROR] Failed to parse {filepath}: {e}")
        return None


def load_reference_caption(scene_name):
    """Loads the reference caption for a given scene."""
    ref_file = os.path.join(REFERENCE_DIR, scene_name + ".txt")

    if not os.path.exists(ref_file):
        return None

    with open(ref_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def compute_metrics(pred, ref):
    """Compute BLEU, ROUGE, and cosine similarity."""
    rouge = Rouge()
    bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=SmoothingFunction().method1)

    try:
        rouge_score = rouge.get_scores(pred, ref)[0]["rouge-l"]["f"]
    except:
        rouge_score = 0.0

    vectorizer = TfidfVectorizer().fit([pred, ref])
    vecs = vectorizer.transform([pred, ref])
    cos_sim = cosine_similarity(vecs[0], vecs[1])[0][0]

    return bleu, rouge_score, cos_sim


def evaluate():
    results = []

    for scene_folder in os.listdir(VLM_DIR):
        scene_path = os.path.join(VLM_DIR, scene_folder)

        if not os.path.isdir(scene_path):
            continue

        print(f"\n🔍 Processing: {scene_folder}")

        # Load reference caption
        ref = load_reference_caption(scene_folder)
        if not ref:
            print("⚠️ No reference caption found. Skipping.")
            continue

        # Find .txt report file
        txt_files = [f for f in os.listdir(scene_path) if f.endswith("_report.txt")]
        if not txt_files:
            print("⚠️ No VLM report found. Skipping.")
            continue

        txt_path = os.path.join(scene_path, txt_files[0])
        pred = extract_description_from_txt(txt_path)

        if not pred:
            print("⚠️ No valid description found.")
            continue

        bleu, rouge_l, cos_sim = compute_metrics(pred, ref)

        results.append((scene_folder, bleu, rouge_l, cos_sim))

    print("\n================= FINAL RESULTS =================")
    for r in results:
        print(f"\nScene: {r[0]}")
        print(f"BLEU:     {r[1]:.4f}")
        print(f"ROUGE-L:  {r[2]:.4f}")
        print(f"Cosine:   {r[3]:.4f}")

    if not results:
        print("❌ No captions evaluated.")


if __name__ == "__main__":
    evaluate()
