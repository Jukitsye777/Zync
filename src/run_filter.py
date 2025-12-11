# run_filter.py
import argparse
from similarity_filter import SimilarityFilter
from config import OUTPUT_JSON

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Video stem name, e.g. 'classroom'")
    parser.add_argument("--prompt", required=True, help="User prompt to match scenes")
    parser.add_argument("--out", default=OUTPUT_JSON)
    args = parser.parse_args()

    sf = SimilarityFilter()
    selected = sf.score_and_select(args.video, args.prompt)
    sf.save_output(selected, output_path=args.out)
    print(f"Selected {len(selected)} clips saved to {args.out}")

if __name__ == "__main__":
    main()
