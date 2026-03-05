import argparse
from similarity_filter import SimilarityFilter
from config import OUTPUT_JSON

def main():
    parser = argparse.ArgumentParser(
        description="Filter video scenes based on semantic similarity to prompt"
    )
    parser.add_argument("--video", required=True, 
                       help="Video stem name, e.g. 'classroom'")
    parser.add_argument("--prompt", required=True, 
                       help="User prompt to match scenes (use '|' to separate multiple clauses)")
    parser.add_argument("--exclude", default=None,
                       help="Scenes to exclude (use '|' to separate multiple exclusions). Example: 'rainy weather | nighttime'")
    parser.add_argument("--out", default=OUTPUT_JSON,
                       help="Output JSON file path")
    parser.add_argument("--mode", default="any", 
                       choices=["any", "all_in_one", "all_distributed"],
                       help="""Match mode:
                       'any' = match ANY clause (OR logic),
                       'all_in_one' = each frame matches ALL clauses (strict AND),
                       'all_distributed' = ALL clauses matched across different frames (recommended for multi-scene prompts)""")
    parser.add_argument("--threshold", type=float, default=None,
                       help="Override similarity threshold (0-1)")
    
    args = parser.parse_args()

    # Initialize filter
    if args.threshold:
        sf = SimilarityFilter(threshold=args.threshold)
        print(f"Using custom threshold: {args.threshold}")
    else:
        sf = SimilarityFilter()
        print(f"Using default threshold: {sf.threshold}")

    print(f"Mode: {args.mode}")
    print(f"Prompt: {args.prompt}")
    if args.exclude:
        print(f"Exclude: {args.exclude}")
    
    
    # Run filtering
    selected = sf.score_and_select(
        args.video, 
        args.prompt, 
        match_mode=args.mode,
        exclude_prompt=args.exclude
    )
    
    if not selected:
        print("No matching scenes found. Try:")
        print("  - Lowering the threshold with --threshold (e.g., --threshold 0.2)")
        print("  - Using 'any' mode instead of 'all_distributed'")
        print("  - Simplifying your prompt")
    else:
        sf.save_output(selected, output_path=args.out)
        print(f"\n✓ Selected {len(selected)} clips saved to {args.out}")

if __name__ == "__main__":
    main()